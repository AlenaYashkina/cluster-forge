from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import cluster_core


class AssignPayload(BaseModel):
    root: str
    image_path: str
    cluster: str
    manual_only: bool = False


class DeletePayload(BaseModel):
    root: str
    image_path: str


class AutoMovePayload(BaseModel):
    root: str
    threshold: float = 0.9
    min_samples: int = 1
    image_path: Optional[str] = None


class UndoPayload(BaseModel):
    root: str


def _resolve_root(root_raw: str) -> Path:
    root_path = cluster_core.resolve_input_path(root_raw)
    if root_path is None or not root_path.is_dir():
        raise HTTPException(status_code=400, detail="Cannot resolve provided root folder")
    return root_path.resolve()


def _normalize_path(raw: str) -> Path:
    candidate = Path(raw)
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Image path does not exist")
    return candidate.resolve()


app = FastAPI(
    title="Manual Cluster API",
    description="Backend helpers extracted from the legacy Streamlit/CLI helpers.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def status() -> Dict[str, Any]:
    return {"status": "ok", "info": "Use /session to drive the React UI."}


@app.get("/session")
def session(
    root: str = Query(..., description="Root directory to inspect"),
    manual_only: bool = Query(False, description="Skip CLIP suggestions/auto-move"),
    threshold: float = Query(0.9, description="Similarity threshold for auto move"),
    min_samples: int = Query(5, description="Minimum samples required for auto move"),
) -> Dict[str, Any]:
    resolved_root = _resolve_root(root)
    history = cluster_core.load_history(resolved_root)
    main_state = cluster_core.load_main_state(resolved_root)
    known_clusters = sorted(cluster_core.gather_known_clusters(history, main_state))
    queue = cluster_core.list_unlabeled_images(resolved_root, known_clusters)
    auto_move_result = {}
    if queue and not manual_only:
        queue, known_clusters, auto_move_result, history, main_state = cluster_core.auto_move_queue(
            resolved_root,
            queue,
            known_clusters,
            threshold=threshold,
            min_samples=min_samples,
        )
    stats = cluster_core.compute_stats(history, main_state, len(queue))
    suggestion = None
    if queue and not manual_only:
        suggestion = cluster_core.suggest_for_image(resolved_root, queue[0])
    return {
        "root": str(resolved_root),
        "queue": queue,
        "known_clusters": known_clusters,
        "stats": stats,
        "auto_move": auto_move_result,
        "suggestion": suggestion,
    }


@app.post("/assign")
def assign(payload: AssignPayload) -> Dict[str, Any]:
    resolved_root = _resolve_root(payload.root)
    image_path = _normalize_path(payload.image_path)
    if not cluster_core.is_within_root(image_path, resolved_root):
        raise HTTPException(status_code=400, detail="Image is not within provided root")
    cluster = payload.cluster.strip()
    if not cluster:
        raise HTTPException(status_code=400, detail="Cluster name is required")
    embedding: Optional[List[float]] = None
    if not payload.manual_only:
        try:
            embedding = cluster_core.embed_image(image_path)
        except Exception:
            embedding = None
    try:
        destination = cluster_core.move_into_cluster(image_path, cluster)
    except (OSError, shutil.Error) as move_err:
        raise HTTPException(status_code=500, detail=f"Assign error: {move_err}") from move_err
    cluster_core.append_log(
        resolved_root,
        {
            "ts": datetime.now().isoformat(),
            "src": str(image_path),
            "dst": str(destination),
            "cluster": cluster,
            "note": "react-ui",
        },
    )
    cluster_core.record_assignment(resolved_root, cluster, embedding, mode="manual")
    return {"status": "ok", "path_after": str(destination)}


@app.post("/delete")
def delete(payload: DeletePayload) -> Dict[str, Any]:
    resolved_root = _resolve_root(payload.root)
    image_path = _normalize_path(payload.image_path)
    if not cluster_core.is_within_root(image_path, resolved_root):
        raise HTTPException(status_code=400, detail="Image is not within provided root")
    try:
        cluster_core.delete_file(image_path)
    except (OSError, PermissionError) as delete_err:
        raise HTTPException(status_code=500, detail=f"Delete error: {delete_err}") from delete_err
    cluster_core.append_log(
        resolved_root,
        {
            "ts": datetime.now().isoformat(),
            "src": str(image_path),
            "dst": "",
            "cluster": "__deleted__",
            "note": "react-ui",
        },
    )
    cluster_core.record_deletion(resolved_root)
    return {"status": "ok"}


@app.post("/auto-move")
def auto_move(payload: AutoMovePayload) -> Dict[str, Any]:
    resolved_root = _resolve_root(payload.root)
    history = cluster_core.load_history(resolved_root)
    main_state = cluster_core.load_main_state(resolved_root)
    known_clusters = cluster_core.gather_known_clusters(history, main_state)
    queue = cluster_core.list_unlabeled_images(resolved_root, known_clusters)
    try:
        target = None
        if payload.image_path:
            target = _normalize_path(payload.image_path)
        elif queue:
            target = Path(queue[0])
        if target is None:
            raise HTTPException(status_code=400, detail="Queue is empty")
        if not cluster_core.is_within_root(target, resolved_root):
            raise HTTPException(status_code=400, detail="Image is not within provided root")
        result = cluster_core.auto_move_image(
            resolved_root,
            target,
            threshold=payload.threshold,
            min_samples=payload.min_samples,
        )
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return {"status": "ok", "result": result}


@app.post("/undo")
def undo(payload: UndoPayload) -> Dict[str, Any]:
    resolved_root = _resolve_root(payload.root)
    try:
        result = cluster_core.undo_last_assignment(resolved_root)
    except (RuntimeError, FileNotFoundError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return {"status": "ok", "result": result}


@app.get("/image")
def image(root: str, path: str) -> FileResponse:
    resolved_root = _resolve_root(root)
    image_path = _normalize_path(path)
    if not cluster_core.is_within_root(image_path, resolved_root):
        raise HTTPException(status_code=400, detail="Image is not within provided root")
    return FileResponse(str(image_path))
