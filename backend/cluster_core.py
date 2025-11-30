from __future__ import annotations

import json
import os
import shutil
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Sequence, Tuple

try:
    import numpy as np
    import torch
    from PIL import Image
    from PIL.Image import Resampling
    from transformers import CLIPModel, CLIPProcessor

    CLIP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional feature
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]
    CLIP_AVAILABLE = False

try:
    import rawpy  # type: ignore[import]
except ImportError:
    rawpy = None  # type: ignore[assignment]

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RAW_EXTS: Set[str] = {".cr2", ".nef", ".arw", ".dng"}
STATE_DIRNAME = ".manual_cluster"
LOG_NAME = "actions.jsonl"
CLUSTER_STATE_DIRNAME = ".clustering"
CLUSTER_STATE_FILE = "state.json"
CLIP_MAX_EDGE = 1024
_CLIP_CACHE: Optional[Tuple[CLIPModel, CLIPProcessor, torch.device, bool]] = None
LOOKALIKES: Dict[str, str] = {
    "а": "a",
    "в": "b",
    "е": "e",
    "к": "k",
    "м": "m",
    "н": "h",
    "о": "o",
    "р": "p",
    "с": "c",
    "т": "t",
    "х": "x",
    "у": "y",
    "ё": "e",
}


def _canonical_token(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).casefold()
    return "".join(LOOKALIKES.get(ch, ch) for ch in token)


def _match_child_by_name(parent: Path, target: str) -> Optional[Path]:
    try:
        entries = list(parent.iterdir())
    except (OSError, PermissionError):
        return None
    target_key = _canonical_token(target)
    for entry in entries:
        if _canonical_token(entry.name) == target_key:
            return entry
    return None


def resolve_input_path(raw: str) -> Optional[Path]:
    clean = raw.strip().strip("\"'")
    if not clean:
        return None
    try:
        candidate = Path(clean).expanduser()
    except (RuntimeError, OSError):
        return None
    if candidate.exists():
        return candidate

    parts = candidate.parts
    if not parts:
        return None
    current = Path(parts[0])
    if not current.exists():
        return None
    for part in parts[1:]:
        next_path = current / part
        if next_path.exists():
            current = next_path
            continue
        match = _match_child_by_name(current, part)
        if match is None:
            return None
        current = match
    return current if current.exists() else None


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def list_unlabeled_images(root: Path, known_clusters: Optional[Set[str]] = None) -> List[str]:
    known_tokens = {_canonical_token(c) for c in known_clusters} if known_clusters else set()
    result: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        pdir = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and (_canonical_token(d) not in known_tokens)
        ]
        for fname in filenames:
            fp = pdir / fname
            if fp.is_file() and is_image_file(fp):
                result.append(str(fp))
    result.sort()
    return result


def auto_move_queue(
    root: Path,
    queue: List[str],
    known_clusters: List[str],
    *,
    threshold: float,
    min_samples: int,
) -> Tuple[List[str], List[str], Dict[str, Any], List[Dict[str, str]], Optional[Dict[str, Any]]]:
    result: Optional[Dict[str, Any]] = None
    while queue:
        target = Path(queue[0])
        result = auto_move_image(root, target, threshold=threshold, min_samples=min_samples)
        if not result.get("moved"):
            break
        history = load_history(root)
        main_state = load_main_state(root)
        known_clusters = sorted(gather_known_clusters(history, main_state))
        queue = list_unlabeled_images(root, known_clusters)
        if not queue:
            break
    history = load_history(root)
    main_state = load_main_state(root)
    known_clusters = sorted(gather_known_clusters(history, main_state))
    queue = list_unlabeled_images(root, known_clusters)
    return queue, known_clusters, result or {}, history, main_state


def ensure_cluster_dir(img_path: Path, cluster: str) -> Path:
    cluster = cluster.strip()
    base = img_path.parent / cluster
    base.mkdir(parents=True, exist_ok=True)
    return base


def move_into_cluster(img_path: Path, cluster: str) -> Path:
    dst_dir = ensure_cluster_dir(img_path, cluster)
    candidate = dst_dir / img_path.name
    if candidate.exists():
        stem, ext = img_path.stem, img_path.suffix
        idx = 1
        while True:
            alt = dst_dir / f"{stem}__dup{idx}{ext}"
            if not alt.exists():
                candidate = alt
                break
            idx += 1
    shutil.move(str(img_path), str(candidate))
    return candidate


def append_log(root: Path, entry: Dict[str, str]) -> None:
    log_dir = root / STATE_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_NAME
    with log_path.open("a", encoding="utf-8") as fh:
        json.dump(entry, fh, ensure_ascii=False)
        fh.write("\n")


def load_history(root: Path) -> List[Dict[str, str]]:
    log_path = root / STATE_DIRNAME / LOG_NAME
    if not log_path.exists():
        return []
    entries: List[Dict[str, str]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_main_state(root: Path) -> Optional[Dict[str, Any]]:
    state_path = root / CLUSTER_STATE_DIRNAME / CLUSTER_STATE_FILE
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def clusters_from_state(state: Optional[Dict[str, Any]]) -> Set[str]:
    if not state:
        return set()
    names: Set[str] = set()
    for key in ("counts", "clusters"):
        for cname in (state.get(key) or {}):
            if cname:
                names.add(str(cname))
    return names


def clusters_from_history(entries: List[Dict[str, str]]) -> Set[str]:
    names: Set[str] = set()
    for entry in entries:
        cname = entry.get("cluster")
        if cname and cname not in {"__deleted__"}:
            names.add(cname)
    return names


def gather_known_clusters(
    history_entries: List[Dict[str, str]], main_state: Optional[Dict[str, Any]]
) -> Set[str]:
    names: Set[str] = set()
    names.update(clusters_from_state(main_state))
    names.update(clusters_from_history(history_entries))
    return names


def compute_stats(
    history_entries: List[Dict[str, str]],
    main_state: Optional[Dict[str, Any]],
    queue_length: int,
) -> Dict[str, Any]:
    manual_total = sum(1 for e in history_entries if e.get("cluster") not in {"__deleted__", None})
    deleted_total = sum(1 for e in history_entries if e.get("cluster") == "__deleted__")
    processed_total = manual_total + deleted_total
    total_seen_estimate = processed_total + queue_length
    processed_pct = (processed_total / total_seen_estimate * 100.0) if total_seen_estimate else 0.0

    cluster_counts: Dict[str, int] = defaultdict(int)
    for entry in history_entries:
        cname = entry.get("cluster")
        if cname and cname not in {"__deleted__"}:
            cluster_counts[cname] += 1

    state_counts = (main_state or {}).get("counts", {})
    return {
        "manual_labeled": manual_total,
        "deleted": deleted_total,
        "queue_length": queue_length,
        "processed_total": processed_total,
        "processed_pct": round(processed_pct, 3),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "state_counts": state_counts,
        "state_summary": {
            "clusters": len(state_counts),
            "manual_total": (main_state or {}).get("manual_total", 0),
            "auto_total": (main_state or {}).get("auto_total", 0),
            "deleted_total": (main_state or {}).get("deleted_total", 0),
        },
    }


def is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate = candidate.resolve()
        root = root.resolve()
    except OSError:
        return False
    return root == candidate or root in candidate.parents


def delete_file(file_path: Path) -> None:
    if file_path.exists():
        file_path.unlink()


def _ensure_clustering_state(root: Path) -> Dict[str, Any]:
    state = load_main_state(root) or {}
    state.setdefault("counts", {})
    state.setdefault("clusters", {})
    state.setdefault("manual_total", 0)
    state.setdefault("auto_total", 0)
    state.setdefault("deleted_total", 0)
    return state


def _vector_to_list(vec: Sequence[Any]) -> List[float]:
    return [float(v) for v in vec]


def _persist_clustering_state(root: Path, state: Dict[str, Any]) -> None:
    safe_clusters: Dict[str, List[List[float]]] = {}
    for name, vecs in state.get("clusters", {}).items():
        safe_clusters[str(name)] = [_vector_to_list(v) for v in vecs]

    safe_counts: Dict[str, int] = {str(name): int(value) for name, value in state.get("counts", {}).items()}

    payload: Dict[str, Any] = {
        "clusters": safe_clusters,
        "counts": safe_counts,
        "manual_total": int(state.get("manual_total", 0)),
        "auto_total": int(state.get("auto_total", 0)),
        "deleted_total": int(state.get("deleted_total", 0)),
    }
    if "version" in state:
        payload["version"] = state["version"]
    if "history" in state:
        payload["history"] = state["history"]

    state_dir = root / CLUSTER_STATE_DIRNAME
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / CLUSTER_STATE_FILE
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def record_assignment(
    root: Path,
    cluster: str,
    embedding: Optional[Sequence[float]],
    *,
    mode: str = "manual",
    state: Optional[Dict[str, Any]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    target_state = state if state is not None else _ensure_clustering_state(root)
    counts = target_state.setdefault("counts", {})
    clusters = target_state.setdefault("clusters", {})
    counts[cluster] = counts.get(cluster, 0) + 1
    if embedding:
        clusters.setdefault(cluster, []).append(_vector_to_list(embedding))
    else:
        clusters.setdefault(cluster, clusters.get(cluster, []))

    if mode == "auto":
        target_state["auto_total"] = target_state.get("auto_total", 0) + 1
    else:
        target_state["manual_total"] = target_state.get("manual_total", 0) + 1

    if save:
        _persist_clustering_state(root, target_state)
    return target_state


def record_deletion(root: Path, *, state: Optional[Dict[str, Any]] = None, save: bool = True) -> Dict[str, Any]:
    target_state = state if state is not None else _ensure_clustering_state(root)
    target_state["deleted_total"] = target_state.get("deleted_total", 0) + 1
    if save:
        _persist_clustering_state(root, target_state)
    return target_state


def _load_clip_model() -> Tuple[CLIPModel, CLIPProcessor, torch.device, bool]:
    if not CLIP_AVAILABLE or torch is None or CLIPModel is None or CLIPProcessor is None:
        raise RuntimeError("CLIP support is not available in this environment.")
    global _CLIP_CACHE
    if _CLIP_CACHE is not None:
        return _CLIP_CACHE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    use_fp16 = device.type == "cuda"
    if use_fp16:
        model = model.half()
    model = model.eval().to(device)
    _CLIP_CACHE = (model, processor, device, use_fp16)
    return _CLIP_CACHE


def _read_image_rgb(path: Path, max_edge: int) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow is required for CLIP embeddings.")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / float(max(w, h))
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        resample = Resampling.LANCZOS if Resampling is not None else Image.LANCZOS
        img = img.resize(new_size, resample)
    return img


def embed_image(path: Path, max_edge: int = CLIP_MAX_EDGE) -> Optional[List[float]]:
    if not CLIP_AVAILABLE:
        return None
    model, processor, device, use_fp16 = _load_clip_model()
    img = _read_image_rgb(path, max_edge)
    with torch.inference_mode():
        with torch.amp.autocast("cuda", enabled=use_fp16):
            inputs = processor(images=img, return_tensors="pt")
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        vec = feats.squeeze(0).detach().cpu().numpy().astype(float)
    return vec.tolist()


def get_centroids(state: Dict[str, Any]) -> Dict[str, Any]:
    if not CLIP_AVAILABLE or np is None:
        return {}
    cents: Dict[str, Any] = {}
    clusters = state.get("clusters", {})
    for cname, vecs in clusters.items():
        if not vecs:
            continue
        mat = np.asarray(vecs, dtype=np.float32)
        norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        centroid = norm.mean(axis=0)
        denom = np.linalg.norm(centroid) + 1e-8
        if denom:
            centroid = centroid / denom
        cents[cname] = centroid
    return cents


def _cos(a: Any, b: Any) -> float:
    if not CLIP_AVAILABLE or np is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def predict_cluster(
    embedding_vec: Sequence[float], state: Dict[str, Any]
) -> Tuple[Optional[str], float, Optional[str], float]:
    cents = get_centroids(state)
    if not cents:
        return None, 0.0, None, 0.0
    emb_arr = np.asarray(embedding_vec, dtype=np.float32)
    sims: List[Tuple[str, float]] = []
    for cname, centroid in cents.items():
        sims.append((cname, _cos(emb_arr, centroid)))
    sims.sort(key=lambda pair: pair[1], reverse=True)
    best_name, best_sim = sims[0]
    if len(sims) > 1:
        second_name, second_sim = sims[1]
    else:
        second_name, second_sim = None, 0.0
    return best_name, best_sim, second_name, second_sim


def suggest_for_image(root: Path, image_path: str) -> Optional[Dict[str, Any]]:
    if not CLIP_AVAILABLE:
        return None
    queue_path = Path(image_path)
    if not queue_path.exists():
        return None
    embedding = embed_image(queue_path)
    if not embedding:
        return None
    state = _ensure_clustering_state(root)
    best, best_score, second, second_score = predict_cluster(embedding, state)
    if best is None:
        return None
    margin = best_score - (second_score or 0.0)
    return {
        "cluster": best,
        "score": round(best_score, 4),
        "second_cluster": second,
        "second_score": round(second_score, 4) if second_score else 0.0,
        "margin": round(max(margin, 0.0), 4),
    }


def _sync_history_counts(root: Path, counts: Dict[str, int]) -> None:
    history_counts = defaultdict(int)
    for entry in load_history(root):
        cname = entry.get("cluster")
        if cname and cname not in {"__deleted__"}:
            history_counts[cname] += 1
    for cname, cnt in history_counts.items():
        counts[cname] = max(counts.get(cname, 0), cnt)


def auto_move_image(
    root: Path,
    image_path: Path,
    *,
    threshold: float,
    min_samples: int,
) -> Dict[str, Any]:
    if not CLIP_AVAILABLE:
        raise RuntimeError("CLIP support is required for auto moving.")
    errors: List[str] = []
    moved: List[Dict[str, Any]] = []
    if not (image_path.exists() and is_image_file(image_path)):
        errors.append("Target image is missing or not supported")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}

    state = _ensure_clustering_state(root)
    counts = state.setdefault("counts", {})
    _sync_history_counts(root, counts)

    embedding = embed_image(image_path)
    if not embedding:
        errors.append("Failed to embed image")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}

    suggestion = predict_cluster(embedding, state)
    best, score, _, _ = suggestion
    if best is None:
        errors.append("No cluster centroids available yet")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}

    cluster_count = counts.get(best, 0)
    if score < threshold:
        errors.append(f"Confidence ({score:.3f}) below threshold")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}
    if cluster_count < min_samples:
        errors.append(f"Cluster {best} requires {min_samples} samples, only {cluster_count} recorded")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}

    try:
        destination = move_into_cluster(image_path, best)
    except (OSError, shutil.Error) as move_err:
        errors.append(f"Move failed: {move_err}")
        return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}

    record_assignment(root, best, embedding, mode="auto", state=state, save=False)
    append_log(
        root,
        {
            "ts": datetime.now().isoformat(),
            "src": str(image_path),
            "dst": str(destination),
            "cluster": best,
            "note": "auto",
        },
    )
    moved.append(
        {
            "src": str(image_path),
            "dst": str(destination),
            "cluster": best,
            "score": round(score, 4),
        }
    )
    _persist_clustering_state(root, state)
    return {"moved": moved, "errors": errors, "summary": {"threshold": threshold, "min_samples": min_samples}}


def undo_last_assignment(root: Path) -> Dict[str, Any]:
    history = load_history(root)
    if not history:
        raise RuntimeError("No actions recorded yet.")
    last_entry = history.pop()
    if last_entry.get("cluster") in {"__deleted__", None}:
        raise RuntimeError("Cannot undo delete actions.")
    src = Path(last_entry.get("src", ""))
    dst = Path(last_entry.get("dst", ""))
    if not dst.exists():
        raise FileNotFoundError("Assigned file no longer exists.")
    target = src
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        stem, suffix = target.stem, target.suffix
        idx = 1
        while True:
            candidate = target.with_name(f"{stem}__undo{idx}{suffix}")
            if not candidate.exists():
                target = candidate
                break
            idx += 1
    shutil.move(str(dst), str(target))
    state = _ensure_clustering_state(root)
    cluster = last_entry.get("cluster")
    counts = state.get("counts", {})
    counts[cluster] = max(0, counts.get(cluster, 0) - 1)
    cluster_vectors = state.get("clusters", {})
    if cluster in cluster_vectors and cluster_vectors[cluster]:
        cluster_vectors[cluster].pop()
        if not cluster_vectors[cluster]:
            cluster_vectors.pop(cluster, None)
    mode = "auto" if last_entry.get("note") == "auto" else "manual"
    total_key = "auto_total" if mode == "auto" else "manual_total"
    state[total_key] = max(0, state.get(total_key, 0) - 1)
    _persist_clustering_state(root, state)
    log_path = root / STATE_DIRNAME / LOG_NAME
    with log_path.open("w", encoding="utf-8") as fh:
        for entry in history:
            json.dump(entry, fh, ensure_ascii=False)
            fh.write("\n")
    return {"cluster": cluster, "restored_path": str(target), "original": str(src)}
