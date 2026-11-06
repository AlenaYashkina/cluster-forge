from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from backend import api, cluster_core


@pytest.fixture()
def single_photo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"dummy jpg; model is mocked")
    monkeypatch.setenv("CLUSTER_FORGE_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(cluster_core, "get_queue_slice", lambda root, **kwargs: {"queue": [str(photo)]})
    removed = []
    monkeypatch.setattr(cluster_core, "remove_paths_from_queue",
                        lambda root, paths: removed.extend(paths))
    return tmp_path, photo, removed


@pytest.mark.parametrize("reason", [
    "Confidence below threshold",
    "Cluster needs more labeled examples",
    "Failed to embed image",
    "Move failed: permission denied",
])
def test_auto_move_rejection_keeps_original_in_manual_queue(single_photo, monkeypatch, reason):
    root, photo, removed = single_photo
    monkeypatch.setattr(cluster_core, "auto_move_image", lambda *a, **kw: {
        "moved": [], "errors": [reason], "summary": {},
    })
    result = api.auto_move(api.AutoMovePayload(root=str(root), threshold=0.95))
    assert result["status"] == "skipped"
    assert result["result"]["errors"] == [reason]
    assert removed == []
    assert photo.exists()


def test_embedding_runtime_exception_keeps_queue(single_photo, monkeypatch):
    root, photo, removed = single_photo
    def raise_error(*args, **kwargs):
        raise RuntimeError("Cannot compute CLIP features")
    monkeypatch.setattr(cluster_core, "auto_move_image", raise_error)
    with pytest.raises(HTTPException) as exc:
        api.auto_move(api.AutoMovePayload(root=str(root)))
    assert exc.value.status_code == 400
    assert removed == []
    assert photo.exists()


def test_success_removes_exactly_the_moved_item(single_photo, monkeypatch):
    root, photo, removed = single_photo
    another = root / "keep.jpg"
    another.write_bytes(b"another image")
    monkeypatch.setattr(cluster_core, "auto_move_image", lambda *a, **kw: {
        "moved": [{"src": str(photo), "dst": str(root / "cluster" / "photo.jpg"), "cluster": "example"}],
        "errors": [],
        "summary": {},
    })
    result = api.auto_move(api.AutoMovePayload(root=str(root)))
    assert result["status"] == "ok"
    assert removed == [str(photo)]
    assert another.exists()


def test_unrelated_or_invalid_move_result_cannot_remove_target(single_photo, monkeypatch):
    root, photo, removed = single_photo
    monkeypatch.setattr(cluster_core, "auto_move_image", lambda *a, **kw: {
        "moved": [{"src": str(root / "other.jpg"), "dst": "destination"}], "errors": [], "summary": {},
    })
    result = api.auto_move(api.AutoMovePayload(root=str(root)))
    assert result["status"] == "skipped"
    assert removed == []
