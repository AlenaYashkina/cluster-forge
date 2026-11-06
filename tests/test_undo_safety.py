from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend import cluster_core


def test_undo_preserves_original_if_path_reused(tmp_path: Path) -> None:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"original")
    destination = cluster_core.move_into_cluster(source, "kitchen")
    cluster_core.append_log(tmp_path, {
        "ts": "2026-11-06T10:00:00",
        "src": str(source), "dst": str(destination),
        "cluster": "kitchen", "note": "manual",
    })
    cluster_core.record_assignment(tmp_path, "kitchen", None, mode="manual")
    source.write_bytes(b"new file should not be replaced")

    result = cluster_core.undo_last_assignment(tmp_path)
    restored = Path(result["restored_path"])
    assert source.read_bytes() == b"new file should not be replaced"
    assert restored.name == "photo__undo1.jpg"
    assert restored.read_bytes() == b"original"
    assert not destination.exists()


def test_undo_rejects_source_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jpg"
    outside.write_bytes(b"do not overwrite")
    saved = tmp_path / "kitchen" / "photo.jpg"
    saved.parent.mkdir()
    saved.write_bytes(b"inside")
    cluster_core.append_log(tmp_path, {
        "ts": "2026-11-06T10:00:00",
        "src": str(outside), "dst": str(saved),
        "cluster": "kitchen", "note": "manual",
    })
    with pytest.raises(ValueError, match="outside"):
        cluster_core.undo_last_assignment(tmp_path)
    assert saved.read_bytes() == b"inside"
    assert outside.read_bytes() == b"do not overwrite"


def test_undo_rejects_destination_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jpg"
    outside.write_bytes(b"external")
    link = tmp_path / "link.jpg"
    link.symlink_to(outside)
    source = tmp_path / "original.jpg"
    cluster_core.append_log(tmp_path, {
        "ts": "2026-11-06T10:00:00",
        "src": str(source), "dst": str(link),
        "cluster": "kitchen", "note": "manual",
    })
    with pytest.raises(ValueError, match="outside"):
        cluster_core.undo_last_assignment(tmp_path)
    assert outside.read_bytes() == b"external"
