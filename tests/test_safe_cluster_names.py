from pathlib import Path

import pytest

from backend.cluster_core import ensure_cluster_dir


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "/tmp/escape", "other/nested", r"..\\escape", "bad\nname"])
def test_reject_unsafe_names(tmp_path: Path, name: str) -> None:
    image = tmp_path / "sample.jpg"
    image.touch()
    with pytest.raises(ValueError):
        ensure_cluster_dir(image, name)


def test_reject_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)
    image = tmp_path / "sample.jpg"
    image.touch()
    with pytest.raises(ValueError):
        ensure_cluster_dir(image, "linked")


def test_allow_plain_category(tmp_path: Path) -> None:
    image = tmp_path / "sample.jpg"
    image.touch()
    assert ensure_cluster_dir(image, "kitchen") == tmp_path / "kitchen"
