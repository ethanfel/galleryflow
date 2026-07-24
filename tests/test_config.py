from pathlib import Path

import pytest

from app.config import AppConfig


def test_finder_pose_model_defaults_to_persistent_model_directory(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "state",
        download_root=tmp_path / "library",
        finder_execution_provider="cuda",
        sqlite_vfs=None,
    )

    assert config.finder_model_path == (tmp_path / "state/models/dinov2-small.onnx")
    assert config.finder_pose_model_path == (tmp_path / "state/models/rtmo-l.onnx")
    assert config.finder_execution_provider == "cuda"
    assert config.finder_pose_enabled is True
    assert config.finder_inference_batch_size == 8
    assert config.finder_cache_max_bytes == 2 * 1024 * 1024 * 1024
    assert config.finder_cache_max_entries == 50_000

    config.ensure_directories()
    assert config.finder_pose_model_path.parent.is_dir()


def test_invalid_finder_execution_provider_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="auto, cuda, or cpu"):
        AppConfig(
            data_dir=tmp_path / "state",
            download_root=tmp_path / "library",
            finder_execution_provider="metal",
            sqlite_vfs=None,
        )


@pytest.mark.parametrize(("requested", "expected"), [("0", 1), ("17", 17), ("999", 64)])
def test_finder_inference_batch_size_is_bounded(
    tmp_path: Path,
    requested: str,
    expected: int,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "state",
        download_root=tmp_path / "library",
        finder_inference_batch_size=int(requested),
        sqlite_vfs=None,
    )

    assert config.finder_inference_batch_size == expected
