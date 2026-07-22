from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _sqlite_vfs() -> str | None:
    value = os.getenv("PORNPIC_WEBUI_SQLITE_VFS")
    if value is not None:
        return (
            None if value.strip().lower() in {"", "default", "none"} else value.strip()
        )
    return "unix-dotfile" if os.name == "posix" else None


def _optional_path_from_env(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else None


@dataclass(slots=True)
class AppConfig:
    app_root: Path = APP_ROOT
    data_dir: Path = field(
        default_factory=lambda: _path_from_env(
            "PORNPIC_WEBUI_DATA_DIR", APP_ROOT / "data"
        )
    )
    download_root: Path = field(
        default_factory=lambda: _path_from_env(
            "PORNPIC_WEBUI_DOWNLOAD_ROOT", APP_ROOT / "data" / "downloads"
        )
    )
    sort_root: Path | None = field(
        default_factory=lambda: _optional_path_from_env("PORNPIC_WEBUI_SORT_ROOT")
    )
    pose_root: Path | None = field(
        default_factory=lambda: _optional_path_from_env("PORNPIC_WEBUI_POSE_ROOT")
    )
    finder_examples_root: Path | None = field(
        default_factory=lambda: _optional_path_from_env(
            "PORNPIC_WEBUI_FINDER_EXAMPLES_ROOT"
        )
    )
    finder_model_path: Path | None = field(
        default_factory=lambda: _optional_path_from_env(
            "PORNPIC_WEBUI_FINDER_MODEL_PATH"
        )
    )
    source_home: str = os.getenv(
        "PORNPIC_WEBUI_SOURCE_HOME", "https://www.pornpics.com"
    )
    request_timeout: float = float(os.getenv("PORNPIC_WEBUI_REQUEST_TIMEOUT", "25"))
    image_timeout: float = float(os.getenv("PORNPIC_WEBUI_IMAGE_TIMEOUT", "45"))
    job_workers: int = max(1, int(os.getenv("PORNPIC_WEBUI_JOB_WORKERS", "2")))
    image_workers: int = max(1, int(os.getenv("PORNPIC_WEBUI_IMAGE_WORKERS", "6")))
    finder_workers: int = max(
        1, min(2, int(os.getenv("PORNPIC_WEBUI_FINDER_WORKERS", "1")))
    )
    finder_network_workers: int = max(
        1, min(8, int(os.getenv("PORNPIC_WEBUI_FINDER_NETWORK_WORKERS", "3")))
    )
    finder_request_delay: float = max(
        0.0, float(os.getenv("PORNPIC_WEBUI_FINDER_REQUEST_DELAY", "0.15"))
    )
    finder_max_examples: int = max(
        1, min(2_000, int(os.getenv("PORNPIC_WEBUI_FINDER_MAX_EXAMPLES", "500")))
    )
    finder_max_gallery_images: int = max(
        1,
        min(
            5_000,
            int(os.getenv("PORNPIC_WEBUI_FINDER_MAX_GALLERY_IMAGES", "2000")),
        ),
    )
    finder_max_image_bytes: int = max(
        250_000,
        int(os.getenv("PORNPIC_WEBUI_FINDER_MAX_IMAGE_BYTES", str(12 * 1024 * 1024))),
    )
    finder_max_image_pixels: int = max(
        1_000_000,
        int(os.getenv("PORNPIC_WEBUI_FINDER_MAX_IMAGE_PIXELS", "40000000")),
    )
    max_image_bytes: int = max(
        1_000_000,
        int(os.getenv("PORNPIC_WEBUI_MAX_IMAGE_BYTES", str(80 * 1024 * 1024))),
    )
    user_agent: str = os.getenv(
        "PORNPIC_WEBUI_USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    )
    media_signing_key: str = field(
        default_factory=lambda: (
            os.getenv("PORNPIC_WEBUI_MEDIA_KEY") or secrets.token_urlsafe(32)
        )
    )
    sqlite_vfs: str | None = field(default_factory=_sqlite_vfs)

    def __post_init__(self) -> None:
        if self.finder_examples_root is None:
            self.finder_examples_root = (self.data_dir / "references").resolve()
        if self.finder_model_path is None:
            self.finder_model_path = (
                self.data_dir / "models" / "dinov2-small.onnx"
            ).resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pornpic_webui.sqlite3"

    @property
    def static_dir(self) -> Path:
        return self.app_root / "static"

    @property
    def sort_root_path(self) -> Path:
        return (self.sort_root or self.download_root).resolve()

    @property
    def pose_root_path(self) -> Path:
        return (self.pose_root or self.sort_root_path / "pose_pairs").resolve()

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.download_root.mkdir(parents=True, exist_ok=True)
        self.sort_root_path.mkdir(parents=True, exist_ok=True)
        self.pose_root_path.mkdir(parents=True, exist_ok=True)
        assert self.finder_examples_root is not None
        assert self.finder_model_path is not None
        self.finder_examples_root.mkdir(parents=True, exist_ok=True)
        self.finder_model_path.parent.mkdir(parents=True, exist_ok=True)


config = AppConfig()
