from __future__ import annotations

import asyncio
import hashlib
import io
import os
import re
import threading
import uuid
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


DINOV2_MODEL_KEY = (
    "dinov2-small-onnx@08c606e3123472a388efa59181b677d428f69bbd:"
    "contain-224x336-v1:cls-l2-fp32"
)
DINOV2_MODEL_URL = (
    "https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/"
    "08c606e3123472a388efa59181b677d428f69bbd/onnx/model.onnx"
)
DINOV2_MODEL_SHA256 = (
    "6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18"
)
DINOV2_EMBEDDING_DIMENSION = 384

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_INFERENCE_THREADS = 4

_MODEL_HASH_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_PAD_COLOR = tuple(round(float(channel) * 255) for channel in _IMAGE_MEAN)
_SUPPORTED_IMAGE_FORMATS = frozenset({"AVIF", "GIF", "JPEG", "PNG", "WEBP"})


class VisionError(RuntimeError):
    """Base error for local visual-embedding operations."""


class ModelPreparationError(VisionError):
    """The pinned ONNX model could not be prepared safely."""


class ModelIntegrityError(ModelPreparationError):
    """The ONNX model does not match its pinned SHA-256 digest."""


class ModelNotPreparedError(VisionError):
    """Inference was requested before the model was prepared."""


class InvalidImageError(VisionError):
    """Input bytes are not a safe, supported image."""


class VisionInferenceError(VisionError):
    """The encoder returned an invalid result or failed to run."""


class _SessionInput(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[Any]: ...


ModelDownloader = Callable[[str, Path], Awaitable[None]]
SessionFactory = Callable[[Path, int], _InferenceSession]


class DinoV2Encoder:
    """CPU-only DINOv2-S encoder backed by a pinned FP32 ONNX model.

    ``prepare`` must be awaited before inference in normal application use. The
    session is constructed lazily on the first ``embed_bytes`` call. Embeddings
    are returned as a float32 array shaped ``(1, 384)`` or ``(2, 384)`` when a
    horizontally mirrored view is requested.

    ``downloader`` and ``session_factory`` are injectable so callers can mirror
    the artifact and tests can run without network access or a real model.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        model_url: str = DINOV2_MODEL_URL,
        model_sha256: str = DINOV2_MODEL_SHA256,
        model_key: str = DINOV2_MODEL_KEY,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
        session_threads: int | None = None,
        downloader: ModelDownloader | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        digest = model_sha256.strip().lower()
        if not _MODEL_HASH_RE.fullmatch(digest):
            raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be positive")
        if max_model_bytes < 1:
            raise ValueError("max_model_bytes must be positive")

        requested_threads = session_threads
        if requested_threads is None:
            requested_threads = os.cpu_count() or 1

        self.model_path = Path(model_path)
        self.model_url = model_url
        self.model_sha256 = digest
        self.model_key = model_key
        self.max_image_bytes = int(max_image_bytes)
        self.max_image_pixels = int(max_image_pixels)
        self.max_model_bytes = int(max_model_bytes)
        self.session_threads = min(
            MAX_INFERENCE_THREADS, max(1, int(requested_threads))
        )

        self._downloader = downloader or self._download_model
        self._session_factory = session_factory or self._create_cpu_session
        self._prepare_lock = asyncio.Lock()
        self._session_lock = threading.Lock()
        self._session: _InferenceSession | None = None
        self._input_name: str | None = None

    async def prepare(self) -> Path:
        """Ensure that the verified model exists, publishing downloads atomically."""

        async with self._prepare_lock:
            if self.model_path.is_symlink():
                raise ModelPreparationError("The model path may not be a symlink")
            if self.model_path.exists():
                if not self.model_path.is_file():
                    raise ModelPreparationError("The model path is not a regular file")
                if await self._sha256_file(self.model_path) == self.model_sha256:
                    return self.model_path

            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ModelPreparationError(
                    f"Could not create model directory: {self.model_path.parent}"
                ) from exc

            temporary_path = self.model_path.with_name(
                f".{self.model_path.name}.{uuid.uuid4().hex}.part"
            )
            try:
                await self._downloader(self.model_url, temporary_path)
                if temporary_path.is_symlink() or not temporary_path.is_file():
                    raise ModelPreparationError(
                        "The model downloader did not create a regular file"
                    )
                actual = await self._sha256_file(temporary_path)
                if actual != self.model_sha256:
                    raise ModelIntegrityError(
                        "Downloaded DINOv2 model failed SHA-256 verification "
                        f"(expected {self.model_sha256}, got {actual})"
                    )
                os.replace(temporary_path, self.model_path)
            except (ModelPreparationError, asyncio.CancelledError):
                raise
            except Exception as exc:
                raise ModelPreparationError(
                    f"Could not prepare DINOv2 model from {self.model_url}"
                ) from exc
            finally:
                temporary_path.unlink(missing_ok=True)

            # A repaired/replaced artifact must receive a fresh ORT session.
            with self._session_lock:
                self._session = None
                self._input_name = None
            return self.model_path

    def embed_bytes(
        self, data: bytes | bytearray | memoryview, *, include_mirror: bool = False
    ) -> np.ndarray:
        """Return one or two L2-normalized 384-dimensional image embeddings."""

        image = self._decode_image(data)
        tensor = self._preprocess_image(image)
        views = [tensor]
        if include_mirror:
            views.append(np.ascontiguousarray(tensor[:, :, ::-1]))
        batch = np.ascontiguousarray(np.stack(views), dtype=np.float32)

        session, input_name = self._get_session()
        try:
            outputs = session.run(None, {input_name: batch})
        except Exception as exc:
            raise VisionInferenceError("DINOv2 ONNX inference failed") from exc
        if not outputs:
            raise VisionInferenceError("DINOv2 returned no output tensors")

        hidden_state = np.asarray(outputs[0])
        if (
            hidden_state.ndim != 3
            or hidden_state.shape[0] != batch.shape[0]
            or hidden_state.shape[1] < 1
            or hidden_state.shape[2] != DINOV2_EMBEDDING_DIMENSION
        ):
            raise VisionInferenceError(
                "Unexpected DINOv2 output shape: " f"{tuple(hidden_state.shape)}"
            )

        class_tokens = np.asarray(hidden_state[:, 0, :], dtype=np.float32)
        if not np.isfinite(class_tokens).all():
            raise VisionInferenceError("DINOv2 returned non-finite embeddings")
        norms = np.linalg.norm(class_tokens, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise VisionInferenceError("DINOv2 returned zero-length embeddings")
        return np.ascontiguousarray(class_tokens / norms, dtype=np.float32)

    @staticmethod
    async def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                # Model verification happens when a Finder scan first needs the
                # artifact; yield so hashing does not monopolize the event loop.
                await asyncio.sleep(0)
        return digest.hexdigest()

    async def _download_model(self, url: str, destination: Path) -> None:
        timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=timeout
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_length = response.headers.get("content-length")
                    if content_length:
                        try:
                            announced_size = int(content_length)
                        except ValueError:
                            announced_size = 0
                        if announced_size > self.max_model_bytes:
                            raise ModelPreparationError(
                                "The model download exceeds the configured size limit"
                            )

                    downloaded = 0
                    with destination.open("xb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > self.max_model_bytes:
                                raise ModelPreparationError(
                                    "The model download exceeds the configured size limit"
                                )
                            handle.write(chunk)
        except ModelPreparationError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise ModelPreparationError("The DINOv2 model download failed") from exc

    def _decode_image(
        self, data: bytes | bytearray | memoryview
    ) -> Image.Image:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        if not data:
            raise InvalidImageError("Image data is empty")
        if len(data) > self.max_image_bytes:
            raise InvalidImageError(
                f"Image exceeds the {self.max_image_bytes}-byte limit"
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in _SUPPORTED_IMAGE_FORMATS:
                        raise InvalidImageError(
                            f"Unsupported image format: {source.format or 'unknown'}"
                        )
                    pixels = int(source.width) * int(source.height)
                    if source.width < 1 or source.height < 1 or pixels < 1:
                        raise InvalidImageError("Image dimensions are invalid")
                    if pixels > self.max_image_pixels:
                        raise InvalidImageError(
                            f"Image exceeds the {self.max_image_pixels}-pixel limit"
                        )
                    source.load()
                    transposed = ImageOps.exif_transpose(source)
                    image = transposed.convert("RGB")
                    image.load()
        except InvalidImageError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise InvalidImageError("Image data could not be decoded safely") from exc
        return image

    @staticmethod
    def _preprocess_image(image: Image.Image) -> np.ndarray:
        if image.height > image.width:
            target_size = (224, 336)
        elif image.width > image.height:
            target_size = (336, 224)
        else:
            target_size = (280, 280)

        contained = ImageOps.contain(
            image, target_size, method=Image.Resampling.BICUBIC
        )
        canvas = Image.new("RGB", target_size, _PAD_COLOR)
        left = (target_size[0] - contained.width) // 2
        top = (target_size[1] - contained.height) // 2
        canvas.paste(contained, (left, top))

        pixels = np.asarray(canvas, dtype=np.float32) / np.float32(255.0)
        normalized = (pixels - _IMAGE_MEAN) / _IMAGE_STD
        return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)

    def _get_session(self) -> tuple[_InferenceSession, str]:
        if self._session is not None and self._input_name is not None:
            return self._session, self._input_name
        with self._session_lock:
            if self._session is not None and self._input_name is not None:
                return self._session, self._input_name
            if self.model_path.is_symlink() or not self.model_path.is_file():
                raise ModelNotPreparedError(
                    "DINOv2 model is unavailable; await prepare() before inference"
                )
            try:
                session = self._session_factory(
                    self.model_path, self.session_threads
                )
                inputs = session.get_inputs()
            except Exception as exc:
                raise VisionInferenceError(
                    "Could not initialize the DINOv2 ONNX session"
                ) from exc
            if not inputs or not isinstance(inputs[0].name, str):
                raise VisionInferenceError("DINOv2 model has no usable input tensor")
            self._session = session
            self._input_name = inputs[0].name
            return session, self._input_name

    @staticmethod
    def _create_cpu_session(
        model_path: Path, session_threads: int
    ) -> _InferenceSession:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = session_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
