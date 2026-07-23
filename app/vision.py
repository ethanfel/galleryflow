from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import os
import re
import threading
import uuid
import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
DINOV2_PATCH_SIZE = 14
DINOV2_SPATIAL_LEVELS = (1, 2, 4)
DINOV2_SPATIAL_DIMENSION = DINOV2_EMBEDDING_DIMENSION * sum(
    level * level for level in DINOV2_SPATIAL_LEVELS
)
DINOV2_SPATIAL_KEY = (
    f"{DINOV2_MODEL_KEY}:patch-spatial-pyramid-v1:"
    "mean-regions-1x1+2x2+4x4:l2-regions+concat"
)

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_MODEL_BYTES = 128 * 1024 * 1024
MAX_INFERENCE_THREADS = 4

_MODEL_HASH_RE = re.compile(r"[0-9a-f]{64}")
_IMAGE_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGE_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
_PAD_COLOR = tuple(round(float(channel) * 255) for channel in _IMAGE_MEAN)
_SUPPORTED_IMAGE_FORMATS = frozenset({"AVIF", "GIF", "JPEG", "PNG", "WEBP"})
_PHASH_SIZE = 32
_PHASH_LOW_FREQUENCY_SIZE = 8


def _orthonormal_dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64) + 0.5
    frequencies = np.arange(size, dtype=np.float64)[:, np.newaxis]
    matrix = np.cos((np.pi / size) * frequencies * positions)
    matrix[0, :] *= 1.0 / np.sqrt(2.0)
    matrix *= np.sqrt(2.0 / size)
    return matrix


_PHASH_DCT_MATRIX = _orthonormal_dct_matrix(_PHASH_SIZE)


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


@dataclass(frozen=True, slots=True)
class DinoV2Description:
    """Global and layout-aware descriptors for the same ordered image views.

    The first row always describes the source image. When ``include_mirror`` is
    requested, the second row describes its horizontal mirror. Both arrays are
    float32, contiguous, and independently L2-normalized row-wise.
    """

    global_embeddings: np.ndarray
    spatial_embeddings: np.ndarray


def _phash_from_image(image: Image.Image) -> int:
    grayscale = image.convert("L").resize(
        (_PHASH_SIZE, _PHASH_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(grayscale, dtype=np.float64)
    coefficients = _PHASH_DCT_MATRIX @ pixels @ _PHASH_DCT_MATRIX.T
    low_frequency = coefficients[
        :_PHASH_LOW_FREQUENCY_SIZE, :_PHASH_LOW_FREQUENCY_SIZE
    ].reshape(-1)
    # Excluding the DC component makes the threshold insensitive to a uniform
    # brightness shift. It remains in the 64-bit output, as in classic pHash.
    threshold = float(np.median(low_frequency[1:]))
    bits = low_frequency >= threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def perceptual_hash64(
    image: Image.Image, *, mirror_invariant: bool = False
) -> int:
    """Return a deterministic 64-bit DCT perceptual hash for a PIL image.

    With ``mirror_invariant=True``, the canonical minimum of the original and
    mirrored hashes is returned. This preserves a single integer cache value
    while making horizontally flipped copies compare identically.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    transposed = ImageOps.exif_transpose(image).convert("RGB")
    original = _phash_from_image(transposed)
    if not mirror_invariant:
        return original
    mirrored = _phash_from_image(ImageOps.mirror(transposed))
    return min(original, mirrored)


def perceptual_hash_bytes(
    data: bytes | bytearray | memoryview,
    *,
    mirror_invariant: bool = False,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> int:
    """Decode supported image bytes and return their 64-bit perceptual hash."""

    image = _decode_image_bytes(
        data,
        max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
    )
    return perceptual_hash64(image, mirror_invariant=mirror_invariant)


def hamming_similarity64(left: int, right: int) -> float:
    """Return normalized 64-bit Hamming similarity in the inclusive [0, 1]."""

    if isinstance(left, bool) or not isinstance(left, int):
        raise TypeError("left must be an integer")
    if isinstance(right, bool) or not isinstance(right, int):
        raise TypeError("right must be an integer")
    if left < 0 or left >= 1 << 64 or right < 0 or right >= 1 << 64:
        raise ValueError("perceptual hashes must be unsigned 64-bit integers")
    return 1.0 - ((left ^ right).bit_count() / 64.0)


def _decode_image_bytes(
    data: bytes | bytearray | memoryview,
    *,
    max_image_bytes: int,
    max_image_pixels: int,
) -> Image.Image:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    if not data:
        raise InvalidImageError("Image data is empty")
    if len(data) > max_image_bytes:
        raise InvalidImageError(f"Image exceeds the {max_image_bytes}-byte limit")

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
                if pixels > max_image_pixels:
                    raise InvalidImageError(
                        f"Image exceeds the {max_image_pixels}-pixel limit"
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


class _SessionInput(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[Any]: ...


ModelDownloader = Callable[[str, Path], Awaitable[None]]
SessionFactory = Callable[..., _InferenceSession]


class DinoV2Encoder:
    """DINOv2-S encoder backed by a pinned FP32 ONNX model.

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
        execution_provider: str = "cpu",
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
        requested_provider = execution_provider.strip().lower()
        if requested_provider not in {"auto", "cuda", "cpu"}:
            raise ValueError("execution_provider must be one of: auto, cuda, cpu")

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
        self.execution_provider = requested_provider
        self.session_threads = min(
            MAX_INFERENCE_THREADS, max(1, int(requested_threads))
        )

        self._downloader = downloader or self._download_model
        self._session_factory = session_factory or self._create_session
        self._prepare_lock = asyncio.Lock()
        self._session_lock = threading.Lock()
        self._session: _InferenceSession | None = None
        self._input_name: str | None = None
        self._active_providers: tuple[str, ...] = ()

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
                self._active_providers = ()
            return self.model_path

    def embed_bytes(
        self, data: bytes | bytearray | memoryview, *, include_mirror: bool = False
    ) -> np.ndarray:
        """Return one or two L2-normalized 384-dimensional image embeddings."""

        hidden_state, _ = self._infer_hidden_state(
            data, include_mirror=include_mirror
        )
        return self._class_embeddings(hidden_state)

    def embed_spatial_bytes(
        self, data: bytes | bytearray | memoryview, *, include_mirror: bool = False
    ) -> np.ndarray:
        """Return layout-aware 1x1 + 2x2 + 4x4 patch descriptors."""

        hidden_state, patch_grid = self._infer_hidden_state(
            data, include_mirror=include_mirror
        )
        return self._spatial_embeddings(hidden_state, patch_grid)

    def describe_bytes(
        self, data: bytes | bytearray | memoryview, *, include_mirror: bool = False
    ) -> DinoV2Description:
        """Return global and spatial descriptors from one ONNX inference."""

        hidden_state, patch_grid = self._infer_hidden_state(
            data, include_mirror=include_mirror
        )
        return DinoV2Description(
            global_embeddings=self._class_embeddings(hidden_state),
            spatial_embeddings=self._spatial_embeddings(hidden_state, patch_grid),
        )

    def provider_status(self) -> dict[str, Any]:
        """Describe the requested and currently active ONNX providers."""

        providers = list(self._active_providers)
        active = providers[0] if providers else None
        wants_cuda = self.execution_provider in {"auto", "cuda"}
        return {
            "requested": self.execution_provider,
            "active": active,
            "providers": providers,
            "cuda_active": "CUDAExecutionProvider" in providers,
            "cpu_fallback": bool(providers)
            and wants_cuda
            and "CUDAExecutionProvider" not in providers,
        }

    def _infer_hidden_state(
        self,
        data: bytes | bytearray | memoryview,
        *,
        include_mirror: bool,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Run a view batch once and validate its complete patch-token grid."""

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
        height, width = batch.shape[2:]
        if height % DINOV2_PATCH_SIZE or width % DINOV2_PATCH_SIZE:
            raise VisionInferenceError(
                "DINOv2 input canvas is not aligned to the patch size"
            )
        patch_grid = (
            height // DINOV2_PATCH_SIZE,
            width // DINOV2_PATCH_SIZE,
        )
        expected_tokens = patch_grid[0] * patch_grid[1] + 1
        if (
            hidden_state.ndim != 3
            or hidden_state.shape[0] != batch.shape[0]
            or hidden_state.shape[1] != expected_tokens
            or hidden_state.shape[2] != DINOV2_EMBEDDING_DIMENSION
        ):
            raise VisionInferenceError(
                "Unexpected DINOv2 output shape: " f"{tuple(hidden_state.shape)}"
            )
        hidden_state = np.asarray(hidden_state, dtype=np.float32)
        if not np.isfinite(hidden_state).all():
            raise VisionInferenceError("DINOv2 returned non-finite embeddings")
        return hidden_state, patch_grid

    @staticmethod
    def _class_embeddings(hidden_state: np.ndarray) -> np.ndarray:
        class_tokens = np.asarray(hidden_state[:, 0, :], dtype=np.float32)
        norms = np.linalg.norm(class_tokens, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise VisionInferenceError("DINOv2 returned zero-length embeddings")
        return np.ascontiguousarray(class_tokens / norms, dtype=np.float32)

    @staticmethod
    def _spatial_embeddings(
        hidden_state: np.ndarray, patch_grid: tuple[int, int]
    ) -> np.ndarray:
        rows, columns = patch_grid
        if any(rows % level or columns % level for level in DINOV2_SPATIAL_LEVELS):
            raise VisionInferenceError(
                "DINOv2 patch grid cannot be partitioned into spatial levels"
            )

        patches = hidden_state[:, 1:, :].reshape(
            hidden_state.shape[0], rows, columns, DINOV2_EMBEDDING_DIMENSION
        )
        regions: list[np.ndarray] = []
        for level in DINOV2_SPATIAL_LEVELS:
            region_height = rows // level
            region_width = columns // level
            for row in range(level):
                for column in range(level):
                    region = patches[
                        :,
                        row * region_height : (row + 1) * region_height,
                        column * region_width : (column + 1) * region_width,
                        :,
                    ].mean(axis=(1, 2), dtype=np.float32)
                    norms = np.linalg.norm(region, axis=1, keepdims=True)
                    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
                        raise VisionInferenceError(
                            "DINOv2 returned a zero-length spatial region"
                        )
                    regions.append(region / norms)

        descriptor = np.concatenate(regions, axis=1, dtype=np.float32)
        if descriptor.shape[1] != DINOV2_SPATIAL_DIMENSION:
            raise VisionInferenceError("DINOv2 spatial descriptor has wrong size")
        norms = np.linalg.norm(descriptor, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
            raise VisionInferenceError("DINOv2 returned an invalid spatial descriptor")
        return np.ascontiguousarray(descriptor / norms, dtype=np.float32)

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
        return _decode_image_bytes(
            data,
            max_image_bytes=self.max_image_bytes,
            max_image_pixels=self.max_image_pixels,
        )

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
                factory = self._session_factory
                try:
                    signature = inspect.signature(factory)
                    signature.bind(
                        self.model_path,
                        self.session_threads,
                        self.execution_provider,
                    )
                except (TypeError, ValueError):
                    session = factory(self.model_path, self.session_threads)
                else:
                    session = factory(
                        self.model_path,
                        self.session_threads,
                        self.execution_provider,
                    )
                inputs = session.get_inputs()
            except Exception as exc:
                raise VisionInferenceError(
                    "Could not initialize the DINOv2 ONNX session"
                ) from exc
            if not inputs or not isinstance(inputs[0].name, str):
                raise VisionInferenceError("DINOv2 model has no usable input tensor")
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                try:
                    providers = tuple(str(value) for value in get_providers())
                except Exception:
                    providers = ()
            else:
                providers = ()
            if not providers:
                providers = (
                    ("CPUExecutionProvider",)
                    if self.execution_provider == "cpu"
                    else ()
                )
            self._session = session
            self._input_name = inputs[0].name
            self._active_providers = providers
            return session, self._input_name

    @staticmethod
    def _create_session(
        model_path: Path, session_threads: int, execution_provider: str
    ) -> _InferenceSession:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = session_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = set(ort.get_available_providers())
        if (
            execution_provider == "cuda"
            and "CUDAExecutionProvider" not in available
        ):
            raise RuntimeError("CUDAExecutionProvider is not available")

        if execution_provider == "cuda":
            # Explicit CUDA is deliberately strict: an operator who selects it
            # should see a broken GPU runtime instead of unknowingly using CPU.
            providers = ["CUDAExecutionProvider"]
        elif execution_provider == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        try:
            return ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=providers,
            )
        except Exception:
            if (
                providers[0] != "CUDAExecutionProvider"
                or execution_provider != "auto"
            ):
                raise
            # A CUDA EP can be advertised while its driver/cuDNN libraries are
            # unusable. Retry once on CPU only for automatic provider selection.
            return ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
