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
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


JOYTAG_MODEL_REVISION = "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
JOYTAG_MODEL_URL = (
    "https://huggingface.co/fancyfeast/joytag/resolve/"
    f"{JOYTAG_MODEL_REVISION}/model.onnx"
)
JOYTAG_TAGS_URL = (
    "https://huggingface.co/fancyfeast/joytag/resolve/"
    f"{JOYTAG_MODEL_REVISION}/top_tags.txt"
)
JOYTAG_MODEL_SHA256 = (
    "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097"
)
JOYTAG_TAGS_SHA256 = (
    "32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0"
)
JOYTAG_TAG_COUNT = 5_813
JOYTAG_IMAGE_SIZE = 448
JOYTAG_MODEL_KEY = (
    f"fancyfeast-joytag-onnx@{JOYTAG_MODEL_REVISION}:"
    "square-pad-white-448:clip-normalize:multilabel-sigmoid-v1"
)

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_MODEL_BYTES = 400 * 1024 * 1024
DEFAULT_MAX_TAGS_BYTES = 2 * 1024 * 1024
MAX_INFERENCE_THREADS = 4

_HASH_RE = re.compile(r"[0-9a-f]{64}")
_SUPPORTED_IMAGE_FORMATS = frozenset({"AVIF", "GIF", "JPEG", "PNG", "WEBP"})
_NORMALIZE_MEAN = np.asarray(
    (0.48145466, 0.4578275, 0.40821073), dtype=np.float32
).reshape(1, 1, 3)
_NORMALIZE_STD = np.asarray(
    (0.26862954, 0.26130258, 0.27577711), dtype=np.float32
).reshape(1, 1, 3)


class JoyTagError(RuntimeError):
    """Base error for JoyTag artifact management and inference."""


class JoyTagPreparationError(JoyTagError):
    pass


class JoyTagIntegrityError(JoyTagPreparationError):
    pass


class JoyTagNotPreparedError(JoyTagError):
    pass


class JoyTagInvalidImageError(JoyTagError):
    pass


class JoyTagInferenceError(JoyTagError):
    pass


class _SessionTensor(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionTensor]: ...

    def get_outputs(self) -> list[_SessionTensor]: ...

    def get_providers(self) -> list[str]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[Any]: ...


ArtifactDownloader = Callable[..., Awaitable[None]]
SessionFactory = Callable[..., _InferenceSession]


class JoyTagAnalyzer:
    """Pinned JoyTag ONNX multi-label image classifier.

    Artifact download and ONNX session construction are lazy. Call
    :meth:`prepare` before inference. ``downloader`` and ``session_factory`` are
    injectable so tests and offline deployments do not need network access or
    the real model.
    """

    def __init__(
        self,
        model_path: Path | str,
        tags_path: Path | str,
        *,
        execution_provider: str = "auto",
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        model_url: str = JOYTAG_MODEL_URL,
        tags_url: str = JOYTAG_TAGS_URL,
        model_sha256: str = JOYTAG_MODEL_SHA256,
        tags_sha256: str = JOYTAG_TAGS_SHA256,
        model_key: str = JOYTAG_MODEL_KEY,
        expected_tag_count: int = JOYTAG_TAG_COUNT,
        max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
        max_tags_bytes: int = DEFAULT_MAX_TAGS_BYTES,
        session_threads: int | None = None,
        downloader: ArtifactDownloader | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        model_digest = model_sha256.strip().lower()
        tags_digest = tags_sha256.strip().lower()
        if not _HASH_RE.fullmatch(model_digest):
            raise ValueError("model_sha256 must be a 64-character hexadecimal digest")
        if not _HASH_RE.fullmatch(tags_digest):
            raise ValueError("tags_sha256 must be a 64-character hexadecimal digest")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be positive")
        if max_image_pixels < 1:
            raise ValueError("max_image_pixels must be positive")
        if max_model_bytes < 1:
            raise ValueError("max_model_bytes must be positive")
        if max_tags_bytes < 1:
            raise ValueError("max_tags_bytes must be positive")
        if expected_tag_count < 1:
            raise ValueError("expected_tag_count must be positive")
        requested_provider = execution_provider.strip().lower()
        if requested_provider not in {"auto", "cuda", "cpu"}:
            raise ValueError("execution_provider must be one of: auto, cuda, cpu")

        requested_threads = session_threads
        if requested_threads is None:
            requested_threads = os.cpu_count() or 1

        self.model_path = Path(model_path)
        self.tags_path = Path(tags_path)
        self.execution_provider = requested_provider
        self.max_image_bytes = int(max_image_bytes)
        self.max_image_pixels = int(max_image_pixels)
        self.model_url = model_url
        self.tags_url = tags_url
        self.model_sha256 = model_digest
        self.tags_sha256 = tags_digest
        self.model_key = model_key
        self.expected_tag_count = int(expected_tag_count)
        self.max_model_bytes = int(max_model_bytes)
        self.max_tags_bytes = int(max_tags_bytes)
        self.session_threads = min(
            MAX_INFERENCE_THREADS, max(1, int(requested_threads))
        )

        self._downloader = downloader or self._download_artifact
        self._session_factory = session_factory or self._create_session
        self._prepare_lock = asyncio.Lock()
        self._session_lock = threading.RLock()
        self._session: _InferenceSession | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._active_providers: tuple[str, ...] = ()
        self._tags: tuple[str, ...] = ()
        self._prepared = False
        self._provider_message = ""

    @property
    def tags(self) -> tuple[str, ...]:
        return self._tags

    @property
    def active_provider(self) -> str | None:
        return self._active_providers[0] if self._active_providers else None

    async def prepare(self) -> tuple[Path, Path]:
        """Verify or download the pinned model and vocabulary atomically."""

        async with self._prepare_lock:
            model_changed = await self._prepare_artifact(
                path=self.model_path,
                url=self.model_url,
                expected_sha256=self.model_sha256,
                max_bytes=self.max_model_bytes,
                label="JoyTag model",
            )
            tags_changed = await self._prepare_artifact(
                path=self.tags_path,
                url=self.tags_url,
                expected_sha256=self.tags_sha256,
                max_bytes=self.max_tags_bytes,
                label="JoyTag tag vocabulary",
            )
            tags = self._load_tags(self.tags_path)
            if model_changed or tags_changed:
                self._reset_session()
            self._tags = tags
            self._prepared = True
            return self.model_path, self.tags_path

    def classify_bytes(
        self, data: bytes | bytearray | memoryview
    ) -> np.ndarray:
        """Return one ``(5813,)`` probability vector."""

        return self.classify_many_bytes([data])[0]

    def classify_many_bytes(
        self,
        payloads: Sequence[bytes | bytearray | memoryview],
    ) -> np.ndarray:
        """Classify one shape-compatible batch and retain caller ordering."""

        if isinstance(payloads, (bytes, bytearray, memoryview)) or not isinstance(
            payloads, Sequence
        ):
            raise TypeError("payloads must be a sequence of bytes-like images")
        self._require_prepared()
        if not payloads:
            return np.empty((0, len(self._tags)), dtype=np.float32)

        tensors = [self._preprocess_image(self._decode_image(data)) for data in payloads]
        batch = np.ascontiguousarray(np.stack(tensors), dtype=np.float32)
        session, input_name, output_name = self._get_session()
        try:
            values = session.run([output_name], {input_name: batch})
        except Exception as exc:
            if (
                self.execution_provider != "auto"
                or "CUDAExecutionProvider" not in self._active_providers
            ):
                raise JoyTagInferenceError("JoyTag ONNX inference failed") from exc
            session, input_name, output_name = self._force_cpu_session(exc)
            try:
                values = session.run([output_name], {input_name: batch})
            except Exception as retry_exc:
                raise JoyTagInferenceError("JoyTag ONNX inference failed") from retry_exc

        if len(values) != 1:
            raise JoyTagInferenceError(
                f"JoyTag returned {len(values)} output tensors; expected one"
            )
        logits = np.asarray(values[0], dtype=np.float32)
        expected_shape = (len(payloads), len(self._tags))
        if logits.shape != expected_shape:
            raise JoyTagInferenceError(
                "Unexpected JoyTag output shape: "
                f"{tuple(logits.shape)}; expected {expected_shape}"
            )
        if not np.isfinite(logits).all():
            raise JoyTagInferenceError("JoyTag returned non-finite logits")
        return self._sigmoid(logits)

    def provider_status(self) -> dict[str, Any]:
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
            "message": self._provider_message,
        }

    async def _prepare_artifact(
        self,
        *,
        path: Path,
        url: str,
        expected_sha256: str,
        max_bytes: int,
        label: str,
    ) -> bool:
        if path.is_symlink():
            raise JoyTagPreparationError(f"The {label} path may not be a symlink")
        if path.exists():
            if not path.is_file():
                raise JoyTagPreparationError(f"The {label} path is not a regular file")
            if path.stat().st_size > max_bytes:
                current_digest = ""
            else:
                current_digest = await self._sha256_file(path)
            if current_digest == expected_sha256:
                return False

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise JoyTagPreparationError(
                f"Could not create {label} directory: {path.parent}"
            ) from exc

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
        try:
            await self._call_downloader(url, temporary, max_bytes)
            if temporary.is_symlink() or not temporary.is_file():
                raise JoyTagPreparationError(
                    f"The {label} downloader did not create a regular file"
                )
            if temporary.stat().st_size > max_bytes:
                raise JoyTagPreparationError(
                    f"The {label} download exceeds the configured size limit"
                )
            actual = await self._sha256_file(temporary)
            if actual != expected_sha256:
                raise JoyTagIntegrityError(
                    f"Downloaded {label} failed SHA-256 verification "
                    f"(expected {expected_sha256}, got {actual})"
                )
            os.replace(temporary, path)
        except (JoyTagPreparationError, asyncio.CancelledError):
            raise
        except Exception as exc:
            raise JoyTagPreparationError(
                f"Could not prepare {label} from {url}"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return True

    async def _call_downloader(
        self, url: str, destination: Path, max_bytes: int
    ) -> None:
        downloader = self._downloader
        try:
            signature = inspect.signature(downloader)
            signature.bind(url, destination, max_bytes)
        except (TypeError, ValueError):
            value = downloader(url, destination)
        else:
            value = downloader(url, destination, max_bytes)
        if inspect.isawaitable(value):
            await value

    @staticmethod
    async def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                await asyncio.sleep(0)
        return digest.hexdigest()

    async def _download_artifact(
        self, url: str, destination: Path, max_bytes: int
    ) -> None:
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
                        if announced_size > max_bytes:
                            raise JoyTagPreparationError(
                                "The JoyTag artifact download exceeds the "
                                "configured size limit"
                            )
                    downloaded = 0
                    with destination.open("xb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            downloaded += len(chunk)
                            if downloaded > max_bytes:
                                raise JoyTagPreparationError(
                                    "The JoyTag artifact download exceeds the "
                                    "configured size limit"
                                )
                            handle.write(chunk)
        except JoyTagPreparationError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise JoyTagPreparationError(
                "The JoyTag artifact download failed"
            ) from exc

    def _load_tags(self, path: Path) -> tuple[str, ...]:
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise JoyTagPreparationError(
                "The JoyTag tag vocabulary is not valid UTF-8 text"
            ) from exc
        tags = tuple(line.strip() for line in raw.splitlines() if line.strip())
        if len(tags) != self.expected_tag_count:
            raise JoyTagPreparationError(
                "Unexpected JoyTag tag count: "
                f"{len(tags)}; expected {self.expected_tag_count}"
            )
        if len(set(tags)) != len(tags):
            raise JoyTagPreparationError("The JoyTag tag vocabulary has duplicates")
        return tags

    def _require_prepared(self) -> None:
        if (
            not self._prepared
            or not self._tags
            or self.model_path.is_symlink()
            or not self.model_path.is_file()
        ):
            raise JoyTagNotPreparedError(
                "JoyTag artifacts are unavailable; await prepare() before inference"
            )

    def _decode_image(
        self, data: bytes | bytearray | memoryview
    ) -> Image.Image:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("JoyTag image data must be bytes-like")
        if not data:
            raise JoyTagInvalidImageError("Image data is empty")
        if len(data) > self.max_image_bytes:
            raise JoyTagInvalidImageError(
                f"Image exceeds the {self.max_image_bytes}-byte limit"
            )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    if source.format not in _SUPPORTED_IMAGE_FORMATS:
                        raise JoyTagInvalidImageError(
                            f"Unsupported image format: {source.format or 'unknown'}"
                        )
                    pixels = int(source.width) * int(source.height)
                    if source.width < 1 or source.height < 1 or pixels < 1:
                        raise JoyTagInvalidImageError("Image dimensions are invalid")
                    if pixels > self.max_image_pixels:
                        raise JoyTagInvalidImageError(
                            f"Image exceeds the {self.max_image_pixels}-pixel limit"
                        )
                    source.load()
                    transposed = ImageOps.exif_transpose(source)
                    if transposed.mode in {"RGBA", "LA"} or (
                        transposed.mode == "P" and "transparency" in transposed.info
                    ):
                        rgba = transposed.convert("RGBA")
                        background = Image.new(
                            "RGBA", rgba.size, (255, 255, 255, 255)
                        )
                        image = Image.alpha_composite(background, rgba).convert("RGB")
                    else:
                        image = transposed.convert("RGB")
                    image.load()
        except JoyTagInvalidImageError:
            raise
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            SyntaxError,
            ValueError,
        ) as exc:
            raise JoyTagInvalidImageError(
                "Image data could not be decoded safely"
            ) from exc
        return image

    @staticmethod
    def _preprocess_image(image: Image.Image) -> np.ndarray:
        width, height = image.size
        side = max(width, height)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        resized = canvas.resize(
            (JOYTAG_IMAGE_SIZE, JOYTAG_IMAGE_SIZE),
            Image.Resampling.BICUBIC,
        )
        pixels = np.asarray(resized, dtype=np.float32) / np.float32(255.0)
        normalized = (pixels - _NORMALIZE_MEAN) / _NORMALIZE_STD
        return np.ascontiguousarray(
            normalized.transpose(2, 0, 1), dtype=np.float32
        )

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        probabilities = np.empty_like(values)
        positive = values >= 0
        probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        exponent = np.exp(values[~positive])
        probabilities[~positive] = exponent / (1.0 + exponent)
        return np.ascontiguousarray(probabilities, dtype=np.float32)

    def _get_session(self) -> tuple[_InferenceSession, str, str]:
        if (
            self._session is not None
            and self._input_name is not None
            and self._output_name is not None
        ):
            return self._session, self._input_name, self._output_name
        with self._session_lock:
            if (
                self._session is not None
                and self._input_name is not None
                and self._output_name is not None
            ):
                return self._session, self._input_name, self._output_name
            try:
                session = self._build_session(self.execution_provider)
            except Exception as exc:
                if self.execution_provider != "auto":
                    raise JoyTagInferenceError(
                        "Could not initialize the JoyTag ONNX session"
                    ) from exc
                try:
                    session = self._build_session("cpu")
                except Exception as retry_exc:
                    raise JoyTagInferenceError(
                        "Could not initialize the JoyTag ONNX session"
                    ) from retry_exc
                self._provider_message = (
                    "CUDA session initialization failed; using CPU"
                )
            return self._publish_session(session)

    def _build_session(self, provider: str) -> _InferenceSession:
        factory = self._session_factory
        try:
            signature = inspect.signature(factory)
            signature.bind(self.model_path, self.session_threads, provider)
        except (TypeError, ValueError):
            return factory(self.model_path, self.session_threads)
        return factory(self.model_path, self.session_threads, provider)

    def _publish_session(
        self, session: _InferenceSession
    ) -> tuple[_InferenceSession, str, str]:
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or not isinstance(inputs[0].name, str)
            or not isinstance(outputs[0].name, str)
        ):
            raise JoyTagInferenceError(
                "Unexpected JoyTag ONNX interface; expected one named input "
                "and one named output"
            )
        get_providers = getattr(session, "get_providers", None)
        if callable(get_providers):
            try:
                providers = tuple(str(value) for value in get_providers())
            except Exception:
                providers = ()
        else:
            providers = ()
        if not providers and self.execution_provider == "cpu":
            providers = ("CPUExecutionProvider",)
        if (
            self.execution_provider == "cuda"
            and providers
            and providers[0] != "CUDAExecutionProvider"
        ):
            raise JoyTagInferenceError(
                "CUDAExecutionProvider was requested but is not active"
            )
        self._session = session
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._active_providers = providers
        return session, self._input_name, self._output_name

    def _force_cpu_session(
        self, original_error: Exception
    ) -> tuple[_InferenceSession, str, str]:
        with self._session_lock:
            if (
                self._session is not None
                and "CUDAExecutionProvider" not in self._active_providers
                and self._input_name is not None
                and self._output_name is not None
            ):
                return self._session, self._input_name, self._output_name
            try:
                session = self._build_session("cpu")
                self._provider_message = (
                    "CUDA inference failed; using CPU for subsequent batches"
                )
                return self._publish_session(session)
            except Exception as exc:
                raise JoyTagInferenceError(
                    "JoyTag CUDA inference failed and CPU fallback could not start"
                ) from exc

    def _reset_session(self) -> None:
        with self._session_lock:
            self._session = None
            self._input_name = None
            self._output_name = None
            self._active_providers = ()
            self._provider_message = ""

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
            providers = ["CUDAExecutionProvider"]
        elif execution_provider == "auto" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
