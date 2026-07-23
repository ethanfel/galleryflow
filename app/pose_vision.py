from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import math
import os
import threading
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import httpx
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


# Official OpenMMLab RTMO-L body7 ONNX SDK artifact.  Both the container and
# the only model member are pinned: a compromised mirror cannot choose an
# unexpected archive member or replace the network with a different model.
RTMO_L_ARCHIVE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
    "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip"
)
RTMO_L_ARCHIVE_SHA256 = (
    "17b361174d759d974879f9fb46d564ae658d004bfa070e6f1c9ad275d3fd6b87"
)
RTMO_L_MODEL_SHA256 = (
    "090096ca90f29163cc4f67137dcc0cd4b2ee95ea0af11764fbfda88dd2ae1140"
)
RTMO_L_MODEL_KEY = (
    "openmmlab-rtmo-l-body7-640@b37118ce_20231211:"
    "onnx-fp32:pose-geometry-v1"
)
RTMO_L_ARCHIVE_MEMBER = "end2end.onnx"
RTMO_INPUT_SIZE = 640
RTMO_KEYPOINT_COUNT = 17

DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_MAX_ARCHIVE_BYTES = 180 * 1024 * 1024
DEFAULT_MAX_MODEL_BYTES = 190 * 1024 * 1024
MAX_INFERENCE_THREADS = 4

# COCO body keypoint order.
COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
COCO_LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)


class PoseVisionError(RuntimeError):
    """Base error for RTMO model management and inference."""


class PoseModelPreparationError(PoseVisionError):
    pass


class PoseModelIntegrityError(PoseModelPreparationError):
    pass


class PoseModelNotPreparedError(PoseVisionError):
    pass


class PoseInvalidImageError(PoseVisionError):
    pass


class PoseInferenceError(PoseVisionError):
    pass


class _SessionInput(Protocol):
    name: str


class _InferenceSession(Protocol):
    def get_inputs(self) -> list[_SessionInput]: ...

    def get_providers(self) -> list[str]: ...

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[Any]: ...


ArchiveDownloader = Callable[[str, Path], Any]
SessionFactory = Callable[[Path, list[str], int], _InferenceSession]
AvailableProviders = Callable[[], Sequence[str]]


@dataclass(frozen=True, slots=True)
class PoseFrame:
    """Detected people expressed in normalized original-image coordinates."""

    keypoints: np.ndarray  # (people, 17, 2), x/y in [0, 1]
    confidences: np.ndarray  # (people, 17)
    boxes: np.ndarray  # (people, 4), xyxy in [0, 1]
    person_scores: np.ndarray  # (people,)
    image_size: tuple[int, int]  # width, height
    model_key: str = RTMO_L_MODEL_KEY
    provider: str = ""

    @property
    def person_count(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def scene_kind(self) -> str:
        return scene_kind(self.person_count)

    def as_dict(self) -> dict[str, Any]:
        return {
            "keypoints": self.keypoints.tolist(),
            "confidences": self.confidences.tolist(),
            "boxes": self.boxes.tolist(),
            "person_scores": self.person_scores.tolist(),
            "person_count": self.person_count,
            "scene_kind": self.scene_kind,
            "image_size": list(self.image_size),
            "model_key": self.model_key,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PoseFrame":
        """Reconstruct a cached frame, rejecting malformed or non-finite JSON."""

        if not isinstance(value, dict):
            raise ValueError("Cached pose frame must be an object")
        try:
            keypoints = np.asarray(value["keypoints"], dtype=np.float32)
            confidences = np.asarray(value["confidences"], dtype=np.float32)
            boxes = np.asarray(value["boxes"], dtype=np.float32)
            person_scores = np.asarray(value["person_scores"], dtype=np.float32)
            image_size_value = value["image_size"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cached pose frame is missing required numeric fields") from exc
        # JSON loses trailing dimensions for arrays with zero rows. Restore the
        # canonical empty shapes emitted by ``as_dict``.
        if keypoints.size == 0:
            keypoints = np.empty((0, RTMO_KEYPOINT_COUNT, 2), dtype=np.float32)
        if confidences.size == 0:
            confidences = np.empty((0, RTMO_KEYPOINT_COUNT), dtype=np.float32)
        if boxes.size == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
        if person_scores.size == 0:
            person_scores = np.empty((0,), dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (RTMO_KEYPOINT_COUNT, 2):
            raise ValueError("Cached pose keypoints have an invalid shape")
        count = int(keypoints.shape[0])
        if confidences.shape != (count, RTMO_KEYPOINT_COUNT):
            raise ValueError("Cached pose confidences have an invalid shape")
        if boxes.shape != (count, 4):
            raise ValueError("Cached pose boxes have an invalid shape")
        if person_scores.shape != (count,):
            raise ValueError("Cached pose person scores have an invalid shape")
        arrays = (keypoints, confidences, boxes, person_scores)
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("Cached pose frame contains non-finite values")
        if keypoints.size and (np.any(keypoints < 0) or np.any(keypoints > 1)):
            raise ValueError("Cached pose keypoints are not normalized")
        if confidences.size and (np.any(confidences < 0) or np.any(confidences > 1)):
            raise ValueError("Cached pose confidences are outside [0, 1]")
        if boxes.size and (
            np.any(boxes < 0)
            or np.any(boxes > 1)
            or np.any(boxes[:, 2] < boxes[:, 0])
            or np.any(boxes[:, 3] < boxes[:, 1])
        ):
            raise ValueError("Cached pose boxes are invalid")
        if person_scores.size and (
            np.any(person_scores < 0) or np.any(person_scores > 1)
        ):
            raise ValueError("Cached pose person scores are outside [0, 1]")
        if (
            not isinstance(image_size_value, (list, tuple))
            or len(image_size_value) != 2
            or any(isinstance(item, bool) for item in image_size_value)
        ):
            raise ValueError("Cached pose image size is invalid")
        try:
            image_size = tuple(int(item) for item in image_size_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Cached pose image size is invalid") from exc
        if any(item <= 0 or item > 1_000_000 for item in image_size):
            raise ValueError("Cached pose image size is invalid")
        model_key = value.get("model_key", RTMO_L_MODEL_KEY)
        provider = value.get("provider", "")
        if not isinstance(model_key, str) or not model_key or len(model_key) > 500:
            raise ValueError("Cached pose model key is invalid")
        if not isinstance(provider, str) or len(provider) > 100:
            raise ValueError("Cached pose provider is invalid")
        return cls(
            keypoints=np.ascontiguousarray(keypoints),
            confidences=np.ascontiguousarray(confidences),
            boxes=np.ascontiguousarray(boxes),
            person_scores=np.ascontiguousarray(person_scores),
            image_size=(image_size[0], image_size[1]),
            model_key=model_key,
            provider=provider,
        )


@dataclass(frozen=True, slots=True)
class PoseGeometryMatch:
    score: float
    shape_score: float
    group_score: float
    coverage: float
    count_score: float
    mirrored: bool
    matched_pairs: tuple[tuple[int, int], ...]
    reference_count: int
    candidate_count: int
    common_joints: int
    mean_joint_confidence: float
    minimum_body_confidence: float

    @property
    def reliable(self) -> bool:
        return bool(
            self.matched_pairs
            and self.common_joints >= 5
            and self.coverage >= 0.45
            and self.mean_joint_confidence >= 0.25
            and self.minimum_body_confidence >= 0.15
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "shape_score": self.shape_score,
            "group_score": self.group_score,
            "coverage": self.coverage,
            "count_score": self.count_score,
            "mirrored": self.mirrored,
            "matched_pairs": [list(pair) for pair in self.matched_pairs],
            "reference_count": self.reference_count,
            "candidate_count": self.candidate_count,
            "common_joints": self.common_joints,
            "mean_joint_confidence": self.mean_joint_confidence,
            "minimum_body_confidence": self.minimum_body_confidence,
            "reliable": self.reliable,
        }


@dataclass(frozen=True, slots=True)
class _ImageTransform:
    width: int
    height: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float


def scene_kind(person_count: int) -> str:
    if person_count <= 0:
        return "none"
    if person_count == 1:
        return "solo"
    if person_count == 2:
        return "couple"
    return "group"


class RTMOPoseEstimator:
    """Pinned RTMO-L ONNX estimator with transparent CUDA-to-CPU fallback.

    Model preparation is lazy and safe for a persistent Docker data volume.
    The ONNX session is also lazy, so merely starting GalleryFlow neither
    downloads a model nor reserves GPU memory.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        execution_provider: str = "auto",
        model_url: str = RTMO_L_ARCHIVE_URL,
        archive_sha256: str = RTMO_L_ARCHIVE_SHA256,
        model_sha256: str = RTMO_L_MODEL_SHA256,
        model_key: str = RTMO_L_MODEL_KEY,
        detection_threshold: float = 0.25,
        joint_threshold: float = 0.15,
        min_visible_joints: int = 3,
        min_mean_joint_score: float = 0.08,
        nms_threshold: float = 0.45,
        max_people: int = 8,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES,
        max_model_bytes: int = DEFAULT_MAX_MODEL_BYTES,
        session_threads: int | None = None,
        downloader: ArchiveDownloader | None = None,
        session_factory: SessionFactory | None = None,
        available_providers: AvailableProviders | None = None,
    ) -> None:
        preference = execution_provider.strip().lower()
        if preference not in {"auto", "cuda", "cpu"}:
            raise ValueError("execution_provider must be auto, cuda, or cpu")
        for name, digest in {
            "archive_sha256": archive_sha256,
            "model_sha256": model_sha256,
        }.items():
            if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                raise ValueError(f"{name} must be a 64-character hexadecimal digest")
        if not 0 <= detection_threshold <= 1:
            raise ValueError("detection_threshold must be between 0 and 1")
        if not 0 <= joint_threshold <= 1:
            raise ValueError("joint_threshold must be between 0 and 1")
        if not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")
        if min_visible_joints < 1 or min_visible_joints > RTMO_KEYPOINT_COUNT:
            raise ValueError("min_visible_joints is outside the COCO keypoint range")
        if max_people < 1:
            raise ValueError("max_people must be positive")

        threads = session_threads if session_threads is not None else (os.cpu_count() or 1)
        self.model_path = Path(model_path)
        self.model_url = model_url
        self.archive_sha256 = archive_sha256.lower()
        self.model_sha256 = model_sha256.lower()
        self.model_key = model_key
        self.requested_provider = preference
        self.detection_threshold = float(detection_threshold)
        self.joint_threshold = float(joint_threshold)
        self.min_visible_joints = int(min_visible_joints)
        self.min_mean_joint_score = float(min_mean_joint_score)
        self.nms_threshold = float(nms_threshold)
        self.max_people = int(max_people)
        self.max_image_bytes = int(max_image_bytes)
        self.max_image_pixels = int(max_image_pixels)
        self.max_archive_bytes = int(max_archive_bytes)
        self.max_model_bytes = int(max_model_bytes)
        self.session_threads = min(MAX_INFERENCE_THREADS, max(1, int(threads)))

        self._downloader = downloader or self._download_archive
        self._session_factory = session_factory or self._create_session
        self._available_providers = available_providers or self._runtime_providers
        self._prepare_lock = asyncio.Lock()
        self._session_lock = threading.Lock()
        self._session: _InferenceSession | None = None
        self._input_name = ""
        self._active_provider = ""
        self._provider_fallback = False
        self._provider_message = ""

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def provider_status(self) -> dict[str, Any]:
        try:
            available = list(self._available_providers())
        except Exception:
            available = []
        return {
            "requested": self.requested_provider,
            "active": self._active_provider or None,
            "available": available,
            "fallback": self._provider_fallback,
            "message": self._provider_message,
        }

    async def prepare(self) -> Path:
        """Verify or atomically install the official model from its SDK zip."""

        async with self._prepare_lock:
            if self.model_path.is_symlink():
                raise PoseModelPreparationError("The pose model path may not be a symlink")
            if self.model_path.exists():
                if not self.model_path.is_file():
                    raise PoseModelPreparationError("The pose model path is not a regular file")
                if await self._sha256_file(self.model_path) == self.model_sha256:
                    return self.model_path

            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise PoseModelPreparationError(
                    f"Could not create pose model directory: {self.model_path.parent}"
                ) from exc

            token = uuid.uuid4().hex
            archive_path = self.model_path.with_name(f".{self.model_path.name}.{token}.zip.part")
            model_part = self.model_path.with_name(f".{self.model_path.name}.{token}.onnx.part")
            try:
                value = self._downloader(self.model_url, archive_path)
                if hasattr(value, "__await__"):
                    await value
                if archive_path.is_symlink() or not archive_path.is_file():
                    raise PoseModelPreparationError("Pose model downloader created no regular file")
                if archive_path.stat().st_size > self.max_archive_bytes:
                    raise PoseModelPreparationError("Pose model archive exceeds the size limit")
                actual_archive = await self._sha256_file(archive_path)
                if actual_archive != self.archive_sha256:
                    raise PoseModelIntegrityError(
                        "Downloaded RTMO-L archive failed SHA-256 verification "
                        f"(expected {self.archive_sha256}, got {actual_archive})"
                    )
                self._extract_verified_model(archive_path, model_part)
                actual_model = await self._sha256_file(model_part)
                if actual_model != self.model_sha256:
                    raise PoseModelIntegrityError(
                        "Extracted RTMO-L model failed SHA-256 verification "
                        f"(expected {self.model_sha256}, got {actual_model})"
                    )
                os.replace(model_part, self.model_path)
            except (PoseModelPreparationError, asyncio.CancelledError):
                raise
            except (OSError, zipfile.BadZipFile, KeyError) as exc:
                raise PoseModelPreparationError("Could not install the RTMO-L ONNX model") from exc
            except Exception as exc:
                raise PoseModelPreparationError(
                    f"Could not prepare RTMO-L model from {self.model_url}"
                ) from exc
            finally:
                archive_path.unlink(missing_ok=True)
                model_part.unlink(missing_ok=True)

            with self._session_lock:
                self._session = None
                self._input_name = ""
                self._active_provider = ""
            return self.model_path

    def infer_bytes(self, data: bytes | bytearray | memoryview) -> PoseFrame:
        image = self._decode_image(data)
        tensor, transform = self._preprocess(image)
        session, input_name = self._get_session()
        try:
            outputs = session.run(None, {input_name: tensor})
        except Exception as exc:
            raise PoseInferenceError("RTMO-L ONNX inference failed") from exc
        return self._decode_outputs(outputs, transform)

    async def _download_archive(self, url: str, destination: Path) -> None:
        timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    announced = response.headers.get("content-length")
                    if announced and int(announced) > self.max_archive_bytes:
                        raise PoseModelPreparationError("Pose model archive exceeds the size limit")
                    total = 0
                    with destination.open("xb") as handle:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > self.max_archive_bytes:
                                raise PoseModelPreparationError(
                                    "Pose model archive exceeds the size limit"
                                )
                            handle.write(chunk)
        except PoseModelPreparationError:
            raise
        except Exception as exc:
            raise PoseModelPreparationError("Could not download the RTMO-L archive") from exc

    @staticmethod
    async def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                await asyncio.sleep(0)
        return digest.hexdigest()

    def _extract_verified_model(self, archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                member = archive.getinfo(RTMO_L_ARCHIVE_MEMBER)
            except KeyError as exc:
                raise PoseModelPreparationError(
                    f"RTMO-L archive has no {RTMO_L_ARCHIVE_MEMBER} member"
                ) from exc
            # Reject directories, links, encrypted entries and zip bombs.  We
            # stream this one exact member; extractall/path traversal is never used.
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            if member.is_dir() or (unix_mode and (unix_mode & 0o170000) == 0o120000):
                raise PoseModelPreparationError("RTMO-L model member is not a regular file")
            if member.flag_bits & 0x1:
                raise PoseModelPreparationError("Encrypted RTMO-L archives are not supported")
            if member.file_size <= 0 or member.file_size > self.max_model_bytes:
                raise PoseModelPreparationError("RTMO-L model member has an invalid size")
            total = 0
            with archive.open(member, "r") as source, destination.open("xb") as target:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_model_bytes:
                        raise PoseModelPreparationError("RTMO-L model exceeds the size limit")
                    target.write(chunk)
            if total != member.file_size:
                raise PoseModelIntegrityError("RTMO-L model member was truncated")

    @staticmethod
    def _runtime_providers() -> Sequence[str]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PoseModelPreparationError("onnxruntime is required for pose inference") from exc
        return ort.get_available_providers()

    @staticmethod
    def _create_session(path: Path, providers: list[str], threads: int) -> _InferenceSession:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise PoseModelPreparationError("onnxruntime is required for pose inference") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), sess_options=options, providers=providers)

    def _provider_order(self) -> list[str]:
        available = set(self._available_providers())
        has_cuda = "CUDAExecutionProvider" in available
        has_cpu = "CPUExecutionProvider" in available
        if not has_cpu and not has_cuda:
            raise PoseModelPreparationError("ONNX Runtime has no CPU or CUDA provider")
        if self.requested_provider == "cpu":
            if not has_cpu:
                raise PoseModelPreparationError("CPUExecutionProvider is unavailable")
            return ["CPUExecutionProvider"]
        if self.requested_provider == "cuda":
            if not has_cuda:
                raise PoseModelPreparationError("CUDAExecutionProvider is unavailable")
            # Do not silently run an explicitly CUDA-requested deployment on
            # CPU. Users can select auto when graceful fallback is wanted.
            return ["CUDAExecutionProvider"]
        if has_cuda:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if has_cpu else ["CUDAExecutionProvider"]
        if has_cpu:
            self._provider_fallback = True
            self._provider_message = "CUDAExecutionProvider is unavailable; using CPU"
            return ["CPUExecutionProvider"]
        raise PoseModelPreparationError("CUDAExecutionProvider is unavailable")

    def _get_session(self) -> tuple[_InferenceSession, str]:
        with self._session_lock:
            if self._session is not None:
                return self._session, self._input_name
            if not self.model_path.is_file():
                raise PoseModelNotPreparedError("Call prepare() before RTMO-L inference")
            providers = self._provider_order()
            try:
                session = self._session_factory(self.model_path, providers, self.session_threads)
            except Exception as exc:
                if providers[0] != "CUDAExecutionProvider" or "CPUExecutionProvider" not in providers:
                    raise PoseInferenceError("Could not create the RTMO-L ONNX session") from exc
                try:
                    session = self._session_factory(
                        self.model_path, ["CPUExecutionProvider"], self.session_threads
                    )
                except Exception as cpu_exc:
                    raise PoseInferenceError("Could not create the RTMO-L ONNX session") from cpu_exc
                self._provider_fallback = True
                self._provider_message = "CUDA session initialization failed; using CPU"
            inputs = session.get_inputs()
            if len(inputs) != 1 or not getattr(inputs[0], "name", ""):
                raise PoseInferenceError("RTMO-L has an unexpected input contract")
            active = list(session.get_providers()) if hasattr(session, "get_providers") else []
            self._active_provider = active[0] if active else providers[0]
            if (
                self.requested_provider == "cuda"
                and self._active_provider != "CUDAExecutionProvider"
            ):
                raise PoseInferenceError(
                    "CUDA was explicitly requested but ONNX Runtime did not activate it"
                )
            if providers[0] == "CUDAExecutionProvider" and self._active_provider != "CUDAExecutionProvider":
                self._provider_fallback = True
                if not self._provider_message:
                    self._provider_message = "ONNX Runtime fell back from CUDA to CPU"
            self._session = session
            self._input_name = inputs[0].name
            return session, self._input_name

    def _decode_image(self, data: bytes | bytearray | memoryview) -> Image.Image:
        payload = bytes(data)
        if not payload:
            raise PoseInvalidImageError("Image is empty")
        if len(payload) > self.max_image_bytes:
            raise PoseInvalidImageError("Image exceeds the pose byte limit")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.width * source.height > self.max_image_pixels:
                    raise PoseInvalidImageError("Image exceeds the pose pixel limit")
                return ImageOps.exif_transpose(source).convert("RGB")
        except PoseInvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise PoseInvalidImageError("Image could not be decoded safely") from exc

    @staticmethod
    def _preprocess(image: Image.Image) -> tuple[np.ndarray, _ImageTransform]:
        width, height = image.size
        scale = min(RTMO_INPUT_SIZE / width, RTMO_INPUT_SIZE / height)
        resized_width = min(RTMO_INPUT_SIZE, max(1, int(width * scale)))
        resized_height = min(RTMO_INPUT_SIZE, max(1, int(height * scale)))
        # MMDeploy's generic BottomupResize uses a centered affine warp, but
        # the official lightweight RTMO ONNX implementation shipped alongside
        # this artifact resizes into the top-left of a 114-valued canvas.
        # Matching that implementation is essential for useful detections.
        pad_left = 0
        pad_top = 0
        resized = image.resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        rgb = np.asarray(resized, dtype=np.uint8)
        canvas = np.full((RTMO_INPUT_SIZE, RTMO_INPUT_SIZE, 3), 114, dtype=np.uint8)
        # The official SDK loads through OpenCV and declares to_rgb=false.
        canvas[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ] = rgb[:, :, ::-1]
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        transform = _ImageTransform(
            width=width,
            height=height,
            scale_x=scale,
            scale_y=scale,
            pad_x=float(pad_left),
            pad_y=float(pad_top),
        )
        return tensor, transform

    def _decode_outputs(self, outputs: Sequence[Any], transform: _ImageTransform) -> PoseFrame:
        if len(outputs) < 2:
            raise PoseInferenceError("RTMO-L returned fewer than two tensors")
        dets = np.asarray(outputs[0], dtype=np.float32)
        keypoints = np.asarray(outputs[1], dtype=np.float32)
        if dets.ndim == 3 and dets.shape[0] == 1:
            dets = dets[0]
        if keypoints.ndim == 4 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if (
            dets.ndim != 2
            or dets.shape[1] < 5
            or keypoints.ndim != 3
            or keypoints.shape[0] != dets.shape[0]
            or keypoints.shape[1:] != (RTMO_KEYPOINT_COUNT, 3)
            or not np.isfinite(dets).all()
            or not np.isfinite(keypoints).all()
        ):
            raise PoseInferenceError(
                f"Unexpected RTMO-L output shapes: {dets.shape}, {keypoints.shape}"
            )

        normalized = keypoints[:, :, :2].copy()
        normalized[:, :, 0] = (
            (normalized[:, :, 0] - transform.pad_x)
            / transform.scale_x
            / transform.width
        )
        normalized[:, :, 1] = (
            (normalized[:, :, 1] - transform.pad_y)
            / transform.scale_y
            / transform.height
        )
        inside = (
            (normalized[:, :, 0] >= 0)
            & (normalized[:, :, 0] <= 1)
            & (normalized[:, :, 1] >= 0)
            & (normalized[:, :, 1] <= 1)
        )
        confidences = np.clip(keypoints[:, :, 2], 0, 1)
        confidences = np.where(inside, confidences, 0).astype(np.float32)
        normalized = np.clip(normalized, 0, 1).astype(np.float32)

        boxes = dets[:, :4].copy()
        boxes[:, 0::2] = (
            (boxes[:, 0::2] - transform.pad_x)
            / transform.scale_x
            / transform.width
        )
        boxes[:, 1::2] = (
            (boxes[:, 1::2] - transform.pad_y)
            / transform.scale_y
            / transform.height
        )
        boxes = np.clip(boxes, 0, 1).astype(np.float32)
        detection_scores = np.clip(dets[:, 4], 0, 1)
        visible = confidences >= self.joint_threshold
        visible_count = visible.sum(axis=1)
        joint_mean = confidences.mean(axis=1)
        keep = (
            (detection_scores >= self.detection_threshold)
            & (visible_count >= self.min_visible_joints)
            & (joint_mean >= self.min_mean_joint_score)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        indices = np.flatnonzero(keep)
        if indices.size:
            indices = indices[
                self._class_agnostic_nms(
                    boxes[indices], detection_scores[indices], self.nms_threshold
                )
            ]
            quality = detection_scores[indices] * (
                0.5 + 0.5 * np.take(joint_mean, indices)
            )
            quality_order = np.argsort(-quality, kind="stable")[: self.max_people]
            indices = indices[quality_order]
            person_scores = quality[quality_order]
        else:
            person_scores = np.empty((0,), dtype=np.float32)

        return PoseFrame(
            keypoints=np.ascontiguousarray(normalized[indices], dtype=np.float32),
            confidences=np.ascontiguousarray(confidences[indices], dtype=np.float32),
            boxes=np.ascontiguousarray(boxes[indices], dtype=np.float32),
            person_scores=np.ascontiguousarray(person_scores, dtype=np.float32),
            image_size=(transform.width, transform.height),
            model_key=self.model_key,
            provider=self._active_provider,
        )

    @staticmethod
    def _class_agnostic_nms(
        boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> np.ndarray:
        """Return deterministic local indices after xyxy IoU suppression."""

        if not len(boxes):
            return np.empty((0,), dtype=np.int64)
        order = np.argsort(-scores, kind="stable")
        keep: list[int] = []
        while order.size:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area_current = max(
                0.0,
                float(
                    (boxes[current, 2] - boxes[current, 0])
                    * (boxes[current, 3] - boxes[current, 1])
                ),
            )
            area_rest = np.maximum(
                0, boxes[rest, 2] - boxes[rest, 0]
            ) * np.maximum(0, boxes[rest, 3] - boxes[rest, 1])
            union = area_current + area_rest - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 1e-12,
            )
            order = rest[iou <= threshold]
        return np.asarray(keep, dtype=np.int64)


def mirror_pose_frame(frame: PoseFrame) -> PoseFrame:
    """Mirror a scene and restore anatomical left/right COCO semantics."""

    keypoints = frame.keypoints.copy()
    keypoints[:, :, 0] = 1.0 - keypoints[:, :, 0]
    confidences = frame.confidences.copy()
    for left, right in COCO_LEFT_RIGHT_PAIRS:
        keypoints[:, [left, right]] = keypoints[:, [right, left]]
        confidences[:, [left, right]] = confidences[:, [right, left]]
    boxes = frame.boxes.copy()
    if boxes.size:
        old_left = boxes[:, 0].copy()
        boxes[:, 0] = 1.0 - boxes[:, 2]
        boxes[:, 2] = 1.0 - old_left
    return PoseFrame(
        keypoints=keypoints,
        confidences=confidences,
        boxes=boxes,
        person_scores=frame.person_scores.copy(),
        image_size=frame.image_size,
        model_key=frame.model_key,
        provider=frame.provider,
    )


def _person_similarity(
    ref_points: np.ndarray,
    ref_conf: np.ndarray,
    cand_points: np.ndarray,
    cand_conf: np.ndarray,
) -> tuple[float, float, int, float]:
    # Very low-confidence joints are missing observations, not coordinates at
    # (0, 0). Confidence remains continuous above this numerical floor.
    mask = (ref_conf >= 0.05) & (cand_conf >= 0.05)
    common_joints = int(mask.sum())
    if common_joints < 3:
        return 0.0, 0.0, common_joints, 0.0
    rc = np.clip(ref_conf[mask], 0, 1)
    cc = np.clip(cand_conf[mask], 0, 1)
    weights = np.sqrt(rc * cc)
    total = float(weights.sum())
    if total <= 1e-6:
        return 0.0, 0.0, common_joints, 0.0
    rp = ref_points[mask]
    cp = cand_points[mask]
    r_center = np.sum(rp * weights[:, None], axis=0) / total
    c_center = np.sum(cp * weights[:, None], axis=0) / total
    rp = rp - r_center
    cp = cp - c_center
    r_scale = math.sqrt(float(np.sum(weights * np.sum(rp * rp, axis=1)) / total))
    c_scale = math.sqrt(float(np.sum(weights * np.sum(cp * cp, axis=1)) / total))
    if r_scale <= 1e-5 or c_scale <= 1e-5:
        return 0.0, 0.0, common_joints, float(np.mean(weights))
    delta = rp / r_scale - cp / c_scale
    rmse = math.sqrt(float(np.sum(weights * np.sum(delta * delta, axis=1)) / total))
    shape = math.exp(-((rmse / 0.62) ** 2))
    # Cosine-like confidence overlap means one missing joint has a small cost,
    # while a candidate supported by only a few of the reference joints cannot
    # receive a deceptively perfect score.
    coverage = float(
        np.sum(np.sqrt(np.clip(ref_conf, 0, 1) * np.clip(cand_conf, 0, 1)))
        / max(
            1e-6,
            math.sqrt(float(np.sum(ref_conf)) * float(np.sum(cand_conf))),
        )
    )
    coverage = max(0.0, min(1.0, coverage))
    return (
        shape * (0.55 + 0.45 * coverage),
        coverage,
        common_joints,
        float(np.mean(weights)),
    )


def _optimal_assignment(similarity: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Maximum-weight rectangular assignment using a small bitmask DP."""

    rows, cols = similarity.shape
    if not rows or not cols:
        return ()
    transpose = rows > cols
    matrix = similarity.T if transpose else similarity
    small, large = matrix.shape
    states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
    for row in range(small):
        next_states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {}
        for mask, (score, pairs) in states.items():
            for col in range(large):
                bit = 1 << col
                if mask & bit:
                    continue
                candidate = (score + float(matrix[row, col]), pairs + ((row, col),))
                previous = next_states.get(mask | bit)
                if previous is None or candidate[0] > previous[0]:
                    next_states[mask | bit] = candidate
        states = next_states
    _, pairs = max(states.values(), key=lambda item: item[0])
    if transpose:
        return tuple((col, row) for row, col in pairs)
    return pairs


def _person_centers_and_scales(frame: PoseFrame) -> tuple[np.ndarray, np.ndarray]:
    centers: list[np.ndarray] = []
    scales: list[float] = []
    for points, confidence in zip(frame.keypoints, frame.confidences, strict=True):
        mask = confidence >= 0.05
        weights = confidence[mask]
        selected = points[mask]
        if selected.size:
            center = np.sum(selected * weights[:, None], axis=0) / max(1e-6, float(weights.sum()))
            scale = math.sqrt(
                float(
                    np.sum(weights * np.sum((selected - center) ** 2, axis=1))
                    / max(1e-6, float(weights.sum()))
                )
            )
        else:
            center = np.asarray((0.5, 0.5), dtype=np.float32)
            scale = 0.0
        centers.append(center)
        scales.append(scale)
    return np.asarray(centers, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def _group_similarity(
    reference: PoseFrame,
    candidate: PoseFrame,
    pairs: tuple[tuple[int, int], ...],
) -> float:
    if len(pairs) < 2:
        return 1.0
    r_centers, r_scales = _person_centers_and_scales(reference)
    c_centers, c_scales = _person_centers_and_scales(candidate)
    r_idx = np.asarray([pair[0] for pair in pairs])
    c_idx = np.asarray([pair[1] for pair in pairs])
    rc = r_centers[r_idx]
    cc = c_centers[c_idx]
    rs = np.maximum(r_scales[r_idx], 1e-4)
    cs = np.maximum(c_scales[c_idx], 1e-4)
    # Compare spacing in units of body size. Translation and overall image
    # scale/crop disappear, but relative placement and person-size ratios stay.
    rc = (rc - rc.mean(axis=0)) / max(1e-4, float(rs.mean()))
    cc = (cc - cc.mean(axis=0)) / max(1e-4, float(cs.mean()))
    center_rmse = math.sqrt(float(np.mean(np.sum((rc - cc) ** 2, axis=1))))
    rs = rs / max(1e-4, float(rs.mean()))
    cs = cs / max(1e-4, float(cs.mean()))
    scale_rmse = math.sqrt(float(np.mean((np.log(rs) - np.log(cs)) ** 2)))
    return math.exp(-((center_rmse / 0.8) ** 2) - ((scale_rmse / 0.55) ** 2))


def _geometry_one_direction(reference: PoseFrame, candidate: PoseFrame) -> PoseGeometryMatch:
    r_count = reference.person_count
    c_count = candidate.person_count
    if not r_count or not c_count:
        return PoseGeometryMatch(
            0, 0, 0, 0, 0, False, (), r_count, c_count, 0, 0, 0
        )
    similarities = np.zeros((r_count, c_count), dtype=np.float32)
    coverages = np.zeros_like(similarities)
    common_joints = np.zeros_like(similarities, dtype=np.int16)
    joint_evidence = np.zeros_like(similarities)
    for r_index in range(r_count):
        for c_index in range(c_count):
            (
                similarities[r_index, c_index],
                coverages[r_index, c_index],
                common_joints[r_index, c_index],
                joint_evidence[r_index, c_index],
            ) = _person_similarity(
                reference.keypoints[r_index],
                reference.confidences[r_index],
                candidate.keypoints[c_index],
                candidate.confidences[c_index],
            )
    pairs = _optimal_assignment(similarities)
    if not pairs:
        return PoseGeometryMatch(
            0, 0, 0, 0, 0, False, (), r_count, c_count, 0, 0, 0
        )
    shape = float(np.mean([similarities[r, c] for r, c in pairs]))
    coverage = float(np.mean([coverages[r, c] for r, c in pairs]))
    group = _group_similarity(reference, candidate, pairs)
    count_score = math.exp(-0.55 * abs(r_count - c_count))
    score = (0.78 * shape + 0.22 * group) * count_score
    return PoseGeometryMatch(
        score=max(0.0, min(1.0, score)),
        shape_score=shape,
        group_score=group,
        coverage=coverage,
        count_score=count_score,
        mirrored=False,
        matched_pairs=pairs,
        reference_count=r_count,
        candidate_count=c_count,
        common_joints=min(int(common_joints[r, c]) for r, c in pairs),
        mean_joint_confidence=float(
            np.mean([joint_evidence[r, c] for r, c in pairs])
        ),
        minimum_body_confidence=min(
            min(
                float(np.mean(reference.confidences[r, 5:])),
                float(np.mean(candidate.confidences[c, 5:])),
            )
            for r, c in pairs
        ),
    )


def pose_geometry_match(
    reference: PoseFrame,
    candidate: PoseFrame,
    *,
    allow_mirror: bool = True,
) -> PoseGeometryMatch:
    """Compare solo, couple, or group geometry independent of crop and scale."""

    direct = _geometry_one_direction(reference, candidate)
    if not allow_mirror or not reference.person_count or not candidate.person_count:
        return direct
    mirrored = _geometry_one_direction(reference, mirror_pose_frame(candidate))
    if mirrored.score <= direct.score:
        return direct
    return PoseGeometryMatch(
        score=mirrored.score,
        shape_score=mirrored.shape_score,
        group_score=mirrored.group_score,
        coverage=mirrored.coverage,
        count_score=mirrored.count_score,
        mirrored=True,
        matched_pairs=mirrored.matched_pairs,
        reference_count=mirrored.reference_count,
        candidate_count=mirrored.candidate_count,
        common_joints=mirrored.common_joints,
        mean_joint_confidence=mirrored.mean_joint_confidence,
        minimum_body_confidence=mirrored.minimum_body_confidence,
    )


def skeleton_svg(
    frame: PoseFrame,
    *,
    width: int = 640,
    height: int = 640,
    joint_threshold: float = 0.15,
) -> str:
    """Render a compact transparent SVG overlay in normalized image space."""

    width = max(1, min(4096, int(width)))
    height = max(1, min(4096, int(height)))
    colors = ("#ff4f81", "#49c6ff", "#ffd166", "#8ee28e", "#c39bff", "#ff9966")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="none">'
    ]
    for person, (points, confidence) in enumerate(
        zip(frame.keypoints, frame.confidences, strict=True)
    ):
        color = colors[person % len(colors)]
        for start, end in COCO_SKELETON:
            if confidence[start] < joint_threshold or confidence[end] < joint_threshold:
                continue
            x1, y1 = points[start] * (width, height)
            x2, y2 = points[end] * (width, height)
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="{color}" stroke-width="3" stroke-linecap="round" opacity=".9"/>'
            )
        for (x, y), score in zip(points, confidence, strict=True):
            if score < joint_threshold:
                continue
            parts.append(
                f'<circle cx="{x * width:.2f}" cy="{y * height:.2f}" r="3.5" '
                f'fill="{color}" stroke="#101116" stroke-width="1"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def skeleton_data_uri(frame: PoseFrame, **kwargs: Any) -> str:
    encoded = base64.b64encode(skeleton_svg(frame, **kwargs).encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"
