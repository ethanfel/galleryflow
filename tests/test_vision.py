from __future__ import annotations

import asyncio
import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageOps

from app.vision import (
    DINOV2_EMBEDDING_DIMENSION,
    DINOV2_MODEL_KEY,
    DINOV2_MODEL_SHA256,
    DINOV2_MODEL_URL,
    DINOV2_SPATIAL_DIMENSION,
    DINOV2_SPATIAL_KEY,
    DINOV2_SPATIAL_LEVELS,
    DinoV2Encoder,
    InvalidImageError,
    ModelIntegrityError,
    ModelNotPreparedError,
    VisionInferenceError,
    hamming_similarity64,
    perceptual_hash64,
    perceptual_hash_bytes,
)


class FakeSession:
    def __init__(self, *, input_name: str = "pixel_values") -> None:
        self.input_name = input_name
        self.batches: list[np.ndarray] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=self.input_name)]

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        batch = np.asarray(input_feed[self.input_name])
        self.batches.append(batch.copy())
        token_count = (batch.shape[2] // 14) * (batch.shape[3] // 14) + 1
        output = np.zeros(
            (batch.shape[0], token_count, DINOV2_EMBEDDING_DIMENSION),
            dtype=np.float32,
        )
        for index in range(batch.shape[0]):
            output[index, 0, index] = float(index + 2)
        return [output]


class SpatialFakeSession(FakeSession):
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names is None
        batch = np.asarray(input_feed[self.input_name])
        self.batches.append(batch.copy())
        rows = batch.shape[2] // 14
        columns = batch.shape[3] // 14
        output = np.zeros(
            (
                batch.shape[0],
                rows * columns + 1,
                DINOV2_EMBEDDING_DIMENSION,
            ),
            dtype=np.float32,
        )
        for view in range(batch.shape[0]):
            output[view, 0, view] = 2.0
            for row in range(rows):
                for column in range(columns):
                    token = 1 + row * columns + column
                    output[view, token, 0] = 1.0
                    output[view, token, 1] = (row + 1) / rows
                    output[view, token, 2] = (column + 1) / columns
                    output[view, token, 3 + view] = 0.5
        return [output]


def image_bytes(
    size: tuple[int, int],
    *,
    color: tuple[int, int, int] = (240, 30, 20),
    image_format: str = "PNG",
    orientation: int | None = None,
) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    options: dict[str, Any] = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[274] = orientation
        options["exif"] = exif
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


def encoder_with_fake_session(
    tmp_path: Path,
    *,
    session: FakeSession | None = None,
    session_threads: int | None = None,
    **values: Any,
) -> tuple[DinoV2Encoder, FakeSession, list[tuple[Path, int]]]:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-model-for-injected-session")
    fake = session or FakeSession()
    calls: list[tuple[Path, int]] = []

    def factory(path: Path, threads: int) -> FakeSession:
        calls.append((path, threads))
        return fake

    encoder = DinoV2Encoder(
        model_path,
        session_factory=factory,
        session_threads=session_threads,
        **values,
    )
    return encoder, fake, calls


def test_model_manifest_is_pinned_to_verified_fp32_artifact() -> None:
    assert DINOV2_MODEL_KEY == (
        "dinov2-small-onnx@08c606e3123472a388efa59181b677d428f69bbd:"
        "contain-224x336-v1:cls-l2-fp32"
    )
    assert DINOV2_MODEL_URL == (
        "https://huggingface.co/onnx-community/dinov2-small-ONNX/resolve/"
        "08c606e3123472a388efa59181b677d428f69bbd/onnx/model.onnx"
    )
    assert DINOV2_MODEL_SHA256 == (
        "6266c3cd72db6953cecdcbfeab9422a9f783d96f1a4e296ba70ffbac43b54a18"
    )
    assert "int8" not in DINOV2_MODEL_URL.lower()
    assert DINOV2_SPATIAL_LEVELS == (1, 2, 4)
    assert DINOV2_SPATIAL_DIMENSION == 8_064
    assert "spatial-pyramid-v1" in DINOV2_SPATIAL_KEY


@pytest.mark.asyncio
async def test_prepare_downloads_verifies_and_atomically_publishes(
    tmp_path: Path,
) -> None:
    payload = b"small deterministic stand-in for the ONNX artifact"
    digest = hashlib.sha256(payload).hexdigest()
    model_path = tmp_path / "nested" / "encoder.onnx"
    downloads: list[tuple[str, Path]] = []

    async def downloader(url: str, destination: Path) -> None:
        downloads.append((url, destination))
        assert not model_path.exists()
        assert destination.parent == model_path.parent
        assert destination.name.startswith(".encoder.onnx.")
        assert destination.name.endswith(".part")
        destination.write_bytes(payload)

    encoder = DinoV2Encoder(
        model_path,
        model_url="https://models.example/dinov2.onnx",
        model_sha256=digest,
        downloader=downloader,
    )

    assert await encoder.prepare() == model_path
    assert model_path.read_bytes() == payload
    assert len(downloads) == 1
    assert list(model_path.parent.glob("*.part")) == []

    # A second prepare verifies the existing artifact and remains network-free.
    assert await encoder.prepare() == model_path
    assert len(downloads) == 1


@pytest.mark.asyncio
async def test_prepare_never_publishes_a_bad_download(tmp_path: Path) -> None:
    model_path = tmp_path / "encoder.onnx"
    original = b"existing corrupt artifact"
    model_path.write_bytes(original)
    expected = hashlib.sha256(b"expected artifact").hexdigest()

    async def downloader(_: str, destination: Path) -> None:
        destination.write_bytes(b"also corrupt")

    encoder = DinoV2Encoder(
        model_path,
        model_sha256=expected,
        downloader=downloader,
    )

    with pytest.raises(ModelIntegrityError, match="SHA-256"):
        await encoder.prepare()
    assert model_path.read_bytes() == original
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.asyncio
async def test_concurrent_prepare_downloads_once(tmp_path: Path) -> None:
    payload = b"valid model"
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def downloader(_: str, destination: Path) -> None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        destination.write_bytes(payload)

    encoder = DinoV2Encoder(
        tmp_path / "model.onnx",
        model_sha256=hashlib.sha256(payload).hexdigest(),
        downloader=downloader,
    )
    first = asyncio.create_task(encoder.prepare())
    await started.wait()
    second = asyncio.create_task(encoder.prepare())
    release.set()
    await first
    await second
    assert calls == 1


def test_embed_uses_portrait_landscape_and_square_canvases(tmp_path: Path) -> None:
    encoder, session, calls = encoder_with_fake_session(tmp_path)

    portrait = encoder.embed_bytes(image_bytes((100, 200)))
    landscape = encoder.embed_bytes(image_bytes((200, 100)))
    square = encoder.embed_bytes(image_bytes((100, 100)))

    assert [batch.shape for batch in session.batches] == [
        (1, 3, 336, 224),
        (1, 3, 224, 336),
        (1, 3, 280, 280),
    ]
    assert portrait.shape == landscape.shape == square.shape == (1, 384)
    assert portrait.dtype == np.float32
    assert np.allclose(np.linalg.norm(portrait, axis=1), 1.0)
    assert calls == [(tmp_path / "model.onnx", 4)]

    # A 1:2 portrait is contained at 168x336 with mean-colored side padding.
    portrait_tensor = session.batches[0][0]
    assert np.max(np.abs(portrait_tensor[:, :, 0])) < 0.02
    assert np.max(np.abs(portrait_tensor[:, :, -1])) < 0.02
    assert np.max(np.abs(portrait_tensor[:, 0, 112])) > 1.0
    assert np.max(np.abs(portrait_tensor[:, -1, 112])) > 1.0


def test_exif_orientation_is_applied_before_canvas_selection(tmp_path: Path) -> None:
    encoder, session, _ = encoder_with_fake_session(tmp_path)

    # EXIF orientation 6 rotates this stored portrait into a landscape image.
    encoded = image_bytes((40, 80), image_format="JPEG", orientation=6)
    encoder.embed_bytes(encoded)

    assert session.batches[0].shape == (1, 3, 224, 336)


def test_mirror_is_a_second_normalized_view(tmp_path: Path) -> None:
    encoder, session, _ = encoder_with_fake_session(tmp_path)
    image = Image.new("RGB", (120, 240), (255, 0, 0))
    image.paste((0, 0, 255), (60, 0, 120, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    embeddings = encoder.embed_bytes(buffer.getvalue(), include_mirror=True)

    batch = session.batches[0]
    assert batch.shape == (2, 3, 336, 224)
    assert np.allclose(batch[1], batch[0, :, :, ::-1])
    assert embeddings.shape == (2, 384)
    assert np.allclose(np.linalg.norm(embeddings, axis=1), (1.0, 1.0))
    assert not np.array_equal(embeddings[0], embeddings[1])


@pytest.mark.parametrize(
    ("size", "patch_grid"),
    [
        ((100, 200), (24, 16)),
        ((200, 100), (16, 24)),
        ((100, 100), (20, 20)),
    ],
)
def test_describe_builds_versioned_spatial_pyramid_in_one_inference(
    tmp_path: Path,
    size: tuple[int, int],
    patch_grid: tuple[int, int],
) -> None:
    session = SpatialFakeSession()
    encoder, _, _ = encoder_with_fake_session(tmp_path, session=session)

    description = encoder.describe_bytes(
        image_bytes(size), include_mirror=True
    )

    assert len(session.batches) == 1
    assert session.batches[0].shape[0] == 2
    assert (
        session.batches[0].shape[2] // 14,
        session.batches[0].shape[3] // 14,
    ) == patch_grid
    assert description.global_embeddings.shape == (2, 384)
    assert description.spatial_embeddings.shape == (2, 8_064)
    assert description.global_embeddings.dtype == np.float32
    assert description.spatial_embeddings.dtype == np.float32
    assert description.global_embeddings.flags.c_contiguous
    assert description.spatial_embeddings.flags.c_contiguous
    assert np.allclose(
        np.linalg.norm(description.global_embeddings, axis=1), 1.0
    )
    assert np.allclose(
        np.linalg.norm(description.spatial_embeddings, axis=1), 1.0
    )
    assert not np.allclose(
        description.spatial_embeddings[0], description.spatial_embeddings[1]
    )


def test_embed_spatial_bytes_uses_region_order_and_normalization(
    tmp_path: Path,
) -> None:
    encoder, _, _ = encoder_with_fake_session(
        tmp_path, session=SpatialFakeSession()
    )

    descriptor = encoder.embed_spatial_bytes(image_bytes((200, 100)))

    assert descriptor.shape == (1, DINOV2_SPATIAL_DIMENSION)
    regions = descriptor.reshape(1, 21, DINOV2_EMBEDDING_DIMENSION)
    # Each region was normalized before concatenation, so all 21 regions carry
    # equal weight after the final concatenated L2 normalization.
    expected_region_norm = 1.0 / np.sqrt(21.0)
    assert np.allclose(
        np.linalg.norm(regions, axis=2), expected_region_norm, atol=1e-6
    )
    assert not np.allclose(regions[:, 1, :], regions[:, -1, :])


def test_exact_patch_grid_and_nonfinite_patch_tokens_are_rejected(
    tmp_path: Path,
) -> None:
    class WrongGridSession(FakeSession):
        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            batch = np.asarray(input_feed[self.input_name])
            return [
                np.ones(
                    (batch.shape[0], 384, DINOV2_EMBEDDING_DIMENSION),
                    dtype=np.float32,
                )
            ]

    wrong_path = tmp_path / "wrong"
    wrong_path.mkdir()
    wrong, _, _ = encoder_with_fake_session(wrong_path, session=WrongGridSession())
    with pytest.raises(VisionInferenceError, match="output shape"):
        wrong.describe_bytes(image_bytes((100, 200)))

    class NonFinitePatchSession(SpatialFakeSession):
        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            output = super().run(output_names, input_feed)[0]
            output[0, -1, 20] = np.nan
            return [output]

    nonfinite_path = tmp_path / "nonfinite"
    nonfinite_path.mkdir()
    nonfinite, _, _ = encoder_with_fake_session(
        nonfinite_path, session=NonFinitePatchSession()
    )
    with pytest.raises(VisionInferenceError, match="non-finite"):
        nonfinite.embed_bytes(image_bytes((100, 200)))


def test_session_is_lazy_discovers_input_name_and_bounds_threads(
    tmp_path: Path,
) -> None:
    session = FakeSession(input_name="images")
    encoder, _, calls = encoder_with_fake_session(
        tmp_path, session=session, session_threads=999
    )
    encoded = image_bytes((20, 30))

    assert calls == []
    encoder.embed_bytes(encoded)
    encoder.embed_bytes(encoded)

    assert calls == [(tmp_path / "model.onnx", 4)]
    assert len(session.batches) == 2


@pytest.mark.parametrize(
    ("requested", "available", "expected", "cpu_fallback"),
    [
        (
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
        ),
        ("auto", ["CPUExecutionProvider"], ["CPUExecutionProvider"], True),
        (
            "cpu",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
            False,
        ),
    ],
)
def test_execution_provider_order_and_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    available: list[str],
    expected: list[str],
    cpu_fallback: bool,
) -> None:
    captured: list[list[str]] = []

    class FakeSessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0
        execution_mode: object | None = None
        graph_optimization_level: object | None = None

    class RuntimeSession(FakeSession):
        def __init__(self, providers: list[str]) -> None:
            super().__init__()
            self.providers = providers

        def get_providers(self) -> list[str]:
            return list(self.providers)

    def inference_session(
        _: str, *, sess_options: Any, providers: list[str]
    ) -> RuntimeSession:
        assert isinstance(sess_options, FakeSessionOptions)
        captured.append(list(providers))
        return RuntimeSession(providers)

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: list(available),
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model_path = tmp_path / f"{requested}-{len(available)}.onnx"
    model_path.write_bytes(b"model")
    encoder = DinoV2Encoder(model_path, execution_provider=requested)

    assert encoder.provider_status()["active"] is None
    encoder.embed_bytes(image_bytes((30, 60)))

    assert captured == [expected]
    assert encoder.provider_status() == {
        "requested": requested,
        "active": expected[0],
        "providers": expected,
        "cuda_active": "CUDAExecutionProvider" in expected,
        "cpu_fallback": cpu_fallback,
    }


def test_three_argument_provider_factory_and_legacy_factory_are_supported(
    tmp_path: Path,
) -> None:
    requested: list[str] = []

    class ProviderSession(FakeSession):
        def get_providers(self) -> list[str]:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ProviderSession()

    def factory(_: Path, __: int, provider: str) -> ProviderSession:
        requested.append(provider)
        return session

    model_path = tmp_path / "three-argument.onnx"
    model_path.write_bytes(b"model")
    encoder = DinoV2Encoder(
        model_path,
        session_factory=factory,
        execution_provider="auto",
    )
    encoder.embed_bytes(image_bytes((40, 80)))

    assert requested == ["auto"]
    assert encoder.provider_status()["cuda_active"] is True


def test_cuda_initialization_failure_retries_once_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[list[str]] = []

    class FakeSessionOptions:
        pass

    class CpuSession(FakeSession):
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

    def inference_session(
        _: str, *, sess_options: Any, providers: list[str]
    ) -> CpuSession:
        assert isinstance(sess_options, FakeSessionOptions)
        attempts.append(list(providers))
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA libraries are unavailable")
        return CpuSession()

    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model_path = tmp_path / "cuda-fallback.onnx"
    model_path.write_bytes(b"model")
    encoder = DinoV2Encoder(model_path, execution_provider="auto")

    encoder.embed_bytes(image_bytes((30, 60)))

    assert attempts == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert encoder.provider_status()["cpu_fallback"] is True


@pytest.mark.parametrize("available", [False, True])
def test_explicit_cuda_never_silently_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
) -> None:
    attempts: list[list[str]] = []

    class FakeSessionOptions:
        pass

    def inference_session(
        _: str, *, sess_options: Any, providers: list[str]
    ) -> FakeSession:
        assert isinstance(sess_options, FakeSessionOptions)
        attempts.append(list(providers))
        raise RuntimeError("CUDA initialization failed")

    provider_list = ["CPUExecutionProvider"]
    if available:
        provider_list.insert(0, "CUDAExecutionProvider")
    fake_ort = SimpleNamespace(
        SessionOptions=FakeSessionOptions,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
        get_available_providers=lambda: provider_list,
        InferenceSession=inference_session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    model_path = tmp_path / f"strict-cuda-{available}.onnx"
    model_path.write_bytes(b"model")
    encoder = DinoV2Encoder(model_path, execution_provider="cuda")

    with pytest.raises(VisionInferenceError, match="initialize"):
        encoder.embed_bytes(image_bytes((30, 60)))

    assert attempts == (
        [["CUDAExecutionProvider"]]
        if available
        else []
    )


def test_embed_requires_a_prepared_model(tmp_path: Path) -> None:
    encoder = DinoV2Encoder(
        tmp_path / "missing.onnx",
        session_factory=lambda *_: FakeSession(),
    )
    with pytest.raises(ModelNotPreparedError, match="prepare"):
        encoder.embed_bytes(image_bytes((20, 30)))


@pytest.mark.parametrize(
    ("values", "payload", "message"),
    [
        ({"max_image_bytes": 4}, b"12345", "byte limit"),
        ({}, b"not an image", "decoded safely"),
        ({"max_image_pixels": 300}, image_bytes((20, 20)), "pixel limit"),
        ({}, image_bytes((20, 20), image_format="BMP"), "Unsupported"),
    ],
)
def test_invalid_or_oversized_images_are_rejected(
    tmp_path: Path, values: dict[str, int], payload: bytes, message: str
) -> None:
    encoder, session, _ = encoder_with_fake_session(tmp_path, **values)
    with pytest.raises(InvalidImageError, match=message):
        encoder.embed_bytes(payload)
    assert session.batches == []


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((1, 257, 383), dtype=np.float32),
        np.zeros((1, 257, 384), dtype=np.float32),
        np.full((1, 257, 384), np.nan, dtype=np.float32),
    ],
)
def test_invalid_model_outputs_are_rejected(
    tmp_path: Path, output: np.ndarray
) -> None:
    class BadSession(FakeSession):
        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            return [output]

    encoder, _, _ = encoder_with_fake_session(tmp_path, session=BadSession())
    with pytest.raises(VisionInferenceError):
        encoder.embed_bytes(image_bytes((20, 20)))


def test_perceptual_hash_survives_resize_and_jpeg_recompression() -> None:
    image = Image.new("RGB", (320, 240), (18, 28, 45))
    pixels = np.asarray(image).copy()
    yy, xx = np.indices((240, 320))
    pixels[..., 0] = (pixels[..., 0] + xx * 3 + yy) % 256
    pixels[..., 1] = (pixels[..., 1] + yy * 2) % 256
    pixels[30:120, 40:150] = (245, 190, 35)
    pixels[115:215, 185:290] = (35, 205, 225)
    image = Image.fromarray(pixels.astype(np.uint8))

    original = io.BytesIO()
    image.save(original, format="PNG")
    recompressed = io.BytesIO()
    image.resize((640, 480), Image.Resampling.BICUBIC).save(
        recompressed,
        format="JPEG",
        quality=38,
        optimize=True,
    )

    original_hash = perceptual_hash_bytes(original.getvalue())
    recompressed_hash = perceptual_hash_bytes(recompressed.getvalue())

    assert 0 <= original_hash < 1 << 64
    assert hamming_similarity64(original_hash, recompressed_hash) >= 0.90


def test_perceptual_hash_can_be_mirror_invariant_and_rejects_distinct_image() -> None:
    image = Image.new("RGB", (256, 192), "black")
    pixels = np.asarray(image).copy()
    yy, xx = np.indices((192, 256))
    mask = ((xx - 76) ** 2) / (56**2) + ((yy - 85) ** 2) / (72**2) < 1
    pixels[mask] = (240, 210, 65)
    pixels[118:175, 150:242] = (30, 150, 245)
    image = Image.fromarray(pixels.astype(np.uint8))
    mirrored = ImageOps.mirror(image)

    original = perceptual_hash64(image, mirror_invariant=True)
    mirror = perceptual_hash64(mirrored, mirror_invariant=True)

    distinct = Image.new("RGB", image.size, "white")
    distinct_pixels = np.asarray(distinct).copy()
    distinct_pixels[:, ::16] = (0, 0, 0)
    distinct_pixels[::19, :] = (220, 20, 40)
    distinct = Image.fromarray(distinct_pixels.astype(np.uint8))
    distinct_hash = perceptual_hash64(distinct, mirror_invariant=True)

    assert original == mirror
    assert hamming_similarity64(original, mirror) == 1.0
    assert hamming_similarity64(original, distinct_hash) < 0.80


def test_hamming_similarity_validates_unsigned_64_bit_hashes() -> None:
    assert hamming_similarity64(0, 0) == 1.0
    assert hamming_similarity64(0, (1 << 64) - 1) == 0.0
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        hamming_similarity64(-1, 0)
    with pytest.raises(TypeError, match="integer"):
        hamming_similarity64(True, 0)
