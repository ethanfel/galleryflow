from __future__ import annotations

import asyncio
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.vision import (
    DINOV2_EMBEDDING_DIMENSION,
    DINOV2_MODEL_KEY,
    DINOV2_MODEL_SHA256,
    DINOV2_MODEL_URL,
    DinoV2Encoder,
    InvalidImageError,
    ModelIntegrityError,
    ModelNotPreparedError,
    VisionInferenceError,
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
