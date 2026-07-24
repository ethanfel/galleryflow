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

from app.joytag import (
    JOYTAG_IMAGE_SIZE,
    JOYTAG_MODEL_KEY,
    JOYTAG_MODEL_REVISION,
    JOYTAG_MODEL_SHA256,
    JOYTAG_MODEL_URL,
    JOYTAG_TAG_COUNT,
    JOYTAG_TAGS_SHA256,
    JOYTAG_TAGS_URL,
    JoyTagAnalyzer,
    JoyTagInferenceError,
    JoyTagIntegrityError,
    JoyTagInvalidImageError,
    JoyTagNotPreparedError,
    JoyTagPreparationError,
)


class FakeSession:
    def __init__(
        self,
        tag_count: int,
        *,
        providers: list[str] | None = None,
        logits: np.ndarray | None = None,
        fail: bool = False,
    ) -> None:
        self.tag_count = tag_count
        self.providers = providers or ["CPUExecutionProvider"]
        self.logits = logits
        self.fail = fail
        self.batches: list[np.ndarray] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="output")]

    def get_providers(self) -> list[str]:
        return self.providers

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        assert output_names == ["output"]
        batch = np.asarray(input_feed["input"])
        self.batches.append(batch.copy())
        if self.fail:
            raise RuntimeError("provider failed")
        if self.logits is not None:
            output = np.asarray(self.logits, dtype=np.float32)
            if output.ndim == 1:
                output = np.repeat(output[np.newaxis, :], batch.shape[0], axis=0)
            return [output]
        summaries = batch.mean(axis=(1, 2, 3), dtype=np.float32)
        return [
            np.stack(
                [
                    np.arange(self.tag_count, dtype=np.float32) + value
                    for value in summaries
                ]
            )
        ]


def image_bytes(
    size: tuple[int, int] = (40, 80),
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    image_format: str = "PNG",
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


async def prepared_analyzer(
    tmp_path: Path,
    *,
    tags: tuple[str, ...] = ("first", "second", "third"),
    session: FakeSession | None = None,
    execution_provider: str = "auto",
    session_factory: Any | None = None,
    max_image_bytes: int = 1024 * 1024,
    max_image_pixels: int = 1_000_000,
) -> tuple[JoyTagAnalyzer, FakeSession | None]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = b"fake JoyTag model"
    vocabulary = "\n".join(tags).encode()
    model_path = tmp_path / "model.onnx"
    tags_path = tmp_path / "top_tags.txt"
    model_path.write_bytes(model)
    tags_path.write_bytes(vocabulary)
    fake = session or FakeSession(len(tags))
    factory = session_factory or (lambda *_: fake)
    analyzer = JoyTagAnalyzer(
        model_path,
        tags_path,
        execution_provider=execution_provider,
        max_image_bytes=max_image_bytes,
        max_image_pixels=max_image_pixels,
        model_sha256=hashlib.sha256(model).hexdigest(),
        tags_sha256=hashlib.sha256(vocabulary).hexdigest(),
        expected_tag_count=len(tags),
        session_factory=factory,
    )
    await analyzer.prepare()
    return analyzer, fake


def test_manifest_is_pinned_to_the_probe_artifacts() -> None:
    assert JOYTAG_MODEL_REVISION == "6b7f16331a6ccf0fdce37d5a9564715f6e772b22"
    assert JOYTAG_MODEL_URL.endswith(f"{JOYTAG_MODEL_REVISION}/model.onnx")
    assert JOYTAG_TAGS_URL.endswith(f"{JOYTAG_MODEL_REVISION}/top_tags.txt")
    assert JOYTAG_MODEL_SHA256 == (
        "f85b7130e6e549b5b0822537007b7482e8c4c8e754c8d9a5bee08e27050e1097"
    )
    assert JOYTAG_TAGS_SHA256 == (
        "32b1963a234af848643b2bbf47d8eff1f1c7889406810c57b980f41b2b9e01d0"
    )
    assert JOYTAG_TAG_COUNT == 5_813
    assert "multilabel-sigmoid-v1" in JOYTAG_MODEL_KEY


@pytest.mark.asyncio
async def test_prepare_downloads_verifies_and_reuses_both_artifacts(
    tmp_path: Path,
) -> None:
    model = b"downloaded model"
    vocabulary = b"one\ntwo\nthree"
    model_path = tmp_path / "nested" / "model.onnx"
    tags_path = tmp_path / "nested" / "top_tags.txt"
    calls: list[tuple[str, str, int]] = []

    async def downloader(url: str, destination: Path, max_bytes: int) -> None:
        calls.append((url, destination.name, max_bytes))
        assert destination.name.startswith(".")
        assert destination.name.endswith(".part")
        destination.write_bytes(vocabulary if url.endswith(".txt") else model)

    analyzer = JoyTagAnalyzer(
        model_path,
        tags_path,
        model_url="https://models.example/model.onnx",
        tags_url="https://models.example/tags.txt",
        model_sha256=hashlib.sha256(model).hexdigest(),
        tags_sha256=hashlib.sha256(vocabulary).hexdigest(),
        expected_tag_count=3,
        max_model_bytes=100,
        max_tags_bytes=50,
        downloader=downloader,
    )

    assert await analyzer.prepare() == (model_path, tags_path)
    assert model_path.read_bytes() == model
    assert tags_path.read_bytes() == vocabulary
    assert analyzer.tags == ("one", "two", "three")
    assert [(url, maximum) for url, _, maximum in calls] == [
        ("https://models.example/model.onnx", 100),
        ("https://models.example/tags.txt", 50),
    ]
    assert list(model_path.parent.glob("*.part")) == []

    await analyzer.prepare()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_prepare_never_replaces_an_artifact_with_a_bad_download(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.onnx"
    tags_path = tmp_path / "tags.txt"
    original = b"existing damaged model"
    model_path.write_bytes(original)

    async def downloader(_: str, destination: Path, __: int) -> None:
        destination.write_bytes(b"another wrong model")

    analyzer = JoyTagAnalyzer(
        model_path,
        tags_path,
        model_sha256=hashlib.sha256(b"expected model").hexdigest(),
        tags_sha256=hashlib.sha256(b"one").hexdigest(),
        expected_tag_count=1,
        downloader=downloader,
    )

    with pytest.raises(JoyTagIntegrityError, match="SHA-256"):
        await analyzer.prepare()
    assert model_path.read_bytes() == original
    assert not tags_path.exists()
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.asyncio
async def test_concurrent_prepare_downloads_each_artifact_once(tmp_path: Path) -> None:
    model = b"model"
    vocabulary = b"one\ntwo"
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def downloader(url: str, destination: Path, _: int) -> None:
        calls.append(url)
        if url.endswith(".onnx"):
            started.set()
            await release.wait()
            destination.write_bytes(model)
        else:
            destination.write_bytes(vocabulary)

    analyzer = JoyTagAnalyzer(
        tmp_path / "model.onnx",
        tmp_path / "tags.txt",
        model_url="https://models.example/model.onnx",
        tags_url="https://models.example/tags.txt",
        model_sha256=hashlib.sha256(model).hexdigest(),
        tags_sha256=hashlib.sha256(vocabulary).hexdigest(),
        expected_tag_count=2,
        downloader=downloader,
    )

    first = asyncio.create_task(analyzer.prepare())
    await started.wait()
    second = asyncio.create_task(analyzer.prepare())
    release.set()
    await first
    await second
    assert calls == [
        "https://models.example/model.onnx",
        "https://models.example/tags.txt",
    ]


@pytest.mark.asyncio
async def test_batch_preprocessing_preserves_order_and_uses_white_square_padding(
    tmp_path: Path,
) -> None:
    session = FakeSession(3)
    analyzer, _ = await prepared_analyzer(tmp_path, session=session)
    payloads = [
        image_bytes((40, 80), color=(255, 0, 0)),
        image_bytes((80, 40), color=(0, 0, 255)),
    ]

    probabilities = analyzer.classify_many_bytes(payloads)

    assert probabilities.shape == (2, 3)
    assert probabilities.dtype == np.float32
    assert probabilities.flags.c_contiguous
    assert len(session.batches) == 1
    batch = session.batches[0]
    assert batch.shape == (2, 3, JOYTAG_IMAGE_SIZE, JOYTAG_IMAGE_SIZE)
    # The portrait has white side padding around its red center.
    assert (
        batch[0, 1, JOYTAG_IMAGE_SIZE // 2, 0]
        > batch[0, 1, JOYTAG_IMAGE_SIZE // 2, JOYTAG_IMAGE_SIZE // 2]
    )
    # Different source colors produce different rows rather than a batch-wide
    # score accidentally copied to every image.
    assert not np.array_equal(probabilities[0], probabilities[1])


@pytest.mark.asyncio
async def test_logits_are_converted_to_stable_probabilities(tmp_path: Path) -> None:
    session = FakeSession(
        3,
        logits=np.asarray([-1000.0, 0.0, 1000.0], dtype=np.float32),
    )
    analyzer, _ = await prepared_analyzer(tmp_path, session=session)

    probabilities = analyzer.classify_many_bytes(
        [image_bytes(), image_bytes(color=(0, 255, 0))]
    )

    assert probabilities.shape == (2, 3)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities[:, 0], 0)
    np.testing.assert_allclose(probabilities[:, 1], 0.5)
    np.testing.assert_allclose(probabilities[:, 2], 1)
    np.testing.assert_allclose(analyzer.classify_bytes(image_bytes()), (0, 0.5, 1))


@pytest.mark.asyncio
async def test_validation_happens_before_batch_inference(tmp_path: Path) -> None:
    session = FakeSession(3)
    analyzer, _ = await prepared_analyzer(
        tmp_path,
        session=session,
        max_image_bytes=2_000,
        max_image_pixels=5_000,
    )

    assert analyzer.classify_many_bytes([]).shape == (0, 3)
    with pytest.raises(TypeError, match="sequence"):
        analyzer.classify_many_bytes(image_bytes())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes-like"):
        analyzer.classify_many_bytes([image_bytes(), object()])  # type: ignore[list-item]
    with pytest.raises(JoyTagInvalidImageError, match="decoded safely"):
        analyzer.classify_many_bytes([b"not an image"])
    with pytest.raises(JoyTagInvalidImageError, match="pixel limit"):
        analyzer.classify_many_bytes([image_bytes((80, 80))])
    assert session.batches == []


@pytest.mark.asyncio
async def test_inference_rejects_wrong_shape_and_non_finite_logits(
    tmp_path: Path,
) -> None:
    wrong = FakeSession(3, logits=np.zeros((2, 3), dtype=np.float32))
    analyzer, _ = await prepared_analyzer(tmp_path / "wrong", session=wrong)
    with pytest.raises(JoyTagInferenceError, match="output shape"):
        analyzer.classify_many_bytes([image_bytes()])

    non_finite = FakeSession(
        3, logits=np.asarray([0.0, np.nan, 1.0], dtype=np.float32)
    )
    analyzer, _ = await prepared_analyzer(
        tmp_path / "non-finite", session=non_finite
    )
    with pytest.raises(JoyTagInferenceError, match="non-finite"):
        analyzer.classify_many_bytes([image_bytes()])


@pytest.mark.asyncio
async def test_auto_provider_falls_back_during_session_initialization(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    cpu = FakeSession(3, providers=["CPUExecutionProvider"])

    def factory(_: Path, __: int, provider: str) -> FakeSession:
        calls.append(provider)
        if provider == "auto":
            raise RuntimeError("CUDA libraries are broken")
        return cpu

    analyzer, _ = await prepared_analyzer(
        tmp_path,
        execution_provider="auto",
        session_factory=factory,
    )

    analyzer.classify_many_bytes([image_bytes()])
    assert calls == ["auto", "cpu"]
    assert analyzer.provider_status() == {
        "requested": "auto",
        "active": "CPUExecutionProvider",
        "providers": ["CPUExecutionProvider"],
        "cuda_active": False,
        "cpu_fallback": True,
        "message": "CUDA session initialization failed; using CPU",
    }


@pytest.mark.asyncio
async def test_auto_provider_retries_failed_cuda_inference_on_cpu(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    cuda = FakeSession(
        3,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        fail=True,
    )
    cpu = FakeSession(
        3,
        providers=["CPUExecutionProvider"],
        logits=np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
    )

    def factory(_: Path, __: int, provider: str) -> FakeSession:
        calls.append(provider)
        return cuda if provider == "auto" else cpu

    analyzer, _ = await prepared_analyzer(
        tmp_path,
        execution_provider="auto",
        session_factory=factory,
    )

    probabilities = analyzer.classify_many_bytes([image_bytes()])
    assert calls == ["auto", "cpu"]
    assert probabilities[0, 1] == pytest.approx(0.7310586)
    assert analyzer.active_provider == "CPUExecutionProvider"
    assert "inference failed" in analyzer.provider_status()["message"]


@pytest.mark.asyncio
async def test_explicit_cuda_never_silently_falls_back(tmp_path: Path) -> None:
    calls: list[str] = []
    cuda = FakeSession(3, providers=["CUDAExecutionProvider"], fail=True)

    def factory(_: Path, __: int, provider: str) -> FakeSession:
        calls.append(provider)
        return cuda

    analyzer, _ = await prepared_analyzer(
        tmp_path,
        execution_provider="cuda",
        session_factory=factory,
    )

    with pytest.raises(JoyTagInferenceError, match="inference failed"):
        analyzer.classify_many_bytes([image_bytes()])
    assert calls == ["cuda"]


@pytest.mark.asyncio
async def test_prepare_rejects_invalid_vocabulary_and_inference_requires_prepare(
    tmp_path: Path,
) -> None:
    model = b"model"
    vocabulary = b"duplicate\nduplicate"
    model_path = tmp_path / "model.onnx"
    tags_path = tmp_path / "tags.txt"
    model_path.write_bytes(model)
    tags_path.write_bytes(vocabulary)
    analyzer = JoyTagAnalyzer(
        model_path,
        tags_path,
        model_sha256=hashlib.sha256(model).hexdigest(),
        tags_sha256=hashlib.sha256(vocabulary).hexdigest(),
        expected_tag_count=2,
        session_factory=lambda *_: FakeSession(2),
    )

    with pytest.raises(JoyTagNotPreparedError):
        analyzer.classify_many_bytes([image_bytes()])
    with pytest.raises(JoyTagPreparationError, match="duplicates"):
        await analyzer.prepare()
