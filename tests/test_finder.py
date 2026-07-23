from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

import app.finder as finder_module
from app.config import AppConfig
from app.db import Database
from app.downloader import EventBroker
from app.finder import FinderService
from app.main import create_app
from app.pose_vision import PoseFrame
from app.security import verify_media_signature


ROOT = "https://www.pornpics.com/"
GALLERY_A = "https://www.pornpics.com/galleries/alpha-79186222/"
GALLERY_B = "https://www.pornpics.com/galleries/beta-79186223/"
GALLERY_C = "https://www.pornpics.com/galleries/broken-79186224/"


@pytest.fixture(autouse=True)
def inline_finder_thread_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid this sandbox's non-terminating default executor during unit tests."""

    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(finder_module.asyncio, "to_thread", inline)


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 18), color).save(output, format="PNG")
    return output.getvalue()


class FakeEncoder:
    model_key = "fake-dinov2-v1"

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.embed_calls = 0

    async def prepare(self) -> Path:
        self.prepare_calls += 1
        return Path("fake.onnx")

    def embed_bytes(self, data: bytes, *, include_mirror: bool = False) -> np.ndarray:
        self.embed_calls += 1
        with Image.open(io.BytesIO(data)) as image:
            red, green, blue = image.convert("RGB").getpixel((0, 0))
        if red < 20 and green < 20 and blue < 20:
            vector = np.asarray([-1.0, 0.0], dtype=np.float32)
        elif blue > red and blue > green:
            vector = np.asarray([0.0, 1.0], dtype=np.float32)
        elif green > red and green > blue:
            vector = np.asarray([0.6, 0.8], dtype=np.float32)
        else:
            vector = np.asarray([1.0, 0.0], dtype=np.float32)
        rows = [vector]
        if include_mirror:
            rows.append(np.asarray([0.0, 1.0], dtype=np.float32))
        return np.stack(rows)


class SpatialFakeEncoder(FakeEncoder):
    model_key = "fake-dinov2-spatial-v1"

    def __init__(self) -> None:
        super().__init__()
        self.describe_calls = 0

    def describe_bytes(
        self, data: bytes, *, include_mirror: bool = False
    ) -> dict[str, np.ndarray]:
        self.describe_calls += 1
        with Image.open(io.BytesIO(data)) as image:
            red, green, blue = image.convert("RGB").getpixel((0, 0))
        if red < 20 and green < 20 and blue < 20:
            vector = np.asarray([-1.0, 0.0], dtype=np.float32)
        elif blue > red and blue > green:
            vector = np.asarray([0.0, 1.0], dtype=np.float32)
        elif green > red and green > blue:
            vector = np.asarray([0.8, 0.6], dtype=np.float32)
        else:
            vector = np.asarray([1.0, 0.0], dtype=np.float32)
        rows = [vector]
        if include_mirror:
            rows.append(vector.copy())
        embeddings = np.stack(rows)
        return {"global": embeddings, "spatial": embeddings}


class FakePoseEstimator:
    model_key = "fake-rtmo-pose-v1"

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.infer_calls = 0

    async def prepare(self) -> Path:
        self.prepare_calls += 1
        return Path("fake-rtmo.onnx")

    def provider_status(self) -> dict[str, object]:
        return {"requested": "cpu", "active": "CPUExecutionProvider"}

    def infer_bytes(self, _: bytes) -> PoseFrame:
        self.infer_calls += 1
        keypoints = np.zeros((1, 17, 2), dtype=np.float32)
        keypoints[0, :, 0] = np.linspace(0.25, 0.75, 17)
        keypoints[0, :, 1] = 0.25 + np.sin(np.linspace(0, np.pi, 17)) * 0.5
        return PoseFrame(
            keypoints=keypoints,
            confidences=np.full((1, 17), 0.9, dtype=np.float32),
            boxes=np.asarray([[0.2, 0.2, 0.8, 0.8]], dtype=np.float32),
            person_scores=np.asarray([0.9], dtype=np.float32),
            image_size=(12, 18),
            model_key=self.model_key,
            provider="CPUExecutionProvider",
        )


class FakeScraper:
    async def browse(self, **_: object) -> dict:
        return {
            "items": [
                {
                    "url": GALLERY_A,
                    "title": "Alpha",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/a.png",
                },
                {
                    "url": GALLERY_B,
                    "title": "Beta",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/b.png",
                },
            ],
            "next_url": None,
        }

    async def gallery(self, url: str) -> dict:
        if url == GALLERY_A:
            images = [
                {
                    "url": "https://cdni.pornpics.com/full/blue.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/blue.png",
                    "ordinal": 7,
                },
                {
                    "url": "https://cdni.pornpics.com/full/black.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/black.png",
                    "ordinal": 8,
                },
            ]
        else:
            images = [
                {
                    "url": "https://cdni.pornpics.com/full/green.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/green.png",
                    "ordinal": 3,
                }
            ]
        return {"url": url, "images": images}


async def fake_media(url: str, _: str) -> bytes:
    if "blue" in url:
        return image_bytes("blue")
    if "green" in url:
        return image_bytes("lime")
    return image_bytes("black")


class TopMatchScraper(FakeScraper):
    async def browse(self, **_: object) -> dict:
        return {
            "items": [
                {
                    "url": GALLERY_A,
                    "title": "Rank every image",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/a.png",
                }
            ],
            "next_url": None,
        }

    async def gallery(self, url: str) -> dict:
        images = [
            {
                "url": f"https://cdni.pornpics.com/full/{color}.png",
                "preview_remote_url": f"https://cdni.pornpics.com/p/{color}.png",
                "ordinal": ordinal,
            }
            for ordinal, color in enumerate(("red", "green", "blue", "black"), 1)
        ]
        return {"url": url, "images": images}


async def top_match_media(url: str, _: str) -> bytes:
    for color in ("red", "green", "blue", "black"):
        if f"/{color}.png" in url:
            return image_bytes("lime" if color == "green" else color)
    raise AssertionError(f"unexpected media URL: {url}")


def configured(tmp_path: Path) -> tuple[AppConfig, Database, int]:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        finder_examples_root=tmp_path / "references",
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        finder_request_delay=0,
        sqlite_vfs=None,
    )
    config.ensure_directories()
    (config.finder_examples_root / "pose").mkdir()
    (config.finder_examples_root / "pose" / "example.png").write_bytes(
        image_bytes("red")
    )
    database = Database(config.db_path)
    database.initialize()
    tag = database.create_pose_tag("Standing", "solo")
    return config, database, int(tag["id"])


@pytest.mark.asyncio
async def test_finder_max_score_ranking_review_and_persistent_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    encoder = FakeEncoder()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=encoder,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        first = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        completed = service.get_scan(first["id"])
        assert completed and completed["status"] == "completed"
        assert completed["progress"] == 100
        assert completed["pose_tag"]["label"] == "Standing"

        service._save_result(
            first["id"],
            {"url": GALLERY_C, "title": "Broken"},
            order=3,
            score=0,
            images_scored=0,
            best=None,
            status="failed",
            error="preview failed",
        )

        results, total = service.results(
            first["id"], review="all", min_score=0, limit=20, offset=0
        )
        assert total == 2
        assert [item["title"] for item in results] == ["Alpha", "Beta"]
        # Alpha is 1.0 because gallery aggregation is max, not an average with black.
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[0]["images_scored"] == 2
        assert results[0]["best_ordinal"] == 7
        # The blue candidate only matches the reference's mirrored fake embedding.
        assert results[0]["best_image_url"].endswith("blue.png")
        assert results[1]["score"] == pytest.approx(0.8)

        service.set_review(first["id"], results[0]["id"], "accepted")
        pending, _ = service.results(
            first["id"], review="pending", min_score=0, limit=20, offset=0
        )
        assert [item["title"] for item in pending] == ["Beta"]

        calls_after_first_scan = encoder.embed_calls
        second = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert encoder.embed_calls == calls_after_first_scan
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_spatial_top_three_exact_gate_and_versioned_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)

    def fake_phash(data: bytes, **_: object) -> int:
        with Image.open(io.BytesIO(data)) as image:
            red, green, blue = image.convert("RGB").getpixel((0, 0))
        if red > green and red > blue:
            return 0
        if green > red and green > blue:
            return (1 << 20) - 1  # Outside the hard eight-bit exact lane.
        if blue > red and blue > green:
            return 0xF  # Four bits away: a strong exact signal.
        return (1 << 64) - 1

    monkeypatch.setattr(finder_module, "_perceptual_hash_bytes", fake_phash)
    config, database, tag_id = configured(tmp_path)
    encoder = SpatialFakeEncoder()
    service = FinderService(
        config,
        database,
        TopMatchScraper(),
        EventBroker(),
        encoder=encoder,
        media_fetcher=top_match_media,
    )
    await service.start()
    try:
        first = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        completed = service.get_scan(first["id"])
        assert completed and completed["status"] == "completed"
        assert "hybrid-spatial-pyramid-v1" in completed["reference_model_key"]

        results, total = service.results(
            first["id"], review="all", min_score=0, limit=20, offset=0
        )
        assert total == 1
        result = results[0]
        assert result["images_scored"] == 4
        assert [match["rank"] for match in result["top_matches"]] == [1, 2, 3]
        assert [match["ordinal"] for match in result["top_matches"]] == [1, 3, 2]
        assert result["best_ordinal"] == result["top_matches"][0]["ordinal"]
        assert result["best_image_url"] == result["top_matches"][0]["image_url"]
        matches_by_ordinal = {
            match["ordinal"]: match for match in result["top_matches"]
        }
        assert matches_by_ordinal[2]["appearance_score"] == pytest.approx(
            (0.8 - 0.20) / 0.65
        )
        assert matches_by_ordinal[2]["exact_score"] == 0
        assert matches_by_ordinal[3]["appearance_score"] == 0
        assert matches_by_ordinal[3]["exact_score"] == pytest.approx(0.9375)
        assert all(match["pose_score"] is None for match in result["top_matches"])
        assert all(match["person_count"] is None for match in result["top_matches"])

        calls_after_first = encoder.describe_calls
        second = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert encoder.describe_calls == calls_after_first

        with database.connect() as connection:
            cached = connection.execute(
                "SELECT model_key, metadata_json FROM finder_embedding_cache"
            ).fetchall()
            stored_matches = connection.execute(
                "SELECT matches_json FROM finder_results WHERE scan_id = ?",
                (first["id"],),
            ).fetchone()[0]
        assert cached
        assert all("hybrid-spatial-pyramid-v1" in row["model_key"] for row in cached)
        assert all(
            json.loads(row["metadata_json"])["analyzer_version"]
            == finder_module.ANALYZER_VERSION
            for row in cached
        )
        assert len(json.loads(stored_matches)) == 3
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_caches_pose_diagnostics_and_skeleton_overlays(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    monkeypatch.setattr(finder_module, "_perceptual_hash_bytes", None)
    config, database, tag_id = configured(tmp_path)
    encoder = SpatialFakeEncoder()
    pose = FakePoseEstimator()
    service = FinderService(
        config,
        database,
        TopMatchScraper(),
        EventBroker(),
        encoder=encoder,
        pose_estimator=pose,
        media_fetcher=top_match_media,
    )
    await service.start()
    try:
        first = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        results, _ = service.results(
            first["id"], review="all", min_score=0, limit=20, offset=0
        )
        match = results[0]["top_matches"][0]
        assert match["pose_reliable"] is True
        assert match["pose_score"] == pytest.approx(1.0)
        assert match["person_count"] == 1
        assert match["pose_common_joints"] == 17
        assert match["skeleton_overlay_url"].startswith(
            "data:image/svg+xml;base64,"
        )
        assert service.status()["pose_ready"] is True
        assert pose.prepare_calls == 1

        inference_calls = pose.infer_calls
        second = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert pose.infer_calls == inference_calls
    finally:
        await service.stop()


def test_finder_labels_and_overlays_only_reliable_pose_assistance() -> None:
    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    keypoints[0, :, 0] = np.linspace(0.25, 0.75, 17)
    keypoints[0, :, 1] = np.linspace(0.2, 0.8, 17)
    confidences = np.full((1, 17), 0.02, dtype=np.float32)
    confidences[0, :5] = 0.9
    sparse = PoseFrame(
        keypoints=keypoints,
        confidences=confidences,
        boxes=np.asarray([[0.2, 0.2, 0.8, 0.8]], dtype=np.float32),
        person_scores=np.asarray([0.9], dtype=np.float32),
        image_size=(640, 960),
        model_key="fake-rtmo-pose-v1",
        provider="CPUExecutionProvider",
    )
    diagnostics = FinderService._pose_diagnostics(
        {"pose": sparse.as_dict()}, (sparse,)
    )
    assert diagnostics["pose_reliable"] is False
    assert diagnostics["skeleton_overlay_url"] == ""

    overlay = "data:image/svg+xml;base64,PHN2Zy8+"

    def normalized(score: float, reliable: bool) -> dict[str, object]:
        return FinderService._normalized_top_matches(
            [
                {
                    "image_url": "https://cdni.pornpics.com/full/test.jpg",
                    "preview_remote_url": "https://cdni.pornpics.com/p/test.jpg",
                    "ordinal": 1,
                    "score": 0.8,
                    "appearance_score": 0.8,
                    "exact_score": 0,
                    "pose_score": score,
                    "pose_reliable": reliable,
                    "skeleton_overlay_url": overlay,
                }
            ]
        )[0]

    unreliable = normalized(0.95, False)
    assert unreliable["match_type"] == "appearance"
    assert unreliable["skeleton_overlay_url"] == ""

    reliable_but_not_assisting = normalized(0.4, True)
    assert reliable_but_not_assisting["match_type"] == "appearance"
    assert reliable_but_not_assisting["skeleton_overlay_url"] == overlay

    pose_assisted = normalized(0.95, True)
    assert pose_assisted["match_type"] == "pose"
    assert pose_assisted["skeleton_overlay_url"] == overlay


@pytest.mark.asyncio
async def test_finder_model_prepare_is_lazy_and_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    # Even a present-but-corrupt artifact must never trigger prepare/download at startup.
    config.finder_model_path.write_bytes(b"corrupt existing model")

    class FlakyEncoder(FakeEncoder):
        async def prepare(self) -> Path:
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                raise RuntimeError("temporary model mirror failure")
            return Path("fake.onnx")

    encoder = FlakyEncoder()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=encoder,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        assert service.status()["available"] is True
        assert service.status()["model_ready"] is False
        assert encoder.prepare_calls == 0
        failed = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(failed["id"])["status"] == "failed"
        assert service.status()["available"] is True

        retried = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(retried["id"])["status"] == "completed"
        assert service.status()["model_ready"] is True
        assert encoder.prepare_calls == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_pause_resume_and_restart_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingScraper(FakeScraper):
        async def browse(self, **kwargs: object) -> dict:
            started.set()
            await release.wait()
            return await super().browse(**kwargs)

    service = FinderService(
        config,
        database,
        BlockingScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    scan = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    await asyncio.wait_for(started.wait(), 30)
    assert service.pause(scan["id"])["status"] == "pausing"
    release.set()
    await asyncio.wait_for(service.queue.join(), 30)
    assert service.get_scan(scan["id"])["status"] == "paused"
    service.resume(scan["id"])
    await asyncio.wait_for(service.queue.join(), 30)
    assert service.get_scan(scan["id"])["status"] == "completed"

    # Simulate an interrupted process after the immutable references were prepared.
    with database.connect() as connection:
        connection.execute(
            "UPDATE finder_scans SET status = 'scanning', finished_at = NULL WHERE id = ?",
            (scan["id"],),
        )
    await service.stop()
    restarted = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await restarted.start()
    try:
        await asyncio.wait_for(restarted.queue.join(), 30)
        assert restarted.get_scan(scan["id"])["status"] == "completed"
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_finder_model_upgrade_restarts_partial_scan_without_mixed_results(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    original = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await original.start()
    scan = original.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    await asyncio.wait_for(original.queue.join(), 30)
    completed = original.get_scan(scan["id"])
    assert completed is not None
    assert completed["pages_completed"] == 1
    assert original.results(
        scan["id"], review="all", min_score=0, limit=20, offset=0
    )[1] == 2
    await original.stop()

    upgraded_encoder = FakeEncoder()
    upgraded_encoder.model_key = "fake-dinov2-v2"
    upgraded = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=upgraded_encoder,
        media_fetcher=fake_media,
    )
    upgraded.ensure_schema()
    upgraded._model_key = upgraded._encoder_key()
    await upgraded._prepare_references(upgraded.get_scan(scan["id"]))

    reset = upgraded.get_scan(scan["id"])
    assert reset is not None
    assert reset["pages_completed"] == 0
    assert reset["next_url"] == ROOT
    assert reset["processed_galleries"] == 0
    assert reset["processed_images"] == 0
    assert upgraded.results(
        scan["id"], review="all", min_score=0, limit=20, offset=0
    )[1] == 0
    assert reset["reference_model_key"] == upgraded._model_key


def test_finder_folder_confinement_and_symlink_rejection(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    root = config.finder_examples_root
    (root / "direct.png").write_bytes(image_bytes("blue"))
    (root / "nested").mkdir()
    (root / "nested" / "sample.png").write_bytes(image_bytes("green"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(image_bytes("black"))
    (root / "linked").symlink_to(outside, target_is_directory=True)
    service = FinderService(config, database, FakeScraper(), EventBroker())

    listing = service.folders()
    assert listing["root"] == str(root)
    assert listing["current"]["path"] == "."
    assert listing["current"]["image_count"] == 1
    assert {item["path"] for item in listing["items"]} == {"nested", "pose"}
    nested = service.folders("nested")
    assert nested["path"] == "nested"
    assert nested["parent"] == "."
    assert nested["current"]["image_count"] == 1
    with pytest.raises(ValueError, match="cannot contain"):
        service._resolve_example_directory("../outside")
    with pytest.raises(ValueError, match="Symlinked"):
        service._resolve_example_directory("linked")
    (root / "linked-image.png").symlink_to(outside / "secret.png")
    with pytest.raises(ValueError, match="Symlinked"):
        service._example_files(root)


def test_finder_accepts_free_relative_and_confined_absolute_library_paths(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=library,
        sort_root=library,
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    relative = "sorted_outpaint/mating press - backview/selected_target_upscaled"
    target = library / relative
    target.mkdir(parents=True)
    (target / "example.png").write_bytes(image_bytes("red"))
    database = Database(config.db_path)
    database.initialize()
    tag = database.create_pose_tag("Mating press backview", "couple")
    service = FinderService(config, database, FakeScraper(), EventBroker())
    service.ensure_schema()

    assert config.finder_examples_root == library.resolve()
    assert service.status()["folder_root"] == str(library.resolve())
    assert service._resolve_example_directory(relative) == (target, relative)
    assert service._resolve_example_directory(str(target)) == (target, relative)
    assert service._resolve_example_directory(str(library)) == (library, ".")

    first = service.create_scan(
        example_directory=relative,
        pose_tag_id=int(tag["id"]),
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    second = service.create_scan(
        example_directory=str(target),
        pose_tag_id=int(tag["id"]),
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    assert first["example_directory"] == relative
    assert second["example_directory"] == relative

    sibling_prefix = tmp_path / "library2"
    sibling_prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    for forbidden in (
        str(sibling_prefix),
        str(outside),
        "/mnt/user/Davinci/Qwen_edit_lora/pornpic",
    ):
        with pytest.raises(ValueError, match="inside"):
            service._resolve_example_directory(forbidden)
    with pytest.raises(ValueError, match="cannot contain"):
        service._resolve_example_directory("sorted_outpaint/../outside")

    linked = library / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        service._resolve_example_directory("linked")


def test_finder_cache_prunes_stale_models_lru_and_transient_pose_errors(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    config.finder_cache_max_entries = 3
    config.finder_cache_max_bytes = 10_000_000
    service = FinderService(config, database, FakeScraper(), EventBroker())
    service.ensure_schema()
    embedding = np.asarray([[1.0, 0.0]], dtype=np.float32)
    metadata = {"analyzer_version": finder_module.ANALYZER_VERSION}

    service._model_key = "obsolete-model"
    service._store_embedding("old", False, embedding, metadata)
    service._model_key = "current-model"
    for index in range(3):
        service._store_embedding(f"current-{index}", False, embedding, metadata)

    assert service._prune_embedding_cache(purge_stale_models=True) == 1
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT model_key FROM finder_embedding_cache"
        ).fetchall()
    assert len(rows) == 3
    assert {row["model_key"] for row in rows} == {"current-model"}

    service._store_embedding("current-extra", False, embedding, metadata)
    assert service._prune_embedding_cache() == 2
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM finder_embedding_cache"
        ).fetchone()[0] == 2

    service._pose_ready = True
    service._store_embedding(
        "transient-pose-failure",
        False,
        embedding,
        {**metadata, "pose_error": "temporary CUDA OOM"},
    )
    assert service._cached_descriptor("transient-pose-failure", False) is None
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM finder_embedding_cache WHERE source_key = ?",
            ("transient-pose-failure",),
        ).fetchone()[0] == 0


def test_finder_schema_migrates_legacy_metadata_and_matches_safely(
    tmp_path: Path,
) -> None:
    config, database, tag_id = configured(tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    vector = np.asarray([1.0, 0.0], dtype="<f4").tobytes()
    with database.connect() as connection:
        connection.executescript(
            """
            CREATE TABLE finder_scans (
                id TEXT PRIMARY KEY, example_directory TEXT NOT NULL,
                reference_fingerprint TEXT NOT NULL DEFAULT '',
                reference_ready INTEGER NOT NULL DEFAULT 0,
                reference_count INTEGER NOT NULL DEFAULT 0,
                pose_tag_id INTEGER NOT NULL, pose_tag_label TEXT NOT NULL,
                pose_tag_slug TEXT NOT NULL, pose_default_role TEXT NOT NULL,
                source_url TEXT NOT NULL, next_url TEXT,
                page_limit INTEGER NOT NULL,
                pages_completed INTEGER NOT NULL DEFAULT 0,
                minimum_score REAL NOT NULL, status TEXT NOT NULL,
                total_galleries INTEGER NOT NULL DEFAULT 0,
                processed_galleries INTEGER NOT NULL DEFAULT 0,
                processed_images INTEGER NOT NULL DEFAULT 0,
                failed_galleries INTEGER NOT NULL DEFAULT 0,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                pause_requested INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, finished_at TEXT
            );
            CREATE TABLE finder_scan_references (
                scan_id TEXT NOT NULL, example_key TEXT NOT NULL,
                mirror_index INTEGER NOT NULL, embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                PRIMARY KEY (scan_id, example_key, mirror_index),
                FOREIGN KEY (scan_id) REFERENCES finder_scans(id) ON DELETE CASCADE
            );
            CREATE TABLE finder_embedding_cache (
                cache_key TEXT PRIMARY KEY, model_key TEXT NOT NULL,
                source_key TEXT NOT NULL, include_mirror INTEGER NOT NULL,
                rows INTEGER NOT NULL, dimensions INTEGER NOT NULL,
                embedding BLOB NOT NULL, created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL
            );
            CREATE TABLE finder_results (
                id TEXT PRIMARY KEY, scan_id TEXT NOT NULL,
                gallery_key TEXT NOT NULL, gallery_url TEXT NOT NULL,
                title TEXT NOT NULL,
                thumbnail_remote_url TEXT NOT NULL DEFAULT '',
                best_image_url TEXT NOT NULL DEFAULT '',
                best_preview_remote_url TEXT NOT NULL DEFAULT '',
                best_ordinal INTEGER, score REAL NOT NULL DEFAULT 0,
                images_scored INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL, review TEXT NOT NULL DEFAULT 'pending',
                error TEXT NOT NULL DEFAULT '', discovered_order INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE (scan_id, gallery_key),
                FOREIGN KEY (scan_id) REFERENCES finder_scans(id) ON DELETE CASCADE
            );
            """
        )
        connection.execute(
            """INSERT INTO finder_scans(
                   id, example_directory, reference_ready, reference_count,
                   pose_tag_id, pose_tag_label, pose_tag_slug, pose_default_role,
                   source_url, page_limit, pages_completed, minimum_score,
                   status, created_at, updated_at, finished_at
               ) VALUES ('legacy', 'pose', 1, 1, ?, 'Standing', 'standing',
                         'solo', ?, 1, 1, 0, 'completed', ?, ?, ?)""",
            (tag_id, ROOT, now, now, now),
        )
        connection.execute(
            """INSERT INTO finder_scan_references(
                   scan_id, example_key, mirror_index, embedding, dimensions
               ) VALUES ('legacy', 'example', 0, ?, 2)""",
            (vector,),
        )
        connection.execute(
            """INSERT INTO finder_embedding_cache(
                   cache_key, model_key, source_key, include_mirror, rows,
                   dimensions, embedding, created_at, last_used_at
               ) VALUES ('old-cache', 'old-model', 'old-source', 0, 1, 2, ?, ?, ?)""",
            (vector, now, now),
        )
        connection.execute(
            """INSERT INTO finder_results(
                   id, scan_id, gallery_key, gallery_url, title,
                   best_image_url, best_preview_remote_url, best_ordinal,
                   score, images_scored, status, discovered_order,
                   created_at, updated_at
               ) VALUES ('legacy-result', 'legacy', 'legacy-gallery', ?,
                         'Legacy result', 'https://cdn.example/full.jpg',
                         'https://cdn.example/preview.jpg', 4, 0.7, 1,
                         'completed', 1, ?, ?)""",
            (GALLERY_A, now, now),
        )

    service = FinderService(config, database, FakeScraper(), EventBroker())
    service.ensure_schema()

    with database.connect() as connection:
        scan_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(finder_scans)")
        }
        reference_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(finder_scan_references)"
            )
        }
        cache_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(finder_embedding_cache)"
            )
        }
        result_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(finder_results)")
        }
        reference_metadata = connection.execute(
            "SELECT metadata_json FROM finder_scan_references"
        ).fetchone()[0]
        cache_metadata = connection.execute(
            "SELECT metadata_json FROM finder_embedding_cache"
        ).fetchone()[0]
    assert "reference_model_key" in scan_columns
    assert "metadata_json" in reference_columns
    assert "metadata_json" in cache_columns
    assert "matches_json" in result_columns
    assert json.loads(reference_metadata) == {}
    assert json.loads(cache_metadata) == {}

    results, total = service.results(
        "legacy", review="all", min_score=0, limit=10, offset=0
    )
    assert total == 1
    assert results[0]["id"] == "legacy-result"
    assert results[0]["top_matches"][0]["rank"] == 1
    assert results[0]["top_matches"][0]["image_url"].endswith("full.jpg")


@pytest.mark.asyncio
async def test_finder_api_signs_best_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        finder_examples_root=tmp_path / "references",
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        finder_request_delay=0,
        sqlite_vfs=None,
    )
    config.ensure_directories()
    (config.finder_examples_root / "example.png").write_bytes(image_bytes("red"))
    app = create_app(config)
    app.state.finder.encoder = FakeEncoder()
    app.state.finder.pose_estimator = FakePoseEstimator()
    app.state.finder.media_fetcher = fake_media
    app.state.finder.scraper = FakeScraper()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            folder_response = await client.get("/api/finder/folders")
            assert folder_response.status_code == 200
            folders = folder_response.json()
            assert folders["root"] == str(config.finder_examples_root)
            assert folders["path"] == "."
            assert folders["current"]["image_count"] == 1
            assert folders["items"] == []
            finder_status = (await client.get("/api/finder/status")).json()
            assert finder_status["folder_root"] == str(config.finder_examples_root)
            assert finder_status["examples_root"] == finder_status["folder_root"]
            outside_response = await client.get(
                "/api/finder/folders", params={"path": str(tmp_path)}
            )
            assert outside_response.status_code == 400
            empty_scans = await client.get("/api/finder/scans")
            assert empty_scans.status_code == 200
            assert empty_scans.json()["items"] == []
            tag = (
                await client.post(
                    "/api/pose-tags",
                    json={"label": "Finder pose", "default_role": "solo"},
                )
            ).json()["tag"]
            response = await client.post(
                "/api/finder/scans",
                json={
                    "example_directory": ".",
                    "pose_tag_id": tag["id"],
                    "source_url": ROOT,
                    "page_limit": 1,
                    "minimum_score": 0,
                },
            )
            assert response.status_code == 202
            scan_id = response.json()["scan"]["id"]
            await asyncio.wait_for(app.state.finder.queue.join(), 30)
            listed_scans = await client.get("/api/finder/scans", params={"limit": 1})
            assert listed_scans.status_code == 200
            assert listed_scans.json()["items"][0]["id"] == scan_id
            result_response = await client.get(
                f"/api/finder/scans/{scan_id}/results",
                params={"review": "all", "min_score": 0},
            )
            assert result_response.status_code == 200
            result = result_response.json()["items"][0]
            assert "best_preview_remote_url" not in result
            assert result["best_preview_url"].startswith("/api/media?")
            request = httpx.URL(result["best_preview_url"])
            assert verify_media_signature(
                request.params["url"],
                request.params["token"],
                config.media_signing_key,
            )
            assert result["top_matches"]
            for rank, match in enumerate(result["top_matches"], start=1):
                assert match["rank"] == rank
                assert "preview_remote_url" not in match
                assert match["preview_url"].startswith("/api/media?")
                match_request = httpx.URL(match["preview_url"])
                assert verify_media_signature(
                    match_request.params["url"],
                    match_request.params["token"],
                    config.media_signing_key,
                )
                assert match["skeleton_overlay_url"].startswith(
                    "data:image/svg+xml;base64,"
                )
