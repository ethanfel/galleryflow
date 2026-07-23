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
from app.security import encode_gallery_id, gallery_key, verify_media_signature


ROOT = "https://www.pornpics.com/"
GALLERY_A = "https://www.pornpics.com/galleries/alpha-79186222/"
GALLERY_B = "https://www.pornpics.com/galleries/beta-79186223/"
GALLERY_C = "https://www.pornpics.com/galleries/broken-79186224/"
PAGE_2 = "https://www.pornpics.com/?page=2"
PAGE_3 = "https://www.pornpics.com/?page=3"


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


class PaginatedScraper(FakeScraper):
    def __init__(self) -> None:
        self.browse_calls: list[str] = []

    async def browse(self, **kwargs: object) -> dict:
        url = str(kwargs["url"])
        self.browse_calls.append(url)
        pages = {
            ROOT: (GALLERY_A, "Alpha", PAGE_2),
            PAGE_2: (GALLERY_B, "Beta", PAGE_3),
            PAGE_3: (GALLERY_C, "Gamma", None),
        }
        gallery_url, title, next_url = pages[url]
        return {
            "items": [
                {
                    "url": gallery_url,
                    "title": title,
                    "thumbnail_remote_url": f"https://cdni.pornpics.com/t/{title}.png",
                }
            ],
            "next_url": next_url,
        }


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


def pose_first_match(
    ordinal: int,
    *,
    appearance: float,
    exact: float = 0,
    pose: float | None = None,
    reliable: bool = False,
    coverage: float = 0.9,
    body_confidence: float = 0.9,
) -> dict[str, object]:
    diagnostics = {
        "pose_score": pose,
        "pose_reliable": reliable,
        "pose_coverage": coverage,
        "pose_body_confidence": body_confidence,
    }
    tier, score = FinderService._ranked_score(
        appearance,
        exact,
        diagnostics,
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
    )
    return {
        "image_url": f"https://cdni.pornpics.com/full/rank-{ordinal}.jpg",
        "preview_remote_url": f"https://cdni.pornpics.com/p/rank-{ordinal}.jpg",
        "ordinal": ordinal,
        "score": score,
        "ranking_tier": tier,
        "appearance_score": appearance,
        "exact_score": exact,
        **diagnostics,
    }


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
async def test_finder_reuses_persistent_corpus_without_media_or_candidate_inference(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    first_encoder = FakeEncoder()
    first_service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=first_encoder,
        media_fetcher=fake_media,
    )
    await first_service.start()
    try:
        first = first_service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(first_service.queue.join(), 30)
        first_results, _ = first_service.results(
            first["id"], review="all", min_score=0, limit=20, offset=0
        )
        corpus = first_service.corpus_status()
        assert corpus == {
            "galleries": 2,
            "images": 3,
            "complete": 2,
            "partial": 0,
            "ready": 3,
            "cache_entries": 4,
            "cache_bytes": corpus["cache_bytes"],
            "max_cache_entries": config.finder_cache_max_entries,
            "max_cache_bytes": config.finder_cache_max_bytes,
        }
        assert corpus["cache_bytes"] > 0
        with database.connect() as connection:
            connection.execute(
                """UPDATE finder_embedding_cache SET last_used_at = '2000-01-01'
                   WHERE source_key LIKE 'url:%'"""
            )
    finally:
        await first_service.stop()

    class LocalOnlyScraper:
        def __init__(self) -> None:
            self.browse_calls = 0
            self.gallery_calls = 0

        async def browse(self, **_: object) -> dict:
            self.browse_calls += 1
            return {"items": [], "next_url": None}

        async def gallery(self, _: str) -> dict:
            self.gallery_calls += 1
            raise AssertionError("A local-only result must not fetch gallery media")

    async def forbidden_media(_: str, __: str) -> bytes:
        raise AssertionError("A reusable corpus descriptor must not be downloaded")

    local_scraper = LocalOnlyScraper()
    second_encoder = FakeEncoder()
    second_service = FinderService(
        config,
        database,
        local_scraper,
        EventBroker(),
        encoder=second_encoder,
        media_fetcher=forbidden_media,
    )
    await second_service.start()
    try:
        # A restart can report reusable rows before lazy model preparation.
        assert second_service.corpus_status()["ready"] == 3
        second = second_service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(second_service.queue.join(), 30)
        completed = second_service.get_scan(second["id"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["corpus_search_complete"] is True
        assert completed["corpus_images_scored"] == 3
        assert completed["corpus_galleries_scored"] == 2
        assert completed["processed_galleries"] == 0
        results, total = second_service.results(
            second["id"], review="all", min_score=0, limit=20, offset=0
        )
        assert total == 2
        assert all(item["online_scanned"] is False for item in results)
        assert [(item["title"], item["score"]) for item in results] == [
            (item["title"], item["score"]) for item in first_results
        ]
        assert second_encoder.embed_calls == 0
        assert local_scraper.browse_calls == 1
        assert local_scraper.gallery_calls == 0
        with database.connect() as connection:
            touched = connection.execute(
                """SELECT COUNT(*) FROM finder_embedding_cache
                   WHERE source_key LIKE 'url:%'
                     AND last_used_at != '2000-01-01'"""
            ).fetchone()[0]
        # Only each candidate gallery's leading descriptor is promoted in LRU;
        # sweeping the corpus must not rewrite every cache row.
        assert touched == 2
    finally:
        await second_service.stop()


@pytest.mark.asyncio
async def test_finder_partial_corpus_refresh_and_pause_preserve_review(
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
        seed = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(seed["id"])["status"] == "completed"
        key = finder_module.gallery_key(GALLERY_A)
        black_source = service._remote_source_key(
            "https://cdni.pornpics.com/p/black.png"
        )
        with database.connect() as connection:
            connection.execute(
                """DELETE FROM finder_corpus_images
                   WHERE gallery_key = ? AND source_key = ?""",
                (key, black_source),
            )
            connection.execute(
                """UPDATE finder_corpus_galleries
                   SET state = 'partial', image_count = 1
                   WHERE gallery_key = ?""",
                (key,),
            )
        assert service.corpus_status()["partial"] == 1
        assert service.corpus_status()["ready"] == 2

        gallery_started = asyncio.Event()
        release_gallery = asyncio.Event()

        class BlockingRefreshScraper(FakeScraper):
            def __init__(self) -> None:
                self.gallery_calls = 0

            async def browse(self, **_: object) -> dict:
                return {
                    "items": [
                        {
                            "url": GALLERY_A,
                            "title": "Alpha refreshed",
                            "thumbnail_remote_url": (
                                "https://cdni.pornpics.com/t/a-new.png"
                            ),
                        }
                    ],
                    "next_url": None,
                }

            async def gallery(self, url: str) -> dict:
                self.gallery_calls += 1
                if self.gallery_calls == 1:
                    gallery_started.set()
                    await release_gallery.wait()
                return await super().gallery(url)

        scraper = BlockingRefreshScraper()
        service.scraper = scraper
        scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(gallery_started.wait(), 30)
        local_results, _ = service.results(
            scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        local_alpha = next(item for item in local_results if item["title"] == "Alpha")
        assert local_alpha["online_scanned"] is False
        original_result_id = local_alpha["id"]
        service.set_review(scan["id"], original_result_id, "accepted")

        assert service.pause(scan["id"])["status"] == "pausing"
        release_gallery.set()
        await asyncio.wait_for(service.queue.join(), 30)
        paused = service.get_scan(scan["id"])
        assert paused["status"] == "paused"
        assert paused["pages_completed"] == 0
        assert paused["corpus_search_complete"] is True
        corpus_counts = (
            paused["corpus_images_scored"],
            paused["corpus_galleries_scored"],
        )

        service.resume(scan["id"])
        await asyncio.wait_for(service.queue.join(), 30)
        completed = service.get_scan(scan["id"])
        assert completed["status"] == "completed"
        assert (
            completed["corpus_images_scored"],
            completed["corpus_galleries_scored"],
        ) == corpus_counts
        results, _ = service.results(
            scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        refreshed = next(item for item in results if item["gallery_key"] == key)
        assert refreshed["id"] == original_result_id
        assert refreshed["review"] == "accepted"
        assert refreshed["online_scanned"] is True
        assert refreshed["title"] == "Alpha refreshed"
        assert scraper.gallery_calls == 2
        with database.connect() as connection:
            gallery = connection.execute(
                """SELECT state, image_count FROM finder_corpus_galleries
                   WHERE gallery_key = ?""",
                (key,),
            ).fetchone()
            associations = connection.execute(
                """SELECT source_key, ordinal FROM finder_corpus_images
                   WHERE gallery_key = ? ORDER BY ordinal""",
                (key,),
            ).fetchall()
        assert gallery["state"] == "complete"
        assert gallery["image_count"] == 2
        assert len(associations) == 2
        assert black_source in {row["source_key"] for row in associations}
        assert service.corpus_status()["ready"] == 3
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_repeated_online_failure_keeps_local_candidate_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        seed = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(seed["id"])["status"] == "completed"

        gallery_started = asyncio.Event()
        release_gallery = asyncio.Event()

        class FailingRefreshScraper:
            def __init__(self) -> None:
                self.gallery_calls = 0

            async def browse(self, **kwargs: object) -> dict:
                return {
                    "items": [
                        {
                            "url": GALLERY_A,
                            "title": "Retry Alpha",
                            "thumbnail_remote_url": (
                                "https://cdni.pornpics.com/t/retry.png"
                            ),
                        }
                    ],
                    "next_url": PAGE_2 if int(kwargs["page"]) == 1 else None,
                }

            async def gallery(self, _: str) -> dict:
                self.gallery_calls += 1
                if self.gallery_calls == 1:
                    gallery_started.set()
                    await release_gallery.wait()
                raise RuntimeError("temporary source failure")

        scraper = FailingRefreshScraper()
        service.scraper = scraper
        scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=2,
            minimum_score=0,
        )
        await asyncio.wait_for(gallery_started.wait(), 30)
        local_results, _ = service.results(
            scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        local = next(item for item in local_results if item["gallery_url"] == GALLERY_A)
        original_id = local["id"]
        original_score = local["score"]
        service.set_review(scan["id"], original_id, "accepted")
        release_gallery.set()
        await asyncio.wait_for(service.queue.join(), 30)

        completed = service.get_scan(scan["id"])
        assert completed["status"] == "completed_with_errors"
        assert completed["processed_galleries"] == 1
        assert completed["failed_galleries"] == 1
        results, _ = service.results(
            scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        preserved = next(item for item in results if item["gallery_url"] == GALLERY_A)
        assert preserved["id"] == original_id
        assert preserved["review"] == "accepted"
        assert preserved["score"] == original_score
        assert preserved["status"] == "completed"
        assert preserved["online_scanned"] is False
        assert preserved["online_refresh_failed"] is True
        assert "temporary source failure" in preserved["error"]
        assert scraper.gallery_calls == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_corpus_search_pause_resumes_idempotently(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        seed = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(seed["id"])["status"] == "completed"

        class EmptyScraper:
            async def browse(self, **_: object) -> dict:
                return {"items": [], "next_url": None}

            async def gallery(self, _: str) -> dict:
                raise AssertionError

        service.scraper = EmptyScraper()
        original_rows = service._corpus_descriptor_rows
        scan_holder: dict[str, str] = {}
        calls = 0

        def pause_after_first_batch(
            after: tuple[str, int, str] | None,
            *,
            limit: int = 256,
        ):
            nonlocal calls
            rows = original_rows(after, limit=limit)
            calls += 1
            if calls == 1:
                service.pause(scan_holder["id"])
            return rows

        monkeypatch.setattr(service, "_corpus_descriptor_rows", pause_after_first_batch)
        scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        scan_holder["id"] = scan["id"]
        await asyncio.wait_for(service.queue.join(), 30)
        paused = service.get_scan(scan["id"])
        assert paused["status"] == "paused"
        assert paused["corpus_search_complete"] is False

        monkeypatch.setattr(service, "_corpus_descriptor_rows", original_rows)
        service.resume(scan["id"])
        await asyncio.wait_for(service.queue.join(), 30)
        completed = service.get_scan(scan["id"])
        assert completed["status"] == "completed"
        assert completed["corpus_search_complete"] is True
        assert completed["corpus_images_scored"] == 3
        assert completed["corpus_galleries_scored"] == 2
        results, total = service.results(
            scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        assert total == 2
        assert len({item["gallery_key"] for item in results}) == 2
    finally:
        await service.stop()


def test_finder_corpus_backfill_unions_historical_top_matches(
    tmp_path: Path,
) -> None:
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    service.ensure_schema()
    scans = [
        service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        for _ in range(2)
    ]
    for scan, ordinals in zip(scans, ((1, 2, 3), (4, 5, 6)), strict=True):
        service._save_result(
            scan["id"],
            {"url": GALLERY_A, "title": "Historical Alpha"},
            order=1,
            score=0,
            images_scored=3,
            best=None,
            status="completed",
            ranking_version=finder_module.CURRENT_RANKING_VERSION,
            top_matches=[
                pose_first_match(ordinal, appearance=0.9 - ordinal / 100)
                for ordinal in ordinals
            ],
        )
    with database.connect() as connection:
        connection.execute(
            """DELETE FROM finder_corpus_meta
               WHERE key = 'historical-backfill'"""
        )
    service.ensure_schema()
    service.ensure_schema()

    with database.connect() as connection:
        gallery = connection.execute(
            """SELECT state, image_count FROM finder_corpus_galleries
               WHERE gallery_key = ?""",
            (finder_module.gallery_key(GALLERY_A),),
        ).fetchone()
        images = connection.execute(
            """SELECT source_key FROM finder_corpus_images
               WHERE gallery_key = ?""",
            (finder_module.gallery_key(GALLERY_A),),
        ).fetchall()
        indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(finder_corpus_images)"
            ).fetchall()
        }
        query, params = service._corpus_descriptor_query(("a", 0, "url:a"))
        query_plan = [
            str(row["detail"])
            for row in connection.execute(
                f"EXPLAIN QUERY PLAN {query}", params
            ).fetchall()
        ]
    assert gallery["state"] == "partial"
    assert gallery["image_count"] == 6
    assert len(images) == 6
    assert len({row["source_key"] for row in images}) == 6
    assert "idx_finder_corpus_images_scan" in indexes
    assert not any("TEMP B-TREE" in detail for detail in query_plan)
    assert any(
        "SEARCH i USING INDEX idx_finder_corpus_images_scan" in detail
        for detail in query_plan
    )
    assert any("idx_finder_embedding_cache_source" in detail for detail in query_plan)


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
        assert match["skeleton_overlay_url"].startswith("data:image/svg+xml;base64,")
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
    diagnostics = FinderService._pose_diagnostics({"pose": sparse.as_dict()}, (sparse,))
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


def test_pose_first_ranking_orders_exact_pose_fallback_then_mismatch() -> None:
    exact = pose_first_match(1, appearance=0.05, exact=0.875)
    correct_pose = pose_first_match(2, appearance=0.10, pose=0.72, reliable=True)
    visual_fallback = pose_first_match(3, appearance=0.98, pose=0.99, reliable=False)
    wrong_pose = pose_first_match(4, appearance=1.0, pose=0.54, reliable=True)

    matches = FinderService._normalized_top_matches(
        [wrong_pose, visual_fallback, correct_pose, exact],
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
    )

    assert [match["ordinal"] for match in matches] == [1, 2, 3]
    assert [match["ranking_tier"] for match in matches] == [3, 2, 1]
    assert matches[1]["score"] == pytest.approx(0.72)
    assert matches[1]["appearance_score"] == pytest.approx(0.10)
    assert matches[2]["score"] == pytest.approx(0.98)
    assert wrong_pose["ranking_tier"] == 0
    assert wrong_pose["score"] == pytest.approx(0.54)


def test_pose_first_uses_evidence_then_appearance_only_as_tiebreakers() -> None:
    low_evidence = pose_first_match(
        1,
        appearance=0.99,
        pose=0.8,
        reliable=True,
        coverage=0.5,
        body_confidence=0.2,
    )
    high_evidence_low_appearance = pose_first_match(
        2,
        appearance=0.1,
        pose=0.8,
        reliable=True,
        coverage=0.9,
        body_confidence=0.9,
    )
    same_evidence_higher_appearance = pose_first_match(
        3,
        appearance=0.2,
        pose=0.8,
        reliable=True,
        coverage=0.9,
        body_confidence=0.9,
    )

    matches = FinderService._normalized_top_matches(
        [
            low_evidence,
            high_evidence_low_appearance,
            same_evidence_higher_appearance,
        ],
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
    )

    assert [match["ordinal"] for match in matches] == [3, 2, 1]


def test_pose_first_persists_gallery_tier_and_uses_leading_lane_threshold(
    tmp_path: Path,
) -> None:
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    service.ensure_schema()
    scan = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0.65,
    )
    assert scan["ranking_version"] == finder_module.CURRENT_RANKING_VERSION
    assert scan["ranking_current"] is True

    pose_hit = pose_first_match(1, appearance=0.1, pose=0.6, reliable=True)
    same_gallery_fallback = pose_first_match(2, appearance=0.99)
    service._save_result(
        scan["id"],
        {"url": GALLERY_A, "title": "Pose hit"},
        order=1,
        score=0,
        images_scored=2,
        best=None,
        status="completed",
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
        top_matches=[same_gallery_fallback, pose_hit],
    )
    service._save_result(
        scan["id"],
        {"url": GALLERY_B, "title": "Visual fallback"},
        order=2,
        score=0,
        images_scored=1,
        best=None,
        status="completed",
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
        top_matches=[pose_first_match(1, appearance=0.99)],
    )
    service._save_result(
        scan["id"],
        {"url": GALLERY_C, "title": "Exact"},
        order=3,
        score=0,
        images_scored=1,
        best=None,
        status="completed",
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
        top_matches=[pose_first_match(1, appearance=0.1, exact=0.875)],
    )

    results, total = service.results(
        scan["id"], review="all", min_score=0, limit=10, offset=0
    )
    assert total == 3
    assert [item["title"] for item in results] == [
        "Exact",
        "Pose hit",
        "Visual fallback",
    ]
    pose_gallery = results[1]
    assert pose_gallery["ranking_tier"] == 2
    assert pose_gallery["score"] == pytest.approx(0.6)
    assert pose_gallery["best_ordinal"] == 1
    assert pose_gallery["above_threshold"] is False
    service.set_review(scan["id"], pose_gallery["id"], "maybe")
    summary = service.get_scan(scan["id"])
    assert summary["candidate_count"] == 2
    assert summary["maybe_count"] == 0
    assert summary["review_counts"] == {
        "pending": 2,
        "maybe": 0,
        "accepted": 0,
        "rejected": 0,
        "total": 2,
    }
    assert service.result_review_counts(scan["id"], min_score=0)["maybe"] == 1
    assert service.result_review_counts(scan["id"], min_score=None)["maybe"] == 0


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
async def test_finder_extend_continues_from_cursor_and_preserves_scan_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    scraper = PaginatedScraper()
    service = FinderService(
        config,
        database,
        scraper,
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        first = service.get_scan(scan["id"])
        assert first is not None
        assert first["status"] == "completed"
        assert first["pages_completed"] == 1
        assert first["processed_galleries"] == 1
        assert first["has_next_page"] is True
        original_reference_count = first["reference_count"]

        first_results, _ = service.results(
            scan["id"], review="all", min_score=0, limit=10, offset=0
        )
        service.set_review(scan["id"], first_results[0]["id"], "accepted")

        extended = service.extend(scan["id"], additional_pages=2)
        assert extended["status"] == "queued"
        assert extended["page_limit"] == 3
        assert extended["pages_completed"] == 1
        assert extended["processed_galleries"] == 1
        assert extended["reference_count"] == original_reference_count
        await asyncio.wait_for(service.queue.join(), 30)

        completed = service.get_scan(scan["id"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["pages_completed"] == 3
        assert completed["processed_galleries"] == 3
        assert completed["reference_count"] == original_reference_count
        assert completed["has_next_page"] is False
        assert scraper.browse_calls == [ROOT, PAGE_2, PAGE_3]
        results, total = service.results(
            scan["id"], review="all", min_score=0, limit=10, offset=0
        )
        assert total == 3
        assert sum(item["review"] == "accepted" for item in results) == 1
        with pytest.raises(finder_module.FinderConflict, match="exhausted"):
            service.extend(scan["id"], additional_pages=1)
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_extend_active_limit_race_and_paused_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    scraper = PaginatedScraper()
    service = FinderService(
        config,
        database,
        scraper,
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        original_finalize = service._finalize_scan_if_done
        extended_at_boundary = False

        def extend_before_finalize(scan_id: str) -> bool:
            nonlocal extended_at_boundary
            if not extended_at_boundary:
                extended_at_boundary = True
                active = service.extend(scan_id, additional_pages=1)
                assert active["status"] == "scanning"
            return original_finalize(scan_id)

        monkeypatch.setattr(service, "_finalize_scan_if_done", extend_before_finalize)
        await asyncio.wait_for(service.queue.join(), 30)
        completed = service.get_scan(scan["id"])
        assert completed is not None
        assert completed["status"] == "completed"
        assert completed["page_limit"] == 2
        assert completed["pages_completed"] == 2
        assert scraper.browse_calls == [ROOT, PAGE_2]

        paused_scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        paused = service.pause(paused_scan["id"])
        assert paused["status"] == "paused"
        extended_paused = service.extend(paused_scan["id"], additional_pages=2)
        assert extended_paused["status"] == "paused"
        assert extended_paused["page_limit"] == 3
        assert extended_paused["pages_completed"] == 0
        service.resume(paused_scan["id"])
        await asyncio.wait_for(service.queue.join(), 30)
        assert service.get_scan(paused_scan["id"])["pages_completed"] == 3
    finally:
        await service.stop()


def test_finder_extend_rejects_canceled_invalid_and_over_limit(
    tmp_path: Path,
) -> None:
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    service.ensure_schema()
    scan = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    service.delete_or_cancel(scan["id"])
    with pytest.raises(finder_module.FinderConflict, match="canceled"):
        service.extend(scan["id"], additional_pages=1)
    for invalid in (False, 0, 51):
        with pytest.raises(ValueError, match="between"):
            service.extend(scan["id"], additional_pages=invalid)

    second = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    with database.connect() as connection:
        connection.execute(
            """UPDATE finder_scans
               SET status = 'paused', pause_requested = 1, page_limit = 499
               WHERE id = ?""",
            (second["id"],),
        )
    with pytest.raises(ValueError, match="at most 500"):
        service.extend(second["id"], additional_pages=2)


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
    assert (
        original.results(scan["id"], review="all", min_score=0, limit=20, offset=0)[1]
        == 2
    )
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
    assert reset["corpus_search_complete"] is False
    assert reset["corpus_images_scored"] == 0
    assert reset["corpus_galleries_scored"] == 0
    assert (
        upgraded.results(scan["id"], review="all", min_score=0, limit=20, offset=0)[1]
        == 0
    )
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

    now = "2026-01-01T00:00:00+00:00"
    with database.connect() as connection:
        connection.execute(
            """INSERT INTO finder_corpus_galleries(
                   gallery_key, gallery_url, title, state, image_count,
                   created_at, updated_at
               ) VALUES ('mapped', ?, 'Mapped', 'partial', 1, ?, ?)""",
            (GALLERY_A, now, now),
        )
        connection.execute(
            """INSERT INTO finder_corpus_images(
                   gallery_key, source_key, image_url, preview_remote_url,
                   ordinal, created_at, updated_at
               ) VALUES (
                   'mapped', 'current-0',
                   'https://cdn.example/full.jpg',
                   'https://cdn.example/preview.jpg', 1, ?, ?
               )""",
            (now, now),
        )
    assert service.corpus_status()["ready"] == 1

    service._store_embedding("current-extra", False, embedding, metadata)
    assert service._prune_embedding_cache() == 2
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finder_embedding_cache"
            ).fetchone()[0]
            == 2
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM finder_corpus_images").fetchone()[
                0
            ]
            == 1
        )
    assert service.corpus_status()["images"] == 1
    assert service.corpus_status()["ready"] == 0

    service._pose_ready = True
    service._store_embedding(
        "transient-pose-failure",
        False,
        embedding,
        {**metadata, "pose_error": "temporary CUDA OOM"},
    )
    assert service._cached_descriptor("transient-pose-failure", False) is None
    with database.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM finder_embedding_cache WHERE source_key = ?",
                ("transient-pose-failure",),
            ).fetchone()[0]
            == 0
        )


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
            """INSERT INTO finder_scans(
                   id, example_directory, reference_ready, reference_count,
                   pose_tag_id, pose_tag_label, pose_tag_slug, pose_default_role,
                   source_url, next_url, page_limit, pages_completed,
                   minimum_score, status, created_at, updated_at
               ) VALUES ('legacy-active', 'pose', 1, 1, ?, 'Standing',
                         'standing', 'solo', ?, ?, 2, 1, 0, 'scanning', ?, ?)""",
            (tag_id, ROOT, PAGE_2, now, now),
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
        cache_before = tuple(
            connection.execute(
                """SELECT cache_key, model_key, source_key, include_mirror,
                          rows, dimensions, embedding, created_at, last_used_at
                   FROM finder_embedding_cache WHERE cache_key = 'old-cache'"""
            ).fetchone()
        )

    service = FinderService(config, database, FakeScraper(), EventBroker())
    service.ensure_schema()
    service.ensure_schema()

    with database.connect() as connection:
        scan_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(finder_scans)")
        }
        reference_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(finder_scan_references)")
        }
        cache_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(finder_embedding_cache)")
        }
        result_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(finder_results)")
        }
        scan_after = connection.execute(
            """SELECT corpus_search_complete, corpus_images_scored,
                      corpus_galleries_scored
               FROM finder_scans WHERE id = 'legacy'"""
        ).fetchone()
        result_after = connection.execute(
            """SELECT online_scanned FROM finder_results
               WHERE id = 'legacy-result'"""
        ).fetchone()
        reference_metadata = connection.execute(
            "SELECT metadata_json FROM finder_scan_references"
        ).fetchone()[0]
        cache_metadata = connection.execute(
            "SELECT metadata_json FROM finder_embedding_cache"
        ).fetchone()[0]
        legacy_scan = connection.execute(
            """SELECT ranking_version FROM finder_scans
               WHERE id = 'legacy'"""
        ).fetchone()
        frozen_scan = connection.execute(
            """SELECT status, ranking_version, error FROM finder_scans
               WHERE id = 'legacy-active'"""
        ).fetchone()
        migrated_result = connection.execute(
            """SELECT ranking_tier FROM finder_results
               WHERE id = 'legacy-result'"""
        ).fetchone()
        indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(finder_results)"
            ).fetchall()
        }
        cache_indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(finder_embedding_cache)"
            ).fetchall()
        }
        cache_after = tuple(
            connection.execute(
                """SELECT cache_key, model_key, source_key, include_mirror,
                          rows, dimensions, embedding, created_at, last_used_at
                   FROM finder_embedding_cache WHERE cache_key = 'old-cache'"""
            ).fetchone()
        )
        corpus_gallery = connection.execute(
            """SELECT state, image_count FROM finder_corpus_galleries
               WHERE gallery_key = 'legacy-gallery'"""
        ).fetchone()
        corpus_images = connection.execute(
            """SELECT source_key, image_url, preview_remote_url, ordinal
               FROM finder_corpus_images
               WHERE gallery_key = 'legacy-gallery'"""
        ).fetchall()
    assert "reference_model_key" in scan_columns
    assert "ranking_version" in scan_columns
    assert "metadata_json" in reference_columns
    assert "metadata_json" in cache_columns
    assert "matches_json" in result_columns
    assert "ranking_tier" in result_columns
    assert "online_scanned" in result_columns
    assert scan_after["corpus_search_complete"] == 1
    assert scan_after["corpus_images_scored"] == 0
    assert scan_after["corpus_galleries_scored"] == 0
    assert result_after["online_scanned"] == 1
    assert json.loads(reference_metadata) == {}
    assert json.loads(cache_metadata) == {}
    assert legacy_scan["ranking_version"] == finder_module.LEGACY_RANKING_VERSION
    assert frozen_scan["ranking_version"] == finder_module.LEGACY_RANKING_VERSION
    assert frozen_scan["status"] == "failed"
    assert "legacy appearance ranking" in frozen_scan["error"]
    assert migrated_result["ranking_tier"] == 1
    assert "idx_finder_results_rank" in indexes
    assert "idx_finder_results_review_rank" in indexes
    assert "idx_finder_embedding_cache_source" in cache_indexes
    assert cache_after == cache_before
    assert corpus_gallery["state"] == "partial"
    assert corpus_gallery["image_count"] == 1
    assert len(corpus_images) == 1
    assert corpus_images[0]["source_key"] == service._remote_source_key(
        "https://cdn.example/preview.jpg"
    )
    assert corpus_images[0]["image_url"].endswith("full.jpg")
    assert corpus_images[0]["ordinal"] == 4

    results, total = service.results(
        "legacy", review="all", min_score=0, limit=10, offset=0
    )
    assert total == 1
    assert results[0]["id"] == "legacy-result"
    assert results[0]["top_matches"][0]["rank"] == 1
    assert results[0]["top_matches"][0]["image_url"].endswith("full.jpg")
    neutral = service.set_review("legacy", "legacy-result", "maybe")
    assert neutral["review"] == "maybe"
    assert neutral["feedback_image_urls"] == []
    # No finder_results rebuild is needed: legacy review columns are plain
    # durable text and remain compatible across repeated schema upgrades.
    service.ensure_schema()
    maybe_results, maybe_total = service.results(
        "legacy", review="maybe", min_score=0, limit=10, offset=0
    )
    assert maybe_total == 1
    assert maybe_results[0]["id"] == "legacy-result"
    with database.connect() as connection:
        assert (
            connection.execute(
                """SELECT review FROM finder_results
                   WHERE id = 'legacy-result'"""
            ).fetchone()[0]
            == "maybe"
        )
    legacy = service.get_scan("legacy")
    assert legacy["maybe_count"] == 1
    assert legacy["review_counts"]["maybe"] == 1
    assert legacy["ranking_current"] is False
    assert legacy["extendable"] is False
    with pytest.raises(finder_module.FinderConflict, match="legacy-ranked"):
        service.extend("legacy", additional_pages=1)

    with database.connect() as connection:
        connection.execute(
            """UPDATE finder_scans
               SET status = 'paused', pause_requested = 1
               WHERE id = 'legacy-active'"""
        )
    with pytest.raises(finder_module.FinderConflict, match="legacy-ranked"):
        service.resume("legacy-active")
    assert "legacy-active" not in service._queued_scan_ids()


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
    fake_scraper = FakeScraper()
    app.state.finder.scraper = fake_scraper
    monkeypatch.setattr(app.state.scraper, "gallery", fake_scraper.gallery)
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
            now = finder_module.utc_now()
            with app.state.db.connect() as connection:
                connection.execute(
                    """INSERT INTO finder_corpus_galleries(
                           gallery_key, gallery_url, title,
                           thumbnail_remote_url, state, image_count,
                           created_at, updated_at
                       ) VALUES (?, ?, 'Historic partial', '', 'partial', 1, ?, ?)""",
                    (gallery_key(GALLERY_A), GALLERY_A, now, now),
                )
                connection.execute(
                    """INSERT INTO finder_corpus_images(
                           gallery_key, source_key, image_url,
                           preview_remote_url, ordinal, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (
                        gallery_key(GALLERY_A),
                        "url:https://cdni.pornpics.com/p/stale.png",
                        "https://cdni.pornpics.com/full/stale.png",
                        "https://cdni.pornpics.com/p/stale.png",
                        now,
                        now,
                    ),
                )
            opened_gallery = await client.get(
                f"/api/galleries/{encode_gallery_id(GALLERY_A)}"
            )
            assert opened_gallery.status_code == 200
            with app.state.db.connect() as connection:
                promoted = connection.execute(
                    """SELECT state, image_count
                       FROM finder_corpus_galleries WHERE gallery_key = ?""",
                    (gallery_key(GALLERY_A),),
                ).fetchone()
                promoted_images = connection.execute(
                    """SELECT image_url FROM finder_corpus_images
                       WHERE gallery_key = ? ORDER BY ordinal""",
                    (gallery_key(GALLERY_A),),
                ).fetchall()
            assert dict(promoted) == {"state": "complete", "image_count": 2}
            assert [row["image_url"] for row in promoted_images] == [
                "https://cdni.pornpics.com/full/blue.png",
                "https://cdni.pornpics.com/full/black.png",
            ]
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
            assert listed_scans.json()["items"][0]["has_next_page"] is False

            for invalid in (True, "2", 1.5, 0, 51, None):
                invalid_extension = await client.post(
                    f"/api/finder/scans/{scan_id}/extend",
                    json={"additional_pages": invalid},
                )
                assert invalid_extension.status_code == 422
            missing_value = await client.post(
                f"/api/finder/scans/{scan_id}/extend",
                json={},
            )
            assert missing_value.status_code == 422

            with app.state.db.connect() as connection:
                connection.execute(
                    """UPDATE finder_scans
                       SET status = 'completed_with_errors', next_url = ?,
                           failed_galleries = 1, finished_at = updated_at
                       WHERE id = ?""",
                    (PAGE_2, scan_id),
                )
            accepted_extension = await client.post(
                f"/api/finder/scans/{scan_id}/extend",
                json={"additional_pages": 1},
            )
            assert accepted_extension.status_code == 202
            accepted_scan = accepted_extension.json()["scan"]
            assert accepted_scan["status"] == "queued"
            assert accepted_scan["page_limit"] == 2
            assert accepted_scan["pages_completed"] == 1
            await asyncio.wait_for(app.state.finder.queue.join(), 30)

            exhausted = await client.post(
                f"/api/finder/scans/{scan_id}/extend",
                json={"additional_pages": 1},
            )
            assert exhausted.status_code == 409

            with app.state.db.connect() as connection:
                connection.execute(
                    """UPDATE finder_scans
                       SET status = 'canceled', next_url = ?,
                           cancel_requested = 1
                       WHERE id = ?""",
                    (PAGE_2, scan_id),
                )
            canceled_extension = await client.post(
                f"/api/finder/scans/{scan_id}/extend",
                json={"additional_pages": 1},
            )
            assert canceled_extension.status_code == 409

            with app.state.db.connect() as connection:
                connection.execute(
                    """UPDATE finder_scans
                       SET status = 'failed', next_url = ?,
                           cancel_requested = 0
                       WHERE id = ?""",
                    (PAGE_2, scan_id),
                )
            failed_extension = await client.post(
                f"/api/finder/scans/{scan_id}/extend",
                json={"additional_pages": 1},
            )
            assert failed_extension.status_code == 409

            missing_extension = await client.post(
                "/api/finder/scans/missing/extend",
                json={"additional_pages": 1},
            )
            assert missing_extension.status_code == 404
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
            selected_image = result["top_matches"][0]["image_url"]
            feedback_before = await client.get(f"/api/finder/feedback/{tag['id']}")
            assert feedback_before.status_code == 200
            assert feedback_before.json()["feedback"]["accepted_samples"] == 0

            reviewed = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={
                    "review": "accepted",
                    "feedback_image_urls": [selected_image],
                },
            )
            assert reviewed.status_code == 200
            reviewed_body = reviewed.json()
            assert reviewed_body["result"]["feedback_image_urls"] == [selected_image]
            assert reviewed_body["feedback"]["accepted_galleries"] == 1
            assert reviewed_body["feedback"]["accepted_samples"] == 1
            assert reviewed_body["feedback"]["applies_to"] == "future_scans"

            invalid_neutral = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={
                    "review": "maybe",
                    "feedback_image_urls": [selected_image],
                },
            )
            assert invalid_neutral.status_code == 400
            neutral = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={"review": "maybe", "feedback_image_urls": []},
            )
            assert neutral.status_code == 200
            assert neutral.json()["result"]["review"] == "maybe"
            assert neutral.json()["result"]["feedback_image_urls"] == []
            assert neutral.json()["feedback"]["accepted_samples"] == 0
            maybe_results = await client.get(
                f"/api/finder/scans/{scan_id}/results",
                params={"review": "maybe", "min_score": 0},
            )
            assert maybe_results.status_code == 200
            assert maybe_results.json()["total"] == 1
            assert maybe_results.json()["items"][0]["id"] == result["id"]
            assert maybe_results.json()["counts"]["maybe"] == 1
            scan_response = await client.get(f"/api/finder/scans/{scan_id}")
            assert scan_response.json()["scan"]["maybe_count"] == 1
            assert scan_response.json()["scan"]["review_counts"]["maybe"] == 1
            restored = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={
                    "review": "accepted",
                    "feedback_image_urls": [selected_image],
                },
            )
            assert restored.status_code == 200

            empty_accept = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={"review": "accepted", "feedback_image_urls": []},
            )
            assert empty_accept.status_code == 400
            unknown_image = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={
                    "review": "rejected",
                    "feedback_image_urls": [
                        "https://cdni.pornpics.com/full/not-proposed.jpg"
                    ],
                },
            )
            assert unknown_image.status_code == 400
            too_many = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={
                    "review": "rejected",
                    "feedback_image_urls": [
                        "https://example.com/1.jpg",
                        "https://example.com/2.jpg",
                        "https://example.com/3.jpg",
                        "https://example.com/4.jpg",
                    ],
                },
            )
            assert too_many.status_code == 422

            empty_reject = await client.patch(
                f"/api/finder/scans/{scan_id}/results/{result['id']}",
                json={"review": "rejected", "feedback_image_urls": []},
            )
            assert empty_reject.status_code == 200
            assert empty_reject.json()["result"]["feedback_image_urls"] == []
            assert empty_reject.json()["feedback"]["rejected_galleries"] == 0
            assert empty_reject.json()["feedback"]["rejected_samples"] == 0

            reset = await client.delete(f"/api/finder/feedback/{tag['id']}")
            assert reset.status_code == 200
            assert reset.json()["feedback"]["rejected_galleries"] == 0
            assert (await client.get("/api/finder/feedback/999999")).status_code == 404


@pytest.mark.asyncio
async def test_finder_feedback_curated_samples_persist_and_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=SpatialFakeEncoder(),
        pose_estimator=FakePoseEstimator(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        first_scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        results, _ = service.results(
            first_scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        result = results[0]
        selected_match = result["top_matches"][0]
        selected_url = selected_match["image_url"]
        review_only = results[1]
        initial_revision = service.feedback_status(tag_id)["revision"]
        rejected_without_samples = service.set_review(
            first_scan["id"],
            review_only["id"],
            "rejected",
            [],
        )
        assert rejected_without_samples["review"] == "rejected"
        assert rejected_without_samples["feedback_image_urls"] == []
        assert service.feedback_status(tag_id)["revision"] == initial_revision
        with pytest.raises(ValueError, match="cannot include"):
            service.set_review(
                first_scan["id"],
                result["id"],
                "maybe",
                [selected_url],
            )
        with pytest.raises(ValueError, match="cannot include"):
            service.set_review(
                first_scan["id"],
                result["id"],
                "pending",
                [selected_url],
            )
        neutral_without_feedback = service.set_review(
            first_scan["id"],
            result["id"],
            "maybe",
            [],
        )
        assert neutral_without_feedback["review"] == "maybe"
        assert neutral_without_feedback["feedback_image_urls"] == []
        assert service.feedback_status(tag_id)["revision"] == initial_revision

        # Feedback always uses the current analyzer cache, even when the result
        # belongs to a scan created under an older DINO/analyzer namespace.
        assert service._model_key != "retired-analyzer-model"
        with database.connect() as connection:
            connection.execute(
                """UPDATE finder_scans SET reference_model_key = ?
                   WHERE id = ?""",
                ("retired-analyzer-model", first_scan["id"]),
            )

        adjacent_url = "https://cdni.pornpics.com/full/adjacent-feedback.png"
        adjacent_preview = "https://cdni.pornpics.com/p/adjacent-feedback.png"
        now = finder_module.utc_now()
        with database.connect() as connection:
            connection.execute(
                """INSERT INTO finder_corpus_images(
                       gallery_key, source_key, image_url, preview_remote_url,
                       ordinal, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result["gallery_key"],
                    service._remote_source_key(adjacent_preview),
                    adjacent_url,
                    adjacent_preview,
                    99,
                    now,
                    now,
                ),
            )
        assert adjacent_url not in {
            match["image_url"] for match in result["top_matches"]
        }
        with database.connect() as connection:
            assert (
                connection.execute(
                    """SELECT COUNT(*) FROM finder_embedding_cache
                       WHERE source_key = ? AND include_mirror = 0""",
                    (service._remote_source_key(adjacent_preview),),
                ).fetchone()[0]
                == 0
            )

        async def unavailable_media(_: str, __: str) -> bytes:
            raise RuntimeError("temporary image failure")

        service.media_fetcher = unavailable_media
        with pytest.raises(finder_module.FinderUnavailable, match="could not load"):
            await service.set_review_ready(
                first_scan["id"],
                result["id"],
                "accepted",
                [adjacent_url],
            )
        unchanged, _ = service.results(
            first_scan["id"], review="maybe", min_score=0, limit=20, offset=0
        )
        assert [item["id"] for item in unchanged] == [result["id"]]
        assert service.feedback_status(tag_id)["accepted_samples"] == 0
        with pytest.raises(finder_module.FinderUnavailable, match="usable pose"):
            service.set_review(
                first_scan["id"],
                result["id"],
                "accepted",
                [adjacent_url],
                require_usable=True,
            )
        assert service.feedback_status(tag_id)["accepted_samples"] == 0

        service.media_fetcher = fake_media
        accepted = await service.set_review_ready(
            first_scan["id"],
            result["id"],
            "accepted",
            [adjacent_url],
        )
        assert accepted["feedback_image_urls"] == [adjacent_url]
        reloaded_adjacent, _ = service.results(
            first_scan["id"], review="accepted", min_score=0, limit=20, offset=0
        )
        assert reloaded_adjacent[0]["feedback_image_urls"] == [adjacent_url]
        adjacent_status = service.feedback_status(tag_id)
        assert adjacent_status["accepted_samples"] == 1
        assert adjacent_status["usable_accepted_samples"] == 1
        with database.connect() as connection:
            saved_pose = connection.execute(
                """SELECT pose_json FROM finder_feedback_samples
                   WHERE origin_result_id = ?""",
                (result["id"],),
            ).fetchone()[0]
        assert saved_pose

        # The user can curate up to three images, then remove one without the
        # deselected sample lingering in persisted feedback.
        all_three = [
            *(match["image_url"] for match in result["top_matches"]),
            adjacent_url,
        ]
        assert len(all_three) == 3
        selected_three = await service.set_review_ready(
            first_scan["id"],
            result["id"],
            "accepted",
            all_three,
        )
        assert set(selected_three["feedback_image_urls"]) == set(all_three)
        three_status = service.feedback_status(tag_id)
        assert three_status["accepted_samples"] == 3
        selected_two_urls = all_three[:2]
        removed_url = all_three[2]
        selected_two = await service.set_review_ready(
            first_scan["id"],
            result["id"],
            "accepted",
            selected_two_urls,
        )
        assert set(selected_two["feedback_image_urls"]) == set(selected_two_urls)
        two_status = service.feedback_status(tag_id)
        assert two_status["accepted_samples"] == 2
        assert two_status["revision"] == three_status["revision"] + 1
        two_reload, _ = service.results(
            first_scan["id"], review="accepted", min_score=0, limit=20, offset=0
        )
        assert set(two_reload[0]["feedback_image_urls"]) == set(selected_two_urls)
        assert removed_url not in two_reload[0]["feedback_image_urls"]
        repeated_two = await service.set_review_ready(
            first_scan["id"],
            result["id"],
            "accepted",
            selected_two_urls,
        )
        assert set(repeated_two["feedback_image_urls"]) == set(selected_two_urls)
        assert service.feedback_status(tag_id)["revision"] == two_status["revision"]

        # The curated subset stays editable after acceptance, and omitting it
        # on a later patch preserves the prior explicit selection.
        edited = service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
            [selected_url],
        )
        assert edited["feedback_image_urls"] == [selected_url]
        preserved = service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
        )
        assert preserved["feedback_image_urls"] == [selected_url]
        reloaded, _ = service.results(
            first_scan["id"], review="accepted", min_score=0, limit=20, offset=0
        )
        assert reloaded[0]["feedback_image_urls"] == [selected_url]

        # A refreshed full-size URL can legitimately retain the same preview
        # source key. Persist the new URL instead of treating it as idempotent
        # and reverting to the old snapshot on reload.
        changed_url = f"{selected_url}?revision=2"
        selected_source_key = service._remote_source_key(
            selected_match["preview_remote_url"]
        )
        with database.connect() as connection:
            changed_rows = connection.execute(
                """UPDATE finder_corpus_images SET image_url = ?
                   WHERE gallery_key = ? AND source_key = ?""",
                (changed_url, result["gallery_key"], selected_source_key),
            ).rowcount
        assert changed_rows == 1
        changed = service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
            [changed_url],
        )
        assert changed["feedback_image_urls"] == [changed_url]
        changed_reload, _ = service.results(
            first_scan["id"], review="accepted", min_score=0, limit=20, offset=0
        )
        assert changed_reload[0]["feedback_image_urls"] == [changed_url]
        with database.connect() as connection:
            connection.execute(
                """UPDATE finder_corpus_images SET image_url = ?
                   WHERE gallery_key = ? AND source_key = ?""",
                (selected_url, result["gallery_key"], selected_source_key),
            )
        restored_url = service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
            [selected_url],
        )
        assert restored_url["feedback_image_urls"] == [selected_url]
        status = service.feedback_status(tag_id)
        assert status["accepted_galleries"] == 1
        assert status["accepted_samples"] == 1
        assert status["usable_accepted_galleries"] == 1
        assert status["usable_accepted_samples"] == 1
        assert status["active"] is False
        first_revision = status["revision"]

        # Repeating an identical review is idempotent and does not create a
        # phantom new learning revision.
        service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
            [selected_url],
        )
        assert service.feedback_status(tag_id)["revision"] == first_revision
        with pytest.raises(ValueError, match="requires at least one"):
            service.set_review(first_scan["id"], result["id"], "accepted", [])
        with pytest.raises(ValueError, match="belong to"):
            service.set_review(
                first_scan["id"],
                result["id"],
                "rejected",
                ["https://cdni.pornpics.com/full/not-proposed.jpg"],
            )

        maybe = service.set_review(
            first_scan["id"],
            result["id"],
            "maybe",
            [],
        )
        assert maybe["review"] == "maybe"
        assert maybe["feedback_image_urls"] == []
        assert service.feedback_status(tag_id)["accepted_samples"] == 0
        maybe_results, maybe_total = service.results(
            first_scan["id"], review="maybe", min_score=0, limit=20, offset=0
        )
        assert maybe_total == 1
        assert [item["id"] for item in maybe_results] == [result["id"]]
        counts = service.result_review_counts(first_scan["id"], min_score=0)
        assert counts == {
            "pending": 0,
            "maybe": 1,
            "accepted": 0,
            "rejected": 1,
            "total": 2,
        }
        scan_counts = service.get_scan(first_scan["id"])
        assert scan_counts["maybe_count"] == 1
        assert scan_counts["review_counts"] == counts
        service.set_review(
            first_scan["id"],
            result["id"],
            "accepted",
            [selected_url],
        )
        first_revision = service.feedback_status(tag_id)["revision"]

        # A scan snapshots feedback when it is created. Later review changes
        # cannot alter that scan, including after a pause/resume or extension.
        frozen_scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        service.set_review(
            first_scan["id"],
            result["id"],
            "rejected",
            [selected_url],
        )
        with database.connect() as connection:
            frozen = connection.execute(
                """SELECT decision, source_key FROM finder_scan_feedback
                   WHERE scan_id = ?""",
                (frozen_scan["id"],),
            ).fetchall()
            frozen_revision = connection.execute(
                """SELECT feedback_revision FROM finder_scans WHERE id = ?""",
                (frozen_scan["id"],),
            ).fetchone()[0]
        assert [(row["decision"], row["source_key"]) for row in frozen] == [
            (
                "accepted",
                service._remote_source_key(
                    result["top_matches"][0]["preview_remote_url"]
                ),
            )
        ]
        assert frozen_revision == first_revision
        assert service.feedback_status(tag_id)["revision"] == first_revision + 1

        await asyncio.wait_for(service.queue.join(), 30)
        deleted = service.delete_or_cancel(first_scan["id"])
        assert deleted["deleted"] is True
        persisted = service.feedback_status(tag_id)
        assert persisted["rejected_galleries"] == 1
        assert persisted["rejected_samples"] == 1

        other_tag = database.create_pose_tag("Different pose", "solo")
        assert service.feedback_status(int(other_tag["id"]))["accepted_samples"] == 0
        reset = service.reset_feedback(tag_id)
        assert reset["accepted_galleries"] == 0
        assert reset["rejected_galleries"] == 0
        with database.connect() as connection:
            # Reset applies only to scans created later; frozen evidence remains.
            assert (
                connection.execute(
                    """SELECT COUNT(*) FROM finder_scan_feedback
                   WHERE scan_id = ?""",
                    (frozen_scan["id"],),
                ).fetchone()[0]
                == 1
            )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_feedback_review_is_authoritative_per_gallery_across_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=SpatialFakeEncoder(),
        pose_estimator=FakePoseEstimator(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        first_scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        first_results, _ = service.results(
            first_scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        first = next(item for item in first_results if item["gallery_url"] == GALLERY_A)
        first_url = first["top_matches"][0]["image_url"]
        await service.set_review_ready(
            first_scan["id"], first["id"], "accepted", [first_url]
        )
        first_revision = service.feedback_status(tag_id)["revision"]
        assert first_revision == 1

        second_scan = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 30)
        second_results, _ = service.results(
            second_scan["id"], review="all", min_score=0, limit=20, offset=0
        )
        second = next(
            item
            for item in second_results
            if item["gallery_key"] == first["gallery_key"]
        )
        replacement_url = next(
            match["image_url"]
            for match in second["top_matches"]
            if match["image_url"] != first_url
        )
        await service.set_review_ready(
            second_scan["id"],
            second["id"],
            "accepted",
            [replacement_url],
        )
        replacement_status = service.feedback_status(tag_id)
        assert replacement_status["revision"] == first_revision + 1
        assert replacement_status["accepted_galleries"] == 1
        assert replacement_status["accepted_samples"] == 1
        with database.connect() as connection:
            decisions = connection.execute(
                """SELECT origin_result_id, decision
                   FROM finder_feedback_decisions
                   WHERE pose_tag_id = ? AND gallery_key = ?""",
                (tag_id, first["gallery_key"]),
            ).fetchall()
            samples = connection.execute(
                """SELECT origin_result_id, image_url
                   FROM finder_feedback_samples"""
            ).fetchall()
        assert [tuple(row) for row in decisions] == [(second["id"], "accepted")]
        assert [tuple(row) for row in samples] == [(second["id"], replacement_url)]

        service.set_review(second_scan["id"], second["id"], "maybe", [])
        neutral_status = service.feedback_status(tag_id)
        assert neutral_status["revision"] == replacement_status["revision"] + 1
        assert neutral_status["accepted_samples"] == 0
        assert neutral_status["rejected_samples"] == 0
        with database.connect() as connection:
            assert (
                connection.execute(
                    """SELECT COUNT(*) FROM finder_feedback_decisions
                       WHERE pose_tag_id = ? AND gallery_key = ?""",
                    (tag_id, first["gallery_key"]),
                ).fetchone()[0]
                == 0
            )

        await service.set_review_ready(
            first_scan["id"], first["id"], "accepted", [first_url]
        )
        restored_status = service.feedback_status(tag_id)
        assert restored_status["revision"] == neutral_status["revision"] + 1
        assert restored_status["accepted_samples"] == 1

        rejected_without_sample = service.set_review(
            second_scan["id"],
            second["id"],
            "rejected",
            [],
        )
        assert rejected_without_sample["review"] == "rejected"
        assert rejected_without_sample["feedback_image_urls"] == []
        cleared_status = service.feedback_status(tag_id)
        assert cleared_status["revision"] == restored_status["revision"] + 1
        assert cleared_status["accepted_samples"] == 0
        assert cleared_status["rejected_samples"] == 0
        with database.connect() as connection:
            assert (
                connection.execute(
                    """SELECT COUNT(*) FROM finder_feedback_decisions
                       WHERE pose_tag_id = ? AND gallery_key = ?""",
                    (tag_id, first["gallery_key"]),
                ).fetchone()[0]
                == 0
            )
    finally:
        await service.stop()


def test_finder_feedback_pose_reranker_is_bounded_and_preserves_lanes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = FakePoseEstimator().infer_bytes(b"")
    candidate_metadata = {"pose": frame.as_dict()}
    pose = {"pose_reliable": True}
    accepted = finder_module._FeedbackProfile(
        revision=3,
        accepted=(("gallery-a", frame), ("gallery-b", frame)),
    )
    rejected = finder_module._FeedbackProfile(
        revision=4,
        rejected=(("gallery-c", frame), ("gallery-d", frame)),
    )

    def affinity(
        _: PoseFrame, exemplars: tuple[tuple[str, PoseFrame], ...]
    ) -> tuple[float, int]:
        return 0.95, len({gallery for gallery, _ in exemplars})

    monkeypatch.setattr(
        FinderService,
        "_feedback_pose_affinity",
        staticmethod(affinity),
    )
    positive_score, positive_adjustment = FinderService._feedback_adjusted_score(
        ranking_tier=2,
        base_score=0.6,
        pose=pose,
        candidate_metadata=candidate_metadata,
        feedback=accepted,
    )
    assert positive_score > 0.6
    assert 0 < positive_adjustment <= finder_module.MAX_FEEDBACK_ADJUSTMENT

    negative_score, negative_adjustment = FinderService._feedback_adjusted_score(
        ranking_tier=2,
        base_score=0.6,
        pose=pose,
        candidate_metadata=candidate_metadata,
        feedback=rejected,
    )
    assert finder_module.POSE_MATCH_FLOOR <= negative_score < 0.6
    assert -finder_module.MAX_FEEDBACK_ADJUSTMENT <= negative_adjustment < 0

    mismatch_score, _ = FinderService._feedback_adjusted_score(
        ranking_tier=0,
        base_score=0.54,
        pose=pose,
        candidate_metadata=candidate_metadata,
        feedback=accepted,
    )
    assert mismatch_score < finder_module.POSE_MATCH_FLOOR

    for tier in (1, 3):
        score, adjustment = FinderService._feedback_adjusted_score(
            ranking_tier=tier,
            base_score=0.8,
            pose=pose,
            candidate_metadata=candidate_metadata,
            feedback=rejected,
        )
        assert score == 0.8
        assert adjustment == 0

    couple = PoseFrame(
        keypoints=np.concatenate(
            [frame.keypoints, np.clip(frame.keypoints + 0.05, 0, 1)]
        ),
        confidences=np.concatenate([frame.confidences, frame.confidences]),
        boxes=np.concatenate([frame.boxes, np.clip(frame.boxes + 0.05, 0, 1)]),
        person_scores=np.concatenate([frame.person_scores, frame.person_scores]),
        image_size=frame.image_size,
        model_key=frame.model_key,
    )
    # Cardinality buckets are independent: couple/group feedback cannot add
    # expensive or misleading comparisons to a solo candidate.
    monkeypatch.undo()
    assert FinderService._feedback_pose_affinity(
        frame,
        (("couple-a", couple), ("couple-b", couple)),
    ) == (None, 0)


def test_finder_feedback_effective_samples_ignore_empty_and_dedupe_sources(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    estimator = FakePoseEstimator()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=SpatialFakeEncoder(),
        pose_estimator=estimator,
    )
    pose_json = json.dumps(estimator.infer_bytes(b"").as_dict())

    # Empty gallery decisions remain useful review history but must not consume
    # the learning cap ahead of older usable exemplars.
    decisions = [
        {
            "decision": "rejected",
            "gallery_key": f"empty-{index}",
            "samples": [],
        }
        for index in range(finder_module.MAX_FEEDBACK_GALLERIES_PER_STATE + 5)
    ]
    decisions.extend(
        [
            {
                "decision": "rejected",
                "gallery_key": "usable-a",
                "samples": [
                    {
                        "source_key": "url:https://cdn.example/a.jpg",
                        "pose_model_key": estimator.model_key,
                        "pose_json": pose_json,
                    }
                ],
            },
            {
                "decision": "rejected",
                "gallery_key": "usable-b",
                "samples": [
                    {
                        "source_key": "url:https://cdn.example/b.jpg",
                        "pose_model_key": estimator.model_key,
                        "pose_json": pose_json,
                    }
                ],
            },
        ]
    )
    selected = service._usable_feedback_samples(decisions)
    assert [(state, gallery) for state, gallery, _, _ in selected] == [
        ("rejected", "usable-a"),
        ("rejected", "usable-b"),
    ]

    # The newest explicit label owns a source globally, so duplicate image URLs
    # cannot masquerade as multiple independent galleries or opposing votes.
    duplicate = {
        "source_key": "url:https://cdn.example/duplicate.jpg",
        "pose_model_key": estimator.model_key,
        "pose_json": pose_json,
    }
    selected = service._usable_feedback_samples(
        [
            {
                "decision": "accepted",
                "gallery_key": "newest",
                "samples": [duplicate],
            },
            {
                "decision": "rejected",
                "gallery_key": "older",
                "samples": [duplicate],
            },
        ]
    )
    assert [(state, gallery) for state, gallery, _, _ in selected] == [
        ("accepted", "newest")
    ]

    many_decisions = [
        {
            "decision": "accepted",
            "gallery_key": "gallery-00",
            "samples": [
                {
                    "source_key": f"url:https://cdn.example/00-{index}.jpg",
                    "pose_model_key": estimator.model_key,
                    "pose_json": pose_json,
                }
                for index in range(3)
            ],
        }
    ]
    many_decisions.extend(
        {
            "decision": "accepted",
            "gallery_key": f"gallery-{index:02d}",
            "samples": [
                {
                    "source_key": f"url:https://cdn.example/{index:02d}.jpg",
                    "pose_model_key": estimator.model_key,
                    "pose_json": pose_json,
                }
            ],
        }
        for index in range(1, 12)
    )
    selected = service._usable_feedback_samples(many_decisions)
    assert len(selected) == finder_module.MAX_FEEDBACK_SAMPLES_PER_STATE
    assert {gallery for _, gallery, _, _ in selected} == {
        f"gallery-{index:02d}"
        for index in range(finder_module.MAX_FEEDBACK_SAMPLES_PER_STATE)
    }


def test_finder_feedback_upgrade_backfills_reviews_without_mutating_cache(
    tmp_path: Path,
) -> None:
    config, database, tag_id = configured(tmp_path)
    estimator = FakePoseEstimator()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=SpatialFakeEncoder(),
        pose_estimator=estimator,
    )
    service.ensure_schema()
    service._model_key = "feedback-migration-model"
    scan = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    preview = "https://cdni.pornpics.com/p/migration-feedback.jpg"
    image_url = "https://cdni.pornpics.com/full/migration-feedback.jpg"
    pose = estimator.infer_bytes(b"")
    service._store_embedding(
        service._remote_source_key(preview),
        False,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        metadata={
            "analyzer_version": finder_module.ANALYZER_VERSION,
            "descriptor_kind": "spatial",
            "pose": pose.as_dict(),
            "person_count": 1,
        },
    )
    service._save_result(
        scan["id"],
        {"url": GALLERY_A, "title": "Historic accepted gallery"},
        order=0,
        score=0,
        images_scored=1,
        best=None,
        status="completed",
        ranking_version=finder_module.CURRENT_RANKING_VERSION,
        top_matches=[
            pose_first_match(
                1,
                appearance=0.4,
                pose=0.9,
                reliable=True,
            )
            | {
                "image_url": image_url,
                "preview_remote_url": preview,
            }
        ],
    )
    with database.connect() as connection:
        connection.execute(
            """UPDATE finder_scans
               SET reference_model_key = ?, status = 'completed'
               WHERE id = ?""",
            (service._model_key, scan["id"]),
        )
        connection.execute(
            """UPDATE finder_results SET review = 'accepted'
               WHERE scan_id = ?""",
            (scan["id"],),
        )
        connection.execute("""DELETE FROM finder_feedback_decisions""")
        connection.execute(
            """DELETE FROM finder_corpus_meta
               WHERE key = 'feedback-backfill'"""
        )
        cache_before = connection.execute(
            """SELECT cache_key, embedding, metadata_json, created_at,
                      last_used_at
               FROM finder_embedding_cache ORDER BY cache_key"""
        ).fetchall()

    service.ensure_schema()
    status = service.feedback_status(tag_id)
    assert status["accepted_galleries"] == 1
    assert status["accepted_samples"] == 1
    assert status["usable_accepted_samples"] == 1
    first_revision = status["revision"]
    with database.connect() as connection:
        cache_after = connection.execute(
            """SELECT cache_key, embedding, metadata_json, created_at,
                      last_used_at
               FROM finder_embedding_cache ORDER BY cache_key"""
        ).fetchall()
        marker = connection.execute(
            """SELECT value FROM finder_corpus_meta
               WHERE key = 'feedback-backfill'"""
        ).fetchone()[0]
    assert [tuple(row) for row in cache_after] == [tuple(row) for row in cache_before]
    assert marker == finder_module.FEEDBACK_BACKFILL_VERSION

    # The marker makes startup migration idempotent.
    service.ensure_schema()
    repeated = service.feedback_status(tag_id)
    assert repeated["revision"] == first_revision
    assert repeated["accepted_samples"] == 1
