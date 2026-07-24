from __future__ import annotations

import asyncio
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app.finder as finder_module
from app.config import AppConfig
from app.db import Database
from app.downloader import EventBroker
from app.finder import FinderConflict, FinderService


ROOT = "https://www.pornpics.com/"
GALLERY = "https://www.pornpics.com/galleries/tag-match-79186222/"
OTHER_GALLERY = "https://www.pornpics.com/galleries/tag-miss-79186223/"


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 18), color).save(output, format="PNG")
    return output.getvalue()


class FakeJoyTag:
    model_key = "fake-joytag-v1"
    tags = ("target_tag", "other_tag")

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.batch_calls: list[int] = []

    async def prepare(self) -> Path:
        self.prepare_calls += 1
        return Path("fake-joytag.onnx")

    def provider_status(self) -> dict[str, object]:
        return {"requested": "cpu", "active": "CPUExecutionProvider"}

    def classify_many_bytes(self, payloads: list[bytes]) -> np.ndarray:
        self.batch_calls.append(len(payloads))
        results = []
        for data in payloads:
            with Image.open(io.BytesIO(data)) as image:
                red, green, blue = image.convert("RGB").getpixel((0, 0))
            if blue > red and blue > green:
                results.append((0.9, 0.1))
            elif green > red and green > blue:
                results.append((0.6, 0.8))
            elif red > green and red > blue:
                results.append((0.85, 0.2))
            else:
                results.append((0.1, 0.1))
        return np.asarray(results, dtype=np.float32)


class MultiTagJoyTag(FakeJoyTag):
    tags = ("target_tag", "pose_tag", "blocked_tag")

    def classify_many_bytes(self, payloads: list[bytes]) -> np.ndarray:
        self.batch_calls.append(len(payloads))
        results = []
        for data in payloads:
            with Image.open(io.BytesIO(data)) as image:
                red, green, blue = image.convert("RGB").getpixel((0, 0))
            if red > 200 and green > 200:
                # Has the primary tag but misses the second required tag.
                results.append((0.95, 0.60, 0.10))
            elif blue > red and blue > green:
                # Qualifies and remains just below the exclusion threshold.
                results.append((0.90, 0.80, 0.39))
            elif green > red and green > blue:
                # Both positives qualify, but equality at the exclusion
                # threshold must veto this image.
                results.append((0.92, 0.82, 0.40))
            elif red > green and red > blue:
                # Reference image.
                results.append((0.95, 0.85, 0.10))
            else:
                results.append((0.10, 0.10, 0.10))
        return np.asarray(results, dtype=np.float32)


class FakeEncoder:
    model_key = "unused-encoder"

    async def prepare(self) -> Path:
        return Path("unused.onnx")


class FakeScraper:
    async def browse(self, **_: object) -> dict:
        return {
            "items": [
                {
                    "url": GALLERY,
                    "title": "One qualifying image",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/a.png",
                },
                {
                    "url": OTHER_GALLERY,
                    "title": "No qualifying image",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/b.png",
                },
            ],
            "next_url": None,
        }

    async def gallery(self, url: str) -> dict:
        colors = ("blue", "black") if url == GALLERY else ("green",)
        return {
            "url": url,
            "images": [
                {
                    "url": f"https://cdni.pornpics.com/full/{color}.png",
                    "preview_remote_url": (
                        f"https://cdni.pornpics.com/preview/{color}.png"
                    ),
                    "ordinal": index,
                }
                for index, color in enumerate(colors, 1)
            ],
        }


class EmptyScraper(FakeScraper):
    async def browse(self, **_: object) -> dict:
        return {"items": [], "next_url": None}

    async def gallery(self, url: str) -> dict:
        raise AssertionError(f"Local corpus search should not open {url}")


class PagingScraper(FakeScraper):
    def __init__(self) -> None:
        self.gallery_calls: list[str] = []

    async def browse(self, **values: object) -> dict:
        page = int(values.get("page") or 1)
        if page == 1:
            return {
                "items": [
                    {
                        "url": GALLERY,
                        "title": "First page",
                        "thumbnail_remote_url": (
                            "https://cdni.pornpics.com/t/a.png"
                        ),
                    }
                ],
                "next_url": "https://www.pornpics.com/page/2/",
            }
        return {
            "items": [
                {
                    "url": OTHER_GALLERY,
                    "title": "Extended page",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/b.png",
                }
            ],
            "next_url": None,
        }

    async def gallery(self, url: str) -> dict:
        self.gallery_calls.append(url)
        return await super().gallery(url)


class MultiTagScraper(FakeScraper):
    async def gallery(self, url: str) -> dict:
        colors = ("blue", "green") if url == GALLERY else ("yellow",)
        return {
            "url": url,
            "images": [
                {
                    "url": f"https://cdni.pornpics.com/full/{color}.png",
                    "preview_remote_url": (
                        f"https://cdni.pornpics.com/preview/{color}.png"
                    ),
                    "ordinal": index,
                }
                for index, color in enumerate(colors, 1)
            ],
        }


async def fake_media(url: str, _: str) -> bytes:
    for color in ("red", "blue", "green", "yellow", "black"):
        if f"/{color}.png" in url:
            return image_bytes(color)
    raise AssertionError(f"Unexpected image URL: {url}")


@pytest.fixture(autouse=True)
def inline_finder_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    async def inline(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(finder_module.asyncio, "to_thread", inline)
    monkeypatch.setattr(
        finder_module, "validate_public_media_url", lambda value: value
    )


def configured(tmp_path: Path) -> tuple[AppConfig, Database, int]:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        finder_examples_root=tmp_path / "references",
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        finder_pose_model_path=tmp_path / "models" / "rtmo.onnx",
        finder_joytag_model_path=tmp_path / "models" / "joytag" / "model.onnx",
        finder_joytag_tags_path=tmp_path / "models" / "joytag" / "tags.txt",
        finder_request_delay=0,
        finder_network_workers=1,
        finder_inference_batch_size=8,
        sqlite_vfs=None,
    )
    config.ensure_directories()
    examples = config.finder_examples_root / "one-reference"
    examples.mkdir()
    (examples / "example.png").write_bytes(image_bytes("red"))
    database = Database(config.db_path)
    database.initialize()
    pose_tag = database.create_pose_tag("Target tag", "solo")
    return config, database, int(pose_tag["id"])


def seed_corpus_gallery(
    database: Database,
    gallery_url: str,
    previews: list[str],
    *,
    title: str = "Corpus gallery",
) -> None:
    now = "2026-01-01T00:00:00+00:00"
    key = finder_module.gallery_key(gallery_url)
    with database.connect() as db:
        db.execute(
            """INSERT INTO finder_corpus_galleries(
                   gallery_key, gallery_url, title, thumbnail_remote_url,
                   state, image_count, created_at, updated_at
               ) VALUES (?, ?, ?, '', 'complete', ?, ?, ?)""",
            (key, gallery_url, title, len(previews), now, now),
        )
        db.executemany(
            """INSERT INTO finder_corpus_images(
                   gallery_key, source_key, image_url, preview_remote_url,
                   ordinal, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    key,
                    FinderService._remote_source_key(preview),
                    preview.replace("/preview/", "/full/"),
                    preview,
                    ordinal,
                    now,
                    now,
                )
                for ordinal, preview in enumerate(previews, 1)
            ],
        )


@pytest.mark.asyncio
async def test_joytag_mode_analyzes_one_reference_filters_before_top_three_and_reuses(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    joytag = FakeJoyTag()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=joytag,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        assert analysis["image_count"] == 1
        assert analysis["provider"] == "CPUExecutionProvider"
        assert analysis["bytes_per_cached_image"] == 2
        assert analysis["images"][0]["scores"]["target_tag"] == 0.851

        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        completed = service.get_scan(scan["id"])
        assert completed
        assert completed["status"] == "completed"
        assert completed["search_mode"] == "joytag"
        assert completed["reference_count"] == 1

        results, total = service.results(
            scan["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )
        assert total == 1
        assert len(results[0]["top_matches"]) == 1
        match = results[0]["top_matches"][0]
        assert match["tag"] == "target_tag"
        assert match["match_type"] == "tag"
        assert match["image_url"].endswith("/blue.png")

        # The public result filter cannot be lowered below the confidence
        # frozen into a Tag scan. Internal completed rows for galleries with no
        # retained match must never become visible as empty candidates.
        lowered_results, lowered_total = service.results(
            scan["id"],
            review="pending",
            min_score=0,
            limit=20,
            offset=0,
        )
        assert lowered_total == 1
        assert len(lowered_results) == 1
        assert service.result_review_counts(scan["id"], min_score=0)["total"] == 1

        calls_after_first_scan = list(joytag.batch_calls)
        second = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert joytag.batch_calls == calls_after_first_scan
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_local_corpus_is_cache_only_and_explicit_index_reuses_vectors(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    now = "2026-01-01T00:00:00+00:00"
    corpus = (
        (
            GALLERY,
            "Cached blue",
            "blue",
        ),
        (
            OTHER_GALLERY,
            "Cached green",
            "green",
        ),
    )
    fetched: list[str] = []

    async def counting_media(url: str, referer: str) -> bytes:
        fetched.append(url)
        return await fake_media(url, referer)

    joytag = FakeJoyTag()
    service = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=joytag,
        media_fetcher=counting_media,
    )
    await service.start()
    try:
        with database.connect() as db:
            for gallery_url, title, color in corpus:
                key = finder_module.gallery_key(gallery_url)
                preview = f"https://cdni.pornpics.com/preview/{color}.png"
                db.execute(
                    """INSERT INTO finder_corpus_galleries(
                           gallery_key, gallery_url, title, thumbnail_remote_url,
                           state, image_count, created_at, updated_at
                       ) VALUES (?, ?, ?, '', 'complete', 1, ?, ?)""",
                    (key, gallery_url, title, now, now),
                )
                db.execute(
                    """INSERT INTO finder_corpus_images(
                           gallery_key, source_key, image_url, preview_remote_url,
                           ordinal, created_at, updated_at
                       ) VALUES (?, ?, ?, ?, 1, ?, ?)""",
                    (
                        key,
                        FinderService._remote_source_key(preview),
                        f"https://cdni.pornpics.com/full/{color}.png",
                        preview,
                        now,
                        now,
                    ),
                )
        analysis = await service.analyze_reference_directory("one-reference")
        calls_after_analysis = list(joytag.batch_calls)
        uncached = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        assert service.get_scan(uncached["id"])["status"] == "completed"
        assert service.results(
            uncached["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )[1] == 0
        assert fetched == []
        assert joytag.batch_calls == calls_after_analysis

        started = service.create_joytag_index_job()
        duplicate = service.create_joytag_index_job()
        assert duplicate["job"]["id"] == started["job"]["id"]
        await asyncio.wait_for(service.joytag_index_queue.join(), 10)
        indexed = service.joytag_index_status()
        assert indexed["job"]["status"] == "completed"
        assert indexed["job"]["total_images"] == 2
        assert indexed["job"]["processed_images"] == 2
        assert indexed["job"]["indexed_images"] == 2
        assert indexed["job"]["failed_images"] == 0
        assert indexed["coverage"]["cached_images"] == 2
        assert indexed["coverage"]["missing_images"] == 0
        assert sorted(url.rsplit("/", 1)[-1] for url in fetched) == [
            "blue.png",
            "green.png",
        ]
        assert joytag.batch_calls[len(calls_after_analysis) :] == [2]

        calls_after_index = list(joytag.batch_calls)
        fetched_after_index = list(fetched)
        first = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        first_results = service.results(
            first["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )[0]
        assert [item["gallery_url"] for item in first_results] == [GALLERY]

        second = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.15,
            mode="joytag",
            joytag_tag="other_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        second_results = service.results(
            second["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )[0]
        assert [item["gallery_url"] for item in second_results] == [OTHER_GALLERY]
        assert fetched == fetched_after_index
        assert joytag.batch_calls == calls_after_index
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_corpus_index_deduplicates_shared_source_keys(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    fetched: list[str] = []

    async def counting_media(url: str, referer: str) -> bytes:
        fetched.append(url)
        return await fake_media(url, referer)

    joytag = FakeJoyTag()
    service = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=joytag,
        media_fetcher=counting_media,
    )
    await service.start()
    try:
        shared = "https://cdni.pornpics.com/preview/blue.png"
        seed_corpus_gallery(database, GALLERY, [shared], title="First association")
        seed_corpus_gallery(
            database,
            OTHER_GALLERY,
            [shared],
            title="Second association",
        )

        before = service.joytag_index_status()["coverage"]
        assert before["total_images"] == 1
        assert before["cached_images"] == 0
        assert before["missing_images"] == 1

        service.create_joytag_index_job()
        await asyncio.wait_for(service.joytag_index_queue.join(), 10)
        final = service.joytag_index_status()
        assert final["job"]["status"] == "completed"
        assert final["job"]["total_images"] == 1
        assert final["job"]["processed_images"] == 1
        assert final["job"]["indexed_images"] == 1
        assert final["coverage"]["total_images"] == 1
        assert final["coverage"]["cached_images"] == 1
        assert final["coverage"]["missing_images"] == 0
        assert fetched == [shared]
        assert joytag.batch_calls == [1]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_corpus_index_cancellation_is_persistent(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_media(url: str, referer: str) -> bytes:
        started.set()
        await release.wait()
        return await fake_media(url, referer)

    service = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=blocking_media,
    )
    await service.start()
    try:
        seed_corpus_gallery(
            database,
            GALLERY,
            [
                "https://cdni.pornpics.com/preview/blue.png",
                "https://cdni.pornpics.com/preview/green.png",
            ],
        )
        created = service.create_joytag_index_job()
        await asyncio.wait_for(started.wait(), 5)
        canceling = service.cancel_joytag_index_job()
        assert canceling["job"]["id"] == created["job"]["id"]
        assert canceling["job"]["status"] == "canceling"
        release.set()
        await asyncio.wait_for(service.joytag_index_queue.join(), 10)
        final = service.joytag_index_status()["job"]
        assert final["status"] == "canceled"
        assert final["cancel_requested"] is True
        assert final["finished_at"]
        with pytest.raises(FinderConflict, match="No JoyTag"):
            service.cancel_joytag_index_job()
    finally:
        release.set()
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_corpus_index_cancel_wins_terminal_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, database, _ = configured(tmp_path)
    service = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        seed_corpus_gallery(
            database,
            GALLERY,
            ["https://cdni.pornpics.com/preview/blue.png"],
        )
        original_finalize = service._finalize_joytag_index_job
        canceled = False

        def cancel_before_finalize(job_id: str, **values: object):
            nonlocal canceled
            if not canceled and values.get("status") == "completed":
                canceled = True
                service.cancel_joytag_index_job()
            return original_finalize(job_id, **values)

        monkeypatch.setattr(
            service,
            "_finalize_joytag_index_job",
            cancel_before_finalize,
        )
        service.create_joytag_index_job()
        await asyncio.wait_for(service.joytag_index_queue.join(), 10)
        final = service.joytag_index_status()["job"]
        assert canceled is True
        assert final["status"] == "canceled"
        assert final["cancel_requested"] is True
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_corpus_index_bounds_failures(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)

    async def failing_media(url: str, _: str) -> bytes:
        raise RuntimeError(f"cannot fetch {url}")

    service = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=failing_media,
    )
    await service.start()
    try:
        seed_corpus_gallery(
            database,
            GALLERY,
            [
                f"https://cdni.pornpics.com/preview/failure-{index}.png"
                for index in range(25)
            ],
        )
        service.create_joytag_index_job()
        await asyncio.wait_for(service.joytag_index_queue.join(), 10)
        final = service.joytag_index_status()
        job = final["job"]
        assert job["status"] == "completed_with_errors"
        assert job["processed_images"] == 25
        assert job["indexed_images"] == 0
        assert job["failed_images"] == 25
        assert job["progress"] == 100.0
        assert len(job["errors"]) == finder_module.MAX_JOYTAG_INDEX_ERRORS
        assert final["coverage"]["missing_images"] == 25
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_corpus_index_restart_reconciles_committed_vectors(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    initial = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    initial.ensure_schema()
    blue = "https://cdni.pornpics.com/preview/blue.png"
    green = "https://cdni.pornpics.com/preview/green.png"
    seed_corpus_gallery(database, GALLERY, [blue, green])
    now = "2026-01-01T00:00:00+00:00"
    with database.connect() as db:
        # This vector committed just before the old process stopped. The job's
        # cursor/counters did not commit with it.
        db.execute(
            """INSERT INTO finder_joytag_cache(
                   model_key, source_key, dimensions, scores,
                   created_at, last_used_at
               ) VALUES ('fake-joytag-v1', ?, 2, ?, ?, ?)""",
            (
                FinderService._remote_source_key(blue),
                bytes((230, 26)),
                now,
                now,
            ),
        )
        db.execute(
            """INSERT INTO finder_joytag_index_jobs(
                   id, model_key, dimensions, status, total_images,
                   cached_images_at_start, created_at, updated_at
               ) VALUES (
                   'restart-window', 'fake-joytag-v1', 2, 'running', 2,
                   0, ?, ?
               )""",
            (now, now),
        )

    fetched: list[str] = []

    async def counting_media(url: str, referer: str) -> bytes:
        fetched.append(url)
        return await fake_media(url, referer)

    restarted = FinderService(
        config,
        database,
        EmptyScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=counting_media,
    )
    await restarted.start()
    try:
        await asyncio.wait_for(restarted.joytag_index_queue.join(), 10)
        final = restarted.joytag_index_status()
        assert final["job"]["status"] == "completed"
        assert final["job"]["processed_images"] == 2
        assert final["job"]["indexed_images"] == 2
        assert final["coverage"]["cached_images"] == 2
        assert [url.rsplit("/", 1)[-1] for url in fetched] == ["green.png"]
    finally:
        await restarted.stop()


@pytest.mark.asyncio
async def test_joytag_extend_uses_saved_cursor_without_rescanning_first_page(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    scraper = PagingScraper()
    service = FinderService(
        config,
        database,
        scraper,
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.55,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        first = service.get_scan(scan["id"])
        assert first["status"] == "completed"
        assert first["pages_completed"] == 1
        assert first["extendable"] is True

        service.extend(scan["id"], additional_pages=1)
        await asyncio.wait_for(service.queue.join(), 10)
        extended = service.get_scan(scan["id"])
        assert extended["status"] == "completed"
        assert extended["pages_completed"] == 2
        assert scraper.gallery_calls == [GALLERY, OTHER_GALLERY]
        assert service.results(
            scan["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )[1] == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_review_selection_is_persistent_and_not_pose_feedback(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        result = service.results(
            scan["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )[0][0]
        manual_url = "https://cdni.pornpics.com/full/black.png"
        saved = await service.set_review_ready(
            scan["id"],
            result["id"],
            "accepted",
            [manual_url],
        )
        assert saved["feedback_image_urls"] == [manual_url]
        feedback = service.feedback_status(pose_tag_id)
        assert feedback["accepted_samples"] == 0
        assert feedback["rejected_samples"] == 0

        reloaded = service.results(
            scan["id"],
            review="accepted",
            min_score=None,
            limit=20,
            offset=0,
        )[0][0]
        assert reloaded["feedback_image_urls"] == [manual_url]
        assert reloaded["feedback_pending_image_urls"] == []

        # Saving the same review without a replacement list preserves the
        # durable manual choice, even if a later corpus refresh no longer has
        # that historical image association.
        with database.connect() as db:
            db.execute(
                "DELETE FROM finder_corpus_images WHERE image_url = ?",
                (manual_url,),
            )
        preserved = await service.set_review_ready(
            scan["id"],
            result["id"],
            "accepted",
            None,
        )
        assert preserved["feedback_image_urls"] == [manual_url]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_scan_rejects_stale_reference_analysis(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        (
            config.finder_examples_root / "one-reference" / "example.png"
        ).write_bytes(image_bytes("blue"))
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        failed = service.get_scan(scan["id"])
        assert failed["status"] == "failed"
        assert "analyze the folder again" in failed["error"].lower()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_scan_rechecks_reference_manifest_after_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        original = service._joytag_scores_many

        async def mutate_after_scoring(*args, **kwargs):
            outcomes = await original(*args, **kwargs)
            if kwargs.get("scan_id"):
                (
                    config.finder_examples_root
                    / "one-reference"
                    / "added.png"
                ).write_bytes(image_bytes("blue"))
            return outcomes

        monkeypatch.setattr(service, "_joytag_scores_many", mutate_after_scoring)
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_tag="target_tag",
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        failed = service.get_scan(scan["id"])
        assert failed["status"] == "failed"
        assert "changed while preparing" in failed["error"].lower()
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_multitag_and_negative_veto_share_cached_vectors(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    joytag = MultiTagJoyTag()
    service = FinderService(
        config,
        database,
        MultiTagScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=joytag,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        assert analysis["tag_catalog"] == [
            "target_tag",
            "pose_tag",
            "blocked_tag",
        ]
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_required_tags=["target_tag", "pose_tag"],
            joytag_excluded_tags=["blocked_tag"],
            joytag_reject_threshold=0.4,
            reference_fingerprint=analysis["fingerprint"],
        )
        assert scan["joytag_tag"] == "target_tag"
        assert scan["joytag_required_tags"] == ["target_tag", "pose_tag"]
        assert scan["joytag_excluded_tags"] == ["blocked_tag"]
        assert scan["joytag_reject_threshold"] == pytest.approx(0.4)

        await asyncio.wait_for(service.queue.join(), 10)
        completed = service.get_scan(scan["id"])
        assert completed["joytag_required_tag_indices"] == [0, 1]
        assert completed["joytag_excluded_tag_indices"] == [2]
        results, total = service.results(
            scan["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )
        assert total == 1
        assert results[0]["gallery_url"] == GALLERY
        # Blue passes both positives and the exclusion. Green has stronger
        # positive scores but is vetoed at blocked_tag == 0.40, and the result
        # is not padded with either that image or the yellow positive-only miss.
        assert len(results[0]["top_matches"]) == 1
        match = results[0]["top_matches"][0]
        assert match["image_url"].endswith("/blue.png")
        assert match["score"] == pytest.approx(0.8)
        assert match["tag"] == "target_tag"
        assert match["tag_score"] == pytest.approx(0.8)
        assert match["required_score"] == pytest.approx(0.8)
        assert match["tag_scores"] == pytest.approx(
            {
                "target_tag": 230 / 255,
                "pose_tag": 204 / 255,
                "blocked_tag": 99 / 255,
            }
        )
        assert match["required_tag_scores"] == pytest.approx(
            {"target_tag": 230 / 255, "pose_tag": 204 / 255}
        )
        assert match["excluded_tag_scores"] == pytest.approx(
            {"blocked_tag": 99 / 255}
        )
        assert match["max_excluded_score"] == pytest.approx(99 / 255)

        calls_after_first = list(joytag.batch_calls)
        # A different logical query uses the same complete cached vectors.
        second = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_required_tags=["target_tag"],
            joytag_excluded_tags=["blocked_tag"],
            joytag_reject_threshold=0.4,
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert joytag.batch_calls == calls_after_first

        missing_reference = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_required_tags=["target_tag", "blocked_tag"],
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        failed = service.get_scan(missing_reference["id"])
        assert failed["status"] == "failed"
        assert "blocked_tag" in failed["error"]
        assert joytag.batch_calls == calls_after_first

        with database.connect() as db:
            cached = db.execute(
                """SELECT COUNT(*) AS count, MIN(dimensions) AS minimum,
                          MAX(dimensions) AS maximum
                   FROM finder_joytag_cache"""
            ).fetchone()
        assert cached["count"] == 4
        assert cached["minimum"] == cached["maximum"] == 3

        with pytest.raises(ValueError, match="unique"):
            service.create_scan(
                example_directory="one-reference",
                pose_tag_id=pose_tag_id,
                source_url=ROOT,
                page_limit=1,
                minimum_score=0.7,
                mode="joytag",
                joytag_required_tags=["target_tag", "target_tag"],
                reference_fingerprint=analysis["fingerprint"],
            )
        with pytest.raises(ValueError, match="disjoint"):
            service.create_scan(
                example_directory="one-reference",
                pose_tag_id=pose_tag_id,
                source_url=ROOT,
                page_limit=1,
                minimum_score=0.7,
                mode="joytag",
                joytag_required_tags=["target_tag"],
                joytag_excluded_tags=["target_tag"],
                reference_fingerprint=analysis["fingerprint"],
            )
        with pytest.raises(ValueError, match="at most 16"):
            service.create_scan(
                example_directory="one-reference",
                pose_tag_id=pose_tag_id,
                source_url=ROOT,
                page_limit=1,
                minimum_score=0.7,
                mode="joytag",
                joytag_required_tags=[f"tag-{index}" for index in range(17)],
                reference_fingerprint=analysis["fingerprint"],
            )
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_joytag_multitag_query_survives_pause_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        MultiTagScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=MultiTagJoyTag(),
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        analysis = await service.analyze_reference_directory("one-reference")
        original_corpus_search = service._search_joytag_corpus
        paused_once = False

        async def pause_after_references(scan, query):
            nonlocal paused_once
            if not paused_once:
                paused_once = True
                service.pause(scan["id"])
            await original_corpus_search(scan, query)

        monkeypatch.setattr(
            service,
            "_search_joytag_corpus",
            pause_after_references,
        )
        scan = service.create_scan(
            example_directory="one-reference",
            pose_tag_id=pose_tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0.7,
            mode="joytag",
            joytag_required_tags=["target_tag", "pose_tag"],
            joytag_excluded_tags=["blocked_tag"],
            joytag_reject_threshold=0.4,
            reference_fingerprint=analysis["fingerprint"],
        )
        await asyncio.wait_for(service.queue.join(), 10)
        paused = service.get_scan(scan["id"])
        assert paused["status"] == "paused"
        assert paused["reference_ready"] is True
        assert paused["joytag_required_tags"] == ["target_tag", "pose_tag"]
        assert paused["joytag_required_tag_indices"] == [0, 1]
        assert paused["joytag_excluded_tags"] == ["blocked_tag"]
        assert paused["joytag_excluded_tag_indices"] == [2]

        monkeypatch.setattr(
            service,
            "_search_joytag_corpus",
            original_corpus_search,
        )
        service.resume(scan["id"])
        await asyncio.wait_for(service.queue.join(), 10)
        completed = service.get_scan(scan["id"])
        assert completed["status"] == "completed"
        assert completed["joytag_required_tags"] == ["target_tag", "pose_tag"]
        assert completed["joytag_excluded_tags"] == ["blocked_tag"]
        results, total = service.results(
            scan["id"],
            review="pending",
            min_score=None,
            limit=20,
            offset=0,
        )
        assert total == 1
        assert [
            match["image_url"].rsplit("/", 1)[-1]
            for match in results[0]["top_matches"]
        ] == ["blue.png"]
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_v212_joytag_scan_and_match_decode_and_extend(
    tmp_path: Path,
) -> None:
    config, database, pose_tag_id = configured(tmp_path)
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        joytagger=FakeJoyTag(),
        media_fetcher=fake_media,
    )
    service.ensure_schema()
    now = "2026-01-01T00:00:00+00:00"
    scan_id = "v212-single-tag"
    with database.connect() as db:
        db.execute(
            """INSERT INTO finder_scans(
                   id, example_directory, reference_fingerprint,
                   reference_model_key, reference_ready, reference_count,
                   pose_tag_id, pose_tag_label, pose_tag_slug, pose_default_role,
                   source_url, next_url, page_limit, pages_completed, minimum_score,
                   ranking_version, search_mode, joytag_tag, joytag_tag_index,
                   status, created_at, updated_at, finished_at
               ) VALUES (?, 'one-reference', ?, 'fake-joytag-v1', 1, 1,
                         ?, 'Target tag', 'target-tag', 'solo', ?, ?, 1, 1, 0.7,
                         'joytag-v1', 'joytag', 'target_tag', 0,
                         'completed', ?, ?, ?)""",
            (scan_id, "a" * 64, pose_tag_id, ROOT, ROOT, now, now, now),
        )
        db.execute(
            """INSERT INTO finder_results(
                   id, scan_id, gallery_key, gallery_url, title,
                   best_image_url, best_preview_remote_url, best_ordinal,
                   score, ranking_tier, matches_json, images_scored, status,
                   discovered_order, created_at, updated_at
               ) VALUES (
                   'v212-result', ?, 'v212-gallery', ?, 'Legacy tag result',
                   'https://cdni.pornpics.com/full/blue.png',
                   'https://cdni.pornpics.com/preview/blue.png', 1, 0.8, 1,
                   ?, 1, 'completed', 1, ?, ?
               )""",
            (
                scan_id,
                GALLERY,
                (
                    '[{"image_url":"https://cdni.pornpics.com/full/blue.png",'
                    '"preview_remote_url":'
                    '"https://cdni.pornpics.com/preview/blue.png",'
                    '"ordinal":1,"score":0.8,"tag":"target_tag",'
                    '"tag_score":0.8,"ranking_tier":1}]'
                ),
                now,
                now,
            ),
        )

    scan = service.get_scan(scan_id)
    assert scan["joytag_tag"] == "target_tag"
    assert scan["joytag_required_tags"] == ["target_tag"]
    assert scan["joytag_excluded_tags"] == []
    assert scan["joytag_reject_threshold"] == pytest.approx(0.4)
    assert scan["joytag_required_tag_indices"] == []

    results, total = service.results(
        scan_id,
        review="pending",
        min_score=None,
        limit=20,
        offset=0,
    )
    assert total == 1
    match = results[0]["top_matches"][0]
    assert match["tag"] == "target_tag"
    assert match["tag_score"] == pytest.approx(0.8)
    assert match["tag_scores"] == pytest.approx({"target_tag": 0.8})
    assert match["required_tag_scores"] == pytest.approx({"target_tag": 0.8})

    await service.start()
    try:
        service.extend(scan_id, additional_pages=1)
        await asyncio.wait_for(service.queue.join(), 10)
        extended = service.get_scan(scan_id)
        assert extended["status"] == "completed"
        assert extended["pages_completed"] == 2
        assert extended["joytag_required_tags"] == ["target_tag"]
        assert extended["joytag_required_tag_indices"] == [0]
        assert extended["joytag_excluded_tag_indices"] == []
    finally:
        await service.stop()
