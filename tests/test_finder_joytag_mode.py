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
from app.finder import FinderService


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


async def fake_media(url: str, _: str) -> bytes:
    for color in ("red", "blue", "green", "black"):
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
async def test_joytag_local_corpus_backfills_once_and_reuses_vector_for_new_tag(
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
        assert sorted(url.rsplit("/", 1)[-1] for url in fetched) == [
            "blue.png",
            "green.png",
        ]

        calls_after_backfill = list(joytag.batch_calls)
        fetched_after_backfill = list(fetched)
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
        assert fetched == fetched_after_backfill
        assert joytag.batch_calls == calls_after_backfill
    finally:
        await service.stop()


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
