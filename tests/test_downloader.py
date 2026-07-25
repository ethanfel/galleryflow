from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from app.config import AppConfig
from app.db import Database
from app.downloader import ActiveDownloadError, DownloadManager, EventBroker
from app.scraper import ScrapeError


GALLERY = "https://www.pornpics.com/galleries/sample-79186222/"


class FakeScraper:
    def __init__(self) -> None:
        self.images = [
            {
                "id": str(index),
                "ordinal": index,
                "url": f"https://cdni.pornpics.com/1280/a/{index}.jpg",
                "filename": f"{index}.jpg",
                "preview_remote_url": f"https://cdni.pornpics.com/460/a/{index}.jpg",
            }
            for index in range(1, 4)
        ]

    async def gallery(self, _: str) -> dict:
        return {
            "id": "gallery",
            "key": "pornpics:gallery:79186222",
            "url": GALLERY,
            "title": "Sample",
            "images": self.images,
        }


@pytest.mark.asyncio
async def test_selective_then_full_download_tracks_partial_and_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    db.create_profile("POV", "POV")
    scraper = FakeScraper()
    manager = DownloadManager(config, db, scraper, EventBroker())
    downloaded_positions: list[int] = []
    event_loop_thread = threading.get_ident()
    status_threads: list[int] = []
    result_threads: list[int] = []
    summary_reads = 0
    original_statuses = db.image_statuses
    original_record_result = db.record_job_item_result
    original_get_summary = db.get_job_summary

    def tracked_statuses(profile: str, gallery_url: str) -> set[str]:
        status_threads.append(threading.get_ident())
        return original_statuses(profile, gallery_url)

    def tracked_record_result(*args, **kwargs):
        result_threads.append(threading.get_ident())
        return original_record_result(*args, **kwargs)

    def tracked_get_summary(job_id: str):
        nonlocal summary_reads
        summary_reads += 1
        return original_get_summary(job_id)

    monkeypatch.setattr(db, "image_statuses", tracked_statuses)
    monkeypatch.setattr(db, "record_job_item_result", tracked_record_result)
    monkeypatch.setattr(db, "get_job_summary", tracked_get_summary)
    monkeypatch.setattr(
        db,
        "job_cancel_requested",
        lambda *_: pytest.fail("downloads should use the manager cancel event"),
    )

    async def fake_download(
        url: str, destination: Path, position: int, *, referer: str
    ) -> Path:
        downloaded_positions.append(position)
        path = destination / f"{position:04d}.jpg"
        path.write_bytes(b"fake-image-content")
        return path

    manager._download_image = fake_download  # type: ignore[method-assign]
    await manager.start()
    try:
        manager.enqueue(
            gallery_url=GALLERY,
            profile="POV",
            image_urls=[scraper.images[0]["url"], scraper.images[1]["url"]],
        )
        await asyncio.wait_for(manager.queue.join(), 5)
        partial = db.status_for_urls([GALLERY], "POV")[GALLERY]
        assert partial["state"] == "partial"
        assert partial["downloaded_images"] == 2
        assert db.list_history("POV") == []

        manager.enqueue(gallery_url=GALLERY, profile="POV")
        await asyncio.wait_for(manager.queue.join(), 5)
        complete = db.status_for_urls([GALLERY], "POV")[GALLERY]
        assert complete["state"] == "complete"
        assert complete["downloaded_images"] == 3
        assert sorted(downloaded_positions) == [1, 2, 3]
        assert len(status_threads) == 2
        assert len(result_threads) == 5
        assert summary_reads == 2
        assert all(thread_id != event_loop_thread for thread_id in status_threads)
        assert all(thread_id != event_loop_thread for thread_id in result_threads)
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_cancel_queued_job_is_terminal(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    manager = DownloadManager(config, db, FakeScraper(), EventBroker())
    job = manager.enqueue(gallery_url=GALLERY, profile="Default")
    with pytest.raises(ActiveDownloadError) as duplicate:
        manager.enqueue(gallery_url=GALLERY, profile="Default")
    assert duplicate.value.job["id"] == job["id"]
    canceled = manager.cancel(job["id"])
    assert canceled and canceled["status"] == "canceled"


@pytest.mark.asyncio
async def test_cancel_during_failed_start_stays_canceled(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
        job_workers=1,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    started = asyncio.Event()
    release = asyncio.Event()

    class FailingScraper:
        async def gallery(self, _: str) -> dict:
            started.set()
            await release.wait()
            raise ScrapeError("source changed")

    manager = DownloadManager(config, db, FailingScraper(), EventBroker())  # type: ignore[arg-type]
    await manager.start()
    try:
        job = manager.enqueue(gallery_url=GALLERY, profile="Default")
        await asyncio.wait_for(started.wait(), 2)
        assert manager.cancel(job["id"])["status"] == "canceling"
        release.set()
        await asyncio.wait_for(manager.queue.join(), 2)
        assert db.get_job(job["id"])["status"] == "canceled"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_failed_gallery_keeps_the_image_error(tmp_path: Path) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
        image_workers=1,
        job_workers=1,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    manager = DownloadManager(config, db, FakeScraper(), EventBroker())

    async def fail_download(*args, **kwargs) -> Path:
        raise RuntimeError("source timed out")

    manager._download_image = fail_download  # type: ignore[method-assign]
    await manager.start()
    try:
        job = manager.enqueue(gallery_url=GALLERY, profile="Default")
        await asyncio.wait_for(manager.queue.join(), 2)
        finished = db.get_job_summary(job["id"])
        assert finished and finished["status"] == "failed"
        assert finished["failed"] == 3
        assert "source timed out" in finished["error"]
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_bounded_gallery_workers_do_not_starve_another_gallery(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
        image_workers=1,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    manager = DownloadManager(config, db, FakeScraper(), EventBroker())
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def process(_: int, gallery: str) -> bool:
        async with manager._image_semaphore:
            order.append(gallery)
            if len(order) == 1:
                first_started.set()
                await release_first.wait()
            await asyncio.sleep(0)
        return True

    first = asyncio.create_task(
        manager._run_bounded_image_work(
            [(index, "A") for index in range(1, 6)],
            process,
        )
    )
    await asyncio.wait_for(first_started.wait(), 1)
    second = asyncio.create_task(manager._run_bounded_image_work([(1, "B")], process))
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.wait_for(asyncio.gather(first, second), 1)

    assert order[:2] == ["A", "B"]


@pytest.mark.asyncio
async def test_bounded_gallery_workers_cancel_siblings_after_an_error(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
        image_workers=2,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    manager = DownloadManager(config, db, FakeScraper(), EventBroker())
    sibling_started = asyncio.Event()
    sibling_canceled = asyncio.Event()

    async def process(_: int, item: str) -> bool:
        if item == "failure":
            await sibling_started.wait()
            raise RuntimeError("database commit failed")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_canceled.set()
            raise

    with pytest.raises(RuntimeError, match="database commit failed"):
        await asyncio.wait_for(
            manager._run_bounded_image_work(
                [(1, "failure"), (2, "sibling")],
                process,
            ),
            1,
        )
    assert sibling_canceled.is_set()


@pytest.mark.asyncio
async def test_cancel_event_prevents_post_transfer_database_commit(
    tmp_path: Path,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
        image_workers=1,
        job_workers=1,
    )
    config.ensure_directories()
    db = Database(config.db_path)
    db.initialize()
    manager = DownloadManager(config, db, FakeScraper(), EventBroker())
    transfer_started = asyncio.Event()
    release_transfer = asyncio.Event()

    async def delayed_download(
        url: str, destination: Path, position: int, *, referer: str
    ) -> Path:
        path = destination / f"{position:04d}.jpg"
        path.write_bytes(b"downloaded-but-not-committed")
        transfer_started.set()
        await release_transfer.wait()
        return path

    manager._download_image = delayed_download  # type: ignore[method-assign]
    await manager.start()
    try:
        job = manager.enqueue(gallery_url=GALLERY, profile="Default")
        await asyncio.wait_for(transfer_started.wait(), 2)
        canceled = manager.cancel(job["id"])
        assert canceled and canceled["status"] == "canceling"
        release_transfer.set()
        await asyncio.wait_for(manager.queue.join(), 2)

        finished = db.get_job_summary(job["id"])
        assert finished and finished["status"] == "canceled"
        assert finished["completed"] == 0
        assert db.image_statuses("Default", GALLERY) == set()
    finally:
        await manager.stop()
