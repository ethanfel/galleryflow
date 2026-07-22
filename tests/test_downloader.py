from __future__ import annotations

import asyncio
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
