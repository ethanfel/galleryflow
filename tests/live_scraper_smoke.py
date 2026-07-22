"""Opt-in live source check; not collected by pytest."""

from __future__ import annotations

import asyncio
import argparse
import tempfile
from pathlib import Path

from app.config import AppConfig
from app.db import Database
from app.downloader import DownloadManager, EventBroker
from app.scraper import PornPicsScraper


async def check(download: bool = False) -> None:
    scraper = PornPicsScraper(AppConfig(request_timeout=20))
    listing = await scraper.browse(query="pov", page=1)
    if not listing["items"]:
        raise RuntimeError("Live search returned no galleries")
    first = listing["items"][0]
    gallery = await scraper.gallery(first["url"])
    if not gallery["images"]:
        raise RuntimeError("Live gallery returned no images")
    direct_gallery = await scraper.browse(url=first["url"])
    if (
        len(direct_gallery["items"]) != 1
        or direct_gallery["items"][0]["key"] != gallery["key"]
    ):
        raise RuntimeError("Pasted gallery URL did not resolve to one gallery card")
    result = {
        "search_items": len(listing["items"]),
        "direct_gallery_items": len(direct_gallery["items"]),
        "gallery_key": gallery["key"],
        "gallery_images": len(gallery["images"]),
        "first_media_host": gallery["images"][0]["url"].split("/", 3)[2],
    }
    if download:
        with tempfile.TemporaryDirectory(prefix="pornpic-live-") as directory:
            root = Path(directory)
            config = AppConfig(
                data_dir=root / "data",
                download_root=root / "downloads",
                sqlite_vfs=None,
            )
            config.ensure_directories()
            database = Database(config.db_path, config.sqlite_vfs)
            database.initialize()
            manager = DownloadManager(config, database, scraper, EventBroker())
            await manager.start()
            try:
                destination = config.download_root / "smoke"
                destination.mkdir()
                path = await manager._download_image(
                    gallery["images"][0]["url"], destination, 1, referer=gallery["url"]
                )
                result["downloaded_bytes"] = path.stat().st_size
            finally:
                await manager.stop()
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    asyncio.run(check(args.download))
