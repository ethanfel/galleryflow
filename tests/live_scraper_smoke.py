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
    category = await scraper.browse(url="https://www.pornpics.com/hardcore/", page=1)
    if not category["items"]:
        raise RuntimeError("Live category returned no galleries")
    if not category["next_url"] or "offset=20" not in category["next_url"]:
        raise RuntimeError("Live category did not expose its JSON scroll cursor")
    category_page_two = await scraper.browse(url=category["next_url"], page=2)
    if not category_page_two["items"]:
        raise RuntimeError("Live category page two returned no galleries")
    category_first_keys = {item["key"] for item in category["items"]}
    category_second_keys = {item["key"] for item in category_page_two["items"]}
    if not category_second_keys - category_first_keys:
        raise RuntimeError("Live category page two returned no new galleries")
    if not category_page_two["next_url"] or "offset=40" not in category_page_two[
        "next_url"
    ]:
        raise RuntimeError("Live category page two did not advance its JSON cursor")
    category_page_three = await scraper.browse(
        url=category_page_two["next_url"], page=3
    )
    if not category_page_three["items"]:
        raise RuntimeError("Live category page three returned no galleries")
    previous_category_keys = category_first_keys | category_second_keys
    category_third_keys = {item["key"] for item in category_page_three["items"]}
    if not category_third_keys - previous_category_keys:
        raise RuntimeError("Live category page three returned no new galleries")

    listing = await scraper.browse(query="pov", page=1)
    if not listing["items"]:
        raise RuntimeError("Live search returned no galleries")
    if not listing["next_url"]:
        raise RuntimeError("Live search did not expose another page")
    second_page = await scraper.browse(url=listing["next_url"], page=2)
    if not second_page["items"]:
        raise RuntimeError("Live search page two returned no galleries")
    first_keys = {item["key"] for item in listing["items"]}
    if first_keys.intersection(item["key"] for item in second_page["items"]):
        raise RuntimeError("Live search pagination repeated page-one galleries")
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
        "category_items": len(category["items"]),
        "category_second_page_items": len(category_page_two["items"]),
        "category_third_page_items": len(category_page_three["items"]),
        "search_items": len(listing["items"]),
        "second_page_items": len(second_page["items"]),
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
