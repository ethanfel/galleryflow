from __future__ import annotations

import json

import pytest

from app.config import AppConfig
from app.scraper import PornPicsScraper, ScrapedPage


@pytest.mark.asyncio
async def test_json_search_preserves_order_and_builds_pagination(monkeypatch) -> None:
    payload = [
        {
            "gid": str(10000000 + index),
            "g_url": f"https://www.pornpics.com/galleries/title-{10000000 + index}/",
            "t_url_460": f"https://cdni.pornpics.com/460/a/{index}.jpg",
            "desc": f"Gallery {index}",
        }
        for index in range(20)
    ]
    scraper = PornPicsScraper(AppConfig())

    async def fake_get(url: str) -> ScrapedPage:
        return ScrapedPage(url, json.dumps(payload))

    monkeypatch.setattr(scraper, "_get_html", fake_get)
    result = await scraper.browse(query="test query", page=1)
    assert len(result["items"]) == 20
    assert result["items"][0]["title"] == "Gallery 0"
    assert result["items"][-1]["key"] == "pornpics:gallery:10000019"
    assert "offset=20" in result["next_url"]


@pytest.mark.asyncio
async def test_current_gallery_markup_extracts_only_ordered_cdni_images(
    monkeypatch,
) -> None:
    html = """
    <html><head><title>Fallback - PornPics.com</title></head><body>
      <div class="title-section"><h1>Current Gallery</h1></div>
      <ul id="tiles">
        <li class="thumbwook"><a class="rel-link" href="https://cdni.pornpics.com/1280/a/first.jpg"><img data-src="https://cdni.pornpics.com/460/a/first.jpg"></a></li>
        <li class="thumbwook"><a class="rel-link" href="https://ads.example/bad.jpg"><img src="https://ads.example/ad.jpg"></a></li>
        <li class="thumbwook"><a class="rel-link" href="https://cdni.pornpics.com/1280/a/second.webp"><img data-src="https://cdni.pornpics.com/460/a/second.webp"></a></li>
      </ul>
      <img src="https://cdni.pornpics.com/1280/unrelated.jpg">
    </body></html>
    """
    url = "https://www.pornpics.com/galleries/current-gallery-79186222/"
    scraper = PornPicsScraper(AppConfig())

    async def fake_get(_: str) -> ScrapedPage:
        return ScrapedPage(url, html)

    monkeypatch.setattr(scraper, "_get_html", fake_get)
    detail = await scraper.gallery(url)
    assert detail["title"] == "Current Gallery"
    assert [item["filename"] for item in detail["images"]] == [
        "first.jpg",
        "second.webp",
    ]
    assert [item["ordinal"] for item in detail["images"]] == [1, 2]

    browse_result = await scraper.browse(url=url)
    assert len(browse_result["items"]) == 1
    assert browse_result["items"][0]["key"] == "pornpics:gallery:79186222"
    assert browse_result["items"][0]["image_count"] == 2
    assert browse_result["next_url"] is None
