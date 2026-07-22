"""Manual browser screenshot helper; not collected by pytest."""

from __future__ import annotations

import subprocess
import shutil
import argparse
import json
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from fastapi.responses import Response
from PIL import Image

from app.config import AppConfig
from app.main import create_app
from app.security import encode_gallery_id, gallery_key


def build_visual_app(temp_root: Path, *, load_more: bool = False):
    config = AppConfig(
        data_dir=temp_root / "data",
        download_root=temp_root / "downloads",
        sqlite_vfs=None,
    )
    app = create_app(config)
    db = app.state.db
    config.ensure_directories()
    db.initialize()
    target_dir = config.sort_root_path / "visual-sort/targets"
    control_dir = config.sort_root_path / "visual-sort/references"
    target_dir.mkdir(parents=True)
    control_dir.mkdir(parents=True)
    Image.new("RGB", (900, 1200), "#453a76").save(target_dir / "sample_target.jpg")
    Image.new("RGB", (900, 1200), "#7d4c72").save(control_dir / "sample_reference.jpg")
    app.state.sorter.ensure_schema()
    sort_session = app.state.sorter.start_session(
        {
            "target_directory": "visual-sort/targets",
            "control_directories": ["visual-sort/references"],
            "mode": "time",
            "threshold_seconds": 50,
            "add_ids": True,
        }
    )
    sample_image = "https://cdni.pornpics.com/460/7/343/79186222/79186222_001_d325.jpg"
    galleries = []
    for index in range(12):
        remote_id = 79186222 + index
        url = f"https://www.pornpics.com/galleries/sample-gallery-{remote_id}/"
        galleries.append(
            {
                "id": encode_gallery_id(url),
                "key": gallery_key(url),
                "url": url,
                "title": [
                    "Midnight studio portrait collection",
                    "Natural light editorial series",
                    "After-hours city gallery",
                    "Soft focus summer collection",
                    "Classic monochrome session",
                    "Warm sunset portrait set",
                ][index % 6],
                "thumbnail_remote_url": sample_image,
                "image_count": 20 + index,
            }
        )
    db.add_history(
        galleries[0]["url"], "Default", galleries[0]["title"], "Default/sample", 20
    )
    db.register_gallery_images(
        galleries[1]["url"],
        [
            {"url": f"https://cdni.pornpics.com/1280/demo/{n}.jpg", "ordinal": n}
            for n in range(1, 11)
        ],
    )
    db.add_profile_image(
        "Default",
        galleries[1]["url"],
        "https://cdni.pornpics.com/1280/demo/1.jpg",
        "Default/demo/0001.jpg",
        1234,
    )

    async def fake_browse(**kwargs: object) -> dict:
        page = int(kwargs.get("page", 1))
        source = str(kwargs.get("url") or "")
        if "offset=20" in source:
            page = 2
        start = 6 if page > 1 else 0
        return {
            "items": [dict(item) for item in galleries[start : start + 6]],
            "source_url": "https://www.pornpics.com/",
            "next_url": (
                "https://www.pornpics.com/?offset=20&limit=20" if page == 1 else None
            ),
            "previous_url": "https://www.pornpics.com/" if page > 1 else None,
        }

    app.state.scraper.browse = fake_browse

    async def fake_media(request=None, url: str = "", token: str = "") -> Response:
        svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1100'><defs><linearGradient id='g' x2='1' y2='1'><stop stop-color='#31295a'/><stop offset='1' stop-color='#121825'/></linearGradient></defs><rect width='800' height='1100' fill='url(#g)'/><circle cx='570' cy='310' r='190' fill='#9b7bfa' opacity='.18'/><path d='M90 860 330 540l150 170 105-125 140 275Z' fill='#ffffff' opacity='.13'/></svg>"""
        return Response(svg, media_type="image/svg+xml")

    async def fake_events(request=None) -> Response:
        return Response(status_code=204)

    async def fake_bootstrap() -> Response:
        session_id = json.dumps(sort_session["id"])
        script = f"localStorage.setItem('galleryflow:sort-session', JSON.stringify({session_id}));"
        if load_more:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('#page-next');if(button&&!button.hidden&&!button.disabled){button.click();clearInterval(poll)}},50)});"
        return Response(script, media_type="application/javascript")

    async def fake_index() -> Response:
        markup = config.static_dir.joinpath("index.html").read_text(encoding="utf-8")
        markup = markup.replace(
            '<script src="/static/app.js" defer></script>',
            '<script src="/manual-bootstrap.js"></script><script src="/static/app.js" defer></script>',
        )
        return Response(markup, media_type="text/html")

    app.add_api_route("/manual-bootstrap.js", fake_bootstrap, methods=["GET"])

    for route in app.routes:
        if getattr(route, "path", None) == "/":
            route.endpoint = fake_index
            route.dependant.call = fake_index
        elif getattr(route, "path", None) == "/api/media":
            route.endpoint = fake_media
            route.dependant.call = fake_media
        elif getattr(route, "path", None) == "/api/events":
            route.endpoint = fake_events
            route.dependant.call = fake_events
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--load-more", action="store_true")
    args = parser.parse_args()
    suffix = (
        "load-more"
        if args.load_more
        else "sort-mobile"
        if args.sort and args.mobile
        else "sort"
        if args.sort
        else "mobile"
        if args.mobile
        else "smoke"
    )
    output = Path(f"/tmp/pornpic-webui-{suffix}.png")
    viewport = "390,844" if args.mobile else "1440,1100"
    with tempfile.TemporaryDirectory(prefix="pornpic-webui-") as directory:
        server = uvicorn.Server(
            uvicorn.Config(
                build_visual_app(Path(directory), load_more=args.load_more),
                host="127.0.0.1",
                port=18101,
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:18101/api/health", timeout=0.2)
                break
            except Exception:
                time.sleep(0.05)
        browser = shutil.which("google-chrome-stable")
        if not browser:
            raise RuntimeError(
                "google-chrome-stable is required for this manual smoke check"
            )
        subprocess.run(
            [
                browser,
                "--headless=new",
                "--no-sandbox",
                "--disable-gpu",
                "--run-all-compositor-stages-before-draw",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-sync",
                "--no-first-run",
                "--virtual-time-budget=1000",
                f"--user-data-dir={Path(directory) / 'chrome-profile'}",
                f"--window-size={viewport}",
                f"--screenshot={output}",
                f"http://127.0.0.1:18101/{'#sort' if args.sort else ''}",
            ],
            check=True,
            timeout=45,
        )
        server.should_exit = True
        thread.join(timeout=5)
    print(output)


if __name__ == "__main__":
    main()
