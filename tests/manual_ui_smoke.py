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


def build_visual_app(
    temp_root: Path,
    *,
    load_more: bool = False,
    open_gallery: bool = False,
    open_lightbox: bool = False,
    open_pose: bool = False,
    open_finder: bool = False,
    finder_exhausted: bool = False,
):
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

    async def fake_gallery(url: str) -> dict:
        gallery = next((item for item in galleries if item["url"] == url), galleries[0])
        return {
            **gallery,
            "images": [
                {
                    "url": f"https://cdni.pornpics.com/1280/manual/{index:03d}.jpg",
                    "preview_remote_url": f"https://cdni.pornpics.com/460/manual/{index:03d}.jpg",
                    "filename": f"manual-{index:03d}.jpg",
                    "ordinal": index,
                }
                for index in range(1, 22)
            ],
        }

    app.state.scraper.gallery = fake_gallery

    async def fake_media(request=None, url: str = "", token: str = "") -> Response:
        if "overlay" in url:
            svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1100'><g fill='none' stroke='#63f2bd' stroke-width='18' stroke-linecap='round' stroke-linejoin='round'><circle cx='400' cy='190' r='58'/><path d='m400 250-20 245m20-180-150 145m150-145 145 110M380 495 245 760m135-265 190 245'/></g><g fill='#ffcf67' stroke='#101017' stroke-width='7'><circle cx='400' cy='250' r='17'/><circle cx='400' cy='315' r='17'/><circle cx='250' cy='460' r='17'/><circle cx='545' cy='425' r='17'/><circle cx='380' cy='495' r='17'/><circle cx='245' cy='760' r='17'/><circle cx='570' cy='740' r='17'/></g></svg>"""
        else:
            color = "#284b63" if "candidate-2" in url else "#57406e" if "candidate-3" in url else "#31295a"
            svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1100'><defs><linearGradient id='g' x2='1' y2='1'><stop stop-color='{color}'/><stop offset='1' stop-color='#121825'/></linearGradient></defs><rect width='800' height='1100' fill='url(#g)'/><circle cx='570' cy='310' r='190' fill='#9b7bfa' opacity='.18'/><path d='M90 860 330 540l150 170 105-125 140 275Z' fill='#ffffff' opacity='.13'/></svg>""".encode()
        return Response(svg, media_type="image/svg+xml")

    finder_scan = {
        "id": "visual-finder",
        "status": "completed",
        "example_directory": "sorted_outpaint/mating press - backview/selected_target_upscaled",
        "pose_tag_id": 1,
        "pose_tag_label": "mating press - backview",
        "source_url": "https://www.pornpics.com/",
        "next_url": None
        if finder_exhausted
        else "https://www.pornpics.com/?page=6",
        "page_limit": 5,
        "pages_completed": 3 if finder_exhausted else 5,
        "processed_galleries": 64,
        "processed_images": 1280,
        "corpus_search_complete": True,
        "corpus_galleries_scored": 418,
        "corpus_images_scored": 8156,
        "candidate_count": 2,
        "minimum_score": 0.65,
        "ranking_version": "pose-first-v1",
        "ranking_current": True,
        "progress_percent": 100,
    }

    async def fake_finder_status(**kwargs: object) -> dict:
        return {"available": True, "model_ready": True, "model_name": "RTMO-L + visual verifier", "device": "CUDA", "folder_root": "/library"}

    async def fake_finder_corpus(**kwargs: object) -> dict:
        return {
            "galleries": 418,
            "images": 8420,
            "complete": 374,
            "partial": 44,
            "ready": 8156,
            "cache_entries": 10820,
            "cache_bytes": 367001600,
            "max_cache_entries": 50000,
            "max_cache_bytes": 2147483648,
        }

    async def fake_finder_folders(**kwargs: object) -> dict:
        return {"folders": [{"path": "sorted_outpaint/mating press - backview/selected_target_upscaled", "image_count": 25}]}

    async def fake_pose_tags(**kwargs: object) -> dict:
        return {"items": [{"id": 1, "label": "mating press - backview", "slug": "mating-press-backview", "default_role": "couple"}]}

    async def fake_finder_scans(**kwargs: object) -> dict:
        return {"scans": [finder_scan]}

    async def fake_finder_scan(**kwargs: object) -> dict:
        return finder_scan

    async def fake_finder_results(**kwargs: object) -> dict:
        def media(name: str) -> str:
            return f"/api/media?url=https%3A%2F%2Fexample.test%2F{name}.jpg&token=visual"

        return {
            "results": [
                {
                    "id": "visual-result-1",
                    "gallery_id": galleries[2]["id"],
                    "gallery_url": galleries[2]["url"],
                    "title": "High-confidence multi-person pose candidate",
                    "rank": 1,
                    "score": 0.96,
                    "ranking_tier": 2,
                    "online_scanned": False,
                    "review": "pending",
                    "images_scored": 24,
                    "image_count": 24,
                    "person_count": 2,
                    "score_breakdown": {"exact": 0.31, "pose": 0.96, "appearance": 0.72},
                    "top_matches": [
                        {"rank": 1, "image_url": "https://example.test/candidate-1.jpg", "preview_url": media("candidate-1"), "ordinal": 12, "score": 0.96, "ranking_tier": 2, "pose_score": 0.96, "pose_reliable": True, "match_type": "pose", "person_count": 2, "skeleton_overlay_url": media("overlay-1")},
                        {"rank": 2, "image_url": "https://example.test/candidate-2.jpg", "preview_url": media("candidate-2"), "ordinal": 8, "score": 0.88, "ranking_tier": 2, "pose_score": 0.88, "pose_reliable": True, "match_type": "pose", "person_count": 2, "skeleton_overlay_url": media("overlay-2")},
                        {"rank": 3, "image_url": "https://example.test/candidate-3.jpg", "preview_url": media("candidate-3"), "ordinal": 19, "score": 0.84, "ranking_tier": 1, "appearance_score": 0.84, "match_type": "visual_fallback", "person_count": 2},
                    ],
                },
                {
                    "id": "visual-result-2",
                    "gallery_id": galleries[3]["id"],
                    "gallery_url": galleries[3]["url"],
                    "title": "Exact source image found in gallery",
                    "rank": 2,
                    "score": 1.0,
                    "ranking_tier": 3,
                    "online_scanned": True,
                    "review": "pending",
                    "images_scored": 21,
                    "image_count": 21,
                    "is_exact": True,
                    "exact_score": 1.0,
                    "best_image_url": "https://example.test/candidate-2.jpg",
                    "best_preview_url": media("candidate-2"),
                    "best_ordinal": 7,
                },
            ]
        }

    async def fake_events(request=None) -> Response:
        return Response(status_code=204)

    async def fake_bootstrap() -> Response:
        session_id = json.dumps(sort_session["id"])
        script = f"localStorage.setItem('galleryflow:sort-session', JSON.stringify({session_id}));"
        if load_more:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('#page-next');if(button&&!button.hidden&&!button.disabled){button.click();clearInterval(poll)}},50)});"
        if open_gallery:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.gallery-open');if(button){button.click();clearInterval(poll)}},50)});"
        if open_lightbox:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.image-preview-button');if(button){button.click();clearInterval(poll)}},50)});"
        if open_pose:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const modal=document.querySelector('#gallery-modal');const button=document.querySelector('[data-gallery-mode=pose]');const image=document.querySelector('.image-option:not(.skeleton-image)');if(modal?.open&&button&&image){button.click();clearInterval(poll)}},50)});"
        if open_finder:
            script += "localStorage.setItem('galleryflow:finder-scan', JSON.stringify('visual-finder'));window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.finder-overlay-toggle:not([hidden])');if(button){button.click();clearInterval(poll)}},50)});"
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
        elif open_finder and getattr(route, "path", None) == "/api/finder/status":
            route.endpoint = fake_finder_status
            route.dependant.call = fake_finder_status
        elif open_finder and getattr(route, "path", None) == "/api/finder/corpus":
            route.endpoint = fake_finder_corpus
            route.dependant.call = fake_finder_corpus
        elif open_finder and getattr(route, "path", None) == "/api/finder/folders":
            route.endpoint = fake_finder_folders
            route.dependant.call = fake_finder_folders
        elif open_finder and getattr(route, "path", None) == "/api/pose-tags":
            route.endpoint = fake_pose_tags
            route.dependant.call = fake_pose_tags
        elif open_finder and getattr(route, "path", None) == "/api/finder/scans":
            route.endpoint = fake_finder_scans
            route.dependant.call = fake_finder_scans
        elif open_finder and getattr(route, "path", None) == "/api/finder/scans/{scan_id}":
            route.endpoint = fake_finder_scan
            route.dependant.call = fake_finder_scan
        elif open_finder and getattr(route, "path", None) == "/api/finder/scans/{scan_id}/results":
            route.endpoint = fake_finder_results
            route.dependant.call = fake_finder_results
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--load-more", action="store_true")
    parser.add_argument("--gallery", action="store_true")
    parser.add_argument("--lightbox", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--finder", action="store_true")
    parser.add_argument("--finder-exhausted", action="store_true")
    args = parser.parse_args()
    finder_mode = args.finder or args.finder_exhausted
    suffix = (
        "finder-exhausted-mobile"
        if args.finder_exhausted and args.mobile
        else "finder-exhausted"
        if args.finder_exhausted
        else "finder-mobile"
        if finder_mode and args.mobile
        else "finder"
        if finder_mode
        else "pose-mobile"
        if args.pose and args.mobile
        else "pose"
        if args.pose
        else "lightbox-mobile"
        if args.lightbox and args.mobile
        else "lightbox"
        if args.lightbox
        else "gallery"
        if args.gallery
        else "load-more"
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
    viewport = (
        "390,1800"
        if args.mobile and finder_mode
        else "390,844"
        if args.mobile
        else "1920,969"
        if args.gallery or args.lightbox or args.pose
        else "1440,1100"
    )
    with tempfile.TemporaryDirectory(prefix="pornpic-webui-") as directory:
        server = uvicorn.Server(
            uvicorn.Config(
                build_visual_app(
                    Path(directory),
                    load_more=args.load_more,
                    open_gallery=args.gallery or args.lightbox or args.pose,
                    open_lightbox=args.lightbox,
                    open_pose=args.pose,
                    open_finder=finder_mode,
                    finder_exhausted=args.finder_exhausted,
                ),
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
                "--force-prefers-reduced-motion",
                "--no-first-run",
                f"--virtual-time-budget={3000 if args.lightbox or args.pose else 1000}",
                f"--user-data-dir={Path(directory) / 'chrome-profile'}",
                f"--window-size={viewport}",
                f"--screenshot={output}",
                f"http://127.0.0.1:18101/{'#finder' if finder_mode else '#sort' if args.sort else ''}",
            ],
            check=True,
            timeout=45,
        )
        server.should_exit = True
        thread.join(timeout=5)
    print(output)


if __name__ == "__main__":
    main()
