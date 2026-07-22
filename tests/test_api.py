from __future__ import annotations

import asyncio
from pathlib import Path

import anyio.to_thread
import httpx
import pytest
from PIL import Image

from app.config import AppConfig
from app.main import create_app
from app.security import encode_gallery_id


GALLERY = "https://www.pornpics.com/galleries/sample-79186222/"


@pytest.mark.asyncio
async def test_api_profile_browse_ignore_and_settings(
    tmp_path: Path, monkeypatch
) -> None:
    # This test runner's syscall sandbox cannot start worker threads. Execute
    # thread-offloaded route work inline here; production still uses workers.
    async def inline_asyncio(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    async def inline_anyio(
        func, *args, abandon_on_cancel=False, cancellable=None, limiter=None
    ):
        return func(*args)

    monkeypatch.setattr(asyncio, "to_thread", inline_asyncio)
    monkeypatch.setattr(anyio.to_thread, "run_sync", inline_anyio)
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    sort_target = config.download_root / "sort-target/target.jpg"
    sort_control = config.download_root / "sort-control/control.jpg"
    sort_target.parent.mkdir(parents=True, exist_ok=True)
    sort_control.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "purple").save(sort_target)
    Image.new("RGB", (8, 8), "pink").save(sort_control)
    app = create_app(config)
    legacy_download = next(
        route for route in app.routes if getattr(route, "path", "") == "/download"
    )
    assert legacy_download.status_code == 200

    async def fake_browse(**_: object) -> dict:
        return {
            "items": [
                {
                    "id": encode_gallery_id(GALLERY),
                    "key": "pornpics:gallery:79186222",
                    "url": GALLERY,
                    "title": "Sample",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/460/a.jpg",
                    "image_count": 3,
                }
            ],
            "source_url": "https://www.pornpics.com/",
            "next_url": None,
            "previous_url": None,
        }

    monkeypatch.setattr(app.state.scraper, "browse", fake_browse)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            assert (await client.get("/api/health")).json()["status"] == "ok"
            assert (
                await client.post("/api/profiles", json={"name": "POV"})
            ).status_code == 201
            response = await client.get("/api/galleries", params={"profile": "POV"})
            card = response.json()["items"][0]
            assert card["saved"] is False and card["ignored"] is False
            assert card["thumbnail_url"].startswith("/api/media?")

            gallery_id = encode_gallery_id(GALLERY)
            assert (
                await client.patch(
                    f"/api/galleries/{gallery_id}",
                    json={"ignored": True, "title": "Sample"},
                )
            ).status_code == 200
            card = (
                await client.get("/api/galleries", params={"profile": "POV"})
            ).json()["items"][0]
            assert card["ignored"] is True

            settings = await client.patch(
                "/api/settings",
                json={
                    "theme": "light",
                    "image_workers": 4,
                    "job_workers": 3,
                    "request_timeout": 35,
                },
            )
            assert settings.status_code == 200
            assert settings.json()["theme"] == "light"
            assert settings.json()["restart_required"] is True

            folders = (await client.get("/api/sort/folders")).json()["items"]
            assert {item["path"] for item in folders} >= {"sort-target", "sort-control"}
            profile = await client.post(
                "/api/sort/profiles",
                json={
                    "name": "Pairs",
                    "target_directory": "sort-target",
                    "control_directories": ["sort-control"],
                    "mode": "time",
                    "threshold_seconds": 50,
                    "add_ids": True,
                },
            )
            assert profile.status_code == 201
            session_response = await client.post(
                "/api/sort/sessions",
                json={
                    "target_directory": "sort-target",
                    "control_directories": ["sort-control"],
                    "mode": "time",
                    "threshold_seconds": 50,
                    "add_ids": True,
                },
            )
            assert session_response.status_code == 201
            session = session_response.json()["session"]
            preview = await client.get(session["current"]["preview_url"])
            assert (
                preview.status_code == 200
                and preview.headers["content-type"] == "image/jpeg"
            )
            action = await client.post(
                f"/api/sort/sessions/{session['id']}/actions",
                json={
                    "kind": "skip",
                    "expected_target": session["current"]["path"],
                },
            )
            assert action.json()["session"]["status"] == "completed"
            undo = await client.post(f"/api/sort/sessions/{session['id']}/undo")
            assert undo.json()["session"]["status"] == "active"

            legacy_sync = await client.get("/sync", params={"profile": "POV"})
            assert GALLERY in legacy_sync.json()["ignores"]
            captured_job = {}

            def fake_enqueue(**values):
                captured_job.update(values)
                return {"id": "legacy-job"}

            monkeypatch.setattr(app.state.downloads, "enqueue", fake_enqueue)
            legacy_download_response = await client.post(
                "/download",
                json={
                    "folder_name": "Legacy sample",
                    "urls": ["https://cdni.pornpics.com/1280/sample.jpg"],
                    "headers": {"Referer": GALLERY},
                    "origin_url": GALLERY,
                    "profile": "POV",
                },
            )
            assert legacy_download_response.status_code == 200
            assert legacy_download_response.json() == {
                "status": "queued",
                "job_id": "legacy-job",
            }
            assert captured_job["profile"] == "POV"

    restarted_config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    restarted = create_app(restarted_config)
    async with restarted.router.lifespan_context(restarted):
        assert restarted_config.request_timeout == 35
        assert restarted_config.image_workers == 4
        assert restarted_config.job_workers == 3
        assert len(restarted.state.downloads._workers) == 3
