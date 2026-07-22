from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppConfig, config as default_config
from .db import Database
from .downloader import ActiveDownloadError, DownloadManager, EventBroker
from .models import (
    DownloadCreate,
    GalleryPatch,
    LegacyDownloadRequest,
    LegacyIgnoreRequest,
    ProfileCreate,
    ProfilePatch,
    SettingsPatch,
    SortActionCreate,
    SortProfileCreate,
    SortSessionCreate,
)
from .scraper import PornPicsScraper, ScrapeError
from .security import (
    UnsafeUrl,
    clean_profile_name,
    confined_path,
    decode_gallery_id,
    safe_folder_name,
    sign_media_url,
    validate_public_media_url,
    validate_source_url,
    verify_media_signature,
)
from .sorter import SortConflict, SorterService, SortNotFound


def create_app(app_config: AppConfig | None = None) -> FastAPI:
    app_config = app_config or default_config
    database = Database(app_config.db_path, app_config.sqlite_vfs)
    scraper = PornPicsScraper(app_config)
    events = EventBroker()
    downloads = DownloadManager(app_config, database, scraper, events)
    sorter = SorterService(app_config, database)
    media_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal media_client
        app_config.ensure_directories()
        database.initialize()
        sorter.ensure_schema()
        stored = database.settings()
        request_timeout = stored.get("request_timeout")
        image_workers = stored.get("image_workers")
        job_workers = stored.get("job_workers")
        if isinstance(request_timeout, (int, float)) and 5 <= request_timeout <= 120:
            app_config.request_timeout = float(request_timeout)
        if isinstance(image_workers, int) and 1 <= image_workers <= 24:
            app_config.image_workers = image_workers
            downloads.set_image_workers(image_workers)
        if isinstance(job_workers, int) and 1 <= job_workers <= 8:
            app_config.job_workers = job_workers
        for profile in database.list_profiles():
            confined_path(app_config.download_root, profile["directory"]).mkdir(
                parents=True, exist_ok=True
            )
        media_client = httpx.AsyncClient(timeout=app_config.image_timeout)
        application.state.media_client = media_client
        try:
            await downloads.start()
            yield
        finally:
            await downloads.stop()
            await media_client.aclose()

    app = FastAPI(
        title="GalleryFlow",
        version=__version__,
        description="A web-only gallery browser, downloader, history tracker, and profile sorter.",
        lifespan=lifespan,
    )
    app.state.config = app_config
    app.state.db = database
    app.state.scraper = scraper
    app.state.downloads = downloads
    app.state.events = events
    app.state.sorter = sorter

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
        )
        return response

    @app.exception_handler(UnsafeUrl)
    async def unsafe_url_handler(_: Request, exc: UnsafeUrl):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(ScrapeError)
    async def scrape_error_handler(_: Request, exc: ScrapeError):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ActiveDownloadError)
    async def active_download_handler(_: Request, exc: ActiveDownloadError):
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "job": exc.job},
        )

    @app.exception_handler(SortConflict)
    async def sort_conflict_handler(_: Request, exc: SortConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SortNotFound)
    async def sort_not_found_handler(_: Request, exc: SortNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    def media_url(remote_url: str) -> str:
        token = sign_media_url(remote_url, app_config.media_signing_key)
        return f"/api/media?url={quote(remote_url, safe='')}&token={token}"

    def decorate_card(item: dict, status: dict[str, bool]) -> dict:
        remote = item.pop("thumbnail_remote_url")
        item["thumbnail_url"] = media_url(remote)
        item.update(status)
        return item

    @app.get("/api/health")
    async def health() -> dict:
        jobs = database.list_jobs(100)
        active = sum(
            j["status"] in {"queued", "starting", "downloading", "canceling"}
            for j in jobs
        )
        return {
            "status": "ok",
            "version": __version__,
            "source": app_config.source_home,
            "active_jobs": active,
            "queue_depth": downloads.queue.qsize(),
        }

    @app.get("/api/galleries")
    async def browse_galleries(
        url: str | None = None,
        q: str | None = Query(default=None, max_length=200),
        page: int = Query(default=1, ge=1, le=1000),
        profile: str = Query(default="Default", max_length=64),
        show_saved: bool = True,
        show_ignored: bool = True,
    ) -> dict:
        profile = clean_profile_name(profile)
        if not database.get_profile(profile):
            raise HTTPException(404, "Profile not found")
        result = await scraper.browse(url=url, query=q, page=page)
        statuses = database.status_for_urls(
            [item["url"] for item in result["items"]], profile
        )
        decorated = [
            decorate_card(item, statuses[item["url"]]) for item in result["items"]
        ]
        if not show_saved:
            decorated = [item for item in decorated if not item["saved"]]
        if not show_ignored:
            decorated = [item for item in decorated if not item["ignored"]]
        result["items"] = decorated
        result["total"] = len(decorated)
        result["page"] = page
        result["pages"] = page + 1 if result["next_url"] else page
        return result

    @app.get("/api/galleries/{gallery_id}")
    async def gallery_detail(
        gallery_id: str, profile: str = Query(default="Default", max_length=64)
    ) -> dict:
        gallery_url = validate_source_url(decode_gallery_id(gallery_id))
        profile = clean_profile_name(profile)
        detail = await scraper.gallery(gallery_url)
        database.register_gallery_images(detail["url"], detail["images"])
        status = database.status_for_urls([detail["url"]], profile)[detail["url"]]
        downloaded_images = database.image_statuses(profile, detail["url"])
        for image in detail["images"]:
            remote = image.pop("preview_remote_url")
            image["preview_url"] = media_url(remote)
            image["downloaded"] = image["url"] in downloaded_images
        detail.update(status)
        return detail

    @app.patch("/api/galleries/{gallery_id}")
    async def patch_gallery(gallery_id: str, payload: GalleryPatch) -> dict:
        gallery_url = validate_source_url(decode_gallery_id(gallery_id))
        database.set_ignored(gallery_url, payload.ignored, payload.title or "")
        events.publish(
            {"type": "gallery", "url": gallery_url, "ignored": payload.ignored}
        )
        return {"url": gallery_url, "ignored": payload.ignored}

    @app.get("/api/media")
    async def proxy_media(request: Request, url: str, token: str) -> StreamingResponse:
        if not verify_media_signature(url, token, app_config.media_signing_key):
            raise HTTPException(403, "Invalid media signature")
        current = await asyncio.to_thread(validate_public_media_url, url)
        client: httpx.AsyncClient = request.app.state.media_client
        response: httpx.Response | None = None
        try:
            for _ in range(6):
                upstream_request = client.build_request(
                    "GET",
                    current,
                    headers={
                        "User-Agent": app_config.user_agent,
                        "Referer": app_config.source_home,
                    },
                )
                response = await client.send(
                    upstream_request, stream=True, follow_redirects=False
                )
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    await response.aclose()
                    if not location:
                        raise HTTPException(502, "Empty media redirect")
                    current = await asyncio.to_thread(
                        validate_public_media_url,
                        str(httpx.URL(current).join(location)),
                    )
                    continue
                if response.status_code >= 400:
                    status = response.status_code
                    await response.aclose()
                    raise HTTPException(502, f"Media host returned HTTP {status}")
                content_type = response.headers.get("content-type", "")
                if not content_type.lower().startswith("image/"):
                    await response.aclose()
                    raise HTTPException(415, "Remote resource is not an image")
                length = response.headers.get("content-length")
                if length and int(length) > app_config.max_image_bytes:
                    await response.aclose()
                    raise HTTPException(413, "Remote image is too large")

                async def stream_body():
                    total = 0
                    try:
                        async for chunk in response.aiter_bytes(256 * 1024):
                            total += len(chunk)
                            if total > app_config.max_image_bytes:
                                break
                            yield chunk
                    finally:
                        await response.aclose()

                return StreamingResponse(
                    stream_body(),
                    media_type=content_type,
                    headers={"Cache-Control": "private, max-age=86400"},
                )
            raise HTTPException(502, "Too many media redirects")
        except Exception:
            if response is not None:
                await response.aclose()
            raise

    @app.post("/api/downloads", status_code=202)
    async def create_download(payload: DownloadCreate) -> dict:
        if payload.gallery_id:
            gallery_url = decode_gallery_id(payload.gallery_id)
        elif payload.gallery_url:
            gallery_url = payload.gallery_url
        else:
            raise HTTPException(422, "gallery_id or gallery_url is required")
        gallery_url = validate_source_url(gallery_url)
        profile = clean_profile_name(payload.profile)
        if not database.get_profile(profile):
            raise HTTPException(404, "Profile not found")
        job = downloads.enqueue(
            gallery_url=gallery_url,
            profile=profile,
            title=payload.title or "",
            image_urls=payload.image_urls,
        )
        return {"job": job}

    @app.get("/api/downloads")
    async def list_downloads(limit: int = Query(default=100, ge=1, le=500)) -> dict:
        return {"items": database.list_jobs(limit)}

    @app.get("/api/downloads/{job_id}/items")
    async def download_items(job_id: str) -> dict:
        if not database.get_job(job_id):
            raise HTTPException(404, "Download job not found")
        return {"items": database.list_job_items(job_id)}

    @app.delete("/api/downloads/{job_id}")
    async def cancel_download(job_id: str) -> dict:
        job = downloads.cancel(job_id)
        if not job:
            raise HTTPException(404, "Download job not found")
        return {"job": job}

    @app.get("/api/events")
    async def event_stream(request: Request) -> StreamingResponse:
        queue = events.subscribe()

        async def generate():
            try:
                yield "event: connected\ndata: {}\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=20)
                        yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n"
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                events.unsubscribe(queue)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/api/profiles")
    async def list_profiles() -> dict:
        return {"items": database.list_profiles()}

    @app.post("/api/profiles", status_code=201)
    async def create_profile(payload: ProfileCreate) -> dict:
        name = clean_profile_name(payload.name)
        directory = safe_folder_name(name, "Default")
        if database.get_profile(name):
            raise HTTPException(409, "Profile already exists")
        profile_path = confined_path(app_config.download_root, directory)
        profile_path.mkdir(parents=True, exist_ok=True)
        try:
            profile = database.create_profile(name, directory)
        except Exception as exc:
            raise HTTPException(
                409, "Profile name or directory already exists"
            ) from exc
        return {"profile": profile}

    @app.patch("/api/profiles/{name}")
    async def rename_profile(name: str, payload: ProfilePatch) -> dict:
        old_name = clean_profile_name(name)
        new_name = clean_profile_name(payload.new_name)
        profile = database.get_profile(old_name)
        if not profile:
            raise HTTPException(404, "Profile not found")
        if (
            database.get_profile(new_name)
            and new_name.casefold() != old_name.casefold()
        ):
            raise HTTPException(409, "Profile already exists")
        if database.has_active_jobs(old_name):
            raise HTTPException(
                409, "Profile cannot be renamed while it has active jobs"
            )
        # The directory is deliberately immutable: a display rename never races file writes.
        database.rename_profile(old_name, new_name, profile["directory"])
        return {"profile": database.get_profile(new_name)}

    @app.delete("/api/profiles/{name}")
    async def delete_profile(name: str) -> dict:
        name = clean_profile_name(name)
        if name.casefold() == "default":
            raise HTTPException(409, "The Default profile cannot be deleted")
        if not database.get_profile(name):
            raise HTTPException(404, "Profile not found")
        try:
            database.delete_profile(name)
        except Exception as exc:
            raise HTTPException(
                409, "Profile is still referenced by history or jobs"
            ) from exc
        return {"deleted": True, "files_preserved": True}

    @app.get("/api/sort/folders")
    async def sort_folders() -> dict:
        return await asyncio.to_thread(sorter.folders)

    @app.get("/api/sort/profiles")
    async def sort_profiles() -> dict:
        return {"items": await asyncio.to_thread(sorter.list_profiles)}

    @app.post("/api/sort/profiles", status_code=201)
    async def save_sort_profile(payload: SortProfileCreate) -> dict:
        profile = await asyncio.to_thread(sorter.save_profile, payload.model_dump())
        events.publish({"type": "sort", "action": "profile", "profile": profile})
        return {"profile": profile}

    @app.delete("/api/sort/profiles/{name}")
    async def delete_sort_profile(name: str) -> dict:
        deleted = await asyncio.to_thread(sorter.delete_profile, name)
        if not deleted:
            raise HTTPException(404, "Sort profile not found")
        events.publish({"type": "sort", "action": "profile_deleted", "name": name})
        return {"deleted": True}

    @app.post("/api/sort/sessions", status_code=201)
    async def start_sort_session(payload: SortSessionCreate) -> dict:
        session = await asyncio.to_thread(sorter.start_session, payload.model_dump())
        events.publish(
            {"type": "sort", "action": "session", "session_id": session["id"]}
        )
        return {"session": session}

    @app.get("/api/sort/sessions/{session_id}")
    async def get_sort_session(session_id: str) -> dict:
        session = await asyncio.to_thread(sorter.get_session, session_id)
        if not session:
            raise HTTPException(404, "Sort session not found")
        return {"session": session}

    @app.post("/api/sort/sessions/{session_id}/actions")
    async def apply_sort_action(session_id: str, payload: SortActionCreate) -> dict:
        session = await asyncio.to_thread(
            sorter.apply_action,
            session_id,
            payload.kind,
            payload.expected_target,
            payload.control_path,
        )
        events.publish(
            {"type": "sort", "action": payload.kind, "session_id": session_id}
        )
        return {"session": session}

    @app.post("/api/sort/sessions/{session_id}/undo")
    async def undo_sort_action(session_id: str) -> dict:
        session = await asyncio.to_thread(sorter.undo, session_id)
        events.publish({"type": "sort", "action": "undo", "session_id": session_id})
        return {"session": session}

    @app.get("/api/sort/media")
    async def sort_media(path: str, token: str) -> FileResponse:
        if not verify_media_signature(
            f"sort:{path}", token, app_config.media_signing_key
        ):
            raise HTTPException(403, "Invalid sorter media signature")
        resolved = await asyncio.to_thread(sorter.resolve_media, path)
        return FileResponse(
            resolved,
            headers={"Cache-Control": "private, max-age=300"},
        )

    @app.get("/api/history")
    async def history(
        profile: str | None = Query(default=None, max_length=64),
        limit: int = Query(default=250, ge=1, le=2000),
    ) -> dict:
        if profile:
            profile = clean_profile_name(profile)
        return {"items": database.list_history(profile, limit)}

    def resolved_settings() -> dict:
        stored = database.settings()
        return {
            "download_root": str(app_config.download_root),
            "sort_root": str(app_config.sort_root_path),
            "source_home": app_config.source_home,
            "request_timeout": stored.get(
                "request_timeout", app_config.request_timeout
            ),
            "image_workers": stored.get("image_workers", app_config.image_workers),
            "job_workers": stored.get("job_workers", app_config.job_workers),
            "theme": stored.get("theme", "dark"),
        }

    @app.get("/api/settings")
    async def get_settings() -> dict:
        return resolved_settings()

    @app.patch("/api/settings")
    async def patch_settings(payload: SettingsPatch) -> dict:
        values = payload.model_dump(exclude_none=True)
        for key, value in values.items():
            database.set_setting(key, value)
        if "request_timeout" in values:
            app_config.request_timeout = values["request_timeout"]
        if "image_workers" in values:
            app_config.image_workers = values["image_workers"]
            downloads.set_image_workers(values["image_workers"])
        restart_required = (
            "job_workers" in values and values["job_workers"] != app_config.job_workers
        )
        if "theme" in values:
            events.publish({"type": "settings", "theme": values["theme"]})
        result = resolved_settings()
        result["restart_required"] = restart_required
        return result

    # Compatibility endpoints make migration from the old desktop client painless.
    @app.get("/sync")
    async def legacy_sync(profile: str = "Default") -> dict:
        profile = clean_profile_name(profile)
        saved, ignored = database.sync(profile)
        return {"downloads": saved, "ignores": ignored}

    @app.post("/ignore")
    async def legacy_ignore(payload: LegacyIgnoreRequest) -> dict:
        url = validate_source_url(payload.url)
        database.set_ignored(url, True)
        return {"status": "ok"}

    @app.post("/download", status_code=200)
    async def legacy_download(payload: LegacyDownloadRequest) -> dict:
        gallery_url = validate_source_url(payload.origin_url)
        profile_name = clean_profile_name(payload.profile)
        if not database.get_profile(profile_name):
            directory = safe_folder_name(profile_name)
            confined_path(app_config.download_root, directory).mkdir(
                parents=True, exist_ok=True
            )
            database.create_profile(profile_name, directory)
        job = downloads.enqueue(
            gallery_url=gallery_url,
            profile=profile_name,
            title=payload.folder_name or "",
            image_urls=payload.urls or None,
        )
        return {"status": "queued", "job_id": job["id"]}

    if app_config.static_dir.exists():
        app.mount(
            "/static", StaticFiles(directory=app_config.static_dir), name="static"
        )

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        path = app_config.static_dir / "index.html"
        if not path.exists():
            return JSONResponse(
                status_code=503,
                content={"detail": "Web assets have not been installed"},
            )
        return FileResponse(path)

    return app


app = create_app()
