from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppConfig, config as default_config
from .db import Database, POSE_ROLES, PoseRevisionConflict
from .downloader import ActiveDownloadError, DownloadManager, EventBroker
from .finder import (
    FinderConflict,
    FinderNotFound,
    FinderService,
    FinderUnavailable,
)
from .models import (
    DownloadCreate,
    FinderReviewPatch,
    FinderScanCreate,
    GalleryPatch,
    LegacyDownloadRequest,
    LegacyIgnoreRequest,
    PoseDraftPut,
    PoseExportCreate,
    PoseTagCreate,
    PoseTagPatch,
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
    canonicalize_url,
    clean_profile_name,
    confined_path,
    decode_gallery_id,
    encode_gallery_id,
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
    finder = FinderService(app_config, database, scraper, events)
    media_client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal media_client
        app_config.ensure_directories()
        database.initialize()
        sorter.ensure_schema()
        finder.ensure_schema()
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
            await finder.start()
            yield
        finally:
            await finder.stop()
            await downloads.stop()
            await media_client.aclose()

    app = FastAPI(
        title="GalleryFlow",
        version=__version__,
        description=(
            "A web-only gallery browser, downloader, pose-pair organizer, "
            "history tracker, and visual sorter."
        ),
        lifespan=lifespan,
    )
    app.state.config = app_config
    app.state.db = database
    app.state.scraper = scraper
    app.state.downloads = downloads
    app.state.events = events
    app.state.sorter = sorter
    app.state.finder = finder

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
            content={"detail": str(exc), "job": downloads.public_job(exc.job)},
        )

    @app.exception_handler(SortConflict)
    async def sort_conflict_handler(_: Request, exc: SortConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SortNotFound)
    async def sort_not_found_handler(_: Request, exc: SortNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FinderNotFound)
    async def finder_not_found_handler(_: Request, exc: FinderNotFound):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FinderConflict)
    async def finder_conflict_handler(_: Request, exc: FinderConflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(FinderUnavailable)
    async def finder_unavailable_handler(_: Request, exc: FinderUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    def media_url(remote_url: str) -> str:
        token = sign_media_url(remote_url, app_config.media_signing_key)
        return f"/api/media?url={quote(remote_url, safe='')}&token={token}"

    def decorate_card(item: dict, status: dict[str, bool]) -> dict:
        remote = item.pop("thumbnail_remote_url")
        item["thumbnail_url"] = media_url(remote)
        item.update(status)
        return item

    def decorate_pose_draft(draft: dict) -> dict:
        draft["gallery_id"] = encode_gallery_id(draft["gallery_url"])
        return draft

    def decorate_finder_result(item: dict) -> dict:
        result = dict(item)
        preview = result.pop("best_preview_remote_url", "")
        thumbnail = result.pop("thumbnail_remote_url", "")
        top_matches: list[dict] = []
        for item_match in result.get("top_matches") or []:
            match = dict(item_match)
            match_preview = match.pop("preview_remote_url", "")
            match["preview_url"] = (
                media_url(match_preview) if match_preview else None
            )
            top_matches.append(match)
        result["top_matches"] = top_matches
        result["best_preview_url"] = media_url(preview) if preview else None
        result["thumbnail_url"] = media_url(thumbnail) if thumbnail else None
        result["gallery_id"] = encode_gallery_id(result["gallery_url"])
        return result

    async def pose_gallery_images(gallery_url: str) -> list[dict]:
        images = database.gallery_images(gallery_url)
        if images:
            return images
        detail = await scraper.gallery(gallery_url)
        database.register_gallery_images(detail["url"], detail["images"])
        return database.gallery_images(detail["url"])

    async def normalize_pose_draft(
        gallery_url: str, payload: PoseDraftPut
    ) -> tuple[dict[str, str | None], list[dict]]:
        images = await pose_gallery_images(gallery_url)
        available = {canonicalize_url(item["url"]): item for item in images}
        controls: dict[str, str | None] = {role: None for role in POSE_ROLES}
        used_controls: set[str] = set()
        for role, value in payload.controls.model_dump().items():
            if value is None:
                continue
            canonical = canonicalize_url(value)
            if canonical not in available:
                raise ValueError("A control image is not part of this gallery")
            if canonical in used_controls:
                raise ValueError("Each control role must use a different image")
            controls[role] = available[canonical]["url"]
            used_controls.add(canonical)

        targets: list[dict] = []
        used_targets: set[str] = set()
        used_ordinals: set[int] = set()
        for requested in payload.targets:
            canonical = canonicalize_url(requested.image_url)
            if canonical not in available:
                raise ValueError("A target image is not part of this gallery")
            if canonical in used_controls:
                raise ValueError("An image cannot be both a control and a target")
            if canonical in used_targets:
                raise ValueError("A target image can have only one pose")
            if not controls[requested.role]:
                raise ValueError(
                    f"The {requested.role} control must be selected before using that role"
                )
            if not database.get_pose_tag(requested.pose_tag_id):
                raise ValueError("A selected pose tag no longer exists")
            ordinal = int(available[canonical]["ordinal"])
            if ordinal in used_ordinals:
                raise ValueError("Gallery image ordinals must be unique")
            targets.append(
                {
                    "image_url": available[canonical]["url"],
                    "ordinal": ordinal,
                    "pose_tag_id": requested.pose_tag_id,
                    "role": requested.role,
                }
            )
            used_targets.add(canonical)
            used_ordinals.add(ordinal)
        return controls, targets

    @app.get("/api/health")
    async def health() -> dict:
        jobs = database.list_job_summaries(100)
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
            image["full_url"] = media_url(image["url"])
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

    @app.get("/api/pose-tags")
    async def list_pose_tags() -> dict:
        return {"items": database.list_pose_tags()}

    @app.post("/api/pose-tags", status_code=201)
    async def create_pose_tag(payload: PoseTagCreate) -> dict:
        try:
            tag = database.create_pose_tag(payload.label, payload.default_role)
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, "A pose tag with this label already exists") from exc
        events.publish({"type": "pose", "action": "tag", "tag": tag})
        return {"tag": tag}

    @app.patch("/api/pose-tags/{tag_id}")
    async def patch_pose_tag(tag_id: int, payload: PoseTagPatch) -> dict:
        try:
            tag = database.update_pose_tag(
                tag_id, label=payload.label, default_role=payload.default_role
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(409, "A pose tag with this label already exists") from exc
        if not tag:
            raise HTTPException(404, "Pose tag not found")
        events.publish({"type": "pose", "action": "tag", "tag": tag})
        return {"tag": tag}

    @app.get("/api/galleries/{gallery_id}/pose-draft")
    async def get_pose_draft(
        gallery_id: str,
        profile: str = Query(default="Default", max_length=64),
    ) -> dict:
        gallery_url = validate_source_url(decode_gallery_id(gallery_id))
        profile = clean_profile_name(profile)
        if not database.get_profile(profile):
            raise HTTPException(404, "Profile not found")
        return {
            "draft": decorate_pose_draft(
                database.get_pose_draft(gallery_url, profile)
            )
        }

    @app.put("/api/galleries/{gallery_id}/pose-draft")
    async def put_pose_draft(
        gallery_id: str,
        payload: PoseDraftPut,
        profile: str = Query(default="Default", max_length=64),
    ) -> Response:
        gallery_url = validate_source_url(decode_gallery_id(gallery_id))
        profile = clean_profile_name(profile)
        if not database.get_profile(profile):
            raise HTTPException(404, "Profile not found")
        controls, targets = await normalize_pose_draft(gallery_url, payload)
        try:
            draft = database.save_pose_draft(
                gallery_url,
                profile,
                payload.expected_revision,
                controls,
                targets,
            )
        except PoseRevisionConflict as exc:
            current = database.get_pose_draft(gallery_url, profile)
            return JSONResponse(
                status_code=409,
                content={
                    "detail": str(exc),
                    "draft": decorate_pose_draft(current),
                },
            )
        draft = decorate_pose_draft(draft)
        events.publish(
            {
                "type": "pose",
                "action": "draft",
                "gallery_id": draft["gallery_id"],
                "profile": draft["profile"],
                "revision": draft["revision"],
            }
        )
        return JSONResponse(content={"draft": draft})

    @app.post("/api/pose-exports", status_code=202)
    async def create_pose_export(payload: PoseExportCreate) -> dict:
        gallery_url = validate_source_url(decode_gallery_id(payload.gallery_id))
        profile = clean_profile_name(payload.profile)
        if not database.get_profile(profile):
            raise HTTPException(404, "Profile not found")
        draft = database.get_pose_draft(gallery_url, profile)
        if draft["revision"] != payload.expected_revision:
            return JSONResponse(
                status_code=409,
                content={
                    "detail": "The pose draft changed before it was exported",
                    "draft": decorate_pose_draft(draft),
                },
            )
        if not draft["targets"]:
            raise HTTPException(422, "Add at least one pose target before exporting")
        for target in draft["targets"]:
            if not draft["controls"].get(target["role"]):
                raise HTTPException(422, f"Missing {target['role']} control")
            if not target.get("pose_slug"):
                raise HTTPException(422, "A pose tag used by this draft no longer exists")
        job = downloads.enqueue_pose_export(
            gallery_url=gallery_url,
            profile=profile,
            draft=draft,
        )
        return {"job": downloads.public_job(job)}

    @app.get("/api/pose-exports")
    async def list_pose_exports(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict:
        return {
            "items": [
                downloads.public_job(job)
                for job in database.list_job_summaries(limit, "pose_export")
            ]
        }

    @app.get("/api/pose-exports/{job_id}/items")
    async def pose_export_items(job_id: str) -> dict:
        job = database.get_job_summary(job_id)
        if not job or job.get("kind") != "pose_export":
            raise HTTPException(404, "Pose export job not found")
        return {"items": database.list_job_items(job_id)}

    @app.delete("/api/pose-exports/{job_id}")
    async def cancel_pose_export(job_id: str) -> dict:
        existing = database.get_job_summary(job_id)
        if not existing or existing.get("kind") != "pose_export":
            raise HTTPException(404, "Pose export job not found")
        job = downloads.cancel(job_id)
        return {"job": job}

    @app.get("/api/finder/folders")
    async def finder_folders(
        path: str = Query(default=".", min_length=1, max_length=500),
    ) -> dict:
        return await asyncio.to_thread(finder.folders, path)

    @app.get("/api/finder/status")
    async def finder_status() -> dict:
        return finder.status()

    @app.post("/api/finder/scans", status_code=202)
    async def create_finder_scan(payload: FinderScanCreate) -> dict:
        scan = finder.create_scan(**payload.model_dump())
        return {"scan": scan}

    @app.get("/api/finder/scans")
    async def list_finder_scans(
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict:
        return {"items": finder.list_scans(limit)}

    @app.get("/api/finder/scans/{scan_id}")
    async def get_finder_scan(scan_id: str) -> dict:
        scan = finder.get_scan(scan_id)
        if not scan:
            raise FinderNotFound("Finder scan not found")
        return {"scan": scan}

    @app.get("/api/finder/scans/{scan_id}/results")
    async def finder_results(
        scan_id: str,
        review: str = Query(
            default="pending", pattern="^(pending|accepted|rejected|all)$"
        ),
        min_score: float | None = Query(default=None, ge=0, le=1),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict:
        items, total = finder.results(
            scan_id,
            review=review,
            min_score=min_score,
            limit=limit,
            offset=offset,
        )
        return {
            "items": [decorate_finder_result(item) for item in items],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @app.patch("/api/finder/scans/{scan_id}/results/{result_id}")
    async def review_finder_result(
        scan_id: str, result_id: str, payload: FinderReviewPatch
    ) -> dict:
        item = finder.set_review(scan_id, result_id, payload.review)
        return {"result": decorate_finder_result(item)}

    @app.post("/api/finder/scans/{scan_id}/pause")
    async def pause_finder_scan(scan_id: str) -> dict:
        return {"scan": finder.pause(scan_id)}

    @app.post("/api/finder/scans/{scan_id}/resume")
    async def resume_finder_scan(scan_id: str) -> dict:
        return {"scan": finder.resume(scan_id)}

    @app.delete("/api/finder/scans/{scan_id}")
    async def delete_finder_scan(scan_id: str) -> dict:
        scan = finder.delete_or_cancel(scan_id)
        return {"deleted": bool(scan.get("deleted")), "scan": scan}

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
        return {
            "items": [
                downloads.public_job(job)
                for job in database.list_job_summaries(limit)
            ]
        }

    @app.get("/api/downloads/{job_id}/items")
    async def download_items(job_id: str) -> dict:
        if not database.get_job_summary(job_id):
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
            "pose_root": str(app_config.pose_root_path),
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
