from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

from .config import AppConfig
from .db import Database, utc_now
from .scraper import PornPicsScraper, ScrapeError
from .security import (
    UnsafeUrl,
    canonicalize_url,
    confined_path,
    safe_folder_name,
    gallery_key,
    validate_public_media_url,
)


CONTENT_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/avif": ".avif",
}
POSE_OUTPUT_PART_RE = re.compile(
    r"^\.g.+-\d+_(?:target|control)\.[A-Za-z0-9]+\.[0-9a-f]{32}\.part$"
)


class ActiveDownloadError(RuntimeError):
    def __init__(self, job: dict, message: str | None = None):
        super().__init__(message or "This gallery already has an active download for the selected profile")
        self.job = job


class PoseExportCanceled(RuntimeError):
    pass


class EventBroker:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def publish(self, event: dict) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(event)


class DownloadManager:
    def __init__(
        self,
        app_config: AppConfig,
        database: Database,
        scraper: PornPicsScraper,
        events: EventBroker,
    ) -> None:
        self.config = app_config
        self.db = database
        self.scraper = scraper
        self.events = events
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._stopping = False
        self._image_semaphore = asyncio.Semaphore(self.config.image_workers)
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._stopping = False
        self._client = httpx.AsyncClient(timeout=self.config.image_timeout)
        await asyncio.to_thread(self._cleanup_orphan_pose_staging)
        for job_id in self.db.queued_job_ids():
            self.queue.put_nowait(job_id)
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"download-worker-{index}")
            for index in range(self.config.job_workers)
        ]

    async def stop(self) -> None:
        self._stopping = True
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def set_image_workers(self, count: int) -> None:
        """Apply a new global media limit to work that has not started yet."""
        self._image_semaphore = asyncio.Semaphore(max(1, count))

    @staticmethod
    def public_job(job: dict | None) -> dict | None:
        if job is None:
            return None
        result = {**job}
        payload = result.pop("payload", None) or {}
        if result.get("kind") == "pose_export":
            result.pop("image_urls", None)
            result["pair_count"] = int(
                result.get("pair_count") or len(payload.get("targets", []))
            )
            if result.get("pose_revision") is None:
                result["pose_revision"] = payload.get("revision")
        return result

    def _publish_job(self, job: dict | None) -> None:
        self.events.publish({"type": "job", "job": self.public_job(job)})

    def _publish_job_id(self, job_id: str) -> None:
        self._publish_job(self.db.get_job_summary(job_id))

    def _cleanup_orphan_pose_staging(self) -> None:
        pose_root = self.config.pose_root_path
        for pattern in (
            "*/selected_target/.*.part",
            "*/selected_control/.*.part",
        ):
            for part in pose_root.glob(pattern):
                if not POSE_OUTPUT_PART_RE.fullmatch(part.name):
                    continue
                try:
                    part.resolve().relative_to(pose_root.resolve())
                except ValueError:
                    continue
                part.unlink(missing_ok=True)
        staging_root = confined_path(pose_root, ".galleryflow-tmp")
        if not staging_root.is_dir():
            return
        for child in staging_root.iterdir():
            if not child.is_dir() or child.is_symlink():
                continue
            job = self.db.get_job_summary(child.name)
            keep = bool(
                job
                and job.get("kind") == "pose_export"
                and job.get("status") == "queued"
                and not job.get("cancel_requested")
            )
            if not keep:
                shutil.rmtree(child, ignore_errors=True)

    def enqueue(
        self,
        *,
        gallery_url: str,
        profile: str,
        title: str = "",
        image_urls: list[str] | None = None,
    ) -> dict:
        active = self.db.active_job_for_gallery(profile, gallery_url)
        if active:
            raise ActiveDownloadError(active)
        job_id = uuid.uuid4().hex
        created_at = utc_now()
        self.db.create_job(
            {
                "id": job_id,
                "gallery_url": gallery_url,
                "title": title,
                "profile": profile,
                "image_urls": image_urls,
                "created_at": created_at,
            }
        )
        self.queue.put_nowait(job_id)
        job = self.db.get_job_summary(job_id) or {}
        self._publish_job(job)
        return self.public_job(job) or {}

    def enqueue_pose_export(
        self,
        *,
        gallery_url: str,
        profile: str,
        draft: dict,
    ) -> dict:
        active = self.db.active_job_for_gallery(None, gallery_url, "pose_export")
        if active:
            raise ActiveDownloadError(
                active,
                "This gallery already has an active pose export",
            )
        controls = {
            role: value
            for role, value in draft["controls"].items()
            if value is not None
        }
        image_urls = list(
            dict.fromkeys(
                [*controls.values(), *(item["image_url"] for item in draft["targets"])]
            )
        )
        job_id = uuid.uuid4().hex
        created_at = utc_now()
        identity = gallery_key(gallery_url).rsplit(":", 1)[-1]
        self.db.create_job(
            {
                "id": job_id,
                "gallery_url": gallery_url,
                "title": f"Pose pairs · gallery {identity}",
                "profile": profile,
                "image_urls": image_urls,
                "kind": "pose_export",
                "payload": {
                    "revision": draft["revision"],
                    "controls": controls,
                    "targets": draft["targets"],
                },
                "pair_count": len(draft["targets"]),
                "pose_revision": draft["revision"],
                "created_at": created_at,
            }
        )
        self.queue.put_nowait(job_id)
        job = self.db.get_job_summary(job_id) or {}
        self._publish_job(job)
        return self.public_job(job) or {}

    def cancel(self, job_id: str) -> dict | None:
        job = self.db.get_job_summary(job_id)
        if not job:
            return None
        terminal = {"completed", "completed_with_errors", "failed", "canceled"}
        if job["status"] in terminal:
            self.db.delete_job(job_id)
            removed = {**job, "deleted": True}
            self._publish_job(removed)
            return self.public_job(removed)
        values = {"cancel_requested": 1}
        if job["status"] == "queued":
            values["status"] = "canceled"
        elif job["status"] not in terminal:
            values["status"] = "canceling"
        self.db.update_job(job_id, **values)
        job = self.db.get_job_summary(job_id)
        self._publish_job(job)
        return self.public_job(job)

    async def _worker(self, index: int) -> None:
        while not self._stopping:
            job_id = await self.queue.get()
            try:
                job = self.db.get_job(job_id)
                if not job or job["status"] == "canceled" or job["cancel_requested"]:
                    continue
                if job.get("kind") == "pose_export":
                    await self._run_pose_export_job(job)
                else:
                    await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except (
                Exception
            ) as exc:  # Defensive boundary: a worker must survive one bad job.
                if self.db.job_cancel_requested(job_id):
                    self.db.update_job(job_id, status="canceled", error="")
                else:
                    self.db.update_job(job_id, status="failed", error=str(exc)[:1000])
                self._publish_job_id(job_id)
            finally:
                self.queue.task_done()

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        self.db.update_job(job_id, status="starting", error="")
        self._publish_job_id(job_id)

        detail = await self.scraper.gallery(job["gallery_url"])
        self.db.register_gallery_images(detail["url"], detail["images"])
        available = {canonicalize_url(item["url"]): item for item in detail["images"]}
        requested = job.get("image_urls")
        if requested:
            selected: list[dict] = []
            for value in requested:
                canonical = canonicalize_url(value)
                if canonical not in available:
                    raise UnsafeUrl(
                        "A selected image is not part of the requested gallery"
                    )
                selected.append(available[canonical])
            images = selected
        else:
            images = list(available.values())

        if not images:
            raise ScrapeError("Gallery has no downloadable images")
        self.db.create_job_items(job_id, images)

        profile = self.db.get_profile(job["profile"])
        if not profile:
            raise ValueError(f"Unknown profile: {job['profile']}")
        profile_root = confined_path(self.config.download_root, profile["directory"])
        profile_root.mkdir(parents=True, exist_ok=True)
        folder_base = safe_folder_name(job.get("title") or detail["title"])
        identity = gallery_key(detail["url"]).rsplit(":", 1)[-1]
        destination = confined_path(profile_root, f"{folder_base}--{identity}")
        destination.mkdir(parents=True, exist_ok=True)

        self.db.update_job(
            job_id,
            title=detail["title"],
            status="downloading",
            total=len(images),
            destination=str(destination.relative_to(self.config.download_root)),
        )
        self._publish_job_id(job_id)

        completed = 0
        failed = 0
        errors: list[str] = []
        progress_lock = asyncio.Lock()

        async def download_one(position: int, item: dict) -> None:
            nonlocal completed, failed
            async with self._image_semaphore:
                if self.db.job_cancel_requested(job_id):
                    return
                try:
                    existing = self.db.image_statuses(job["profile"], detail["url"])
                    if item["url"] in existing:
                        self.db.update_job_item(
                            job_id, item["url"], status="skipped", attempts=0
                        )
                        async with progress_lock:
                            completed += 1
                        return
                    self.db.update_job_item(
                        job_id, item["url"], status="downloading", attempts=1
                    )
                    final_path = await self._download_image(
                        item["url"], destination, position, referer=detail["url"]
                    )
                    relative_path = str(
                        final_path.relative_to(self.config.download_root)
                    )
                    byte_count = final_path.stat().st_size
                    self.db.add_profile_image(
                        job["profile"],
                        detail["url"],
                        item["url"],
                        relative_path,
                        byte_count,
                    )
                    self.db.update_job_item(
                        job_id,
                        item["url"],
                        status="completed",
                        byte_count=byte_count,
                        relative_path=relative_path,
                        error="",
                    )
                    async with progress_lock:
                        completed += 1
                except Exception as exc:
                    self.db.update_job_item(
                        job_id, item["url"], status="failed", error=str(exc)[:500]
                    )
                    async with progress_lock:
                        failed += 1
                        errors.append(f"Image {position}: {exc}")
                finally:
                    async with progress_lock:
                        self.db.update_job(
                            job_id,
                            completed=completed,
                            failed=failed,
                            error="; ".join(errors[-5:])[:1000],
                        )
                        self._publish_job_id(job_id)

        tasks = [
            asyncio.create_task(download_one(int(item.get("ordinal", index)), item))
            for index, item in enumerate(images, start=1)
        ]
        await asyncio.gather(*tasks)

        if self.db.job_cancel_requested(job_id):
            status = "canceled"
        elif completed == len(images) and failed == 0:
            status = "completed"
            all_downloaded = self.db.image_statuses(job["profile"], detail["url"])
            if len(all_downloaded.intersection(available)) >= len(available):
                self.db.add_history(
                    detail["url"],
                    job["profile"],
                    detail["title"],
                    str(destination.relative_to(self.config.download_root)),
                    len(available),
                )
        elif completed:
            status = "completed_with_errors"
        else:
            status = "failed"
        self.db.update_job(job_id, status=status)
        self._publish_job_id(job_id)

    async def _run_pose_export_job(self, job: dict) -> None:
        job_id = job["id"]
        payload = job.get("payload") or {}
        controls = payload.get("controls") or {}
        targets = payload.get("targets") or []
        if not targets:
            raise ValueError("The pose export has no target images")

        ordered_urls: list[str] = []
        canonical_urls: set[str] = set()
        for url in [*controls.values(), *(item["image_url"] for item in targets)]:
            canonical = canonicalize_url(url)
            if canonical not in canonical_urls:
                canonical_urls.add(canonical)
                ordered_urls.append(url)
        images = [
            {"url": url, "ordinal": index}
            for index, url in enumerate(ordered_urls, start=1)
        ]
        self.db.create_job_items(job_id, images)

        pose_root = self.config.pose_root_path
        pose_root.mkdir(parents=True, exist_ok=True)
        staging = confined_path(pose_root, f".galleryflow-tmp/{job_id}")
        staging.mkdir(parents=True, exist_ok=True)
        self.db.update_job(
            job_id,
            status="downloading",
            total=len(images),
            completed=0,
            failed=0,
            destination=str(pose_root),
            error="",
        )
        self._publish_job_id(job_id)

        completed = 0
        failed = 0
        errors: list[str] = []
        progress_lock = asyncio.Lock()
        cached: dict[str, Path] = {}
        previous_items = {
            item["image_url"]: item for item in self.db.list_job_items(job_id)
        }

        async def fetch_one(position: int, url: str) -> None:
            nonlocal completed, failed
            async with self._image_semaphore:
                if self.db.job_cancel_requested(job_id):
                    return
                try:
                    previous = previous_items.get(url) or {}
                    previous_path: Path | None = None
                    if previous.get("status") == "completed" and previous.get(
                        "relative_path"
                    ):
                        candidate = confined_path(
                            pose_root, str(previous["relative_path"])
                        )
                        if candidate.is_file():
                            previous_path = candidate
                    if previous_path is None:
                        self.db.update_job_item(
                            job_id, url, status="downloading", attempts=1, error=""
                        )
                        previous_path = await self._download_image(
                            url, staging, position, referer=job["gallery_url"]
                        )
                        relative_path = str(previous_path.relative_to(pose_root))
                        self.db.update_job_item(
                            job_id,
                            url,
                            status="completed",
                            attempts=1,
                            byte_count=previous_path.stat().st_size,
                            relative_path=relative_path,
                            error="",
                        )
                    cached[canonicalize_url(url)] = previous_path
                    async with progress_lock:
                        completed += 1
                except Exception as exc:
                    self.db.update_job_item(
                        job_id, url, status="failed", error=str(exc)[:500]
                    )
                    async with progress_lock:
                        failed += 1
                        errors.append(f"Image {position}: {exc}")
                finally:
                    async with progress_lock:
                        self.db.update_job(
                            job_id,
                            completed=completed,
                            failed=failed,
                            error="; ".join(errors[-5:])[:1000],
                        )
                        self._publish_job_id(job_id)

        await asyncio.gather(
            *(
                asyncio.create_task(fetch_one(index, url))
                for index, url in enumerate(ordered_urls, start=1)
            )
        )

        if self.db.job_cancel_requested(job_id):
            status = "canceled"
        elif failed:
            status = "failed"
        else:
            last_cancel_poll = 0.0
            cancellation_seen = False

            def should_cancel_materialization() -> bool:
                nonlocal last_cancel_poll, cancellation_seen
                if self._stopping or cancellation_seen:
                    return True
                now = time.monotonic()
                if now - last_cancel_poll >= 0.2:
                    last_cancel_poll = now
                    cancellation_seen = self.db.job_cancel_requested(job_id)
                return cancellation_seen

            try:
                await asyncio.to_thread(
                    self._materialize_pose_pairs,
                    job["gallery_url"],
                    pose_root,
                    controls,
                    targets,
                    cached,
                    should_cancel_materialization,
                )
                status = (
                    "canceled"
                    if self.db.job_cancel_requested(job_id)
                    else "completed"
                )
            except PoseExportCanceled:
                status = "canceled"
            except Exception as exc:
                status = "failed"
                errors.append(str(exc))
        self.db.update_job(
            job_id,
            status=status,
            error="; ".join(errors[-5:])[:1000],
        )
        self._publish_job_id(job_id)
        await asyncio.to_thread(shutil.rmtree, staging, ignore_errors=True)
        for url in ordered_urls:
            self.db.update_job_item(job_id, url, relative_path="")

    def _materialize_pose_pairs(
        self,
        gallery_url: str,
        pose_root: Path,
        controls: dict[str, str],
        targets: list[dict],
        cached: dict[str, Path],
        should_cancel=None,
    ) -> None:
        identity = safe_folder_name(
            gallery_key(gallery_url).rsplit(":", 1)[-1], "gallery"
        )
        plans: list[tuple[Path, Path, Path, Path]] = []
        for target in targets:
            if should_cancel and should_cancel():
                raise PoseExportCanceled("Pose export canceled")
            pose_directory = confined_path(pose_root, target["pose_slug"])
            target_directory = confined_path(pose_directory, "selected_target")
            control_directory = confined_path(pose_directory, "selected_control")
            target_directory.mkdir(parents=True, exist_ok=True)
            control_directory.mkdir(parents=True, exist_ok=True)
            ordinal = int(target["ordinal"])
            stem = f"g{identity}-{ordinal:04d}"
            target_source = cached[canonicalize_url(target["image_url"])]
            control_source = cached[canonicalize_url(controls[target["role"]])]
            target_path = confined_path(
                target_directory, f"{stem}_target{target_source.suffix.lower()}"
            )
            control_path = confined_path(
                control_directory, f"{stem}_control{control_source.suffix.lower()}"
            )
            self._preflight_pose_identity(pose_root, stem, target_path)
            self._preflight_output(target_source, target_path)
            self._preflight_output(control_source, control_path)
            plans.append(
                (target_source, target_path, control_source, control_path)
            )

        created_paths: list[Path] = []
        try:
            for target_source, target_path, control_source, control_path in plans:
                if should_cancel and should_cancel():
                    raise PoseExportCanceled("Pose export canceled")
                if self._copy_without_overwrite(
                    control_source, control_path, should_cancel
                ):
                    created_paths.append(control_path)
                if should_cancel and should_cancel():
                    raise PoseExportCanceled("Pose export canceled")
                if self._copy_without_overwrite(
                    target_source, target_path, should_cancel
                ):
                    created_paths.append(target_path)
                if should_cancel and should_cancel():
                    raise PoseExportCanceled("Pose export canceled")
        except Exception:
            for path in reversed(created_paths):
                path.unlink(missing_ok=True)
            raise

    @classmethod
    def _preflight_output(cls, source: Path, target: Path) -> None:
        if target.exists() and not cls._files_identical(source, target):
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    @staticmethod
    def _preflight_pose_identity(
        pose_root: Path, stem: str, desired_target: Path
    ) -> None:
        for existing in pose_root.glob(f"*/selected_target/{stem}_target.*"):
            try:
                existing.resolve().relative_to(pose_root.resolve())
            except ValueError as exc:
                raise ValueError("Pose output path escapes the configured root") from exc
            if existing != desired_target:
                pose_name = existing.parent.parent.name
                raise FileExistsError(
                    f"{stem} was already exported under pose '{pose_name}'"
                )

    @classmethod
    def _copy_without_overwrite(
        cls, source: Path, target: Path, should_cancel=None
    ) -> bool:
        if target.exists():
            if cls._files_identical(source, target):
                return False
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        for stale in target.parent.glob(f".{target.name}.*.part"):
            stale.unlink(missing_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
        try:
            with source.open("rb") as source_file, temporary.open("xb") as target_file:
                while True:
                    if should_cancel and should_cancel():
                        raise PoseExportCanceled("Pose export canceled")
                    chunk = source_file.read(1024 * 1024)
                    if not chunk:
                        break
                    target_file.write(chunk)
            os.link(temporary, target)
            return True
        except FileExistsError:
            if target.exists() and cls._files_identical(source, target):
                return False
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _files_identical(left: Path, right: Path) -> bool:
        if left.stat().st_size != right.stat().st_size:
            return False
        left_hash = hashlib.sha256()
        right_hash = hashlib.sha256()
        with left.open("rb") as left_file, right.open("rb") as right_file:
            for chunk in iter(lambda: left_file.read(1024 * 1024), b""):
                left_hash.update(chunk)
            for chunk in iter(lambda: right_file.read(1024 * 1024), b""):
                right_hash.update(chunk)
        return left_hash.digest() == right_hash.digest()

    async def _download_image(
        self, url: str, destination: Path, position: int, *, referer: str
    ) -> Path:
        current = await asyncio.to_thread(validate_public_media_url, url)
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8,*/*;q=0.5",
            "Referer": referer,
        }
        if self._client is None:
            raise RuntimeError("Download manager is not running")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                current = await asyncio.to_thread(validate_public_media_url, url)
                for _ in range(6):
                    async with self._client.stream(
                        "GET", current, headers=headers, follow_redirects=False
                    ) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise RuntimeError(
                                    "Image host returned an empty redirect"
                                )
                            current = await asyncio.to_thread(
                                validate_public_media_url, urljoin(current, location)
                            )
                            continue
                        if response.status_code == 429 or response.status_code >= 500:
                            raise httpx.HTTPStatusError(
                                f"Transient media HTTP {response.status_code}",
                                request=response.request,
                                response=response,
                            )
                        response.raise_for_status()
                        content_type = (
                            response.headers.get("content-type", "")
                            .split(";", 1)[0]
                            .lower()
                        )
                        if not content_type.startswith("image/"):
                            raise RuntimeError(
                                f"Unexpected media type: {content_type or 'unknown'}"
                            )
                        extension = CONTENT_EXTENSIONS.get(content_type)
                        if not extension:
                            extension = Path(urlsplit(current).path).suffix.lower()
                            if extension not in {
                                ".jpg",
                                ".jpeg",
                                ".png",
                                ".webp",
                                ".gif",
                                ".avif",
                            }:
                                extension = (
                                    mimetypes.guess_extension(content_type) or ".jpg"
                                )
                        final_path = confined_path(
                            destination, f"{position:04d}{extension}"
                        )
                        part_path = confined_path(
                            destination, f".{position:04d}{extension}.part"
                        )
                        total = 0
                        try:
                            with part_path.open("wb") as output:
                                async for chunk in response.aiter_bytes(256 * 1024):
                                    total += len(chunk)
                                    if total > self.config.max_image_bytes:
                                        raise RuntimeError(
                                            "Image exceeds configured size limit"
                                        )
                                    output.write(chunk)
                            if total == 0:
                                raise RuntimeError("Image response was empty")
                            await asyncio.to_thread(self._verify_image, part_path)
                            os.replace(part_path, final_path)
                        finally:
                            if part_path.exists():
                                part_path.unlink()
                        return final_path
                raise RuntimeError("Too many image redirects")
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
                    continue
                raise
        raise RuntimeError(str(last_error or "Image download failed"))

    @staticmethod
    def _verify_image(path: Path) -> None:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            raise RuntimeError("Downloaded file is not a valid image") from exc
