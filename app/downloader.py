from __future__ import annotations

import asyncio
import mimetypes
import os
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


class ActiveDownloadError(RuntimeError):
    def __init__(self, job: dict):
        super().__init__(
            "This gallery already has an active download for the selected profile"
        )
        self.job = job


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
        job = self.db.get_job(job_id) or {}
        self.events.publish({"type": "job", "job": job})
        return job

    def cancel(self, job_id: str) -> dict | None:
        job = self.db.get_job(job_id)
        if not job:
            return None
        terminal = {"completed", "completed_with_errors", "failed", "canceled"}
        if job["status"] in terminal:
            self.db.delete_job(job_id)
            removed = {**job, "deleted": True}
            self.events.publish({"type": "job", "job": removed})
            return removed
        values = {"cancel_requested": 1}
        if job["status"] == "queued":
            values["status"] = "canceled"
        elif job["status"] not in terminal:
            values["status"] = "canceling"
        self.db.update_job(job_id, **values)
        job = self.db.get_job(job_id)
        self.events.publish({"type": "job", "job": job})
        return job

    async def _worker(self, index: int) -> None:
        while not self._stopping:
            job_id = await self.queue.get()
            try:
                job = self.db.get_job(job_id)
                if not job or job["status"] == "canceled" or job["cancel_requested"]:
                    continue
                await self._run_job(job)
            except asyncio.CancelledError:
                raise
            except (
                Exception
            ) as exc:  # Defensive boundary: a worker must survive one bad job.
                current = self.db.get_job(job_id)
                if current and current["cancel_requested"]:
                    self.db.update_job(job_id, status="canceled", error="")
                else:
                    self.db.update_job(job_id, status="failed", error=str(exc)[:1000])
                self.events.publish({"type": "job", "job": self.db.get_job(job_id)})
            finally:
                self.queue.task_done()

    async def _run_job(self, job: dict) -> None:
        job_id = job["id"]
        self.db.update_job(job_id, status="starting", error="")
        self.events.publish({"type": "job", "job": self.db.get_job(job_id)})

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
        self.events.publish({"type": "job", "job": self.db.get_job(job_id)})

        completed = 0
        failed = 0
        errors: list[str] = []
        progress_lock = asyncio.Lock()

        async def download_one(position: int, item: dict) -> None:
            nonlocal completed, failed
            async with self._image_semaphore:
                current = self.db.get_job(job_id)
                if not current or current["cancel_requested"]:
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
                        self.events.publish(
                            {"type": "job", "job": self.db.get_job(job_id)}
                        )

        tasks = [
            asyncio.create_task(download_one(int(item.get("ordinal", index)), item))
            for index, item in enumerate(images, start=1)
        ]
        await asyncio.gather(*tasks)

        final = self.db.get_job(job_id) or {}
        if final.get("cancel_requested"):
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
        self.events.publish({"type": "job", "job": self.db.get_job(job_id)})

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
