from __future__ import annotations

import asyncio
import ctypes
import errno
import hashlib
import mimetypes
import os
import re
import shutil
import stat
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Awaitable, Callable, TypeVar
from urllib.parse import urljoin, urlsplit

import httpx
from PIL import Image

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the process lock only.
    fcntl = None  # type: ignore[assignment]

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
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_NOREPLACE_UNAVAILABLE = frozenset(
    {
        errno.EPERM,
        errno.EOPNOTSUPP,
        errno.ENOSYS,
        errno.EINVAL,
        errno.EXDEV,
    }
)
_HARDLINK_UNAVAILABLE = _RENAME_NOREPLACE_UNAVAILABLE | {errno.EACCES}
_FLOCK_UNAVAILABLE = frozenset(
    {
        errno.EACCES,
        errno.EPERM,
        errno.EOPNOTSUPP,
        errno.ENOSYS,
        errno.EINVAL,
    }
)
_POSE_PUBLISH_PROCESS_LOCK = threading.Lock()
_POSE_PUBLISH_LOCK_STATE = threading.local()
WorkItem = TypeVar("WorkItem")

try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _LIBC_RENAMEAT2 = _LIBC.renameat2
except (AttributeError, OSError):
    _LIBC_RENAMEAT2 = None
else:
    _LIBC_RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _LIBC_RENAMEAT2.restype = ctypes.c_int


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish source without replacing an existing target."""

    if _LIBC_RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS), source)
    ctypes.set_errno(0)
    result = _LIBC_RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(target),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    error = OSError(error_number, os.strerror(error_number), source)
    error.filename2 = target
    raise error


@contextmanager
def _pose_publish_lock(lock_directory: Path, should_cancel=None) -> Iterator[None]:
    """Serialize the replace fallback in this process and, when supported, beyond it."""

    lock_depth = getattr(_POSE_PUBLISH_LOCK_STATE, "depth", 0)
    if lock_depth:
        if should_cancel and should_cancel():
            raise PoseExportCanceled("Pose export canceled")
        _POSE_PUBLISH_LOCK_STATE.depth = lock_depth + 1
        try:
            yield
        finally:
            _POSE_PUBLISH_LOCK_STATE.depth = lock_depth
        return

    acquired_process_lock = False
    lock_file = None
    flock_acquired = False
    try:
        while not acquired_process_lock:
            if should_cancel and should_cancel():
                raise PoseExportCanceled("Pose export canceled")
            acquired_process_lock = _POSE_PUBLISH_PROCESS_LOCK.acquire(timeout=0.05)

        if should_cancel and should_cancel():
            raise PoseExportCanceled("Pose export canceled")
        if fcntl is not None:
            lock_path = lock_directory / ".galleryflow-publish.lock"
            lock_flags = os.O_CREAT | os.O_RDWR
            lock_flags |= getattr(os, "O_CLOEXEC", 0)
            lock_flags |= getattr(os, "O_NOFOLLOW", 0)
            lock_fd = os.open(lock_path, lock_flags, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
                    raise OSError(
                        errno.EINVAL,
                        "Pose publish lock is not a regular file",
                        lock_path,
                    )
                lock_file = os.fdopen(lock_fd, "r+b", buffering=0)
            except Exception:
                os.close(lock_fd)
                raise
            while True:
                if should_cancel and should_cancel():
                    raise PoseExportCanceled("Pose export canceled")
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    flock_acquired = True
                    break
                except BlockingIOError as exc:
                    if exc.errno in {errno.EAGAIN, errno.EWOULDBLOCK}:
                        time.sleep(0.05)
                        continue
                    if exc.errno in _FLOCK_UNAVAILABLE:
                        break
                    raise
                except OSError as exc:
                    if exc.errno in _FLOCK_UNAVAILABLE:
                        break
                    raise
        _POSE_PUBLISH_LOCK_STATE.depth = 1
        try:
            yield
        finally:
            _POSE_PUBLISH_LOCK_STATE.depth = 0
    finally:
        if lock_file is not None:
            if flock_acquired:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                lock_file.close()
            except OSError:
                pass
        if acquired_process_lock:
            _POSE_PUBLISH_PROCESS_LOCK.release()


class ActiveDownloadError(RuntimeError):
    def __init__(self, job: dict, message: str | None = None):
        super().__init__(
            message
            or "This gallery already has an active download for the selected profile"
        )
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
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_events_lock = threading.Lock()

    async def start(self) -> None:
        self._stopping = False
        self._client = httpx.AsyncClient(timeout=self.config.image_timeout)
        await asyncio.to_thread(self._cleanup_orphan_pose_staging)
        queued_job_ids = await asyncio.to_thread(self.db.queued_job_ids)
        for job_id in queued_job_ids:
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

    def _cancel_event(self, job_id: str) -> threading.Event:
        with self._cancel_events_lock:
            return self._cancel_events.setdefault(job_id, threading.Event())

    def _discard_cancel_event(self, job_id: str) -> None:
        with self._cancel_events_lock:
            self._cancel_events.pop(job_id, None)

    async def _run_bounded_image_work(
        self,
        work_items: list[tuple[int, WorkItem]],
        process: Callable[[int, WorkItem], Awaitable[bool]],
    ) -> None:
        """Run bounded per-job loops so one large gallery cannot monopolize waiters."""

        pending = deque(work_items)

        async def worker() -> None:
            while pending:
                position, item = pending.popleft()
                if not await process(position, item):
                    return

        worker_count = min(len(work_items), max(1, self.config.image_workers))
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        try:
            await asyncio.gather(*workers)
        except BaseException:
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

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
        self._cancel_event(job_id)
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
        self._cancel_event(job_id)
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
            self._discard_cancel_event(job_id)
            removed = {**job, "deleted": True}
            self._publish_job(removed)
            return self.public_job(removed)
        values = {"cancel_requested": 1}
        if job["status"] == "queued":
            values["status"] = "canceled"
        elif job["status"] not in terminal:
            values["status"] = "canceling"
        job = self.db.update_job(job_id, **values)
        self._cancel_event(job_id).set()
        self._publish_job(job)
        return self.public_job(job)

    async def _worker(self, index: int) -> None:
        while not self._stopping:
            job_id = await self.queue.get()
            cancel_event = self._cancel_event(job_id)
            try:
                job = await asyncio.to_thread(self.db.get_job, job_id)
                if not job or job["status"] == "canceled" or job["cancel_requested"]:
                    cancel_event.set()
                    continue
                if job.get("kind") == "pose_export":
                    await self._run_pose_export_job(job, cancel_event)
                else:
                    await self._run_job(job, cancel_event)
            except asyncio.CancelledError:
                raise
            except (
                Exception
            ) as exc:  # Defensive boundary: a worker must survive one bad job.
                values = (
                    {"status": "canceled", "error": ""}
                    if cancel_event.is_set()
                    else {"status": "failed", "error": str(exc)[:1000]}
                )
                summary = await asyncio.to_thread(
                    self.db.finish_job,
                    job_id,
                    values["status"],
                    values["error"],
                )
                self._publish_job(summary)
            finally:
                self._discard_cancel_event(job_id)
                self.queue.task_done()

    async def _run_job(
        self, job: dict, cancel_event: threading.Event | None = None
    ) -> None:
        job_id = job["id"]
        cancel_event = cancel_event or self._cancel_event(job_id)
        summary = await asyncio.to_thread(
            self.db.update_job, job_id, status="starting", error=""
        )
        self._publish_job(summary)

        detail = await self.scraper.gallery(job["gallery_url"])
        await asyncio.to_thread(
            self.db.register_gallery_images, detail["url"], detail["images"]
        )
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
        await asyncio.to_thread(self.db.create_job_items, job_id, images)

        profile = await asyncio.to_thread(self.db.get_profile, job["profile"])
        if not profile:
            raise ValueError(f"Unknown profile: {job['profile']}")
        profile_root = await asyncio.to_thread(
            confined_path, self.config.download_root, profile["directory"]
        )
        await asyncio.to_thread(profile_root.mkdir, parents=True, exist_ok=True)
        folder_base = safe_folder_name(job.get("title") or detail["title"])
        identity = gallery_key(detail["url"]).rsplit(":", 1)[-1]
        destination = await asyncio.to_thread(
            confined_path, profile_root, f"{folder_base}--{identity}"
        )
        await asyncio.to_thread(destination.mkdir, parents=True, exist_ok=True)
        existing = {
            canonicalize_url(url)
            for url in await asyncio.to_thread(
                self.db.image_statuses, job["profile"], detail["url"]
            )
        }
        downloaded_urls = set(existing)

        summary = await asyncio.to_thread(
            self.db.update_job,
            job_id,
            title=detail["title"],
            status="downloading",
            total=len(images),
            completed=0,
            failed=0,
            destination=str(destination.relative_to(self.config.download_root)),
            error="",
        )
        self._publish_job(summary)

        completed = 0
        failed = 0
        errors: list[str] = []
        progress_lock = asyncio.Lock()

        async def download_one(position: int, item: dict) -> bool:
            nonlocal completed, failed
            if cancel_event.is_set():
                return False

            item_status = "skipped"
            item_error = ""
            attempts = 0
            byte_count = 0
            relative_path = ""
            profile_name: str | None = None
            canonical = canonicalize_url(item["url"])
            async with self._image_semaphore:
                if cancel_event.is_set():
                    return False
                try:
                    if canonical not in existing:
                        attempts = 1
                        await asyncio.to_thread(
                            self.db.update_job_item,
                            job_id,
                            item["url"],
                            status="downloading",
                            attempts=attempts,
                            error="",
                        )
                        if cancel_event.is_set():
                            return False
                        final_path = await self._download_image(
                            item["url"],
                            destination,
                            position,
                            referer=detail["url"],
                        )
                        if cancel_event.is_set():
                            return False
                        relative_path = str(
                            final_path.relative_to(self.config.download_root)
                        )
                        file_stat = await asyncio.to_thread(final_path.stat)
                        byte_count = file_stat.st_size
                        item_status = "completed"
                        profile_name = job["profile"]
                except Exception as exc:
                    item_status = "failed"
                    item_error = str(exc)[:500]

            if cancel_event.is_set():
                return False
            async with progress_lock:
                if cancel_event.is_set():
                    return False
                completed_delta = int(item_status in {"completed", "skipped"})
                failed_delta = int(item_status == "failed")
                if failed_delta:
                    errors.append(f"Image {position}: {item_error}")
                summary, committed = await asyncio.to_thread(
                    self.db.record_job_item_result,
                    job_id,
                    item["url"],
                    status=item_status,
                    completed_delta=completed_delta,
                    failed_delta=failed_delta,
                    item_error=item_error,
                    job_error="; ".join(errors[-5:])[:1000],
                    attempts=attempts,
                    byte_count=byte_count,
                    relative_path=relative_path,
                    profile=profile_name,
                    gallery_url=(detail["url"] if profile_name is not None else None),
                )
                if not committed:
                    cancel_event.set()
                    self._publish_job(summary)
                    return False
                completed += completed_delta
                failed += failed_delta
                if completed_delta:
                    downloaded_urls.add(canonical)
                self._publish_job(summary)
            return True

        work_items = [
            (int(item.get("ordinal", index)), item)
            for index, item in enumerate(images, start=1)
        ]
        await self._run_bounded_image_work(work_items, download_one)

        if cancel_event.is_set():
            status = "canceled"
        elif completed == len(images) and failed == 0:
            status = "completed"
            if set(available).issubset(downloaded_urls):
                await asyncio.to_thread(
                    self.db.add_history,
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
        summary = await asyncio.to_thread(
            self.db.finish_job,
            job_id,
            status,
            "; ".join(errors[-5:])[:1000],
        )
        self._publish_job(summary)

    async def _run_pose_export_job(
        self, job: dict, cancel_event: threading.Event | None = None
    ) -> None:
        job_id = job["id"]
        cancel_event = cancel_event or self._cancel_event(job_id)
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
        await asyncio.to_thread(self.db.create_job_items, job_id, images)

        pose_root = self.config.pose_root_path
        await asyncio.to_thread(pose_root.mkdir, parents=True, exist_ok=True)
        staging = await asyncio.to_thread(
            confined_path, pose_root, f".galleryflow-tmp/{job_id}"
        )
        await asyncio.to_thread(staging.mkdir, parents=True, exist_ok=True)
        summary = await asyncio.to_thread(
            self.db.update_job,
            job_id,
            status="downloading",
            total=len(images),
            completed=0,
            failed=0,
            destination=str(pose_root),
            error="",
        )
        self._publish_job(summary)

        completed = 0
        failed = 0
        errors: list[str] = []
        progress_lock = asyncio.Lock()
        cached: dict[str, Path] = {}
        previous_items = {
            item["image_url"]: item
            for item in await asyncio.to_thread(self.db.list_job_items, job_id)
        }

        async def fetch_one(position: int, url: str) -> bool:
            nonlocal completed, failed
            if cancel_event.is_set():
                return False

            item_status = "completed"
            item_error = ""
            attempts: int | None = None
            byte_count = 0
            relative_path = ""
            previous_path: Path | None = None
            async with self._image_semaphore:
                if cancel_event.is_set():
                    return False
                try:
                    previous = previous_items.get(url) or {}
                    if previous.get("status") == "completed" and previous.get(
                        "relative_path"
                    ):
                        candidate = await asyncio.to_thread(
                            confined_path,
                            pose_root,
                            str(previous["relative_path"]),
                        )
                        if await asyncio.to_thread(candidate.is_file):
                            previous_path = candidate
                            relative_path = str(previous["relative_path"])
                            byte_count = int(previous.get("byte_count") or 0)
                    if previous_path is None:
                        attempts = 1
                        await asyncio.to_thread(
                            self.db.update_job_item,
                            job_id,
                            url,
                            status="downloading",
                            attempts=attempts,
                            error="",
                        )
                        if cancel_event.is_set():
                            return False
                        previous_path = await self._download_image(
                            url, staging, position, referer=job["gallery_url"]
                        )
                        if cancel_event.is_set():
                            return False
                        relative_path = str(previous_path.relative_to(pose_root))
                        file_stat = await asyncio.to_thread(previous_path.stat)
                        byte_count = file_stat.st_size
                except Exception as exc:
                    item_status = "failed"
                    item_error = str(exc)[:500]

            if cancel_event.is_set():
                return False
            async with progress_lock:
                if cancel_event.is_set():
                    return False
                completed_delta = int(item_status == "completed")
                failed_delta = int(item_status == "failed")
                if failed_delta:
                    errors.append(f"Image {position}: {item_error}")
                summary, committed = await asyncio.to_thread(
                    self.db.record_job_item_result,
                    job_id,
                    url,
                    status=item_status,
                    completed_delta=completed_delta,
                    failed_delta=failed_delta,
                    item_error=item_error,
                    job_error="; ".join(errors[-5:])[:1000],
                    attempts=attempts,
                    byte_count=byte_count,
                    relative_path=relative_path,
                )
                if not committed:
                    cancel_event.set()
                    self._publish_job(summary)
                    return False
                completed += completed_delta
                failed += failed_delta
                if previous_path is not None:
                    cached[canonicalize_url(url)] = previous_path
                self._publish_job(summary)
            return True

        await self._run_bounded_image_work(
            list(enumerate(ordered_urls, start=1)),
            fetch_one,
        )

        if cancel_event.is_set():
            status = "canceled"
        elif failed:
            status = "failed"
        else:

            def should_cancel_materialization() -> bool:
                return self._stopping or cancel_event.is_set()

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
                if cancel_event.is_set():
                    status = "canceled"
                else:
                    await asyncio.to_thread(
                        self.db.record_pose_outputs,
                        job["gallery_url"],
                        job["profile"],
                        job_id,
                        targets,
                    )
                    status = "completed"
            except PoseExportCanceled:
                status = "canceled"
            except Exception as exc:
                status = "failed"
                errors.append(str(exc))
        summary = await asyncio.to_thread(
            self.db.finish_job,
            job_id,
            status,
            "; ".join(errors[-5:])[:1000],
        )
        self._publish_job(summary)
        await asyncio.to_thread(shutil.rmtree, staging, ignore_errors=True)
        await asyncio.to_thread(self.db.clear_job_item_paths, job_id)

    def _materialize_pose_pairs(
        self,
        gallery_url: str,
        pose_root: Path,
        controls: dict[str, str],
        targets: list[dict],
        cached: dict[str, Path],
        should_cancel=None,
    ) -> None:
        with _pose_publish_lock(pose_root, should_cancel):
            self._materialize_pose_pairs_locked(
                gallery_url,
                pose_root,
                controls,
                targets,
                cached,
                should_cancel,
            )

    def _materialize_pose_pairs_locked(
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
            plans.append((target_source, target_path, control_source, control_path))

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
                raise ValueError(
                    "Pose output path escapes the configured root"
                ) from exc
            if existing != desired_target:
                pose_name = existing.parent.parent.name
                raise FileExistsError(
                    f"{stem} was already exported under pose '{pose_name}'"
                )

    @classmethod
    def _copy_without_overwrite(
        cls, source: Path, target: Path, should_cancel=None
    ) -> bool:
        def existing_result() -> bool:
            if os.path.lexists(target):
                if target.is_file() and not target.is_symlink():
                    try:
                        if cls._files_identical(source, target):
                            return False
                    except FileNotFoundError:
                        pass
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")

        if os.path.lexists(target):
            return existing_result()
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
                target_file.flush()
                os.fsync(target_file.fileno())
            if should_cancel and should_cancel():
                raise PoseExportCanceled("Pose export canceled")
            try:
                os.link(temporary, target)
                return True
            except FileExistsError:
                return existing_result()
            except OSError as exc:
                if exc.errno not in _HARDLINK_UNAVAILABLE:
                    raise

            if should_cancel and should_cancel():
                raise PoseExportCanceled("Pose export canceled")
            try:
                _rename_noreplace(temporary, target)
                return True
            except FileExistsError:
                return existing_result()
            except OSError as exc:
                if exc.errno not in _RENAME_NOREPLACE_UNAVAILABLE:
                    raise

            # Some Unraid shfs and SMB/CIFS mounts reject both hard links and
            # renameat2(RENAME_NOREPLACE). The complete, fsynced staging file
            # can still be published safely for GalleryFlow writers by
            # serializing, checking once more, and using a same-directory
            # replace. Keep the sidecar lock file: unlinking it would let
            # waiters lock different inodes.
            with _pose_publish_lock(target.parent, should_cancel):
                if os.path.lexists(target):
                    return existing_result()
                if should_cancel and should_cancel():
                    raise PoseExportCanceled("Pose export canceled")
                os.replace(temporary, target)
                return True
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
                        final_path = await asyncio.to_thread(
                            confined_path,
                            destination,
                            f"{position:04d}{extension}",
                        )
                        part_path = await asyncio.to_thread(
                            confined_path,
                            destination,
                            f".{position:04d}{extension}.part",
                        )
                        total = 0
                        try:
                            output = await asyncio.to_thread(part_path.open, "wb")
                            try:
                                async for chunk in response.aiter_bytes(1024 * 1024):
                                    total += len(chunk)
                                    if total > self.config.max_image_bytes:
                                        raise RuntimeError(
                                            "Image exceeds configured size limit"
                                        )
                                    await asyncio.to_thread(output.write, chunk)
                            finally:
                                await asyncio.to_thread(output.close)
                            if total == 0:
                                raise RuntimeError("Image response was empty")
                            await asyncio.to_thread(self._verify_image, part_path)
                            await asyncio.to_thread(os.replace, part_path, final_path)
                        finally:
                            await asyncio.to_thread(part_path.unlink, missing_ok=True)
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
