from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urljoin

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import AppConfig
from .db import Database, utc_now
from .security import (
    canonicalize_url,
    gallery_key,
    validate_public_media_url,
    validate_source_url,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
ACTIVE_STATUSES = {"queued", "preparing", "scanning", "pausing", "canceling"}
TERMINAL_STATUSES = {"completed", "completed_with_errors", "failed", "canceled"}
REVIEW_STATES = {"pending", "accepted", "rejected"}
ANALYZER_VERSION = "hybrid-spatial-pyramid-v1+rtmo-l-geometry-v1+phash64-gate-v1"
LEGACY_RANKING_VERSION = "appearance-v1"
CURRENT_RANKING_VERSION = "pose-first-v1"
CORPUS_BACKFILL_VERSION = "top-matches-v1"
CORPUS_SCAN_GUARD_VERSION = "pre-corpus-scans-v1"
POSE_MATCH_FLOOR = 0.55
EXACT_HASH_MAX_DISTANCE = 8
MAX_SCAN_PAGES = 500
MAX_EXTEND_PAGES = 50

try:
    from .vision import perceptual_hash_bytes as _perceptual_hash_bytes
except ImportError:  # pragma: no cover - vision is an optional Finder dependency
    _perceptual_hash_bytes = None


class FinderNotFound(LookupError):
    pass


class FinderConflict(RuntimeError):
    pass


class FinderUnavailable(RuntimeError):
    pass


class _FinderPaused(RuntimeError):
    pass


class _FinderCanceled(RuntimeError):
    pass


MediaFetcher = Callable[[str, str], Awaitable[bytes]]


@dataclass(frozen=True, slots=True)
class _ImageDescriptor:
    appearance: np.ndarray
    metadata: dict[str, Any]


class FinderService:
    """Persistent visual gallery scanner, isolated from download jobs and history."""

    def __init__(
        self,
        config: AppConfig,
        database: Database,
        scraper: Any,
        events: Any,
        *,
        encoder: Any | None = None,
        pose_estimator: Any | None = None,
        media_fetcher: MediaFetcher | None = None,
    ) -> None:
        self.config = config
        self.database = database
        self.scraper = scraper
        self.events = events
        self.encoder = encoder
        self.pose_estimator = pose_estimator
        self.media_fetcher = media_fetcher
        assert config.finder_examples_root is not None
        assert config.finder_model_path is not None
        assert config.finder_pose_model_path is not None
        self.examples_root = config.finder_examples_root.resolve()
        self.model_path = config.finder_model_path.resolve()
        self.pose_model_path = config.finder_pose_model_path.resolve()
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._client: httpx.AsyncClient | None = None
        self._network_semaphore = asyncio.Semaphore(config.finder_network_workers)
        self._embedding_semaphore = asyncio.Semaphore(1)
        self._prepare_lock = asyncio.Lock()
        self._rate_lock = asyncio.Lock()
        self._next_request_at = 0.0
        self._lock = threading.RLock()
        self._stopping = False
        self._available = True
        self._ready = False
        self._pose_ready = False
        self._prepare_error = ""
        self._pose_error = ""
        self._model_key = ""
        self._last_event_at: dict[str, float] = {}
        self._cache_writes_since_prune = 0

    def ensure_schema(self) -> None:
        with self._lock, self.database.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS finder_scans (
                    id TEXT PRIMARY KEY,
                    example_directory TEXT NOT NULL,
                    reference_fingerprint TEXT NOT NULL DEFAULT '',
                    reference_model_key TEXT NOT NULL DEFAULT '',
                    reference_ready INTEGER NOT NULL DEFAULT 0,
                    reference_count INTEGER NOT NULL DEFAULT 0,
                    pose_tag_id INTEGER NOT NULL,
                    pose_tag_label TEXT NOT NULL,
                    pose_tag_slug TEXT NOT NULL,
                    pose_default_role TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    next_url TEXT,
                    page_limit INTEGER NOT NULL,
                    pages_completed INTEGER NOT NULL DEFAULT 0,
                    minimum_score REAL NOT NULL,
                    ranking_version TEXT NOT NULL DEFAULT 'pose-first-v1',
                    status TEXT NOT NULL,
                    total_galleries INTEGER NOT NULL DEFAULT 0,
                    processed_galleries INTEGER NOT NULL DEFAULT 0,
                    processed_images INTEGER NOT NULL DEFAULT 0,
                    failed_galleries INTEGER NOT NULL DEFAULT 0,
                    corpus_search_complete INTEGER NOT NULL DEFAULT 0,
                    corpus_images_scored INTEGER NOT NULL DEFAULT 0,
                    corpus_galleries_scored INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    pause_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    finished_at TEXT
                );
                CREATE TABLE IF NOT EXISTS finder_scan_references (
                    scan_id TEXT NOT NULL,
                    example_key TEXT NOT NULL,
                    mirror_index INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    dimensions INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (scan_id, example_key, mirror_index),
                    FOREIGN KEY (scan_id) REFERENCES finder_scans(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS finder_embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_key TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    include_mirror INTEGER NOT NULL,
                    rows INTEGER NOT NULL,
                    dimensions INTEGER NOT NULL,
                    embedding BLOB NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS finder_results (
                    id TEXT PRIMARY KEY,
                    scan_id TEXT NOT NULL,
                    gallery_key TEXT NOT NULL,
                    gallery_url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    thumbnail_remote_url TEXT NOT NULL DEFAULT '',
                    best_image_url TEXT NOT NULL DEFAULT '',
                    best_preview_remote_url TEXT NOT NULL DEFAULT '',
                    best_ordinal INTEGER,
                    score REAL NOT NULL DEFAULT 0,
                    ranking_tier INTEGER NOT NULL DEFAULT 1,
                    matches_json TEXT NOT NULL DEFAULT '[]',
                    images_scored INTEGER NOT NULL DEFAULT 0,
                    online_scanned INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    review TEXT NOT NULL DEFAULT 'pending',
                    error TEXT NOT NULL DEFAULT '',
                    discovered_order INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (scan_id, gallery_key),
                    FOREIGN KEY (scan_id) REFERENCES finder_scans(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS finder_corpus_galleries (
                    gallery_key TEXT PRIMARY KEY,
                    gallery_url TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    thumbnail_remote_url TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT 'partial'
                        CHECK (state IN ('complete', 'partial')),
                    image_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS finder_corpus_images (
                    gallery_key TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    preview_remote_url TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (gallery_key, source_key),
                    FOREIGN KEY (gallery_key)
                        REFERENCES finder_corpus_galleries(gallery_key)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS finder_corpus_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_finder_scans_updated
                    ON finder_scans(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_finder_results_score
                    ON finder_results(scan_id, score DESC, discovered_order);
                CREATE INDEX IF NOT EXISTS idx_finder_results_review
                    ON finder_results(scan_id, review, score DESC);
                CREATE INDEX IF NOT EXISTS idx_finder_embedding_cache_source
                    ON finder_embedding_cache(
                        model_key, source_key, include_mirror
                    );
                CREATE INDEX IF NOT EXISTS idx_finder_corpus_images_source
                    ON finder_corpus_images(source_key);
                CREATE INDEX IF NOT EXISTS idx_finder_corpus_images_scan
                    ON finder_corpus_images(
                        gallery_key, ordinal, source_key
                    );
                CREATE INDEX IF NOT EXISTS idx_finder_corpus_galleries_state
                    ON finder_corpus_galleries(state, updated_at DESC);
                """
            )
            migrations = {
                "finder_scans": {
                    "reference_model_key": "TEXT NOT NULL DEFAULT ''",
                    "ranking_version": ("TEXT NOT NULL DEFAULT 'appearance-v1'"),
                    "corpus_search_complete": "INTEGER NOT NULL DEFAULT 0",
                    "corpus_images_scored": "INTEGER NOT NULL DEFAULT 0",
                    "corpus_galleries_scored": "INTEGER NOT NULL DEFAULT 0",
                },
                "finder_scan_references": {
                    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                },
                "finder_embedding_cache": {
                    "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                },
                "finder_results": {
                    "matches_json": "TEXT NOT NULL DEFAULT '[]'",
                    "ranking_tier": "INTEGER NOT NULL DEFAULT 1",
                    "online_scanned": "INTEGER NOT NULL DEFAULT 1",
                },
            }
            for table, additions in migrations.items():
                columns = {
                    str(row[1])
                    for row in db.execute(f"PRAGMA table_info({table})").fetchall()
                }
                for column, definition in additions.items():
                    if column not in columns:
                        db.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
                        )
            db.execute(
                """CREATE INDEX IF NOT EXISTS idx_finder_results_rank
                   ON finder_results(
                       scan_id, ranking_tier DESC, score DESC, discovered_order
                   )"""
            )
            db.execute(
                """CREATE INDEX IF NOT EXISTS idx_finder_results_review_rank
                   ON finder_results(
                       scan_id, review, ranking_tier DESC, score DESC
                   )"""
            )
            db.execute(
                """CREATE INDEX IF NOT EXISTS idx_finder_results_online
                   ON finder_results(scan_id, online_scanned, status)"""
            )
            scan_guard = db.execute(
                """SELECT value FROM finder_corpus_meta
                   WHERE key = 'scan-guard'"""
            ).fetchone()
            if not scan_guard or scan_guard["value"] != CORPUS_SCAN_GUARD_VERSION:
                # Existing and paused scans predate corpus pre-search. Mark
                # them searched so an upgrade cannot inject local results into
                # the middle of durable pagination/review state.
                db.execute(
                    """UPDATE finder_scans
                       SET corpus_search_complete = 1"""
                )
                db.execute(
                    """INSERT INTO finder_corpus_meta(key, value, updated_at)
                       VALUES ('scan-guard', ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                           value = excluded.value,
                           updated_at = excluded.updated_at""",
                    (CORPUS_SCAN_GUARD_VERSION, utc_now()),
                )
            backfill = db.execute(
                """SELECT value FROM finder_corpus_meta
                   WHERE key = 'historical-backfill'"""
            ).fetchone()
            if not backfill or backfill["value"] != CORPUS_BACKFILL_VERSION:
                # The marker and backfill commit together. Rows are streamed so
                # historical skeleton overlays do not inflate startup memory.
                self._backfill_corpus_associations(db)
                db.execute(
                    """INSERT INTO finder_corpus_meta(key, value, updated_at)
                       VALUES ('historical-backfill', ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                           value = excluded.value,
                           updated_at = excluded.updated_at""",
                    (CORPUS_BACKFILL_VERSION, utc_now()),
                )
            db.execute(
                """UPDATE finder_scans
                   SET status = 'canceled', finished_at = ?, updated_at = ?
                   WHERE cancel_requested = 1 AND status IN
                       ('queued','preparing','scanning','pausing','canceling')""",
                (utc_now(), utc_now()),
            )
            db.execute(
                """UPDATE finder_scans
                   SET status = 'paused', updated_at = ?
                   WHERE pause_requested = 1 AND cancel_requested = 0
                     AND status IN ('preparing','scanning','pausing')""",
                (utc_now(),),
            )
            db.execute(
                """UPDATE finder_scans
                   SET status = 'queued', updated_at = ?
                   WHERE pause_requested = 0 AND cancel_requested = 0
                     AND status IN ('preparing','scanning','pausing','canceling')""",
                (utc_now(),),
            )
            now = utc_now()
            db.execute(
                """UPDATE finder_scans
                   SET status = 'failed', finished_at = COALESCE(finished_at, ?),
                       updated_at = ?, error = ?
                   WHERE ranking_version != ?
                     AND status IN (
                         'queued','preparing','scanning','pausing','paused',
                         'canceling'
                     )""",
                (
                    now,
                    now,
                    "This Finder scan uses legacy appearance ranking and cannot "
                    "continue after the pose-first ranking upgrade. Start a new scan.",
                    CURRENT_RANKING_VERSION,
                ),
            )

    @staticmethod
    def _remote_source_key(url: str) -> str:
        """Return the exact cache namespace used for a remote preview image."""

        return f"url:{canonicalize_url(url)}"

    def _backfill_corpus_associations(self, db: Any) -> None:
        """Recover partial gallery membership from durable historical top matches.

        Finder has always persisted only the best three images for each result,
        so these rows can seed useful local searches but cannot prove that a
        gallery is complete. Complete galleries are intentionally skipped:
        replaying an old result after an online refresh must not resurrect stale
        preview URLs that the refresh removed.
        """

        complete_keys = {
            str(row["gallery_key"])
            for row in db.execute(
                """SELECT gallery_key FROM finder_corpus_galleries
                   WHERE state = 'complete'"""
            ).fetchall()
        }
        cursor = db.execute(
            """SELECT gallery_key, gallery_url, title, thumbnail_remote_url,
                      matches_json, best_image_url, best_preview_remote_url,
                      best_ordinal, created_at, updated_at
               FROM finder_results
               ORDER BY id"""
        )
        while True:
            rows = cursor.fetchmany(32)
            if not rows:
                break
            for row in rows:
                if str(row["gallery_key"]) in complete_keys:
                    continue
                self._backfill_corpus_result(db, row)
        db.execute(
            """UPDATE finder_corpus_galleries
               SET image_count = (
                   SELECT COUNT(*) FROM finder_corpus_images
                   WHERE gallery_key = finder_corpus_galleries.gallery_key
               )
               WHERE state = 'partial'"""
        )

    def _backfill_corpus_result(self, db: Any, row: Any) -> None:
        try:
            decoded = json.loads(str(row["matches_json"] or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = []
        raw_matches = decoded if isinstance(decoded, list) else []
        if not raw_matches and row["best_image_url"] and row["best_preview_remote_url"]:
            raw_matches = [
                {
                    "image_url": row["best_image_url"],
                    "preview_remote_url": row["best_preview_remote_url"],
                    "ordinal": row["best_ordinal"] or 0,
                }
            ]
        recovered: dict[str, tuple[str, str, int]] = {}
        for match in raw_matches:
            if not isinstance(match, dict):
                continue
            try:
                preview = str(
                    match.get("preview_remote_url") or match.get("preview_url") or ""
                )
                image_url = str(match.get("image_url") or match.get("url") or "")
                if not preview or not image_url:
                    continue
                source_key = self._remote_source_key(preview)
                ordinal = int(match.get("ordinal") or 0)
            except (TypeError, ValueError):
                continue
            recovered[source_key] = (image_url, preview, max(0, ordinal))
        if not recovered:
            return
        now = utc_now()
        created_at = str(row["created_at"] or now)
        updated_at = str(row["updated_at"] or now)
        db.execute(
            """INSERT INTO finder_corpus_galleries(
                   gallery_key, gallery_url, title, thumbnail_remote_url,
                   state, image_count, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'partial', ?, ?, ?)
               ON CONFLICT(gallery_key) DO UPDATE SET
                   gallery_url = excluded.gallery_url,
                   title = CASE
                       WHEN excluded.title != '' THEN excluded.title
                       ELSE finder_corpus_galleries.title
                   END,
                   thumbnail_remote_url = CASE
                       WHEN excluded.thumbnail_remote_url != ''
                           THEN excluded.thumbnail_remote_url
                       ELSE finder_corpus_galleries.thumbnail_remote_url
                   END,
                   image_count = MAX(
                       finder_corpus_galleries.image_count,
                       excluded.image_count
                   ),
                   updated_at = MAX(
                       finder_corpus_galleries.updated_at,
                       excluded.updated_at
                   )""",
            (
                row["gallery_key"],
                row["gallery_url"],
                str(row["title"] or ""),
                str(row["thumbnail_remote_url"] or ""),
                len(recovered),
                created_at,
                updated_at,
            ),
        )
        db.executemany(
            """INSERT INTO finder_corpus_images(
                   gallery_key, source_key, image_url, preview_remote_url,
                   ordinal, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(gallery_key, source_key) DO UPDATE SET
                   image_url = excluded.image_url,
                   preview_remote_url = excluded.preview_remote_url,
                   ordinal = excluded.ordinal,
                   updated_at = excluded.updated_at""",
            [
                (
                    row["gallery_key"],
                    source_key,
                    image_url,
                    preview,
                    ordinal,
                    created_at,
                    updated_at,
                )
                for source_key, (image_url, preview, ordinal) in recovered.items()
            ],
        )

    async def start(self) -> None:
        self._stopping = False
        self.ensure_schema()
        self._client = httpx.AsyncClient(timeout=self.config.image_timeout)
        created_default_encoder = False
        if self.encoder is None:
            try:
                from .vision import DinoV2Encoder

                self.encoder = DinoV2Encoder(
                    self.model_path,
                    execution_provider=self.config.finder_execution_provider,
                )
                created_default_encoder = True
            except Exception as exc:
                self._available = False
                self._prepare_error = str(exc)[:1000]
        # Injected encoders are commonly tests or custom integrations. Only pair
        # the built-in DINO encoder with the built-in RTMO estimator implicitly;
        # callers can still inject both explicitly.
        if (
            self.pose_estimator is None
            and self.config.finder_pose_enabled
            and created_default_encoder
        ):
            try:
                from .pose_vision import RTMOPoseEstimator

                self.pose_estimator = RTMOPoseEstimator(
                    self.pose_model_path,
                    execution_provider=self.config.finder_execution_provider,
                    max_image_bytes=self.config.finder_max_image_bytes,
                    max_image_pixels=self.config.finder_max_image_pixels,
                )
            except Exception as exc:
                # Pose is an enhancement. Spatial matching remains usable when
                # RTMO cannot be initialized or downloaded.
                self._pose_error = str(exc)[:1000]
        if self._available:
            for scan_id in self._queued_scan_ids():
                self.queue.put_nowait(scan_id)
            self._workers = [
                asyncio.create_task(self._worker(index), name=f"finder-worker-{index}")
                for index in range(self.config.finder_workers)
            ]

    async def _ensure_encoder_ready(self) -> None:
        if not self._available or self.encoder is None:
            raise FinderUnavailable(
                self._prepare_error or "Finder vision support is unavailable"
            )
        async with self._prepare_lock:
            if not self._ready:
                try:
                    await self.encoder.prepare()
                    self._ready = True
                    self._prepare_error = ""
                except Exception as exc:
                    self._ready = False
                    self._prepare_error = str(exc)[:1000]
                    raise FinderUnavailable(
                        self._prepare_error or "Finder model preparation failed"
                    ) from exc
            if self.pose_estimator is not None and not self._pose_ready:
                try:
                    await self.pose_estimator.prepare()
                    self._pose_ready = True
                    self._pose_error = ""
                except Exception as exc:
                    self._pose_ready = False
                    self._pose_error = str(exc)[:1000]
            model_key = self._encoder_key()
            if model_key != self._model_key:
                self._model_key = model_key
                await asyncio.to_thread(
                    self._prune_embedding_cache, purge_stale_models=True
                )

    async def stop(self) -> None:
        self._stopping = True
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _encoder_key(self) -> str:
        supplied = getattr(self.encoder, "model_key", None)
        if isinstance(supplied, str) and supplied:
            identity = supplied
        else:
            try:
                details = self.model_path.stat()
                identity = f"{self.model_path}:{details.st_size}:{details.st_mtime_ns}"
            except OSError:
                identity = (
                    f"{type(self.encoder).__module__}.{type(self.encoder).__name__}"
                )
        if self.pose_estimator is None:
            pose_identity = "disabled"
        else:
            pose_key = getattr(self.pose_estimator, "model_key", "rtmo")
            pose_identity = (
                str(pose_key) if self._pose_ready else f"{pose_key}:unavailable"
            )
        return f"{identity}:{ANALYZER_VERSION}:pose={pose_identity}"

    def status(self) -> dict[str, Any]:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) AS count FROM finder_scans GROUP BY status"
            ).fetchall()
        counts = {row["status"]: int(row["count"]) for row in rows}
        encoder_provider = getattr(self.encoder, "provider_status", None)
        pose_provider = getattr(self.pose_estimator, "provider_status", None)
        encoder_provider_status = (
            encoder_provider() if callable(encoder_provider) else {}
        )
        pose_provider_status = pose_provider() if callable(pose_provider) else {}
        active_devices = [
            str(value)
            for value in (
                encoder_provider_status.get("active"),
                pose_provider_status.get("active"),
            )
            if value
        ]
        return {
            "available": self._available,
            "model_ready": self._ready,
            "error": self._prepare_error,
            "model_path": str(self.model_path),
            "pose_model_path": str(self.pose_model_path),
            "pose_enabled": self.pose_estimator is not None,
            "pose_ready": self._pose_ready,
            "pose_error": self._pose_error,
            "execution_provider": self.config.finder_execution_provider,
            "providers": {
                "appearance": encoder_provider_status,
                "pose": pose_provider_status,
            },
            "device": ", ".join(dict.fromkeys(active_devices)),
            "backend": "spatial DINOv2 + RTMO-L"
            if self.pose_estimator
            else "spatial DINOv2",
            "analyzer_version": ANALYZER_VERSION,
            "ranking_version": CURRENT_RANKING_VERSION,
            "folder_root": str(self.examples_root),
            # Kept for clients written against the 2.2 API.
            "examples_root": str(self.examples_root),
            "queue_depth": self.queue.qsize(),
            "active": sum(counts.get(item, 0) for item in ACTIVE_STATUSES),
            "paused": counts.get("paused", 0),
        }

    def corpus_status(self) -> dict[str, int]:
        """Summarize reusable gallery membership and descriptor storage."""

        with self._lock, self.database.connect() as db:
            galleries = db.execute(
                """SELECT COUNT(*) AS galleries,
                          COALESCE(SUM(state = 'complete'), 0) AS complete,
                          COALESCE(SUM(state = 'partial'), 0) AS partial
                   FROM finder_corpus_galleries"""
            ).fetchone()
            images = int(
                db.execute("SELECT COUNT(*) FROM finder_corpus_images").fetchone()[0]
            )
            cache = db.execute(
                """SELECT COUNT(*) AS entries,
                          COALESCE(
                              SUM(length(embedding) + length(metadata_json)), 0
                          ) AS stored_bytes
                   FROM finder_embedding_cache"""
            ).fetchone()
            model_key = self._model_key
            if not model_key:
                latest = db.execute(
                    """SELECT reference_model_key FROM finder_scans
                       WHERE reference_model_key != '' AND ranking_version = ?
                       ORDER BY updated_at DESC LIMIT 1""",
                    (CURRENT_RANKING_VERSION,),
                ).fetchone()
                model_key = str(latest["reference_model_key"]) if latest else ""
            ready = 0
            if model_key:
                ready = int(
                    db.execute(
                        """SELECT COUNT(*)
                           FROM finder_corpus_images i
                           WHERE EXISTS (
                               SELECT 1 FROM finder_embedding_cache c
                               WHERE c.model_key = ?
                                 AND c.source_key = i.source_key
                                 AND c.include_mirror = 0
                           )""",
                        (model_key,),
                    ).fetchone()[0]
                )
        return {
            "galleries": int(galleries["galleries"]),
            "images": images,
            "complete": int(galleries["complete"]),
            "partial": int(galleries["partial"]),
            "ready": ready,
            "cache_entries": int(cache["entries"]),
            "cache_bytes": max(0, int(cache["stored_bytes"] or 0)),
            "max_cache_entries": int(self.config.finder_cache_max_entries),
            "max_cache_bytes": int(self.config.finder_cache_max_bytes),
        }

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )

    def _open_example_directory(self, directory: Path) -> int:
        """Open a directory one component at a time without following links."""

        try:
            relative = directory.relative_to(self.examples_root)
        except ValueError as exc:
            raise ValueError(
                "Example directory escapes the Finder folder root"
            ) from exc
        try:
            descriptor = os.open(self.examples_root, self._directory_flags())
            for part in relative.parts:
                try:
                    child = os.open(
                        part,
                        self._directory_flags(),
                        dir_fd=descriptor,
                    )
                except Exception:
                    os.close(descriptor)
                    raise
                os.close(descriptor)
                descriptor = child
            return descriptor
        except OSError as exc:
            raise ValueError("Could not safely open the example directory") from exc

    @staticmethod
    def _directory_summary(descriptor: int) -> tuple[int, bool]:
        image_count = 0
        has_children = False
        for name in os.listdir(descriptor):
            try:
                details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except OSError:
                continue
            if stat.S_ISDIR(details.st_mode):
                has_children = True
            elif (
                stat.S_ISREG(details.st_mode)
                and Path(name).suffix.lower() in IMAGE_EXTENSIONS
            ):
                image_count += 1
        return image_count, has_children

    def folders(self, value: str = ".") -> dict[str, Any]:
        """Return one safe, shallow level for the optional folder browser."""

        current, normalized = self._resolve_example_directory(value)
        descriptor = self._open_example_directory(current)
        try:
            image_count, has_children = self._directory_summary(descriptor)
            items: list[dict[str, Any]] = []
            for name in sorted(
                os.listdir(descriptor), key=lambda item: (item.casefold(), item)
            ):
                try:
                    details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                    if not stat.S_ISDIR(details.st_mode):
                        continue
                    child_descriptor = os.open(
                        name,
                        self._directory_flags(),
                        dir_fd=descriptor,
                    )
                except OSError:
                    # This also excludes symlinked and inaccessible directories.
                    continue
                try:
                    child_images, child_has_children = self._directory_summary(
                        child_descriptor
                    )
                finally:
                    os.close(child_descriptor)
                path = name if normalized == "." else f"{normalized}/{name}"
                items.append(
                    {
                        "path": path,
                        "absolute_path": str(self.examples_root / path),
                        "name": name,
                        "image_count": child_images,
                        "has_children": child_has_children,
                    }
                )
        finally:
            os.close(descriptor)
        parent = None
        if normalized != ".":
            parent_path = Path(normalized).parent
            parent = "." if parent_path == Path(".") else parent_path.as_posix()
        return {
            "root": str(self.examples_root),
            "path": normalized,
            "parent": parent,
            "current": {
                "path": normalized,
                "absolute_path": str(current),
                "name": "Library root" if normalized == "." else current.name,
                "image_count": image_count,
                "has_children": has_children,
            },
            "items": items,
        }

    def _resolve_example_directory(self, value: str) -> tuple[Path, str]:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("Example directory is required")
        supplied = Path(raw)
        if ".." in supplied.parts:
            raise ValueError("Example directory cannot contain '..'")
        if supplied.is_absolute():
            try:
                relative = supplied.relative_to(self.examples_root)
            except ValueError as exc:
                raise ValueError(
                    "Absolute example directory must stay inside the Finder folder root"
                ) from exc
        else:
            relative = supplied
        lexical = self.examples_root / relative
        current = self.examples_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlinked example directories are not allowed")
        try:
            resolved = lexical.resolve(strict=True)
        except OSError as exc:
            raise ValueError("Example directory does not exist") from exc
        if (
            resolved != self.examples_root
            and self.examples_root not in resolved.parents
        ):
            raise ValueError("Example directory escapes the Finder folder root")
        if not resolved.is_dir():
            raise ValueError("Example directory does not exist")
        normalized = (
            "."
            if resolved == self.examples_root
            else resolved.relative_to(self.examples_root).as_posix()
        )
        return resolved, normalized

    def _example_files(self, directory: Path) -> list[Path]:
        files: list[Path] = []
        descriptor = self._open_example_directory(directory)
        try:
            for name in sorted(
                os.listdir(descriptor), key=lambda item: (item.casefold(), item)
            ):
                try:
                    details = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError(f"Could not inspect example file: {name}") from exc
                if stat.S_ISLNK(details.st_mode):
                    raise ValueError("Symlinked example images are not allowed")
                path = directory / name
                if (
                    stat.S_ISREG(details.st_mode)
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    if details.st_size > self.config.finder_max_image_bytes:
                        raise ValueError(f"Example image is too large: {path.name}")
                    files.append(path)
        finally:
            os.close(descriptor)
        if not files:
            raise ValueError("Example directory contains no supported images")
        if len(files) > self.config.finder_max_examples:
            raise ValueError(
                f"Example directory exceeds the {self.config.finder_max_examples} image limit"
            )
        return files

    def _read_example_file(self, path: Path) -> bytes:
        directory_descriptor = self._open_example_directory(path.parent)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path.name, flags, dir_fd=directory_descriptor)
        except OSError as exc:
            raise ValueError(
                f"Could not safely open example image: {path.name}"
            ) from exc
        finally:
            os.close(directory_descriptor)
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise ValueError("Example images must be regular files")
            if details.st_size > self.config.finder_max_image_bytes:
                raise ValueError(f"Example image is too large: {path.name}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 256 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.config.finder_max_image_bytes:
                    raise ValueError(f"Example image is too large: {path.name}")
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _decode_scan(row: Any) -> dict[str, Any]:
        scan = dict(row)
        scan["cancel_requested"] = bool(scan["cancel_requested"])
        scan["pause_requested"] = bool(scan["pause_requested"])
        scan["reference_ready"] = bool(scan["reference_ready"])
        scan["corpus_search_complete"] = bool(scan["corpus_search_complete"])
        scan["has_next_page"] = bool(scan.get("next_url"))
        scan["ranking_current"] = scan.get("ranking_version") == CURRENT_RANKING_VERSION
        scan["extendable"] = bool(
            scan["ranking_current"]
            and scan["has_next_page"]
            and int(scan["page_limit"]) < MAX_SCAN_PAGES
            and not scan["cancel_requested"]
            and scan["status"]
            in {
                "queued",
                "preparing",
                "scanning",
                "pausing",
                "paused",
                "completed",
                "completed_with_errors",
            }
        )
        if scan["status"] in {"completed", "completed_with_errors"}:
            scan["progress"] = 100.0
        else:
            page_limit = max(1, int(scan["page_limit"]))
            scan["progress"] = round(
                min(int(scan["pages_completed"]), page_limit) / page_limit * 100,
                1,
            )
        scan["pose_tag"] = {
            "id": scan.pop("pose_tag_id"),
            "label": scan.pop("pose_tag_label"),
            "slug": scan.pop("pose_tag_slug"),
            "default_role": scan.pop("pose_default_role"),
        }
        return scan

    def _scan_query(
        self,
        where: str = "",
        params: tuple[Any, ...] = (),
        *,
        limit: int | None = None,
    ) -> list[dict]:
        query = f"""SELECT s.*,
                       (SELECT COUNT(*) FROM finder_results r
                        WHERE r.scan_id = s.id AND r.status = 'completed'
                          AND r.score >= s.minimum_score) AS candidate_count
                    FROM finder_scans s {where}
                    ORDER BY s.created_at DESC"""
        values: list[Any] = list(params)
        if limit is not None:
            query += " LIMIT ?"
            values.append(limit)
        with self._lock, self.database.connect() as db:
            rows = db.execute(query, values).fetchall()
        return [self._decode_scan(row) for row in rows]

    def list_scans(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._scan_query(limit=limit)

    def get_scan(self, scan_id: str) -> dict[str, Any] | None:
        rows = self._scan_query("WHERE s.id = ?", (scan_id,))
        return rows[0] if rows else None

    def _queued_scan_ids(self) -> list[str]:
        with self._lock, self.database.connect() as db:
            return [
                row[0]
                for row in db.execute(
                    """SELECT id FROM finder_scans
                       WHERE status = 'queued' AND cancel_requested = 0
                         AND pause_requested = 0 AND ranking_version = ?
                       ORDER BY created_at""",
                    (CURRENT_RANKING_VERSION,),
                ).fetchall()
            ]

    def create_scan(
        self,
        *,
        example_directory: str,
        pose_tag_id: int,
        source_url: str,
        page_limit: int,
        minimum_score: float,
    ) -> dict[str, Any]:
        if not self._available:
            raise FinderUnavailable(
                self._prepare_error or "Finder vision support is unavailable"
            )
        _, normalized_directory = self._resolve_example_directory(example_directory)
        source_url = validate_source_url(source_url)
        tag = self.database.get_pose_tag(pose_tag_id)
        if not tag:
            raise FinderNotFound("Pose tag not found")
        scan_id = uuid.uuid4().hex
        now = utc_now()
        with self._lock, self.database.connect() as db:
            db.execute(
                """INSERT INTO finder_scans(
                       id, example_directory, pose_tag_id, pose_tag_label,
                       pose_tag_slug, pose_default_role, source_url, next_url,
                       page_limit, minimum_score, ranking_version, status,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (
                    scan_id,
                    normalized_directory,
                    tag["id"],
                    tag["label"],
                    tag["slug"],
                    tag["default_role"],
                    source_url,
                    source_url,
                    page_limit,
                    minimum_score,
                    CURRENT_RANKING_VERSION,
                    now,
                    now,
                ),
            )
        self.queue.put_nowait(scan_id)
        scan = self.get_scan(scan_id) or {}
        self._publish(scan, force=True)
        return scan

    def _update_scan(self, scan_id: str, **values: Any) -> None:
        allowed = {
            "reference_fingerprint",
            "reference_model_key",
            "reference_ready",
            "reference_count",
            "ranking_version",
            "next_url",
            "pages_completed",
            "status",
            "total_galleries",
            "processed_galleries",
            "processed_images",
            "failed_galleries",
            "corpus_search_complete",
            "corpus_images_scored",
            "corpus_galleries_scored",
            "cancel_requested",
            "pause_requested",
            "error",
            "finished_at",
        }
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.database.connect() as db:
            db.execute(
                f"UPDATE finder_scans SET {columns} WHERE id = ?",
                [*values.values(), scan_id],
            )

    def _control_flags(self, scan_id: str) -> tuple[bool, bool]:
        with self._lock, self.database.connect() as db:
            row = db.execute(
                """SELECT cancel_requested, pause_requested
                   FROM finder_scans WHERE id = ?""",
                (scan_id,),
            ).fetchone()
        return (True, False) if not row else (bool(row[0]), bool(row[1]))

    def pause(self, scan_id: str) -> dict[str, Any]:
        scan = self.get_scan(scan_id)
        if not scan:
            raise FinderNotFound("Finder scan not found")
        if scan["status"] in TERMINAL_STATUSES:
            raise FinderConflict("A finished scan cannot be paused")
        status = "paused" if scan["status"] in {"queued", "paused"} else "pausing"
        self._update_scan(scan_id, pause_requested=1, status=status)
        scan = self.get_scan(scan_id) or {}
        self._publish(scan, force=True)
        return scan

    def resume(self, scan_id: str) -> dict[str, Any]:
        scan = self.get_scan(scan_id)
        if not scan:
            raise FinderNotFound("Finder scan not found")
        if scan["status"] != "paused":
            raise FinderConflict("Only a paused Finder scan can be resumed")
        if scan.get("ranking_version") != CURRENT_RANKING_VERSION:
            raise FinderConflict(
                "A legacy-ranked Finder scan cannot be resumed; start a new scan"
            )
        if not self._available:
            raise FinderUnavailable(
                self._prepare_error or "Finder vision support is unavailable"
            )
        self._update_scan(
            scan_id,
            pause_requested=0,
            cancel_requested=0,
            status="queued",
            error="",
            finished_at=None,
        )
        self.queue.put_nowait(scan_id)
        scan = self.get_scan(scan_id) or {}
        self._publish(scan, force=True)
        return scan

    def extend(self, scan_id: str, *, additional_pages: int) -> dict[str, Any]:
        """Extend a scan from its durable pagination cursor.

        Completed scans are queued again. Active scans observe the larger limit
        on their next loop iteration, while paused scans intentionally remain
        paused until the user resumes them.
        """

        if (
            isinstance(additional_pages, bool)
            or not isinstance(additional_pages, int)
            or not 1 <= additional_pages <= MAX_EXTEND_PAGES
        ):
            raise ValueError(
                f"Additional pages must be between 1 and {MAX_EXTEND_PAGES}"
            )
        if not self._available:
            raise FinderUnavailable(
                self._prepare_error or "Finder vision support is unavailable"
            )

        enqueue = False
        with self._lock, self.database.connect() as db:
            row = db.execute(
                """SELECT status, page_limit, pages_completed, next_url,
                          cancel_requested, pause_requested, ranking_version
                   FROM finder_scans WHERE id = ?""",
                (scan_id,),
            ).fetchone()
            if not row:
                raise FinderNotFound("Finder scan not found")
            if row["ranking_version"] != CURRENT_RANKING_VERSION:
                raise FinderConflict(
                    "A legacy-ranked Finder scan cannot be extended; start a new scan"
                )

            status = str(row["status"])
            if bool(row["cancel_requested"]) or status in {"canceling", "canceled"}:
                raise FinderConflict("A canceled Finder scan cannot be extended")
            if status == "failed":
                raise FinderConflict("A failed Finder scan cannot be extended")
            if status not in {
                "queued",
                "preparing",
                "scanning",
                "pausing",
                "paused",
                "completed",
                "completed_with_errors",
            }:
                raise FinderConflict(
                    f"A Finder scan in the '{status}' state cannot be extended"
                )
            if not row["next_url"]:
                raise FinderConflict(
                    "This Finder scan exhausted the available source pages"
                )

            page_limit = int(row["page_limit"])
            new_limit = page_limit + additional_pages
            if new_limit > MAX_SCAN_PAGES:
                raise ValueError(
                    f"A Finder scan may contain at most {MAX_SCAN_PAGES} pages"
                )

            resume_completed = status in {"completed", "completed_with_errors"}
            next_status = "queued" if resume_completed else status
            if resume_completed:
                enqueue = True

            now = utc_now()
            if resume_completed:
                db.execute(
                    """UPDATE finder_scans
                       SET page_limit = ?, status = ?, finished_at = NULL,
                           error = '', updated_at = ?
                       WHERE id = ?""",
                    (new_limit, next_status, now, scan_id),
                )
            else:
                db.execute(
                    """UPDATE finder_scans
                       SET page_limit = ?, status = ?, updated_at = ?
                       WHERE id = ?""",
                    (new_limit, next_status, now, scan_id),
                )

        if enqueue:
            self.queue.put_nowait(scan_id)
        scan = self.get_scan(scan_id) or {}
        self._publish(scan, force=True)
        return scan

    def delete_or_cancel(self, scan_id: str) -> dict[str, Any]:
        scan = self.get_scan(scan_id)
        if not scan:
            raise FinderNotFound("Finder scan not found")
        if scan["status"] in TERMINAL_STATUSES:
            with self._lock, self.database.connect() as db:
                db.execute("DELETE FROM finder_scans WHERE id = ?", (scan_id,))
            event = {**scan, "deleted": True}
            self._publish(event, force=True)
            return event
        status = "canceled" if scan["status"] in {"queued", "paused"} else "canceling"
        self._update_scan(
            scan_id,
            cancel_requested=1,
            pause_requested=0,
            status=status,
            finished_at=utc_now() if status == "canceled" else None,
        )
        scan = self.get_scan(scan_id) or {}
        self._publish(scan, force=True)
        return scan

    @classmethod
    def _normalized_top_matches(
        cls,
        matches: list[dict[str, Any]] | None,
        *,
        ranking_version: str = LEGACY_RANKING_VERSION,
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in matches or []:
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0))
                appearance = float(item.get("appearance_score", score))
                exact = float(item.get("exact_score", 0))
                ordinal = int(item.get("ordinal"))
                pose = (
                    None
                    if item.get("pose_score") is None
                    else float(item["pose_score"])
                )
                person_count = (
                    None
                    if item.get("person_count") is None
                    else int(item["person_count"])
                )
            except (TypeError, ValueError):
                continue
            numeric_scores = (score, appearance, exact)
            if pose is not None:
                numeric_scores = (*numeric_scores, pose)
            if not all(np.isfinite(value) for value in numeric_scores):
                continue
            image_url = str(item.get("image_url") or item.get("url") or "")
            preview = str(
                item.get("preview_remote_url") or item.get("preview_url") or ""
            )
            if not image_url or not preview:
                continue
            optional_scores: dict[str, float] = {}
            for name in ("pose_coverage", "pose_body_confidence"):
                try:
                    value = float(item[name])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    optional_scores[name] = max(0.0, min(1.0, value))
            try:
                common_joints = max(0, int(item.get("pose_common_joints", 0)))
            except (TypeError, ValueError):
                common_joints = 0
            try:
                ranking_tier = max(0, min(3, int(item.get("ranking_tier", 1))))
            except (TypeError, ValueError):
                ranking_tier = 1
            overlay = str(item.get("skeleton_overlay_url") or "")
            pose_reliable = bool(item.get("pose_reliable"))
            if (
                not pose_reliable
                or not overlay.startswith("data:image/svg+xml;base64,")
                or len(overlay) > 200_000
            ):
                overlay = ""
            normalized_item = {
                "rank": 0,
                "image_url": image_url,
                "preview_remote_url": preview,
                "ordinal": ordinal,
                "score": max(0.0, min(1.0, score)),
                "ranking_tier": ranking_tier,
                "appearance_score": max(0.0, min(1.0, appearance)),
                "exact_score": max(0.0, min(1.0, exact)),
                "pose_score": None if pose is None else max(0.0, min(1.0, pose)),
                "person_count": (
                    None if person_count is None else max(0, person_count)
                ),
                "pose_reliable": pose_reliable,
                "pose_coverage": optional_scores.get("pose_coverage"),
                "pose_body_confidence": optional_scores.get("pose_body_confidence"),
                "pose_common_joints": common_joints,
                "pose_mirrored": bool(item.get("pose_mirrored")),
                "skeleton_overlay_url": overlay,
                "is_exact": exact > 0,
                "match_type": (
                    "exact"
                    if ranking_tier == 3 or exact > 0
                    else "pose"
                    if ranking_tier == 2
                    or (
                        ranking_version == LEGACY_RANKING_VERSION
                        and pose_reliable
                        and pose is not None
                        and pose > 0.5
                    )
                    else "pose_mismatch"
                    if ranking_tier == 0 and pose_reliable
                    else "appearance"
                ),
            }
            normalized_item["pose_evidence"] = cls._pose_evidence(normalized_item)
            normalized.append(normalized_item)
        if ranking_version == CURRENT_RANKING_VERSION:
            normalized.sort(
                key=lambda item: (
                    -item["ranking_tier"],
                    -item["score"],
                    -item["pose_evidence"],
                    -(item["pose_coverage"] or 0.0),
                    -item["appearance_score"],
                    item["ordinal"],
                    item["image_url"],
                )
            )
        else:
            normalized.sort(
                key=lambda item: (
                    -item["score"],
                    -item["appearance_score"],
                    item["ordinal"],
                    item["image_url"],
                )
            )
        top = normalized[:3]
        for rank, item in enumerate(top, start=1):
            item["rank"] = rank
        return top

    @classmethod
    def _decode_result(
        cls,
        row: Any,
        *,
        ranking_version: str = LEGACY_RANKING_VERSION,
    ) -> dict[str, Any]:
        item = dict(row)
        online_state = int(item.get("online_scanned", 1))
        item["online_scanned"] = online_state > 0
        item["online_refresh_failed"] = online_state < 0
        raw_matches = item.pop("matches_json", "[]")
        try:
            decoded = json.loads(str(raw_matches or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = []
        matches = cls._normalized_top_matches(
            decoded if isinstance(decoded, list) else [],
            ranking_version=ranking_version,
        )
        if (
            not matches
            and item.get("best_image_url")
            and item.get("best_preview_remote_url")
        ):
            matches = cls._normalized_top_matches(
                [
                    {
                        "image_url": item["best_image_url"],
                        "preview_remote_url": item["best_preview_remote_url"],
                        "ordinal": item.get("best_ordinal") or 0,
                        "score": item.get("score") or 0,
                        "ranking_tier": item.get("ranking_tier", 1),
                        "appearance_score": item.get("score") or 0,
                        "exact_score": 0,
                        "pose_score": None,
                        "person_count": None,
                    }
                ],
                ranking_version=ranking_version,
            )
        item["top_matches"] = matches
        return item

    def results(
        self,
        scan_id: str,
        *,
        review: str,
        min_score: float | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        scan = self.get_scan(scan_id)
        if not scan:
            raise FinderNotFound("Finder scan not found")
        threshold = scan["minimum_score"] if min_score is None else min_score
        clauses = ["scan_id = ?", "status = 'completed'", "score >= ?"]
        params: list[Any] = [scan_id, threshold]
        if review != "all":
            clauses.append("review = ?")
            params.append(review)
        where = " AND ".join(clauses)
        order_by = (
            "ranking_tier DESC, score DESC, discovered_order, id"
            if scan.get("ranking_version") == CURRENT_RANKING_VERSION
            else "score DESC, discovered_order, id"
        )
        with self._lock, self.database.connect() as db:
            total = int(
                db.execute(
                    f"SELECT COUNT(*) FROM finder_results WHERE {where}", params
                ).fetchone()[0]
            )
            rows = db.execute(
                f"""SELECT * FROM finder_results WHERE {where}
                    ORDER BY {order_by} LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
        items = [
            self._decode_result(
                row,
                ranking_version=str(
                    scan.get("ranking_version") or LEGACY_RANKING_VERSION
                ),
            )
            for row in rows
        ]
        for item in items:
            item["above_threshold"] = item["score"] >= scan["minimum_score"]
        return items, total

    def set_review(self, scan_id: str, result_id: str, review: str) -> dict[str, Any]:
        if review not in REVIEW_STATES:
            raise ValueError("Invalid Finder review state")
        if not self.get_scan(scan_id):
            raise FinderNotFound("Finder scan not found")
        with self._lock, self.database.connect() as db:
            updated = db.execute(
                """UPDATE finder_results SET review = ?, updated_at = ?
                   WHERE id = ? AND scan_id = ?""",
                (review, utc_now(), result_id, scan_id),
            )
            if not updated.rowcount:
                raise FinderNotFound("Finder result not found")
            row = db.execute(
                "SELECT * FROM finder_results WHERE id = ?", (result_id,)
            ).fetchone()
        scan = self.get_scan(scan_id) or {}
        item = self._decode_result(
            row,
            ranking_version=str(scan.get("ranking_version") or LEGACY_RANKING_VERSION),
        )
        item["above_threshold"] = item["score"] >= scan.get("minimum_score", 0)
        return item

    def _publish(self, scan: dict[str, Any], *, force: bool = False) -> None:
        if not scan:
            return
        now = time.monotonic()
        previous = self._last_event_at.get(scan["id"], 0)
        if not force and now - previous < 0.5:
            return
        self._last_event_at[scan["id"]] = now
        keys = {
            "id",
            "status",
            "pages_completed",
            "page_limit",
            "has_next_page",
            "total_galleries",
            "processed_galleries",
            "processed_images",
            "failed_galleries",
            "corpus_search_complete",
            "corpus_images_scored",
            "corpus_galleries_scored",
            "candidate_count",
            "ranking_version",
            "ranking_current",
            "extendable",
            "progress",
            "error",
            "updated_at",
            "deleted",
        }
        self.events.publish(
            {
                "type": "finder",
                "scan_id": scan["id"],
                "scan": {key: value for key, value in scan.items() if key in keys},
            }
        )

    def _check_control(self, scan_id: str) -> None:
        cancel, pause = self._control_flags(scan_id)
        if self._stopping or cancel:
            raise _FinderCanceled("Finder scan canceled")
        if pause:
            raise _FinderPaused("Finder scan paused")

    async def _wait_for_request_slot(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            wait = self._next_request_at - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_request_at = time.monotonic() + self.config.finder_request_delay

    @staticmethod
    def _normalized_embeddings(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError("Vision encoder returned an invalid embedding")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if not np.all(np.isfinite(array)) or np.any(norms <= 1e-12):
            raise ValueError("Vision encoder returned an unusable embedding")
        return np.ascontiguousarray(array / norms, dtype=np.float32)

    @staticmethod
    def _decode_metadata(value: Any) -> dict[str, Any]:
        try:
            decoded = json.loads(str(value or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _spatial_appearance(description: Any) -> np.ndarray:
        if isinstance(description, dict):
            for name in ("spatial", "spatial_embeddings"):
                if name in description and description[name] is not None:
                    return FinderService._normalized_embeddings(description[name])
        else:
            for name in ("spatial", "spatial_embeddings"):
                value = getattr(description, name, None)
                if value is not None:
                    return FinderService._normalized_embeddings(value)
        raise ValueError("Vision encoder description has no spatial appearance")

    def _descriptor_metadata(self, data: bytes, descriptor_kind: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "analyzer_version": ANALYZER_VERSION,
            "descriptor_kind": descriptor_kind,
            "phash64": None,
            "pose": None,
            "person_count": None,
        }
        if _perceptual_hash_bytes is not None:
            try:
                value = _perceptual_hash_bytes(
                    data,
                    mirror_invariant=True,
                    max_image_bytes=self.config.finder_max_image_bytes,
                    max_image_pixels=self.config.finder_max_image_pixels,
                )
            except Exception:
                value = None
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value < 1 << 64
            ):
                metadata["phash64"] = f"{value:016x}"
        if self._pose_ready and self.pose_estimator is not None:
            try:
                frame = self.pose_estimator.infer_bytes(data)
                metadata["pose"] = frame.as_dict()
                metadata["person_count"] = frame.person_count
            except Exception as exc:
                # One difficult image must not make its entire gallery fail.
                metadata["pose_error"] = str(exc)[:300]
        return metadata

    def _cache_key(self, source_key: str, include_mirror: bool) -> str:
        raw = f"{self._model_key}\0{int(include_mirror)}\0{source_key}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cached_descriptor(
        self, source_key: str, include_mirror: bool
    ) -> _ImageDescriptor | None:
        key = self._cache_key(source_key, include_mirror)
        with self._lock, self.database.connect() as db:
            row = db.execute(
                """SELECT rows, dimensions, embedding, metadata_json
                   FROM finder_embedding_cache
                   WHERE cache_key = ? AND model_key = ?""",
                (key, self._model_key),
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE finder_embedding_cache SET last_used_at = ? WHERE cache_key = ?",
                (utc_now(), key),
            )
        rows = int(row["rows"])
        dimensions = int(row["dimensions"])
        raw = bytes(row["embedding"])
        if rows <= 0 or dimensions <= 0 or len(raw) != rows * dimensions * 4:
            with self._lock, self.database.connect() as db:
                db.execute(
                    "DELETE FROM finder_embedding_cache WHERE cache_key = ?", (key,)
                )
            return None
        metadata = self._decode_metadata(row["metadata_json"])
        if metadata.get("analyzer_version") != ANALYZER_VERSION:
            with self._lock, self.database.connect() as db:
                db.execute(
                    "DELETE FROM finder_embedding_cache WHERE cache_key = ?", (key,)
                )
            return None
        if metadata.get("pose_error") and self._pose_ready:
            # Inference failures can be transient (for example a temporary CUDA
            # OOM). Do not turn one failure into a permanent pose-less cache hit.
            with self._lock, self.database.connect() as db:
                db.execute(
                    "DELETE FROM finder_embedding_cache WHERE cache_key = ?", (key,)
                )
            return None
        appearance = np.frombuffer(raw, dtype="<f4").reshape(rows, dimensions).copy()
        return _ImageDescriptor(appearance, metadata)

    def _cached_embedding(
        self, source_key: str, include_mirror: bool
    ) -> np.ndarray | None:
        descriptor = self._cached_descriptor(source_key, include_mirror)
        return descriptor.appearance if descriptor is not None else None

    def _store_embedding(
        self,
        source_key: str,
        include_mirror: bool,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        key = self._cache_key(source_key, include_mirror)
        now = utc_now()
        with self._lock, self.database.connect() as db:
            db.execute(
                """INSERT INTO finder_embedding_cache(
                       cache_key, model_key, source_key, include_mirror,
                       rows, dimensions, embedding, metadata_json,
                       created_at, last_used_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                       rows = excluded.rows,
                       dimensions = excluded.dimensions,
                       embedding = excluded.embedding,
                       metadata_json = excluded.metadata_json,
                       last_used_at = excluded.last_used_at""",
                (
                    key,
                    self._model_key,
                    source_key,
                    int(include_mirror),
                    int(embedding.shape[0]),
                    int(embedding.shape[1]),
                    embedding.astype("<f4", copy=False).tobytes(),
                    json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True),
                    now,
                    now,
                ),
            )
            self._cache_writes_since_prune += 1
            prune_due = self._cache_writes_since_prune >= 500
            if prune_due:
                self._cache_writes_since_prune = 0
        return prune_due

    def _prune_embedding_cache(self, *, purge_stale_models: bool = False) -> int:
        """Bound persistent descriptor storage and discard obsolete namespaces."""

        removed = 0
        with self._lock, self.database.connect() as db:
            if purge_stale_models and self._model_key:
                cursor = db.execute(
                    "DELETE FROM finder_embedding_cache WHERE model_key != ?",
                    (self._model_key,),
                )
                removed += max(0, int(cursor.rowcount))
            summary = db.execute(
                """SELECT COUNT(*) AS entries,
                          COALESCE(SUM(length(embedding) + length(metadata_json)), 0)
                              AS stored_bytes
                   FROM finder_embedding_cache"""
            ).fetchone()
            total_entries = int(summary["entries"])
            total_bytes = max(0, int(summary["stored_bytes"] or 0))
            max_entries = self.config.finder_cache_max_entries
            max_bytes = self.config.finder_cache_max_bytes
            if total_entries <= max_entries and total_bytes <= max_bytes:
                return removed

            # Leave 10% headroom so a large scan does not prune on every batch.
            target_entries = max(0, int(max_entries * 0.9))
            target_bytes = max(0, int(max_bytes * 0.9))
            rows = db.execute(
                """SELECT cache_key,
                          length(embedding) + length(metadata_json) AS stored_bytes
                   FROM finder_embedding_cache
                   ORDER BY last_used_at ASC, created_at ASC, cache_key ASC"""
            ).fetchall()
            keys: list[tuple[str]] = []
            for row in rows:
                if total_entries <= target_entries and total_bytes <= target_bytes:
                    break
                keys.append((str(row["cache_key"]),))
                total_entries -= 1
                total_bytes -= max(0, int(row["stored_bytes"] or 0))
            if keys:
                db.executemany(
                    "DELETE FROM finder_embedding_cache WHERE cache_key = ?", keys
                )
                removed += len(keys)
        return removed

    async def _describe_bytes(
        self, data: bytes, source_key: str, *, include_mirror: bool
    ) -> _ImageDescriptor:
        cached = self._cached_descriptor(source_key, include_mirror)
        if cached is not None:
            return cached
        self._validate_image_bytes(data)
        async with self._embedding_semaphore:
            cached = self._cached_descriptor(source_key, include_mirror)
            if cached is not None:
                return cached
            describe = getattr(self.encoder, "describe_bytes", None)
            if callable(describe):
                value = await asyncio.to_thread(
                    describe, data, include_mirror=include_mirror
                )
                appearance = self._spatial_appearance(value)
                descriptor_kind = "spatial"
            else:
                value = await asyncio.to_thread(
                    self.encoder.embed_bytes, data, include_mirror=include_mirror
                )
                appearance = self._normalized_embeddings(value)
                descriptor_kind = "legacy-global"
            metadata = await asyncio.to_thread(
                self._descriptor_metadata, data, descriptor_kind
            )
        prune_due = self._store_embedding(
            source_key, include_mirror, appearance, metadata=metadata
        )
        if prune_due:
            await asyncio.to_thread(self._prune_embedding_cache)
        return _ImageDescriptor(appearance, metadata)

    async def _embed_bytes(
        self, data: bytes, source_key: str, *, include_mirror: bool
    ) -> np.ndarray:
        descriptor = await self._describe_bytes(
            data, source_key, include_mirror=include_mirror
        )
        return descriptor.appearance

    def _validate_image_bytes(self, data: bytes) -> None:
        if not data:
            raise ValueError("Image is empty")
        if len(data) > self.config.finder_max_image_bytes:
            raise ValueError("Image exceeds the Finder size limit")
        try:
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("Image dimensions are invalid")
                if width * height > self.config.finder_max_image_pixels:
                    raise ValueError("Image exceeds the Finder pixel limit")
                image.verify()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Resource is not a valid image") from exc

    async def _fetch_media(self, url: str, referer: str) -> bytes:
        current = await asyncio.to_thread(validate_public_media_url, url)
        if self.media_fetcher is not None:
            async with self._network_semaphore:
                await self._wait_for_request_slot()
                data = await self.media_fetcher(current, referer)
            self._validate_image_bytes(data)
            return data
        if self._client is None:
            raise RuntimeError("Finder network client is not running")
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
            "Referer": referer,
        }
        async with self._network_semaphore:
            for _ in range(6):
                await self._wait_for_request_slot()
                async with self._client.stream(
                    "GET", current, headers=headers, follow_redirects=False
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError("Image host returned an empty redirect")
                        current = await asyncio.to_thread(
                            validate_public_media_url, urljoin(current, location)
                        )
                        continue
                    response.raise_for_status()
                    media_type = response.headers.get("content-type", "").lower()
                    if not media_type.startswith("image/"):
                        raise ValueError("Remote resource is not an image")
                    length = response.headers.get("content-length")
                    if length and int(length) > self.config.finder_max_image_bytes:
                        raise ValueError("Image exceeds the Finder size limit")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes(256 * 1024):
                        total += len(chunk)
                        if total > self.config.finder_max_image_bytes:
                            raise ValueError("Image exceeds the Finder size limit")
                        chunks.append(chunk)
                    data = b"".join(chunks)
                    self._validate_image_bytes(data)
                    return data
        raise RuntimeError("Too many image redirects")

    async def _remote_descriptor(self, url: str, referer: str) -> _ImageDescriptor:
        canonical = canonicalize_url(url)
        source_key = self._remote_source_key(canonical)
        cached = self._cached_descriptor(source_key, False)
        if cached is not None:
            return cached
        data = await self._fetch_media(canonical, referer)
        return await self._describe_bytes(data, source_key, include_mirror=False)

    async def _remote_embedding(self, url: str, referer: str) -> np.ndarray:
        descriptor = await self._remote_descriptor(url, referer)
        return descriptor.appearance

    def _references_are_current(self, scan_id: str) -> bool:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                """SELECT metadata_json FROM finder_scan_references
                   WHERE scan_id = ?""",
                (scan_id,),
            ).fetchall()
        return bool(rows) and all(
            self._decode_metadata(row["metadata_json"]).get("analyzer_version")
            == ANALYZER_VERSION
            for row in rows
        )

    async def _prepare_references(self, scan: dict[str, Any]) -> np.ndarray:
        references_current = scan.get(
            "reference_model_key"
        ) == self._model_key and self._references_are_current(scan["id"])
        if scan["reference_ready"] and references_current:
            return self._load_scan_references(scan["id"])
        if scan["reference_ready"]:
            with self._lock, self.database.connect() as db:
                db.execute(
                    "DELETE FROM finder_scan_references WHERE scan_id = ?",
                    (scan["id"],),
                )
                db.execute(
                    "DELETE FROM finder_results WHERE scan_id = ?", (scan["id"],)
                )
                db.execute(
                    """UPDATE finder_scans
                       SET reference_fingerprint = '', reference_model_key = '',
                           reference_ready = 0, reference_count = 0,
                           next_url = source_url, pages_completed = 0,
                           total_galleries = 0, processed_galleries = 0,
                           processed_images = 0, failed_galleries = 0,
                           corpus_search_complete = 0,
                           corpus_images_scored = 0,
                           corpus_galleries_scored = 0,
                           status = 'preparing', error = '', finished_at = NULL,
                           updated_at = ?
                       WHERE id = ?""",
                    (utc_now(), scan["id"]),
                )
            scan = {
                **scan,
                "reference_fingerprint": "",
                "reference_model_key": "",
                "reference_ready": False,
                "reference_count": 0,
                "next_url": scan["source_url"],
                "pages_completed": 0,
                "total_galleries": 0,
                "processed_galleries": 0,
                "processed_images": 0,
                "failed_galleries": 0,
                "corpus_search_complete": False,
                "corpus_images_scored": 0,
                "corpus_galleries_scored": 0,
            }
        directory, _ = self._resolve_example_directory(scan["example_directory"])
        files = self._example_files(directory)
        manifest: list[tuple[Path, str, str]] = []
        fingerprint = hashlib.sha256()
        for path in files:
            self._check_control(scan["id"])
            data = await asyncio.to_thread(self._read_example_file, path)
            self._validate_image_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            relative = path.relative_to(directory).as_posix()
            fingerprint.update(relative.encode())
            fingerprint.update(b"\0")
            fingerprint.update(digest.encode())
            fingerprint.update(b"\0")
            manifest.append((path, relative, digest))
        fingerprint_value = fingerprint.hexdigest()
        if (
            scan["reference_fingerprint"]
            and scan["reference_fingerprint"] != fingerprint_value
        ):
            raise FinderConflict(
                "Example images changed before the Finder reference was ready"
            )
        if not scan["reference_fingerprint"]:
            with self._lock, self.database.connect() as db:
                db.execute(
                    "DELETE FROM finder_scan_references WHERE scan_id = ?",
                    (scan["id"],),
                )
            self._update_scan(
                scan["id"], reference_fingerprint=fingerprint_value, status="preparing"
            )
        for path, relative, digest in manifest:
            self._check_control(scan["id"])
            data = await asyncio.to_thread(self._read_example_file, path)
            if hashlib.sha256(data).hexdigest() != digest:
                raise FinderConflict("Example images changed while preparing the scan")
            descriptor = await self._describe_bytes(
                data, f"sha256:{digest}", include_mirror=True
            )
            with self._lock, self.database.connect() as db:
                for mirror_index, vector in enumerate(descriptor.appearance):
                    metadata = {
                        **descriptor.metadata,
                        "view_index": mirror_index,
                        "mirrored": bool(mirror_index),
                    }
                    db.execute(
                        """INSERT INTO finder_scan_references(
                               scan_id, example_key, mirror_index, embedding,
                               dimensions, metadata_json
                           ) VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(scan_id, example_key, mirror_index) DO UPDATE SET
                               embedding = excluded.embedding,
                               dimensions = excluded.dimensions,
                               metadata_json = excluded.metadata_json""",
                        (
                            scan["id"],
                            f"{relative}:{digest}",
                            mirror_index,
                            vector.astype("<f4", copy=False).tobytes(),
                            int(vector.shape[0]),
                            json.dumps(metadata, separators=(",", ":"), sort_keys=True),
                        ),
                    )
        references = self._load_scan_references(scan["id"])
        self._update_scan(
            scan["id"],
            reference_ready=1,
            reference_count=int(references.shape[0]),
            reference_model_key=self._model_key,
            status="scanning",
        )
        return references

    def _load_scan_references(self, scan_id: str) -> np.ndarray:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                """SELECT embedding, dimensions FROM finder_scan_references
                   WHERE scan_id = ? ORDER BY example_key, mirror_index""",
                (scan_id,),
            ).fetchall()
        vectors: list[np.ndarray] = []
        dimensions: int | None = None
        for row in rows:
            current = int(row["dimensions"])
            raw = bytes(row["embedding"])
            if current <= 0 or len(raw) != current * 4:
                raise ValueError("Stored Finder reference is invalid")
            if dimensions is not None and current != dimensions:
                raise ValueError("Finder reference dimensions do not match")
            dimensions = current
            vectors.append(np.frombuffer(raw, dtype="<f4").copy())
        if not vectors:
            raise ValueError("Finder scan has no reference embeddings")
        return self._normalized_embeddings(np.stack(vectors))

    @staticmethod
    def _metadata_phash(metadata: dict[str, Any]) -> int | None:
        raw = metadata.get("phash64")
        if not isinstance(raw, str) or len(raw) != 16:
            return None
        try:
            value = int(raw, 16)
        except ValueError:
            return None
        return value if 0 <= value < 1 << 64 else None

    def _load_reference_hashes(self, scan_id: str) -> tuple[int, ...]:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                """SELECT metadata_json FROM finder_scan_references
                   WHERE scan_id = ? ORDER BY example_key, mirror_index""",
                (scan_id,),
            ).fetchall()
        hashes = {
            value
            for row in rows
            if (
                value := self._metadata_phash(
                    self._decode_metadata(row["metadata_json"])
                )
            )
            is not None
        }
        return tuple(sorted(hashes))

    @staticmethod
    def _metadata_pose(metadata: dict[str, Any]) -> Any | None:
        value = metadata.get("pose")
        if not isinstance(value, dict):
            return None
        try:
            from .pose_vision import PoseFrame

            return PoseFrame.from_dict(value)
        except (ImportError, ValueError, TypeError):
            return None

    def _load_reference_poses(self, scan_id: str) -> tuple[Any, ...]:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                """SELECT metadata_json FROM finder_scan_references
                   WHERE scan_id = ? AND mirror_index = 0
                   ORDER BY example_key""",
                (scan_id,),
            ).fetchall()
        frames = [
            self._metadata_pose(self._decode_metadata(row["metadata_json"]))
            for row in rows
        ]
        return tuple(frame for frame in frames if frame is not None)

    @staticmethod
    def _pose_diagnostics(
        candidate_metadata: dict[str, Any], references: tuple[Any, ...]
    ) -> dict[str, Any]:
        candidate = FinderService._metadata_pose(candidate_metadata)
        if candidate is None:
            return {
                "pose_score": None,
                "pose_reliable": False,
                "person_count": None,
                "skeleton_overlay_url": "",
            }
        if not references:
            return {
                "pose_score": None,
                "pose_reliable": False,
                "person_count": candidate.person_count,
                "skeleton_overlay_url": "",
            }
        try:
            from .pose_vision import pose_geometry_match

            matches = [
                pose_geometry_match(reference, candidate, allow_mirror=True)
                for reference in references
            ]
        except (ImportError, ValueError, TypeError):
            matches = []
        if not matches:
            return {
                "pose_score": None,
                "pose_reliable": False,
                "person_count": candidate.person_count,
                "skeleton_overlay_url": "",
            }
        best = max(
            matches,
            key=lambda match: (
                bool(match.reliable),
                float(match.score),
                float(match.coverage),
            ),
        )
        overlay = ""
        if best.reliable and candidate.person_count:
            try:
                from .pose_vision import skeleton_data_uri

                overlay = skeleton_data_uri(
                    candidate,
                    width=candidate.image_size[0],
                    height=candidate.image_size[1],
                )
            except (ImportError, ValueError, TypeError):
                overlay = ""
        return {
            "pose_score": max(0.0, min(1.0, float(best.score))),
            "pose_reliable": bool(best.reliable),
            "pose_coverage": max(0.0, min(1.0, float(best.coverage))),
            "pose_common_joints": max(0, int(best.common_joints)),
            "pose_body_confidence": max(
                0.0,
                min(
                    1.0,
                    float(
                        getattr(
                            best,
                            "minimum_body_confidence",
                            best.mean_joint_confidence,
                        )
                    ),
                ),
            ),
            "pose_mirrored": bool(best.mirrored),
            "person_count": candidate.person_count,
            "skeleton_overlay_url": overlay,
        }

    @staticmethod
    def _appearance_score(raw_score: float, *, spatial: bool) -> float:
        if spatial:
            return float(np.clip((raw_score - 0.20) / 0.65, 0.0, 1.0))
        # Compatibility for injected/legacy encoders that only implement
        # embed_bytes. Their scores were historically raw cosine similarity.
        return max(0.0, min(1.0, raw_score))

    @staticmethod
    def _exact_score(candidate: int | None, references: tuple[int, ...]) -> float:
        if candidate is None or not references:
            return 0.0
        distance = min((candidate ^ reference).bit_count() for reference in references)
        if distance > EXACT_HASH_MAX_DISTANCE:
            return 0.0
        # Once inside the strict duplicate gate, retain a strong score so a
        # resized/recompressed source copy is not hidden by appearance ranking.
        return max(0.0, 1.0 - distance / 64.0)

    @staticmethod
    def _pose_evidence(pose: dict[str, Any]) -> float:
        try:
            coverage = float(pose.get("pose_coverage") or 0)
            body_confidence = float(pose.get("pose_body_confidence") or 0)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(coverage) or not np.isfinite(body_confidence):
            return 0.0
        return max(0.0, min(1.0, coverage)) * max(0.0, min(1.0, body_confidence / 0.45))

    @staticmethod
    def _combined_score(
        appearance_score: float,
        exact_score: float,
        pose: dict[str, Any],
    ) -> float:
        if exact_score > 0:
            return max(appearance_score, exact_score)
        pose_score = pose.get("pose_score")
        if not pose.get("pose_reliable") or not isinstance(pose_score, (int, float)):
            return appearance_score
        evidence = FinderService._pose_evidence(pose)
        # Reliable geometry is a bounded positive tie-breaker. It cannot erase
        # a strong crop/appearance match, which is important for extreme poses
        # where even RTMO can see only part of a body.
        bonus = (
            0.12
            * evidence
            * max(0.0, float(pose_score) - 0.5)
            * (1.0 - appearance_score)
        )
        return max(0.0, min(1.0, appearance_score + bonus))

    @classmethod
    def _ranked_score(
        cls,
        appearance_score: float,
        exact_score: float,
        pose: dict[str, Any],
        *,
        ranking_version: str,
    ) -> tuple[int, float]:
        appearance = max(0.0, min(1.0, float(appearance_score)))
        exact = max(0.0, min(1.0, float(exact_score)))
        if ranking_version != CURRENT_RANKING_VERSION:
            return 1, cls._combined_score(appearance, exact, pose)
        if exact > 0:
            return 3, exact
        pose_score = pose.get("pose_score")
        try:
            normalized_pose = float(pose_score)
        except (TypeError, ValueError):
            normalized_pose = float("nan")
        if pose.get("pose_reliable") and np.isfinite(normalized_pose):
            normalized_pose = max(0.0, min(1.0, normalized_pose))
            tier = 2 if normalized_pose >= POSE_MATCH_FLOOR else 0
            return tier, normalized_pose
        return 1, appearance

    def _claim_scan(self, scan_id: str) -> bool:
        with self._lock, self.database.connect() as db:
            result = db.execute(
                """UPDATE finder_scans
                   SET status = 'preparing', updated_at = ?, error = ''
                   WHERE id = ? AND status = 'queued'
                     AND ranking_version = ?
                     AND cancel_requested = 0 AND pause_requested = 0""",
                (utc_now(), scan_id, CURRENT_RANKING_VERSION),
            )
        return bool(result.rowcount)

    async def _worker(self, _: int) -> None:
        while not self._stopping:
            scan_id = await self.queue.get()
            try:
                if not self._claim_scan(scan_id):
                    continue
                scan = self.get_scan(scan_id)
                if not scan:
                    continue
                await self._ensure_encoder_ready()
                await self._run_scan(scan)
            except asyncio.CancelledError:
                raise
            except _FinderCanceled:
                self._update_scan(
                    scan_id,
                    status="canceled",
                    cancel_requested=1,
                    pause_requested=0,
                    finished_at=utc_now(),
                    error="",
                )
            except _FinderPaused:
                self._update_scan(scan_id, status="paused", pause_requested=1)
            except Exception as exc:
                cancel, pause = self._control_flags(scan_id)
                if cancel:
                    self._update_scan(
                        scan_id,
                        status="canceled",
                        finished_at=utc_now(),
                        error="",
                    )
                elif pause:
                    self._update_scan(scan_id, status="paused")
                else:
                    self._update_scan(
                        scan_id,
                        status="failed",
                        error=str(exc)[:1000],
                        finished_at=utc_now(),
                    )
            finally:
                # The scan's durable status is already written. Release queue
                # joiners before the final read/event publication, which can be
                # comparatively slow on Unraid/FUSE-backed SQLite volumes.
                self.queue.task_done()
                final = self.get_scan(scan_id)
                if final:
                    self._publish(final, force=True)

    def _result_complete(self, scan_id: str, gallery_url: str) -> bool:
        key = gallery_key(gallery_url)
        with self._lock, self.database.connect() as db:
            row = db.execute(
                """SELECT status FROM finder_results
                   WHERE scan_id = ? AND gallery_key = ?
                     AND online_scanned = 1""",
                (scan_id, key),
            ).fetchone()
        return bool(row and row["status"] == "completed")

    def _index_corpus_gallery(
        self,
        card: dict[str, Any],
        detail: dict[str, Any],
        images: list[dict[str, Any]],
    ) -> None:
        """Replace one gallery's membership after a complete detail fetch."""

        gallery_url = validate_source_url(
            str(detail.get("url") or card.get("url") or "")
        )
        key = gallery_key(gallery_url)
        now = utc_now()
        associations: dict[str, tuple[str, str, int]] = {}
        for index, image in enumerate(images, start=1):
            preview = str(image.get("preview_remote_url") or "")
            image_url = str(image.get("url") or "")
            if not preview or not image_url:
                continue
            try:
                canonical_preview = canonicalize_url(preview)
                source_key = self._remote_source_key(preview)
                ordinal = max(0, int(image.get("ordinal") or index))
            except (TypeError, ValueError):
                continue
            associations[source_key] = (
                image_url,
                canonical_preview,
                ordinal,
            )
        with self._lock, self.database.connect() as db:
            db.execute(
                """INSERT INTO finder_corpus_galleries(
                       gallery_key, gallery_url, title, thumbnail_remote_url,
                       state, image_count, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, 'complete', ?, ?, ?)
                   ON CONFLICT(gallery_key) DO UPDATE SET
                       gallery_url = excluded.gallery_url,
                       title = excluded.title,
                       thumbnail_remote_url = excluded.thumbnail_remote_url,
                       state = 'complete',
                       image_count = excluded.image_count,
                       updated_at = excluded.updated_at""",
                (
                    key,
                    gallery_url,
                    str(detail.get("title") or card.get("title") or "Untitled gallery")[
                        :300
                    ],
                    str(
                        detail.get("thumbnail_remote_url")
                        or card.get("thumbnail_remote_url")
                        or ""
                    ),
                    len(associations),
                    now,
                    now,
                ),
            )
            # Detail pages are the authoritative full association list. Delete
            # stale preview URLs/ordinals before inserting the refreshed list.
            db.execute(
                "DELETE FROM finder_corpus_images WHERE gallery_key = ?",
                (key,),
            )
            db.executemany(
                """INSERT INTO finder_corpus_images(
                       gallery_key, source_key, image_url, preview_remote_url,
                       ordinal, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        key,
                        source_key,
                        image_url,
                        preview,
                        ordinal,
                        now,
                        now,
                    )
                    for source_key, (image_url, preview, ordinal) in (
                        associations.items()
                    )
                ],
            )

    def _save_result(
        self,
        scan_id: str,
        card: dict,
        *,
        order: int,
        score: float,
        images_scored: int,
        best: dict | None,
        status: str,
        ranking_version: str = LEGACY_RANKING_VERSION,
        error: str = "",
        top_matches: list[dict[str, Any]] | None = None,
        online_scanned: bool = True,
    ) -> None:
        now = utc_now()
        key = gallery_key(card["url"])
        if top_matches is None and best is not None:
            top_matches = [
                {
                    "image_url": best.get("url"),
                    "preview_remote_url": best.get("preview_remote_url"),
                    "ordinal": best.get("ordinal"),
                    "score": score,
                    "appearance_score": score,
                    "exact_score": 0,
                    "pose_score": None,
                    "person_count": None,
                }
            ]
        matches = self._normalized_top_matches(
            top_matches, ranking_version=ranking_version
        )
        ranking_tier = 1 if ranking_version == LEGACY_RANKING_VERSION else 0
        if matches:
            leading = matches[0]
            best = {
                "url": leading["image_url"],
                "preview_remote_url": leading["preview_remote_url"],
                "ordinal": leading["ordinal"],
            }
            score = leading["score"]
            ranking_tier = leading["ranking_tier"]
        matches_json = json.dumps(matches, separators=(",", ":"), sort_keys=True)
        with self._lock, self.database.connect() as db:
            db.execute(
                """INSERT INTO finder_results(
                       id, scan_id, gallery_key, gallery_url, title,
                       thumbnail_remote_url, best_image_url,
                       best_preview_remote_url, best_ordinal, score,
                       ranking_tier, matches_json, images_scored, status, error,
                       online_scanned, discovered_order,
                       created_at, updated_at
                   ) VALUES (
                       ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                   )
                   ON CONFLICT(scan_id, gallery_key) DO UPDATE SET
                       gallery_url = excluded.gallery_url,
                       title = excluded.title,
                       thumbnail_remote_url = excluded.thumbnail_remote_url,
                       best_image_url = excluded.best_image_url,
                       best_preview_remote_url = excluded.best_preview_remote_url,
                       best_ordinal = excluded.best_ordinal,
                       score = excluded.score,
                       ranking_tier = excluded.ranking_tier,
                       matches_json = excluded.matches_json,
                       images_scored = excluded.images_scored,
                       status = excluded.status,
                       error = excluded.error,
                       online_scanned = MAX(
                           finder_results.online_scanned,
                           excluded.online_scanned
                       ),
                       discovered_order = excluded.discovered_order,
                       updated_at = excluded.updated_at""",
                (
                    uuid.uuid4().hex,
                    scan_id,
                    key,
                    card["url"],
                    str(card.get("title") or "Untitled gallery")[:300],
                    str(card.get("thumbnail_remote_url") or ""),
                    str(best.get("url") if best else ""),
                    str(best.get("preview_remote_url") if best else ""),
                    int(best.get("ordinal")) if best else None,
                    float(score),
                    ranking_tier,
                    matches_json,
                    images_scored,
                    status,
                    error[:1000],
                    int(online_scanned),
                    order,
                    now,
                    now,
                ),
            )

    def _preserve_local_result_after_online_error(
        self,
        scan_id: str,
        card: dict[str, Any],
        *,
        order: int,
        error: str,
    ) -> bool:
        """Record an online attempt without erasing a usable local candidate."""

        key = gallery_key(card["url"])
        now = utc_now()
        with self._lock, self.database.connect() as db:
            updated = db.execute(
                """UPDATE finder_results
                   SET gallery_url = ?,
                       title = CASE WHEN ? != '' THEN ? ELSE title END,
                       thumbnail_remote_url = CASE
                           WHEN ? != '' THEN ? ELSE thumbnail_remote_url
                       END,
                       online_scanned = -1,
                       images_scored = 0,
                       error = ?,
                       discovered_order = ?,
                       updated_at = ?
                   WHERE scan_id = ? AND gallery_key = ?
                     AND online_scanned <= 0 AND status = 'completed'""",
                (
                    card["url"],
                    str(card.get("title") or "")[:300],
                    str(card.get("title") or "")[:300],
                    str(card.get("thumbnail_remote_url") or ""),
                    str(card.get("thumbnail_remote_url") or ""),
                    error[:1000],
                    order,
                    now,
                    scan_id,
                    key,
                ),
            )
        return bool(updated.rowcount)

    def _progress_counts(self, scan_id: str) -> dict[str, int]:
        with self._lock, self.database.connect() as db:
            row = db.execute(
                """SELECT COUNT(*) AS processed,
                          COALESCE(SUM(images_scored), 0) AS images,
                          COALESCE(SUM(
                              CASE WHEN status = 'failed' OR error != ''
                                  THEN 1 ELSE 0 END
                          ), 0) AS failed
                   FROM finder_results
                   WHERE scan_id = ? AND online_scanned != 0""",
                # Corpus-preloaded candidates are not web crawl progress.
                (scan_id,),
            ).fetchone()
        return {
            "processed_galleries": int(row["processed"]),
            "processed_images": int(row["images"]),
            "failed_galleries": int(row["failed"]),
        }

    def _finalize_scan_if_done(self, scan_id: str) -> bool:
        """Atomically finalize only when no extension has moved the limit.

        The extension endpoint and this check share the service lock. This
        closes the small race where a worker had observed its old limit just
        before another request increased it.
        """

        with self._lock, self.database.connect() as db:
            scan = db.execute(
                """SELECT status, pages_completed, page_limit, next_url,
                          cancel_requested, pause_requested
                   FROM finder_scans WHERE id = ?""",
                (scan_id,),
            ).fetchone()
            if not scan:
                raise _FinderCanceled("Finder scan was deleted")
            if (
                bool(scan["cancel_requested"])
                or bool(scan["pause_requested"])
                or scan["status"] != "scanning"
            ):
                return False
            if scan["next_url"] and int(scan["pages_completed"]) < int(
                scan["page_limit"]
            ):
                return False

            counts = db.execute(
                """SELECT COUNT(*) AS processed,
                          COALESCE(SUM(images_scored), 0) AS images,
                          COALESCE(SUM(
                              CASE WHEN status = 'failed' OR error != ''
                                  THEN 1 ELSE 0 END
                          ), 0) AS failed
                   FROM finder_results
                   WHERE scan_id = ? AND online_scanned != 0""",
                (scan_id,),
            ).fetchone()
            processed = int(counts["processed"])
            images = int(counts["images"])
            failed = int(counts["failed"])
            status = "completed_with_errors" if failed else "completed"
            now = utc_now()
            updated = db.execute(
                """UPDATE finder_scans
                   SET status = ?, total_galleries = ?,
                       processed_galleries = ?, processed_images = ?,
                       failed_galleries = ?, finished_at = ?, updated_at = ?
                   WHERE id = ? AND status = 'scanning'
                     AND cancel_requested = 0 AND pause_requested = 0""",
                (
                    status,
                    processed,
                    processed,
                    images,
                    failed,
                    now,
                    now,
                    scan_id,
                ),
            )
            return bool(updated.rowcount)

    def _missing_on_page(self, scan_id: str, cards: list[dict]) -> int:
        keys = [gallery_key(card["url"]) for card in cards]
        if not keys:
            return 0
        placeholders = ",".join("?" for _ in keys)
        with self._lock, self.database.connect() as db:
            count = int(
                db.execute(
                    f"""SELECT COUNT(*) FROM finder_results
                        WHERE scan_id = ? AND online_scanned != 0
                          AND gallery_key IN ({placeholders})""",
                    [scan_id, *keys],
                ).fetchone()[0]
            )
        return len(set(keys)) - count

    def _corpus_descriptor_query(
        self,
        after: tuple[str, int, str] | None,
        *,
        limit: int = 256,
    ) -> tuple[str, list[Any]]:
        condition = ""
        params: list[Any] = [self._model_key]
        if after is not None:
            gallery, ordinal, source_key = after
            condition = """
                AND (i.gallery_key, i.ordinal, i.source_key) > (?, ?, ?)
            """
            params.extend((gallery, ordinal, source_key))
        params.append(limit)
        query = f"""SELECT i.gallery_key, i.source_key, i.image_url,
                           i.preview_remote_url, i.ordinal,
                           g.gallery_url, g.title, g.thumbnail_remote_url,
                           c.cache_key, c.rows, c.dimensions, c.embedding,
                           c.metadata_json
                    FROM finder_corpus_images i
                         INDEXED BY idx_finder_corpus_images_scan
                    CROSS JOIN finder_corpus_galleries g
                    CROSS JOIN finder_embedding_cache c
                         INDEXED BY idx_finder_embedding_cache_source
                    WHERE g.gallery_key = i.gallery_key
                      AND c.model_key = ?
                      AND c.source_key = i.source_key
                      AND c.include_mirror = 0
                      {condition}
                    ORDER BY i.gallery_key, i.ordinal, i.source_key
                    LIMIT ?"""
        return query, params

    def _corpus_descriptor_rows(
        self,
        after: tuple[str, int, str] | None,
        *,
        limit: int = 256,
    ) -> list[Any]:
        query, params = self._corpus_descriptor_query(after, limit=limit)
        with self._lock, self.database.connect() as db:
            return db.execute(query, params).fetchall()

    def _descriptor_from_corpus_row(
        self,
        row: Any,
        expected_dimensions: int,
    ) -> _ImageDescriptor | None:
        rows = int(row["rows"])
        dimensions = int(row["dimensions"])
        raw = bytes(row["embedding"])
        if (
            rows <= 0
            or dimensions != expected_dimensions
            or len(raw) != rows * dimensions * 4
        ):
            return None
        metadata = FinderService._decode_metadata(row["metadata_json"])
        if metadata.get("analyzer_version") != ANALYZER_VERSION:
            return None
        if metadata.get("pose_error") and self._pose_ready:
            # Let a later online visit retry the transient pose inference
            # failure through the normal descriptor path.
            return None
        appearance = np.frombuffer(raw, dtype="<f4").reshape(rows, dimensions).copy()
        if not np.all(np.isfinite(appearance)):
            return None
        norms = np.linalg.norm(appearance, axis=1)
        if np.any(norms <= 1e-12):
            return None
        return _ImageDescriptor(appearance, metadata)

    def _score_corpus_rows(
        self,
        rows: list[Any],
        references: np.ndarray,
        reference_hashes: tuple[int, ...],
        reference_poses: tuple[Any, ...],
        ranking_version: str,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        dimensions = int(references.shape[1])
        decoded: list[tuple[Any, _ImageDescriptor]] = []
        for row in rows:
            descriptor = self._descriptor_from_corpus_row(row, dimensions)
            if descriptor is None:
                continue
            decoded.append((row, descriptor))
        if not decoded:
            return {}
        vectors = np.stack([descriptor.appearance[0] for _, descriptor in decoded])
        raw_scores = np.max(references @ vectors.T, axis=0)
        scored_by_source: dict[tuple[str, str], dict[str, Any]] = {}
        for (row, descriptor), raw_score in zip(decoded, raw_scores, strict=True):
            metadata = descriptor.metadata
            spatial = metadata.get("descriptor_kind") == "spatial"
            appearance_score = self._appearance_score(
                float(raw_score),
                spatial=spatial,
            )
            exact_score = (
                self._exact_score(
                    self._metadata_phash(metadata),
                    reference_hashes,
                )
                if spatial
                else 0.0
            )
            pose = self._pose_diagnostics(metadata, reference_poses)
            ranking_tier, score = self._ranked_score(
                appearance_score,
                exact_score,
                pose,
                ranking_version=ranking_version,
            )
            scored_by_source[(str(row["gallery_key"]), str(row["source_key"]))] = {
                "image_url": str(row["image_url"]),
                "preview_remote_url": str(row["preview_remote_url"]),
                "ordinal": int(row["ordinal"]),
                "score": score,
                "ranking_tier": ranking_tier,
                "appearance_score": appearance_score,
                "exact_score": exact_score,
                "_cache_key": str(row["cache_key"]),
                **pose,
            }
        return scored_by_source

    def _touch_corpus_cache_rows(self, cache_keys: set[str]) -> None:
        if not cache_keys:
            return
        now = utc_now()
        with self._lock, self.database.connect() as db:
            db.executemany(
                """UPDATE finder_embedding_cache
                   SET last_used_at = ? WHERE cache_key = ?""",
                [(now, key) for key in sorted(cache_keys)],
            )

    async def _search_corpus(
        self,
        scan: dict[str, Any],
        references: np.ndarray,
        reference_hashes: tuple[int, ...],
        reference_poses: tuple[Any, ...],
    ) -> None:
        """Score reusable descriptors locally before visiting source pages."""

        current = self.get_scan(scan["id"])
        if not current or current["corpus_search_complete"]:
            return
        self._check_control(scan["id"])
        self._update_scan(
            scan["id"],
            corpus_images_scored=0,
            corpus_galleries_scored=0,
        )
        ranking_version = str(current.get("ranking_version") or LEGACY_RANKING_VERSION)
        after: tuple[str, int, str] | None = None
        active_key = ""
        active_card: dict[str, Any] | None = None
        active_matches: list[dict[str, Any]] = []
        active_images = 0
        images_scored = 0
        galleries_scored = 0
        selected_cache_keys: set[str] = set()
        minimum_score = float(current["minimum_score"])

        def save_active() -> None:
            nonlocal active_card, active_matches, active_images, galleries_scored
            if active_card is None or not active_matches:
                active_card = None
                active_matches = []
                active_images = 0
                return
            matches = self._normalized_top_matches(
                active_matches,
                ranking_version=ranking_version,
            )
            if matches[0]["score"] >= minimum_score:
                leading = matches[0]
                for raw_match in active_matches:
                    if (
                        raw_match["image_url"] == leading["image_url"]
                        and raw_match["preview_remote_url"]
                        == leading["preview_remote_url"]
                        and raw_match["ordinal"] == leading["ordinal"]
                    ):
                        selected_cache_keys.add(str(raw_match["_cache_key"]))
                        break
            self._save_result(
                scan["id"],
                active_card,
                order=galleries_scored,
                score=matches[0]["score"],
                images_scored=active_images,
                best=None,
                status="completed",
                ranking_version=ranking_version,
                top_matches=matches,
                online_scanned=False,
            )
            galleries_scored += 1
            active_card = None
            active_matches = []
            active_images = 0

        while True:
            self._check_control(scan["id"])
            rows = await asyncio.to_thread(self._corpus_descriptor_rows, after)
            if not rows:
                break
            scored_by_source = await asyncio.to_thread(
                self._score_corpus_rows,
                rows,
                references,
                reference_hashes,
                reference_poses,
                ranking_version,
            )

            for row in rows:
                self._check_control(scan["id"])
                gallery = str(row["gallery_key"])
                if active_key and gallery != active_key:
                    save_active()
                if gallery != active_key:
                    active_key = gallery
                    active_card = {
                        "url": str(row["gallery_url"]),
                        "title": str(row["title"] or "Untitled gallery"),
                        "thumbnail_remote_url": str(row["thumbnail_remote_url"] or ""),
                    }
                match = scored_by_source.get((gallery, str(row["source_key"])))
                if match is not None:
                    active_matches.append(match)
                    active_images += 1
                    images_scored += 1
            final_row = rows[-1]
            after = (
                str(final_row["gallery_key"]),
                int(final_row["ordinal"]),
                str(final_row["source_key"]),
            )
            self._update_scan(
                scan["id"],
                corpus_images_scored=images_scored,
                corpus_galleries_scored=galleries_scored,
            )
            progress = self.get_scan(scan["id"])
            if progress:
                self._publish(progress)
        save_active()
        self._check_control(scan["id"])
        await asyncio.to_thread(
            self._touch_corpus_cache_rows,
            selected_cache_keys,
        )
        self._update_scan(
            scan["id"],
            corpus_search_complete=1,
            corpus_images_scored=images_scored,
            corpus_galleries_scored=galleries_scored,
        )
        progress = self.get_scan(scan["id"])
        if progress:
            self._publish(progress, force=True)

    async def _score_gallery(
        self,
        scan: dict[str, Any],
        card: dict,
        order: int,
        references: np.ndarray,
        reference_hashes: tuple[int, ...],
        reference_poses: tuple[Any, ...],
    ) -> None:
        ranking_version = str(scan.get("ranking_version") or LEGACY_RANKING_VERSION)
        gallery_url = validate_source_url(card["url"])
        if self._result_complete(scan["id"], gallery_url):
            return
        self._check_control(scan["id"])
        try:
            async with self._network_semaphore:
                await self._wait_for_request_slot()
                detail = await self.scraper.gallery(gallery_url)
            images = list(detail.get("images") or [])
            if not images:
                raise ValueError("Gallery contains no preview images")
            if len(images) > self.config.finder_max_gallery_images:
                raise ValueError("Gallery exceeds the Finder image-count limit")
            for index, image in enumerate(images, start=1):
                image.setdefault("ordinal", index)
            await asyncio.to_thread(
                self._index_corpus_gallery,
                card,
                detail,
                images,
            )

            async def score_image(image: dict) -> dict[str, Any] | None:
                self._check_control(scan["id"])
                preview = str(image.get("preview_remote_url") or "")
                original = str(image.get("url") or "")
                await asyncio.to_thread(validate_public_media_url, preview)
                await asyncio.to_thread(validate_public_media_url, original)
                try:
                    candidate = await self._remote_descriptor(preview, gallery_url)
                except (_FinderPaused, _FinderCanceled):
                    raise
                except Exception:
                    return None
                vector = candidate.appearance[0]
                raw_score = float(np.max(references @ vector))
                spatial = candidate.metadata.get("descriptor_kind") == "spatial"
                appearance_score = self._appearance_score(raw_score, spatial=spatial)
                exact_score = (
                    self._exact_score(
                        self._metadata_phash(candidate.metadata), reference_hashes
                    )
                    if spatial
                    else 0.0
                )
                pose = self._pose_diagnostics(candidate.metadata, reference_poses)
                ranking_tier, score = self._ranked_score(
                    appearance_score,
                    exact_score,
                    pose,
                    ranking_version=ranking_version,
                )
                return {
                    "image_url": original,
                    "preview_remote_url": preview,
                    "ordinal": int(image["ordinal"]),
                    "score": score,
                    "ranking_tier": ranking_tier,
                    "appearance_score": appearance_score,
                    "exact_score": exact_score,
                    **pose,
                }

            scored: list[dict[str, Any]] = []
            batch_size = self.config.finder_network_workers
            for start in range(0, len(images), batch_size):
                self._check_control(scan["id"])
                outcomes = await asyncio.gather(
                    *(
                        score_image(image)
                        for image in images[start : start + batch_size]
                    ),
                    return_exceptions=True,
                )
                for outcome in outcomes:
                    if isinstance(outcome, (_FinderPaused, _FinderCanceled)):
                        raise outcome
                    if isinstance(outcome, dict):
                        scored.append(outcome)
            if not scored:
                raise ValueError("No gallery preview image could be scored")
            top_matches = self._normalized_top_matches(
                scored, ranking_version=ranking_version
            )
            self._save_result(
                scan["id"],
                {**card, "url": detail.get("url") or gallery_url},
                order=order,
                score=top_matches[0]["score"],
                images_scored=len(scored),
                best=None,
                status="completed",
                ranking_version=ranking_version,
                top_matches=top_matches,
            )
        except (_FinderPaused, _FinderCanceled):
            raise
        except Exception as exc:
            if not self._preserve_local_result_after_online_error(
                scan["id"],
                card,
                order=order,
                error=str(exc),
            ):
                self._save_result(
                    scan["id"],
                    card,
                    order=order,
                    score=0,
                    images_scored=0,
                    best=None,
                    status="failed",
                    ranking_version=ranking_version,
                    error=str(exc),
                )

    async def _run_scan(self, scan: dict[str, Any]) -> None:
        if scan.get("ranking_version") != CURRENT_RANKING_VERSION:
            raise FinderConflict(
                "A legacy-ranked Finder scan cannot continue; start a new scan"
            )
        references = await self._prepare_references(scan)
        reference_hashes = self._load_reference_hashes(scan["id"])
        reference_poses = self._load_reference_poses(scan["id"])
        self._check_control(scan["id"])
        await self._search_corpus(
            scan,
            references,
            reference_hashes,
            reference_poses,
        )
        self._check_control(scan["id"])
        self._update_scan(scan["id"], status="scanning", error="")
        while True:
            self._check_control(scan["id"])
            current = self.get_scan(scan["id"])
            if not current:
                raise _FinderCanceled("Finder scan was deleted")
            if current["pages_completed"] >= current["page_limit"]:
                if self._finalize_scan_if_done(scan["id"]):
                    return
                continue
            page_url = current.get("next_url")
            if not page_url:
                if self._finalize_scan_if_done(scan["id"]):
                    return
                continue
            page_url = validate_source_url(page_url)
            await self._wait_for_request_slot()
            page_number = int(current["pages_completed"]) + 1
            page = await self.scraper.browse(url=page_url, page=page_number)
            cards: list[dict] = []
            seen: set[str] = set()
            for card in page.get("items") or []:
                try:
                    card_url = validate_source_url(card["url"])
                    key = gallery_key(card_url)
                except (KeyError, ValueError):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                cards.append({**card, "url": card_url})
            counts = self._progress_counts(scan["id"])
            missing = self._missing_on_page(scan["id"], cards)
            total = max(
                int(current["total_galleries"]),
                counts["processed_galleries"] + missing,
            )
            self._update_scan(scan["id"], total_galleries=total, **counts)

            batch_size = self.config.finder_network_workers
            order_base = int(current["pages_completed"]) * 10_000
            for start in range(0, len(cards), batch_size):
                self._check_control(scan["id"])
                batch = cards[start : start + batch_size]
                outcomes = await asyncio.gather(
                    *(
                        self._score_gallery(
                            current,
                            card,
                            order_base + start + index,
                            references,
                            reference_hashes,
                            reference_poses,
                        )
                        for index, card in enumerate(batch)
                    ),
                    return_exceptions=True,
                )
                for outcome in outcomes:
                    if isinstance(outcome, (_FinderPaused, _FinderCanceled)):
                        raise outcome
                    if isinstance(outcome, BaseException):
                        raise outcome
                counts = self._progress_counts(scan["id"])
                self._update_scan(scan["id"], **counts)
                update = self.get_scan(scan["id"])
                if update:
                    self._publish(update)

            next_url = page.get("next_url")
            if next_url:
                next_url = validate_source_url(next_url)
            self._update_scan(
                scan["id"],
                pages_completed=page_number,
                next_url=next_url,
            )
            if not next_url:
                if self._finalize_scan_if_done(scan["id"]):
                    return
