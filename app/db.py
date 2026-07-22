from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .security import gallery_key


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path, vfs: str | None = None):
        self.path = path
        self.vfs = vfs
        self._lock = threading.RLock()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.vfs:
            connection = sqlite3.connect(
                f"file:{self.path.resolve()}?vfs={self.vfs}", uri=True, timeout=30
            )
        else:
            connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock, self.connect() as db:
            # Dot-file locks work on Unraid/NFS/FUSE mounts where POSIX byte-range
            # locks can fail. WAL remains available by setting SQLITE_VFS empty
            # on a local filesystem.
            db.execute(f"PRAGMA journal_mode = {'DELETE' if self.vfs else 'WAL'}")
            db.execute("PRAGMA synchronous = NORMAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    name TEXT PRIMARY KEY COLLATE NOCASE,
                    directory TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS history (
                    gallery_key TEXT NOT NULL,
                    url TEXT NOT NULL,
                    profile TEXT NOT NULL COLLATE NOCASE,
                    title TEXT NOT NULL DEFAULT '',
                    destination TEXT NOT NULL DEFAULT '',
                    image_count INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (gallery_key, profile),
                    FOREIGN KEY (profile) REFERENCES profiles(name) ON UPDATE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ignored (
                    gallery_key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    gallery_url TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    profile TEXT NOT NULL COLLATE NOCASE,
                    requested_images TEXT,
                    status TEXT NOT NULL,
                    total INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    destination TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (profile) REFERENCES profiles(name) ON UPDATE CASCADE
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS gallery_images (
                    gallery_key TEXT NOT NULL,
                    gallery_url TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    filename TEXT NOT NULL DEFAULT '',
                    first_seen_at TEXT NOT NULL,
                    PRIMARY KEY (gallery_key, image_url)
                );
                CREATE TABLE IF NOT EXISTS profile_images (
                    profile TEXT NOT NULL COLLATE NOCASE,
                    gallery_key TEXT NOT NULL,
                    gallery_url TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    byte_count INTEGER NOT NULL DEFAULT 0,
                    downloaded_at TEXT NOT NULL,
                    PRIMARY KEY (profile, gallery_key, image_url),
                    FOREIGN KEY (profile) REFERENCES profiles(name) ON UPDATE CASCADE
                );
                CREATE TABLE IF NOT EXISTS job_items (
                    job_id TEXT NOT NULL,
                    image_url TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    byte_count INTEGER NOT NULL DEFAULT 0,
                    relative_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (job_id, image_url),
                    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_profile ON history(profile, completed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_profile_images_gallery
                    ON profile_images(profile, gallery_key);
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO profiles(name, directory, created_at) VALUES (?, ?, ?)",
                ("Default", "Default", utc_now()),
            )
            db.execute(
                """UPDATE jobs
                   SET status = 'canceled', error = ''
                   WHERE cancel_requested = 1
                     AND status IN ('queued', 'starting', 'downloading', 'canceling')"""
            )
            db.execute(
                """UPDATE jobs
                   SET status = 'queued', error = ''
                   WHERE cancel_requested = 0
                     AND status IN ('starting', 'downloading', 'canceling')"""
            )
            db.execute(
                "UPDATE job_items SET status = 'pending' WHERE status = 'downloading'"
            )

    @staticmethod
    def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as db:
            rows = db.execute(
                """
                SELECT p.name, p.directory, p.created_at,
                       COUNT(h.url) AS download_count,
                       COALESCE(SUM(h.image_count), 0) AS image_count
                FROM profiles p LEFT JOIN history h ON h.profile = p.name
                GROUP BY p.name ORDER BY p.name COLLATE NOCASE
                """
            ).fetchall()
            return self._rows(rows)

    def get_profile(self, name: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as db:
            row = db.execute(
                "SELECT * FROM profiles WHERE name = ?", (name,)
            ).fetchone()
            return dict(row) if row else None

    def create_profile(self, name: str, directory: str) -> dict[str, Any]:
        with self._lock, self.connect() as db:
            db.execute(
                "INSERT INTO profiles(name, directory, created_at) VALUES (?, ?, ?)",
                (name, directory, utc_now()),
            )
        return self.get_profile(name) or {}

    def rename_profile(self, name: str, new_name: str, directory: str) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "UPDATE profiles SET name = ?, directory = ? WHERE name = ?",
                (new_name, directory, name),
            )

    def delete_profile(self, name: str) -> None:
        with self._lock, self.connect() as db:
            db.execute("DELETE FROM profiles WHERE name = ?", (name,))

    def status_for_urls(
        self, urls: list[str], profile: str
    ) -> dict[str, dict[str, Any]]:
        if not urls:
            return {}
        keyed = {url: gallery_key(url) for url in urls}
        result: dict[str, dict[str, Any]] = {
            url: {
                "saved": False,
                "ignored": False,
                "partial": False,
                "state": "new",
                "downloaded_images": 0,
                "total_images": 0,
            }
            for url in urls
        }
        keys = list(dict.fromkeys(keyed.values()))
        placeholders = ",".join("?" for _ in keys)
        with self._lock, self.connect() as db:
            saved = db.execute(
                f"SELECT gallery_key FROM history WHERE profile = ? AND gallery_key IN ({placeholders})",
                [profile, *keys],
            ).fetchall()
            ignored = db.execute(
                f"SELECT gallery_key FROM ignored WHERE gallery_key IN ({placeholders})",
                keys,
            ).fetchall()
            downloaded = db.execute(
                f"""SELECT gallery_key, COUNT(*) AS count FROM profile_images
                    WHERE profile = ? AND gallery_key IN ({placeholders}) GROUP BY gallery_key""",
                [profile, *keys],
            ).fetchall()
            totals = db.execute(
                f"""SELECT gallery_key, COUNT(*) AS count FROM gallery_images
                    WHERE gallery_key IN ({placeholders}) GROUP BY gallery_key""",
                keys,
            ).fetchall()
        saved_keys = {row["gallery_key"] for row in saved}
        ignored_keys = {row["gallery_key"] for row in ignored}
        downloaded_counts = {row["gallery_key"]: row["count"] for row in downloaded}
        total_counts = {row["gallery_key"]: row["count"] for row in totals}
        for url, key in keyed.items():
            item = result[url]
            item["saved"] = key in saved_keys
            item["ignored"] = key in ignored_keys
            item["downloaded_images"] = downloaded_counts.get(key, 0)
            item["total_images"] = total_counts.get(key, 0)
            item["partial"] = bool(item["downloaded_images"] and not item["saved"])
            item["state"] = (
                "complete"
                if item["saved"]
                else ("partial" if item["partial"] else "new")
            )
        return result

    def sync(self, profile: str) -> tuple[list[str], list[str]]:
        with self._lock, self.connect() as db:
            saved = [
                r[0]
                for r in db.execute(
                    "SELECT url FROM history WHERE profile = ?", (profile,)
                )
            ]
            ignored = [r[0] for r in db.execute("SELECT url FROM ignored")]
        return saved, ignored

    def set_ignored(self, url: str, ignored: bool, title: str = "") -> None:
        key = gallery_key(url)
        with self._lock, self.connect() as db:
            if ignored:
                db.execute(
                    "INSERT INTO ignored(gallery_key, url, title, created_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(gallery_key) DO UPDATE SET url = excluded.url, title = excluded.title",
                    (key, url, title, utc_now()),
                )
            else:
                db.execute("DELETE FROM ignored WHERE gallery_key = ?", (key,))

    def add_history(
        self, url: str, profile: str, title: str, destination: str, image_count: int
    ) -> None:
        key = gallery_key(url)
        with self._lock, self.connect() as db:
            db.execute(
                """
                INSERT INTO history(gallery_key, url, profile, title, destination, image_count, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(gallery_key, profile) DO UPDATE SET
                    url = excluded.url, title = excluded.title, destination = excluded.destination,
                    image_count = excluded.image_count, completed_at = excluded.completed_at
                """,
                (key, url, profile, title, destination, image_count, utc_now()),
            )

    def register_gallery_images(
        self, gallery_url: str, images: list[dict[str, Any]]
    ) -> None:
        key = gallery_key(gallery_url)
        now = utc_now()
        with self._lock, self.connect() as db:
            for index, image in enumerate(images, start=1):
                db.execute(
                    """INSERT INTO gallery_images(
                           gallery_key, gallery_url, image_url, ordinal, filename, first_seen_at
                       ) VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(gallery_key, image_url) DO UPDATE SET
                           gallery_url = excluded.gallery_url, ordinal = excluded.ordinal,
                           filename = excluded.filename""",
                    (
                        key,
                        gallery_url,
                        image["url"],
                        image.get("ordinal", index),
                        image.get("filename", ""),
                        now,
                    ),
                )

    def image_statuses(self, profile: str, gallery_url: str) -> set[str]:
        key = gallery_key(gallery_url)
        with self._lock, self.connect() as db:
            return {
                row[0]
                for row in db.execute(
                    "SELECT image_url FROM profile_images WHERE profile = ? AND gallery_key = ?",
                    (profile, key),
                )
            }

    def add_profile_image(
        self,
        profile: str,
        gallery_url: str,
        image_url: str,
        relative_path: str,
        byte_count: int,
    ) -> None:
        key = gallery_key(gallery_url)
        with self._lock, self.connect() as db:
            db.execute(
                """INSERT INTO profile_images(
                       profile, gallery_key, gallery_url, image_url, relative_path,
                       byte_count, downloaded_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(profile, gallery_key, image_url) DO UPDATE SET
                       gallery_url = excluded.gallery_url,
                       relative_path = excluded.relative_path,
                       byte_count = excluded.byte_count,
                       downloaded_at = excluded.downloaded_at""",
                (
                    profile,
                    key,
                    gallery_url,
                    image_url,
                    relative_path,
                    byte_count,
                    utc_now(),
                ),
            )

    def create_job_items(self, job_id: str, images: list[dict[str, Any]]) -> None:
        with self._lock, self.connect() as db:
            for index, image in enumerate(images, start=1):
                db.execute(
                    """INSERT OR IGNORE INTO job_items(job_id, image_url, ordinal, status)
                       VALUES (?, ?, ?, 'pending')""",
                    (job_id, image["url"], image.get("ordinal", index)),
                )

    def update_job_item(self, job_id: str, image_url: str, **values: Any) -> None:
        allowed = {"status", "attempts", "byte_count", "relative_path", "error"}
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.connect() as db:
            db.execute(
                f"UPDATE job_items SET {columns} WHERE job_id = ? AND image_url = ?",
                [*values.values(), job_id, image_url],
            )

    def list_job_items(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock, self.connect() as db:
            return self._rows(
                db.execute(
                    "SELECT * FROM job_items WHERE job_id = ? ORDER BY ordinal",
                    (job_id,),
                ).fetchall()
            )

    def list_history(
        self, profile: str | None, limit: int = 250
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM history"
        params: list[Any] = []
        if profile:
            query += " WHERE profile = ?"
            params.append(profile)
        query += " ORDER BY completed_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self.connect() as db:
            return self._rows(db.execute(query, params).fetchall())

    def create_job(self, job: dict[str, Any]) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                """
                INSERT INTO jobs(
                    id, gallery_url, title, profile, requested_images, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job["id"],
                    job["gallery_url"],
                    job.get("title", ""),
                    job["profile"],
                    json.dumps(job.get("image_urls"))
                    if job.get("image_urls") is not None
                    else None,
                    job["created_at"],
                    job["created_at"],
                ),
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "title",
            "status",
            "total",
            "completed",
            "failed",
            "destination",
            "error",
            "cancel_requested",
        }
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.connect() as db:
            db.execute(
                f"UPDATE jobs SET {columns} WHERE id = ?", [*values.values(), job_id]
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._decode_job(dict(row)) if row else None

    @staticmethod
    def _decode_job(job: dict[str, Any]) -> dict[str, Any]:
        raw = job.get("requested_images")
        job["image_urls"] = json.loads(raw) if raw else None
        job["cancel_requested"] = bool(job.get("cancel_requested"))
        job.pop("requested_images", None)
        total = int(job.get("total") or 0)
        done = int(job.get("completed") or 0) + int(job.get("failed") or 0)
        job["progress"] = round(done / total * 100, 1) if total else 0
        return job

    def list_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self.connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._decode_job(dict(row)) for row in rows]

    def active_job_for_gallery(
        self, profile: str, gallery_url: str
    ) -> dict[str, Any] | None:
        target_key = gallery_key(gallery_url)
        with self._lock, self.connect() as db:
            rows = db.execute(
                """SELECT * FROM jobs
                   WHERE profile = ?
                     AND status IN ('queued', 'starting', 'downloading', 'canceling')
                   ORDER BY created_at""",
                (profile,),
            ).fetchall()
        for row in rows:
            job = self._decode_job(dict(row))
            if gallery_key(job["gallery_url"]) == target_key:
                return job
        return None

    def delete_job(self, job_id: str) -> None:
        with self._lock, self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def has_active_jobs(self, profile: str) -> bool:
        with self._lock, self.connect() as db:
            row = db.execute(
                """SELECT 1 FROM jobs WHERE profile = ?
                   AND status IN ('queued', 'starting', 'downloading', 'canceling') LIMIT 1""",
                (profile,),
            ).fetchone()
        return row is not None

    def queued_job_ids(self) -> list[str]:
        with self._lock, self.connect() as db:
            return [
                r[0]
                for r in db.execute(
                    """SELECT id FROM jobs
                   WHERE status = 'queued' AND cancel_requested = 0
                   ORDER BY created_at"""
                )
            ]

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "INSERT INTO settings(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(value)),
            )

    def settings(self) -> dict[str, Any]:
        with self._lock, self.connect() as db:
            rows = db.execute("SELECT key, value FROM settings").fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}
