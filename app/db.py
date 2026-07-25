from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from .security import gallery_key


POSE_ROLES = ("solo", "couple", "group")
DATABASE_INITIALIZE_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0)
DATABASE_STARTUP_CONNECTION_TIMEOUT = 1.0
logger = logging.getLogger(__name__)


class PoseRevisionConflict(RuntimeError):
    def __init__(self, current: dict[str, Any]):
        super().__init__("The pose draft was changed in another browser")
        self.current = current


class DatabaseStartupLockedError(RuntimeError):
    pass


def pose_slug(label: str) -> str:
    normalized = unicodedata.normalize("NFKD", label)
    ascii_label = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_label).strip("-") or "pose"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_sqlite_lock_error(error: sqlite3.OperationalError) -> bool:
    error_code = getattr(error, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        base_code = error_code & 0xFF
        if base_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            return True
    message = str(error).casefold()
    return any(
        phrase in message
        for phrase in (
            "database is locked",
            "database table is locked",
            "database schema is locked",
        )
    )


class Database:
    def __init__(self, path: Path, vfs: str | None = None):
        self.path = path
        self.vfs = vfs
        self._lock = threading.RLock()
        self._startup_connection_timeout: float | None = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        timeout = (
            self._startup_connection_timeout
            if self._startup_connection_timeout is not None
            else 30.0
        )
        if self.vfs:
            connection = sqlite3.connect(
                f"file:{self.path.resolve()}?vfs={self.vfs}", uri=True, timeout=timeout
            )
        else:
            connection = sqlite3.connect(self.path, timeout=timeout)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self, *additional_initializers: Callable[[], None]) -> None:
        initializers = (self._initialize_once, *additional_initializers)
        for retry, delay in enumerate((*DATABASE_INITIALIZE_RETRY_DELAYS, None)):
            try:
                self._startup_connection_timeout = DATABASE_STARTUP_CONNECTION_TIMEOUT
                try:
                    for initializer in initializers:
                        initializer()
                finally:
                    self._startup_connection_timeout = None
                return
            except sqlite3.OperationalError as exc:
                if not is_sqlite_lock_error(exc):
                    raise
                if delay is None:
                    lock_hint = (
                        f" If every instance is stopped, remove only the stale "
                        f"lock directory {self.path}.lock; never delete the "
                        "SQLite database."
                        if self.vfs == "unix-dotfile"
                        else ""
                    )
                    raise DatabaseStartupLockedError(
                        f"SQLite database remained locked during startup: "
                        f"{self.path}. Stop every GalleryFlow container and "
                        f"SQLite tool that uses this database.{lock_hint}"
                    ) from exc
                logger.warning(
                    "SQLite database is locked during startup; retrying in %.2fs "
                    "(attempt %d/%d). Ensure only one GalleryFlow container uses %s.",
                    delay,
                    retry + 1,
                    len(DATABASE_INITIALIZE_RETRY_DELAYS) + 1,
                    self.path,
                )
                time.sleep(delay)

    def _initialize_once(self) -> None:
        with self._lock, self.connect() as db:
            # Dot-file locks work on Unraid/NFS/FUSE mounts where POSIX byte-range
            # locks can fail. WAL remains available by setting SQLITE_VFS empty
            # on a local filesystem.
            desired_journal_mode = "delete" if self.vfs else "wal"
            current_journal_mode = str(
                db.execute("PRAGMA journal_mode").fetchone()[0]
            ).casefold()
            if current_journal_mode != desired_journal_mode:
                applied_journal_mode = str(
                    db.execute(
                        f"PRAGMA journal_mode = {desired_journal_mode.upper()}"
                    ).fetchone()[0]
                ).casefold()
                if applied_journal_mode != desired_journal_mode:
                    raise sqlite3.OperationalError(
                        f"could not set SQLite journal mode to {desired_journal_mode}"
                    )
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
                    kind TEXT NOT NULL DEFAULT 'download',
                    payload TEXT,
                    pair_count INTEGER NOT NULL DEFAULT 0,
                    pose_revision INTEGER,
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
                CREATE TABLE IF NOT EXISTS pose_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    slug TEXT NOT NULL UNIQUE,
                    default_role TEXT NOT NULL CHECK(default_role IN ('solo', 'couple', 'group')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pose_drafts (
                    gallery_key TEXT NOT NULL,
                    gallery_url TEXT NOT NULL,
                    profile TEXT NOT NULL COLLATE NOCASE,
                    revision INTEGER NOT NULL,
                    controls TEXT NOT NULL,
                    targets TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (gallery_key, profile),
                    FOREIGN KEY (profile) REFERENCES profiles(name)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS pose_outputs (
                    gallery_key TEXT NOT NULL,
                    gallery_url TEXT NOT NULL DEFAULT '',
                    ordinal INTEGER NOT NULL,
                    image_url TEXT NOT NULL DEFAULT '',
                    pose_slug TEXT NOT NULL,
                    pose_label TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'solo'
                        CHECK(role IN ('solo', 'couple', 'group')),
                    profile TEXT NOT NULL DEFAULT '',
                    job_id TEXT NOT NULL DEFAULT '',
                    exported_at TEXT NOT NULL,
                    PRIMARY KEY (gallery_key, ordinal)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_history_profile ON history(profile, completed_at DESC);
                CREATE INDEX IF NOT EXISTS idx_profile_images_gallery
                    ON profile_images(profile, gallery_key);
                CREATE INDEX IF NOT EXISTS idx_pose_outputs_pose
                    ON pose_outputs(pose_slug, exported_at DESC);
                """
            )
            job_columns = {
                row[1] for row in db.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "kind" not in job_columns:
                db.execute(
                    "ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'download'"
                )
            if "payload" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN payload TEXT")
            if "pair_count" not in job_columns:
                db.execute(
                    "ALTER TABLE jobs ADD COLUMN pair_count INTEGER NOT NULL DEFAULT 0"
                )
            if "pose_revision" not in job_columns:
                db.execute("ALTER TABLE jobs ADD COLUMN pose_revision INTEGER")
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

    def gallery_images(self, gallery_url: str) -> list[dict[str, Any]]:
        key = gallery_key(gallery_url)
        with self._lock, self.connect() as db:
            return self._rows(
                db.execute(
                    """SELECT image_url AS url, ordinal, filename
                       FROM gallery_images WHERE gallery_key = ? ORDER BY ordinal""",
                    (key,),
                ).fetchall()
            )

    def list_pose_tags(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as db:
            return self._rows(
                db.execute(
                    """SELECT * FROM pose_tags
                       ORDER BY updated_at DESC, label COLLATE NOCASE"""
                ).fetchall()
            )

    def ensure_pose_folder_tags(self, folder_names: list[str]) -> list[dict[str, Any]]:
        """Import existing first-level pose output folders as reusable tags."""

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for folder_name in folder_names:
            slug = str(folder_name)
            label = " ".join(slug.split())
            folded = label.casefold()
            if (
                not label
                or len(label) > 80
                or slug.startswith(".")
                or any(ord(character) < 32 for character in slug)
                or folded in seen
            ):
                continue
            seen.add(folded)
            candidates.append((label, slug))

        if candidates:
            now = utc_now()
            with self._lock, self.connect() as db:
                rows = db.execute("SELECT label, slug FROM pose_tags").fetchall()
                labels = {str(row["label"]).casefold() for row in rows}
                slugs = {str(row["slug"]) for row in rows}
                for label, slug in candidates:
                    if label.casefold() in labels or slug in slugs:
                        continue
                    db.execute(
                        """INSERT INTO pose_tags(
                               label, slug, default_role, created_at, updated_at
                           ) VALUES (?, ?, 'solo', ?, ?)""",
                        (label, slug, now, now),
                    )
                    labels.add(label.casefold())
                    slugs.add(slug)
        return self.list_pose_tags()

    def get_pose_tag(self, tag_id: int) -> dict[str, Any] | None:
        with self._lock, self.connect() as db:
            row = db.execute(
                "SELECT * FROM pose_tags WHERE id = ?", (tag_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_pose_tag(self, label: str, default_role: str) -> dict[str, Any]:
        label = " ".join(label.split()).rstrip("/\\").rstrip()
        if not label:
            raise ValueError("Pose label cannot be empty")
        if default_role not in POSE_ROLES:
            raise ValueError("Invalid control role")
        base = pose_slug(label)
        now = utc_now()
        with self._lock, self.connect() as db:
            candidate = base
            suffix = 2
            while db.execute(
                "SELECT 1 FROM pose_tags WHERE slug = ?", (candidate,)
            ).fetchone():
                candidate = f"{base}-{suffix}"
                suffix += 1
            db.execute(
                """INSERT INTO pose_tags(label, slug, default_role, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (label, candidate, default_role, now, now),
            )
            tag_id = int(db.execute("SELECT last_insert_rowid()").fetchone()[0])
        return self.get_pose_tag(tag_id) or {}

    def import_pose_outputs(self, records: list[dict[str, Any]]) -> None:
        """Remember pose files discovered in an existing output library."""

        if not records:
            return
        with self._lock, self.connect() as db:
            db.executemany(
                """INSERT INTO pose_outputs(
                       gallery_key, gallery_url, ordinal, image_url, pose_slug,
                       pose_label, role, profile, job_id, exported_at
                   ) VALUES (?, '', ?, '', ?, ?, 'solo', '', '', ?)
                   ON CONFLICT(gallery_key, ordinal) DO NOTHING""",
                (
                    (
                        record["gallery_key"],
                        int(record["ordinal"]),
                        record["pose_slug"],
                        record["pose_label"],
                        record["exported_at"],
                    )
                    for record in records
                ),
            )

    def record_pose_outputs(
        self,
        gallery_url: str,
        profile: str,
        job_id: str,
        targets: list[dict[str, Any]],
    ) -> None:
        if not targets:
            return
        key = gallery_key(gallery_url)
        now = utc_now()
        with self._lock, self.connect() as db:
            db.executemany(
                """INSERT INTO pose_outputs(
                       gallery_key, gallery_url, ordinal, image_url, pose_slug,
                       pose_label, role, profile, job_id, exported_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(gallery_key, ordinal) DO UPDATE SET
                       gallery_url = excluded.gallery_url,
                       image_url = excluded.image_url,
                       pose_slug = excluded.pose_slug,
                       pose_label = excluded.pose_label,
                       role = excluded.role,
                       profile = excluded.profile,
                       job_id = excluded.job_id,
                       exported_at = excluded.exported_at""",
                (
                    (
                        key,
                        gallery_url,
                        int(target["ordinal"]),
                        target["image_url"],
                        target["pose_slug"],
                        target.get("pose_label") or target["pose_slug"],
                        target.get("role") or "solo",
                        profile,
                        job_id,
                        now,
                    )
                    for target in targets
                ),
            )

    def pose_outputs_for_urls(
        self, gallery_urls: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        keys = list(dict.fromkeys(gallery_key(url) for url in gallery_urls))
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._lock, self.connect() as db:
            rows = db.execute(
                f"""SELECT * FROM pose_outputs
                    WHERE gallery_key IN ({placeholders})
                    ORDER BY exported_at DESC, ordinal""",
                keys,
            ).fetchall()
        result = {key: [] for key in keys}
        for row in rows:
            item = dict(row)
            result[item["gallery_key"]].append(item)
        return result

    def update_pose_tag(
        self,
        tag_id: int,
        *,
        label: str | None = None,
        default_role: str | None = None,
    ) -> dict[str, Any] | None:
        values: dict[str, Any] = {}
        if label is not None:
            label = " ".join(label.split()).rstrip("/\\").rstrip()
            if not label:
                raise ValueError("Pose label cannot be empty")
            values["label"] = label
        if default_role is not None:
            if default_role not in POSE_ROLES:
                raise ValueError("Invalid control role")
            values["default_role"] = default_role
        if not values:
            return self.get_pose_tag(tag_id)
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.connect() as db:
            result = db.execute(
                f"UPDATE pose_tags SET {columns} WHERE id = ?",
                [*values.values(), tag_id],
            )
            if result.rowcount == 0:
                return None
        return self.get_pose_tag(tag_id)

    @staticmethod
    def _decode_pose_draft(row: dict[str, Any]) -> dict[str, Any]:
        row["controls"] = json.loads(row["controls"])
        row["targets"] = json.loads(row["targets"])
        row.pop("gallery_key", None)
        return row

    def get_pose_draft(self, gallery_url: str, profile: str) -> dict[str, Any]:
        key = gallery_key(gallery_url)
        with self._lock, self.connect() as db:
            row = db.execute(
                """SELECT * FROM pose_drafts
                   WHERE gallery_key = ? AND profile = ?""",
                (key, profile),
            ).fetchone()
            if not row:
                return {
                    "gallery_url": gallery_url,
                    "profile": profile,
                    "revision": 0,
                    "controls": {role: None for role in POSE_ROLES},
                    "targets": [],
                    "updated_at": None,
                }
            draft = self._decode_pose_draft(dict(row))
            tags = {
                item["id"]: dict(item)
                for item in db.execute("SELECT * FROM pose_tags").fetchall()
            }
        for target in draft["targets"]:
            tag = tags.get(target["pose_tag_id"])
            if tag:
                target.update(
                    pose_slug=tag["slug"],
                    pose_label=tag["label"],
                )
        return draft

    def save_pose_draft(
        self,
        gallery_url: str,
        profile: str,
        expected_revision: int,
        controls: dict[str, str | None],
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        key = gallery_key(gallery_url)
        now = utc_now()
        with self._lock, self.connect() as db:
            current_row = db.execute(
                """SELECT * FROM pose_drafts
                   WHERE gallery_key = ? AND profile = ?""",
                (key, profile),
            ).fetchone()
            current_revision = int(current_row["revision"]) if current_row else 0
            if current_revision != expected_revision:
                current = (
                    self._decode_pose_draft(dict(current_row))
                    if current_row
                    else {
                        "gallery_url": gallery_url,
                        "profile": profile,
                        "revision": 0,
                        "controls": {role: None for role in POSE_ROLES},
                        "targets": [],
                        "updated_at": None,
                    }
                )
                raise PoseRevisionConflict(current)
            revision = current_revision + 1
            db.execute(
                """INSERT INTO pose_drafts(
                       gallery_key, gallery_url, profile, revision, controls, targets, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(gallery_key, profile) DO UPDATE SET
                       gallery_url = excluded.gallery_url,
                       revision = excluded.revision,
                       controls = excluded.controls,
                       targets = excluded.targets,
                       updated_at = excluded.updated_at""",
                (
                    key,
                    gallery_url,
                    profile,
                    revision,
                    json.dumps(controls, separators=(",", ":")),
                    json.dumps(targets, separators=(",", ":")),
                    now,
                ),
            )
            pose_tag_ids = {
                int(target["pose_tag_id"])
                for target in targets
                if target.get("pose_tag_id") is not None
            }
            if pose_tag_ids:
                db.executemany(
                    "UPDATE pose_tags SET updated_at = ? WHERE id = ?",
                    ((now, tag_id) for tag_id in pose_tag_ids),
                )
        return self.get_pose_draft(gallery_url, profile)

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

    def record_job_item_result(
        self,
        job_id: str,
        image_url: str,
        *,
        status: str,
        completed_delta: int = 0,
        failed_delta: int = 0,
        item_error: str = "",
        job_error: str = "",
        attempts: int | None = None,
        byte_count: int = 0,
        relative_path: str = "",
        profile: str | None = None,
        gallery_url: str | None = None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Persist one transfer result and its job progress atomically."""

        if status not in {"completed", "failed", "skipped"}:
            raise ValueError("Invalid terminal job item status")
        if completed_delta < 0 or failed_delta < 0:
            raise ValueError("Job progress increments cannot be negative")
        if profile is not None and (status != "completed" or not gallery_url):
            raise ValueError(
                "Completed profile images require a gallery URL and completed status"
            )

        item_values: dict[str, Any] = {
            "status": status,
            "byte_count": byte_count,
            "relative_path": relative_path,
            "error": item_error,
        }
        if attempts is not None:
            item_values["attempts"] = attempts
        item_columns = ", ".join(f"{key} = ?" for key in item_values)
        now = utc_now()
        with self._lock, self.connect() as db:
            progress = db.execute(
                """UPDATE jobs
                   SET completed = completed + ?, failed = failed + ?,
                       error = ?, updated_at = ?
                   WHERE id = ? AND cancel_requested = 0""",
                (completed_delta, failed_delta, job_error, now, job_id),
            )
            if progress.rowcount != 1:
                row = self._job_summary_row(db, job_id)
                summary = self._decode_job_summary(dict(row)) if row else None
                return summary, False

            updated = db.execute(
                f"UPDATE job_items SET {item_columns} "
                "WHERE job_id = ? AND image_url = ?",
                [*item_values.values(), job_id, image_url],
            )
            if updated.rowcount != 1:
                raise ValueError("Download job item not found")

            if profile is not None:
                assert gallery_url is not None
                key = gallery_key(gallery_url)
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
                        now,
                    ),
                )

            row = self._job_summary_row(db, job_id)
        summary = self._decode_job_summary(dict(row)) if row else None
        return summary, True

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
                    kind, payload, pair_count, pose_revision, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["id"],
                    job["gallery_url"],
                    job.get("title", ""),
                    job["profile"],
                    json.dumps(job.get("image_urls"))
                    if job.get("image_urls") is not None
                    else None,
                    job.get("kind", "download"),
                    json.dumps(job.get("payload"), separators=(",", ":"))
                    if job.get("payload") is not None
                    else None,
                    int(job.get("pair_count") or 0),
                    job.get("pose_revision"),
                    job["created_at"],
                    job["created_at"],
                ),
            )

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any] | None:
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
            return self.get_job_summary(job_id)
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.connect() as db:
            db.execute(
                f"UPDATE jobs SET {columns} WHERE id = ?", [*values.values(), job_id]
            )
            row = self._job_summary_row(db, job_id)
        return self._decode_job_summary(dict(row)) if row else None

    def finish_job(
        self, job_id: str, status: str, error: str = ""
    ) -> dict[str, Any] | None:
        """Publish a terminal state without racing a concurrent cancellation."""

        terminal = {"completed", "completed_with_errors", "failed", "canceled"}
        if status not in terminal:
            raise ValueError("Invalid terminal job status")
        now = utc_now()
        with self._lock, self.connect() as db:
            db.execute(
                """UPDATE jobs
                   SET status = CASE
                           WHEN cancel_requested = 1 THEN 'canceled'
                           ELSE ?
                       END,
                       error = CASE
                           WHEN cancel_requested = 1 THEN ''
                           ELSE ?
                       END,
                       updated_at = ?
                   WHERE id = ?""",
                (status, error, now, job_id),
            )
            row = self._job_summary_row(db, job_id)
        return self._decode_job_summary(dict(row)) if row else None

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._decode_job(dict(row)) if row else None

    @staticmethod
    def _decode_job(job: dict[str, Any]) -> dict[str, Any]:
        raw = job.get("requested_images")
        job["image_urls"] = json.loads(raw) if raw else None
        raw_payload = job.get("payload")
        job["payload"] = json.loads(raw_payload) if raw_payload else None
        job["kind"] = job.get("kind") or "download"
        job["pair_count"] = int(job.get("pair_count") or 0)
        if job["kind"] == "pose_export" and job["payload"]:
            if not job["pair_count"]:
                job["pair_count"] = len(job["payload"].get("targets", []))
            if job.get("pose_revision") is None:
                job["pose_revision"] = job["payload"].get("revision")
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

    @staticmethod
    def _decode_job_summary(job: dict[str, Any]) -> dict[str, Any]:
        job["cancel_requested"] = bool(job.get("cancel_requested"))
        job["kind"] = job.get("kind") or "download"
        job["pair_count"] = int(job.get("pair_count") or 0)
        total = int(job.get("total") or 0)
        done = int(job.get("completed") or 0) + int(job.get("failed") or 0)
        job["progress"] = round(done / total * 100, 1) if total else 0
        return job

    def get_job_summary(self, job_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as db:
            row = self._job_summary_row(db, job_id)
        return self._decode_job_summary(dict(row)) if row else None

    @staticmethod
    def _job_summary_row(db: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
        return db.execute(
            """SELECT id, gallery_url, title, profile, status, total, completed,
                      failed, destination, error, cancel_requested, kind,
                      pair_count, pose_revision, created_at, updated_at
               FROM jobs WHERE id = ?""",
            (job_id,),
        ).fetchone()

    def clear_job_item_paths(self, job_id: str) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "UPDATE job_items SET relative_path = '' WHERE job_id = ?",
                (job_id,),
            )

    def list_job_summaries(
        self, limit: int = 100, kind: str | None = None
    ) -> list[dict[str, Any]]:
        where = " WHERE kind = ?" if kind is not None else ""
        params: list[Any] = [kind] if kind is not None else []
        params.append(limit)
        with self._lock, self.connect() as db:
            rows = db.execute(
                f"""SELECT id, gallery_url, title, profile, status, total, completed,
                           failed, destination, error, cancel_requested, kind,
                           pair_count, pose_revision, created_at, updated_at
                    FROM jobs{where} ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()
        return [self._decode_job_summary(dict(row)) for row in rows]

    def job_cancel_requested(self, job_id: str) -> bool:
        with self._lock, self.connect() as db:
            row = db.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return row is None or bool(row[0])

    def active_job_for_gallery(
        self, profile: str | None, gallery_url: str, kind: str = "download"
    ) -> dict[str, Any] | None:
        target_key = gallery_key(gallery_url)
        profile_clause = " AND profile = ?" if profile is not None else ""
        params: list[Any] = [kind]
        if profile is not None:
            params.append(profile)
        with self._lock, self.connect() as db:
            rows = db.execute(
                f"""SELECT * FROM jobs
                   WHERE kind = ?{profile_clause}
                     AND status IN ('queued', 'starting', 'downloading', 'canceling')
                   ORDER BY created_at""",
                params,
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

    def list_jobs_by_kind(self, kind: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock, self.connect() as db:
            rows = db.execute(
                """SELECT * FROM jobs WHERE kind = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (kind, limit),
            ).fetchall()
        return [self._decode_job(dict(row)) for row in rows]

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
