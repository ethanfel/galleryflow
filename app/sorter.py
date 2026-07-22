from __future__ import annotations

import json
import shutil
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import AppConfig
from .db import Database, utc_now
from .security import clean_profile_name, confined_path, sign_media_url


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".avif",
}
OUTPUT_DIRECTORIES = {
    "selected_control",
    "selected_target",
    "control_selected_solo_woman",
    "selected_target_solo_woman",
    "selected_target_no_control",
    "skipped_target",
}
SESSION_RETENTION = 200


class SortConflict(RuntimeError):
    pass


class SortNotFound(LookupError):
    pass


class SorterService:
    """Persistent, server-side replacement for the two legacy image sorters."""

    def __init__(self, config: AppConfig, database: Database):
        self.config = config
        self.database = database
        self.root = config.sort_root_path.resolve()
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        with self._lock, self.database.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sort_profiles (
                    name TEXT PRIMARY KEY COLLATE NOCASE,
                    target_directory TEXT NOT NULL,
                    control_directories TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    threshold_seconds REAL NOT NULL,
                    add_ids INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sort_sessions (
                    id TEXT PRIMARY KEY,
                    target_directory TEXT NOT NULL,
                    control_directories TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    threshold_seconds REAL NOT NULL,
                    add_ids INTEGER NOT NULL,
                    target_total INTEGER NOT NULL,
                    match_counter INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sort_target_files (
                    session_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    stem TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY (session_id, relative_path),
                    FOREIGN KEY (session_id) REFERENCES sort_sessions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sort_control_files (
                    session_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    stem TEXT NOT NULL,
                    PRIMARY KEY (session_id, relative_path),
                    FOREIGN KEY (session_id) REFERENCES sort_sessions(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS sort_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    target_source TEXT NOT NULL,
                    target_destination TEXT NOT NULL,
                    control_source TEXT NOT NULL DEFAULT '',
                    control_copy TEXT NOT NULL DEFAULT '',
                    renamed INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    undone_at TEXT,
                    FOREIGN KEY (session_id) REFERENCES sort_sessions(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sort_targets_pending
                    ON sort_target_files(session_id, status, ordinal);
                CREATE INDEX IF NOT EXISTS idx_sort_controls_time
                    ON sort_control_files(session_id, mtime);
                CREATE INDEX IF NOT EXISTS idx_sort_controls_stem
                    ON sort_control_files(session_id, stem);
                CREATE INDEX IF NOT EXISTS idx_sort_actions_active
                    ON sort_actions(session_id, undone_at, id DESC);
                """
            )
        self.reconcile_incomplete()

    @staticmethod
    def _stem(filename: str) -> str:
        return Path(filename).stem.split("_", 1)[0]

    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _relative(self, path: Path) -> str:
        resolved = path.resolve()
        relative = resolved.relative_to(self.root)
        return "." if relative == Path(".") else relative.as_posix()

    def _resolve(
        self,
        value: str,
        *,
        directory: bool = False,
        file: bool = False,
    ) -> tuple[Path, str]:
        raw = (value or "").strip()
        if not raw:
            raise ValueError("A path relative to the configured sort root is required")
        candidate_path = Path(raw)
        if candidate_path.is_absolute():
            raise ValueError(
                "Sorter paths must be relative to the configured sort root"
            )
        candidate = confined_path(self.root, raw)
        if directory and not candidate.is_dir():
            raise ValueError(f"Sorter directory does not exist: {raw}")
        if file and (not candidate.is_file() or not self._is_image(candidate)):
            raise ValueError(f"Sorter image does not exist: {raw}")
        return candidate, self._relative(candidate)

    def _scan_images(self, directory: Path) -> list[Path]:
        images: list[Path] = []
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            raise ValueError(
                f"Could not scan sorter directory: {directory.name}"
            ) from exc
        for path in entries:
            try:
                if path.is_symlink():
                    continue
                resolved = path.resolve()
                resolved.relative_to(self.root)
                if path.is_file() and self._is_image(path):
                    images.append(resolved)
            except (OSError, ValueError):
                continue
        return sorted(images, key=lambda item: (item.name.casefold(), item.name))

    def folders(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        truncated = False
        with self._lock:
            pending = deque([self.root])
            while pending:
                directory = pending.popleft()
                try:
                    resolved = directory.resolve()
                    resolved.relative_to(self.root)
                    if directory != self.root and directory.is_symlink():
                        continue
                    children = sorted(
                        (item for item in directory.iterdir() if item.is_dir()),
                        key=lambda item: (item.name.casefold(), item.name),
                    )
                except (OSError, ValueError):
                    continue
                relative = self._relative(resolved)
                try:
                    image_count = len(self._scan_images(resolved))
                except ValueError:
                    image_count = 0
                items.append(
                    {
                        "path": relative,
                        "name": "Sort root" if relative == "." else relative,
                        "image_count": image_count,
                    }
                )
                if len(items) >= 2_000:
                    truncated = bool(children or pending)
                    break
                for child in children:
                    try:
                        child.resolve().relative_to(self.root)
                    except (OSError, ValueError):
                        continue
                    pending.append(child)
        return {"root": str(self.root), "items": items, "truncated": truncated}

    def _options(self, data: dict[str, Any]) -> dict[str, Any]:
        target, target_relative = self._resolve(
            str(data.get("target_directory") or ""), directory=True
        )
        mode = str(data.get("mode") or "time").lower()
        if mode not in {"time", "stem"}:
            raise ValueError("Sorter mode must be 'time' or 'stem'")
        try:
            threshold = float(data.get("threshold_seconds", 50))
        except (TypeError, ValueError) as exc:
            raise ValueError("Time threshold must be a number") from exc
        if not 0 <= threshold <= 3_600:
            raise ValueError("Time threshold must be between 0 and 3600 seconds")

        controls: list[tuple[Path, str]] = []
        raw_controls = data.get("control_directories") or []
        if not isinstance(raw_controls, list):
            raise ValueError("Control directories must be a list")
        if len(raw_controls) > 1_000:
            raise ValueError("Too many control directories")
        for value in raw_controls:
            path, relative = self._resolve(str(value), directory=True)
            if path == target or relative in {item[1] for item in controls}:
                continue
            controls.append((path, relative))
        if not controls:
            try:
                siblings = sorted(
                    (item for item in target.parent.iterdir() if item.is_dir()),
                    key=lambda item: (item.name.casefold(), item.name),
                )
            except OSError as exc:
                raise ValueError("Could not discover sibling control folders") from exc
            for sibling in siblings:
                if (
                    sibling == target
                    or sibling.name in OUTPUT_DIRECTORIES
                    or sibling.is_symlink()
                ):
                    continue
                try:
                    sibling.resolve().relative_to(self.root)
                except (OSError, ValueError):
                    continue
                controls.append((sibling.resolve(), self._relative(sibling)))
        return {
            "target": target,
            "target_directory": target_relative,
            "controls": controls,
            "control_directories": [item[1] for item in controls],
            "mode": mode,
            "threshold_seconds": threshold,
            "add_ids": bool(data.get("add_ids", True)),
        }

    @staticmethod
    def _decode_profile(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["control_directories"] = json.loads(item["control_directories"] or "[]")
        item["add_ids"] = bool(item["add_ids"])
        return item

    def list_profiles(self) -> list[dict[str, Any]]:
        with self._lock, self.database.connect() as db:
            rows = db.execute(
                "SELECT * FROM sort_profiles ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [self._decode_profile(row) for row in rows]

    def save_profile(self, data: dict[str, Any]) -> dict[str, Any]:
        name = clean_profile_name(str(data.get("name") or ""))
        options = self._options(data)
        # An empty list means "auto-discover siblings" and must remain empty in
        # a reusable profile rather than freezing today's sibling inventory.
        stored_controls = (
            options["control_directories"] if data.get("control_directories") else []
        )
        now = utc_now()
        with self._lock, self.database.connect() as db:
            db.execute(
                """INSERT INTO sort_profiles(
                       name, target_directory, control_directories, mode,
                       threshold_seconds, add_ids, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                       target_directory = excluded.target_directory,
                       control_directories = excluded.control_directories,
                       mode = excluded.mode,
                       threshold_seconds = excluded.threshold_seconds,
                       add_ids = excluded.add_ids,
                       updated_at = excluded.updated_at""",
                (
                    name,
                    options["target_directory"],
                    json.dumps(stored_controls),
                    options["mode"],
                    options["threshold_seconds"],
                    int(options["add_ids"]),
                    now,
                    now,
                ),
            )
            row = db.execute(
                "SELECT * FROM sort_profiles WHERE name = ?", (name,)
            ).fetchone()
        return self._decode_profile(row)

    def delete_profile(self, name: str) -> bool:
        name = clean_profile_name(name)
        with self._lock, self.database.connect() as db:
            cursor = db.execute("DELETE FROM sort_profiles WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def start_session(self, data: dict[str, Any]) -> dict[str, Any]:
        self.reconcile_incomplete()
        options = self._options(data)
        targets = self._scan_images(options["target"])
        controls: list[Path] = []
        for directory, _ in options["controls"]:
            controls.extend(self._scan_images(directory))
        controls = list(dict.fromkeys(controls))
        target_records: list[tuple[Path, Any]] = []
        for path in targets:
            try:
                target_records.append((path, path.stat()))
            except OSError:
                continue
        control_records: list[tuple[Path, Any]] = []
        for path in controls:
            try:
                control_records.append((path, path.stat()))
            except OSError:
                continue
        session_id = uuid.uuid4().hex
        now = utc_now()
        with self._lock, self.database.connect() as db:
            unresolved = db.execute(
                """SELECT 1
                   FROM sort_sessions s
                   JOIN sort_target_files t ON t.session_id = s.id
                   WHERE s.target_directory = ?
                     AND t.status IN ('applying', 'undoing')
                   LIMIT 1""",
                (options["target_directory"],),
            ).fetchone()
            if unresolved:
                raise SortConflict(
                    "This folder has an interrupted sorter operation; resolve its files before rescanning"
                )
            previous = db.execute(
                """SELECT id FROM sort_sessions
                   WHERE target_directory = ? AND status = 'active'""",
                (options["target_directory"],),
            ).fetchall()
            previous_ids = [row["id"] for row in previous]
            if previous_ids:
                placeholders = ",".join("?" for _ in previous_ids)
                db.execute(
                    f"UPDATE sort_sessions SET status = 'superseded', updated_at = ? "
                    f"WHERE id IN ({placeholders})",
                    (now, *previous_ids),
                )
                db.execute(
                    f"UPDATE sort_target_files SET status = 'superseded' "
                    f"WHERE status = 'pending' AND session_id IN ({placeholders})",
                    previous_ids,
                )
            db.execute(
                """INSERT INTO sort_sessions(
                       id, target_directory, control_directories, mode,
                       threshold_seconds, add_ids, target_total, match_counter,
                       status, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)""",
                (
                    session_id,
                    options["target_directory"],
                    json.dumps(options["control_directories"]),
                    options["mode"],
                    options["threshold_seconds"],
                    int(options["add_ids"]),
                    len(target_records),
                    "active" if target_records else "completed",
                    now,
                    now,
                ),
            )
            for ordinal, (path, stat) in enumerate(target_records, start=1):
                db.execute(
                    """INSERT INTO sort_target_files(
                           session_id, relative_path, filename, mtime, stem, ordinal
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        self._relative(path),
                        path.name,
                        stat.st_mtime,
                        self._stem(path.name),
                        ordinal,
                    ),
                )
            for path, stat in control_records:
                db.execute(
                    """INSERT OR IGNORE INTO sort_control_files(
                           session_id, relative_path, filename, mtime, stem
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        self._relative(path),
                        path.name,
                        stat.st_mtime,
                        self._stem(path.name),
                    ),
                )
            db.execute(
                """DELETE FROM sort_sessions
                   WHERE status IN ('completed', 'superseded')
                     AND id NOT IN (
                         SELECT session_id FROM sort_target_files
                         WHERE status IN ('applying', 'undoing')
                     )
                     AND id NOT IN (
                         SELECT id FROM sort_sessions
                         ORDER BY updated_at DESC LIMIT ?
                     )""",
                (SESSION_RETENTION,),
            )
        return self.get_session(session_id) or {}

    @staticmethod
    def _decode_session(row: Any) -> dict[str, Any]:
        item = dict(row)
        item["control_directories"] = json.loads(item["control_directories"] or "[]")
        item["add_ids"] = bool(item["add_ids"])
        return item

    def _preview_url(self, relative_path: str) -> str:
        signed_value = f"sort:{relative_path}"
        token = sign_media_url(signed_value, self.config.media_signing_key)
        return f"/api/sort/media?path={quote(relative_path, safe='')}&token={token}"

    def _file_payload(
        self, row: Any, *, target_mtime: float | None = None
    ) -> dict[str, Any]:
        item = dict(row)
        timestamp = float(item["mtime"])
        payload: dict[str, Any] = {
            "path": item["relative_path"],
            "name": item["filename"],
            "preview_url": self._preview_url(item["relative_path"]),
            "modified_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
        }
        parent = Path(item["relative_path"]).parent.as_posix()
        payload["folder"] = "Sort root" if parent == "." else parent
        if target_mtime is not None:
            payload["delta_seconds"] = round(abs(timestamp - target_mtime), 3)
        return payload

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        self.reconcile_incomplete()
        with self._lock, self.database.connect() as db:
            session_row = db.execute(
                "SELECT * FROM sort_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not session_row:
                return None
            session = self._decode_session(session_row)
            current = None
            if session["status"] != "superseded":
                while True:
                    current = db.execute(
                        """SELECT * FROM sort_target_files
                           WHERE session_id = ? AND status = 'pending'
                           ORDER BY ordinal LIMIT 1""",
                        (session_id,),
                    ).fetchone()
                    if not current:
                        break
                    try:
                        self._resolve(current["relative_path"], file=True)
                        break
                    except ValueError:
                        db.execute(
                            """UPDATE sort_target_files SET status = 'missing'
                               WHERE session_id = ? AND relative_path = ?""",
                            (session_id, current["relative_path"]),
                        )
            pending = db.execute(
                """SELECT COUNT(*) FROM sort_target_files
                   WHERE session_id = ? AND status = 'pending'""",
                (session_id,),
            ).fetchone()[0]
            recovering = db.execute(
                """SELECT COUNT(*) FROM sort_target_files
                   WHERE session_id = ? AND status IN ('applying', 'undoing')""",
                (session_id,),
            ).fetchone()[0]
            remaining = int(pending) + int(recovering)
            processed = db.execute(
                """SELECT COUNT(*) FROM sort_target_files
                   WHERE session_id = ? AND status = 'processed'""",
                (session_id,),
            ).fetchone()[0]
            missing = db.execute(
                """SELECT COUNT(*) FROM sort_target_files
                   WHERE session_id = ? AND status = 'missing'""",
                (session_id,),
            ).fetchone()[0]
            can_undo = (
                not recovering
                and session["status"] != "superseded"
                and db.execute(
                    """SELECT 1 FROM sort_actions
                   WHERE session_id = ? AND undone_at IS NULL LIMIT 1""",
                    (session_id,),
                ).fetchone()
                is not None
            )
            matches: list[Any] = []
            if current:
                if session["mode"] == "time":
                    low = current["mtime"] - session["threshold_seconds"]
                    high = current["mtime"] + session["threshold_seconds"]
                    matches = db.execute(
                        """SELECT * FROM sort_control_files
                           WHERE session_id = ? AND mtime BETWEEN ? AND ?
                           ORDER BY ABS(mtime - ?), relative_path COLLATE NOCASE
                           LIMIT 200""",
                        (session_id, low, high, current["mtime"]),
                    ).fetchall()
                else:
                    matches = db.execute(
                        """SELECT * FROM sort_control_files
                           WHERE session_id = ? AND stem = ?
                           ORDER BY relative_path COLLATE NOCASE LIMIT 200""",
                        (session_id, current["stem"]),
                    ).fetchall()
        current_payload = self._file_payload(current) if current else None
        match_payloads: list[dict[str, Any]] = []
        if current:
            for match in matches:
                try:
                    self._resolve(match["relative_path"], file=True)
                except ValueError:
                    continue
                match_payloads.append(
                    self._file_payload(match, target_mtime=float(current["mtime"]))
                )
        status = (
            "superseded"
            if session["status"] == "superseded"
            else "recovering"
            if recovering
            else "completed"
            if not pending
            else "active"
        )
        stored_status = "active" if status == "recovering" else status
        if stored_status != session["status"]:
            with self._lock, self.database.connect() as db:
                db.execute(
                    "UPDATE sort_sessions SET status = ?, updated_at = ? WHERE id = ?",
                    (stored_status, utc_now(), session_id),
                )
        total = int(session.pop("target_total"))
        return {
            **session,
            "status": status,
            "processed": processed,
            "remaining": int(remaining),
            "missing": int(missing),
            "recovering": int(recovering),
            "total": total,
            "can_undo": can_undo,
            "current": current_payload,
            "matches": match_payloads,
        }

    def _session_and_current(
        self, db: Any, session_id: str
    ) -> tuple[dict[str, Any], Any]:
        session_row = db.execute(
            "SELECT * FROM sort_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not session_row:
            raise SortNotFound("Sort session not found")
        if session_row["status"] != "active":
            raise SortConflict("This sort session is no longer active")
        current = db.execute(
            """SELECT * FROM sort_target_files
               WHERE session_id = ? AND status = 'pending'
               ORDER BY ordinal LIMIT 1""",
            (session_id,),
        ).fetchone()
        if not current:
            raise SortConflict("This sort session has no pending target")
        return self._decode_session(session_row), current

    @staticmethod
    def _unique_destination(directory: Path, filename: str) -> Path:
        candidate = directory / filename
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        for number in range(1, 100_000):
            candidate = directory / f"{stem}_copy{number}{suffix}"
            if not candidate.exists() and not candidate.is_symlink():
                return candidate
        raise SortConflict("Could not allocate a collision-safe destination filename")

    def apply_action(
        self,
        session_id: str,
        kind: str,
        expected_target: str,
        control_path: str | None = None,
    ) -> dict[str, Any]:
        if kind not in {"match", "solo", "no_control", "skip"}:
            raise ValueError("Unknown sorter action")
        with self._lock:
            with self.database.connect() as db:
                session, current = self._session_and_current(db, session_id)
                if current["relative_path"] != expected_target:
                    raise SortConflict(
                        "The current target changed; refresh before acting"
                    )
                target_source, _ = self._resolve(current["relative_path"], file=True)
                control_source: Path | None = None
                control_relative = ""
                if kind in {"match", "solo"}:
                    if not control_path:
                        raise ValueError("A control image is required for this action")
                    control_row = db.execute(
                        """SELECT * FROM sort_control_files
                           WHERE session_id = ? AND relative_path = ?""",
                        (session_id, control_path),
                    ).fetchone()
                    if not control_row:
                        raise ValueError(
                            "Control image is not part of this sort session"
                        )
                    if session["mode"] == "time" and abs(
                        float(control_row["mtime"]) - float(current["mtime"])
                    ) > float(session["threshold_seconds"]):
                        raise SortConflict(
                            "Control image is outside the session time threshold"
                        )
                    if (
                        session["mode"] == "stem"
                        and control_row["stem"] != current["stem"]
                    ):
                        raise SortConflict(
                            "Control image no longer matches the target filename"
                        )
                    control_source, control_relative = self._resolve(
                        control_path, file=True
                    )

                target_root, _ = self._resolve(
                    session["target_directory"], directory=True
                )
                output_names = {
                    "match": ("selected_target", "selected_control"),
                    "solo": (
                        "selected_target_solo_woman",
                        "control_selected_solo_woman",
                    ),
                    "no_control": ("selected_target_no_control", None),
                    "skip": ("skipped_target", None),
                }
                target_folder = confined_path(target_root, output_names[kind][0])
                target_folder.mkdir(parents=True, exist_ok=True)
                prefix = ""
                renamed = bool(kind in {"match", "solo"} and session["add_ids"])
                if renamed:
                    prefix = f"id{int(session['match_counter']):03d}_"
                target_destination = self._unique_destination(
                    target_folder, prefix + target_source.name
                )
                control_copy: Path | None = None
                if control_source is not None:
                    control_folder = confined_path(
                        target_root, output_names[kind][1] or ""
                    )
                    control_folder.mkdir(parents=True, exist_ok=True)
                    control_copy = self._unique_destination(
                        control_folder, prefix + control_source.name
                    )

                action = db.execute(
                    """INSERT INTO sort_actions(
                           session_id, kind, target_source, target_destination,
                           control_source, control_copy, renamed, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        kind,
                        current["relative_path"],
                        self._relative(target_destination),
                        control_relative,
                        self._relative(control_copy) if control_copy else "",
                        int(renamed),
                        utc_now(),
                    ),
                )
                action_id = int(action.lastrowid)
                db.execute(
                    """UPDATE sort_target_files SET status = 'applying'
                       WHERE session_id = ? AND relative_path = ?""",
                    (session_id, current["relative_path"]),
                )
                # Persist intent before touching files so a restart can safely
                # finish or roll back an interrupted operation.
                db.commit()

                try:
                    if control_source is not None and control_copy is not None:
                        shutil.copy2(control_source, control_copy)
                    shutil.move(str(target_source), str(target_destination))
                except Exception:
                    source_exists = target_source.exists() or target_source.is_symlink()
                    destination_exists = (
                        target_destination.exists() or target_destination.is_symlink()
                    )
                    if source_exists and destination_exists:
                        # A concurrent writer occupied the destination while the
                        # operation failed. Neither file is known to be ours, so
                        # preserve both and leave the durable journal applying.
                        raise
                    if destination_exists:
                        shutil.move(str(target_destination), str(target_source))
                    if control_copy and (
                        control_copy.exists() or control_copy.is_symlink()
                    ):
                        control_copy.unlink()
                    db.execute(
                        "UPDATE sort_actions SET undone_at = ? WHERE id = ?",
                        (utc_now(), action_id),
                    )
                    db.execute(
                        """UPDATE sort_target_files SET status = 'pending'
                           WHERE session_id = ? AND relative_path = ?""",
                        (session_id, current["relative_path"]),
                    )
                    db.commit()
                    raise

                try:
                    db.execute(
                        """UPDATE sort_target_files SET status = 'processed'
                           WHERE session_id = ? AND relative_path = ?""",
                        (session_id, current["relative_path"]),
                    )
                    remaining = db.execute(
                        """SELECT COUNT(*) FROM sort_target_files
                           WHERE session_id = ? AND status = 'pending'""",
                        (session_id,),
                    ).fetchone()[0]
                    db.execute(
                        """UPDATE sort_sessions
                           SET status = ?, match_counter = match_counter + ?, updated_at = ?
                           WHERE id = ?""",
                        (
                            "active" if remaining else "completed",
                            int(renamed),
                            utc_now(),
                            session_id,
                        ),
                    )
                    db.commit()
                except Exception:
                    # The committed 'applying' journal is intentionally left in
                    # place; get_session/startup reconciliation repairs it.
                    raise
        return self.get_session(session_id) or {}

    def undo(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            with self.database.connect() as db:
                session_row = db.execute(
                    "SELECT * FROM sort_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if not session_row:
                    raise SortNotFound("Sort session not found")
                if session_row["status"] == "superseded":
                    raise SortConflict("A superseded sort session cannot be undone")
                action = db.execute(
                    """SELECT * FROM sort_actions
                       WHERE session_id = ? AND undone_at IS NULL
                       ORDER BY id DESC LIMIT 1""",
                    (session_id,),
                ).fetchone()
                if not action:
                    raise SortConflict("There is no sorter action to undo")
                source, _ = self._resolve(action["target_source"])
                destination, _ = self._resolve(action["target_destination"], file=True)
                if source.exists():
                    raise SortConflict("The original target path is already occupied")
                control_copy: Path | None = None
                if action["control_copy"]:
                    control_copy, _ = self._resolve(action["control_copy"])
                db.execute(
                    """UPDATE sort_target_files SET status = 'undoing'
                       WHERE session_id = ? AND relative_path = ?""",
                    (session_id, action["target_source"]),
                )
                db.commit()
                removed_control = False
                try:
                    shutil.move(str(destination), str(source))
                    if control_copy and control_copy.exists():
                        control_copy.unlink()
                        removed_control = True
                    db.execute(
                        "UPDATE sort_actions SET undone_at = ? WHERE id = ?",
                        (utc_now(), action["id"]),
                    )
                    db.execute(
                        """UPDATE sort_target_files SET status = 'pending'
                           WHERE session_id = ? AND relative_path = ?""",
                        (session_id, action["target_source"]),
                    )
                    db.execute(
                        """UPDATE sort_sessions
                           SET status = 'active',
                               match_counter = MAX(1, match_counter - ?),
                               updated_at = ? WHERE id = ?""",
                        (int(action["renamed"]), utc_now(), session_id),
                    )
                    db.commit()
                except Exception:
                    db.rollback()
                    source_exists = source.exists() or source.is_symlink()
                    destination_exists = (
                        destination.exists() or destination.is_symlink()
                    )
                    if source_exists and destination_exists:
                        # The destination was repopulated while undo was in
                        # flight. Preserve both and keep the journal undoing so
                        # recovery remains explicit instead of deleting either.
                        raise
                    if source_exists:
                        shutil.move(str(source), str(destination))
                    if removed_control and action["control_source"]:
                        original, _ = self._resolve(action["control_source"], file=True)
                        shutil.copy2(original, control_copy)
                    db.execute(
                        "UPDATE sort_actions SET undone_at = NULL WHERE id = ?",
                        (action["id"],),
                    )
                    db.execute(
                        """UPDATE sort_target_files SET status = 'processed'
                           WHERE session_id = ? AND relative_path = ?""",
                        (session_id, action["target_source"]),
                    )
                    db.execute(
                        """UPDATE sort_sessions
                           SET status = ?, match_counter = ?, updated_at = ?
                           WHERE id = ?""",
                        (
                            session_row["status"],
                            session_row["match_counter"],
                            utc_now(),
                            session_id,
                        ),
                    )
                    db.commit()
                    raise
        return self.get_session(session_id) or {}

    def reconcile_incomplete(self) -> None:
        """Recover file operations interrupted between their journal commits."""
        with self._lock, self.database.connect() as db:
            applying = db.execute(
                """SELECT a.*
                   FROM sort_actions a
                   JOIN sort_target_files t
                     ON t.session_id = a.session_id
                    AND t.relative_path = a.target_source
                   WHERE t.status = 'applying' AND a.undone_at IS NULL
                   ORDER BY a.id"""
            ).fetchall()
            affected_sessions: set[str] = set()
            for action in applying:
                affected_sessions.add(action["session_id"])
                try:
                    source, _ = self._resolve(action["target_source"])
                    destination, _ = self._resolve(action["target_destination"])
                    control_copy = (
                        self._resolve(action["control_copy"])[0]
                        if action["control_copy"]
                        else None
                    )
                    complete = (
                        destination.is_file()
                        and not source.exists()
                        and (control_copy is None or control_copy.is_file())
                    )
                    if complete:
                        db.execute(
                            """UPDATE sort_target_files SET status = 'processed'
                               WHERE session_id = ? AND relative_path = ?""",
                            (action["session_id"], action["target_source"]),
                        )
                        if action["renamed"]:
                            db.execute(
                                """UPDATE sort_sessions
                                   SET match_counter = match_counter + 1
                                   WHERE id = ?""",
                                (action["session_id"],),
                            )
                        continue

                    if destination.exists() or destination.is_symlink():
                        if source.exists():
                            # Both paths existing is ambiguous (external race or
                            # manual recovery). Preserve both and keep the journal
                            # pending until an operator removes/renames one.
                            continue
                        else:
                            shutil.move(str(destination), str(source))
                    if control_copy and (
                        control_copy.exists() or control_copy.is_symlink()
                    ):
                        control_copy.unlink()
                    target_status = (
                        "pending"
                        if source.is_file() and self._is_image(source)
                        else "missing"
                    )
                    db.execute(
                        """UPDATE sort_target_files SET status = ?
                           WHERE session_id = ? AND relative_path = ?""",
                        (target_status, action["session_id"], action["target_source"]),
                    )
                    db.execute(
                        "UPDATE sort_actions SET undone_at = ? WHERE id = ?",
                        (utc_now(), action["id"]),
                    )
                except (OSError, ValueError):
                    # Storage may only be temporarily unavailable. Keeping the
                    # journal marker makes a later read/startup retry safe.
                    continue

            undoing = db.execute(
                """SELECT a.*
                   FROM sort_actions a
                   JOIN sort_target_files t
                     ON t.session_id = a.session_id
                    AND t.relative_path = a.target_source
                   WHERE t.status = 'undoing' AND a.undone_at IS NULL
                   ORDER BY a.id"""
            ).fetchall()
            for action in undoing:
                affected_sessions.add(action["session_id"])
                try:
                    source, _ = self._resolve(action["target_source"])
                    destination, _ = self._resolve(action["target_destination"])
                    control_copy = (
                        self._resolve(action["control_copy"])[0]
                        if action["control_copy"]
                        else None
                    )
                    if source.exists() and destination.exists():
                        # A concurrent or manual change makes the intended state
                        # ambiguous. Preserve both and keep recovery visible.
                        continue
                    if not source.exists() and destination.exists():
                        shutil.move(str(destination), str(source))
                    if control_copy and (
                        control_copy.exists() or control_copy.is_symlink()
                    ):
                        control_copy.unlink()
                    target_status = (
                        "pending"
                        if source.is_file() and self._is_image(source)
                        else "missing"
                    )
                    db.execute(
                        """UPDATE sort_target_files SET status = ?
                           WHERE session_id = ? AND relative_path = ?""",
                        (target_status, action["session_id"], action["target_source"]),
                    )
                    db.execute(
                        "UPDATE sort_actions SET undone_at = ? WHERE id = ?",
                        (utc_now(), action["id"]),
                    )
                    if action["renamed"]:
                        db.execute(
                            """UPDATE sort_sessions
                               SET match_counter = MAX(1, match_counter - 1)
                               WHERE id = ?""",
                            (action["session_id"],),
                        )
                except (OSError, ValueError):
                    continue

            for session_id in affected_sessions:
                session = db.execute(
                    "SELECT status FROM sort_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if not session or session["status"] == "superseded":
                    continue
                work = db.execute(
                    """SELECT COUNT(*) FROM sort_target_files
                       WHERE session_id = ?
                         AND status IN ('pending', 'applying', 'undoing')""",
                    (session_id,),
                ).fetchone()[0]
                db.execute(
                    "UPDATE sort_sessions SET status = ?, updated_at = ? WHERE id = ?",
                    ("active" if work else "completed", utc_now(), session_id),
                )
            db.commit()

    def resolve_media(self, relative_path: str) -> Path:
        path, _ = self._resolve(relative_path, file=True)
        return path
