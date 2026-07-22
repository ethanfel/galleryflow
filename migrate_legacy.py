#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from app.config import config
from app.db import Database
from app.security import (
    UnsafeUrl,
    clean_profile_name,
    gallery_key,
    safe_folder_name,
    validate_source_url,
)
from app.sorter import SorterService


def read_source_database(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    if not path.exists():
        return [], []
    source = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    snapshot = sqlite3.connect(":memory:")
    try:
        source.backup(snapshot)
        tables = {
            row[0]
            for row in snapshot.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        history = (
            [
                (str(row[0]), str(row[1] or "Default"))
                for row in snapshot.execute("SELECT url, profile FROM history")
            ]
            if "history" in tables
            else []
        )
        ignored = (
            [str(row[0]) for row in snapshot.execute("SELECT url FROM ignored")]
            if "ignored" in tables
            else []
        )
        return history, ignored
    finally:
        snapshot.close()
        source.close()


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]


def collect(
    legacy_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, str], list[str]]:
    saved: dict[str, dict[str, str]] = defaultdict(dict)
    ignored: dict[str, str] = {}
    invalid: list[str] = []

    db_history, db_ignored = read_source_database(legacy_dir / "gallery_data.db")
    candidates: list[tuple[str, str]] = list(db_history)
    profile_files = list(legacy_dir.glob("history_*.txt"))
    for path in profile_files:
        profile = path.stem.removeprefix("history_") or "Default"
        candidates.extend((url, profile) for url in read_lines(path))
    # The generic file is an older standalone subset. Only use it when no
    # profile-aware database or history files exist, otherwise it invents a
    # duplicate Default profile for the same galleries.
    if not db_history and not profile_files:
        candidates.extend(
            (url, "Default") for url in read_lines(legacy_dir / "download_history.txt")
        )

    for raw_url, raw_profile in candidates:
        try:
            url = validate_source_url(raw_url)
            profile = clean_profile_name(raw_profile)
            saved[profile][gallery_key(url)] = url
        except (ValueError, UnsafeUrl):
            invalid.append(raw_url)

    ignore_candidates = list(db_ignored)
    for name in ("server_ignore_history.txt", "ignored_history.txt"):
        ignore_candidates.extend(read_lines(legacy_dir / name))
    for raw_url in ignore_candidates:
        try:
            url = validate_source_url(raw_url)
            ignored[gallery_key(url)] = url
        except (ValueError, UnsafeUrl):
            invalid.append(raw_url)
    return saved, ignored, invalid


def collect_sort_profiles(
    legacy_dir: Path, sort_root: Path
) -> tuple[list[dict], int, int]:
    source = legacy_dir / "sorter_profiles.json"
    if not source.exists():
        return [], 0, 0
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 1, 0
    if not isinstance(payload, dict):
        return [], 1, 0

    root = sort_root.resolve()
    profiles: list[dict] = []
    skipped_profiles = 0
    stale_control_paths = 0

    def relative_path(value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            relative = Path(value).expanduser().resolve().relative_to(root)
        except (OSError, ValueError):
            return None
        return "." if relative == Path(".") else relative.as_posix()

    for raw_name, raw in payload.items():
        if not isinstance(raw, dict):
            skipped_profiles += 1
            continue
        try:
            name = clean_profile_name(str(raw_name))
        except ValueError:
            skipped_profiles += 1
            continue
        target = relative_path(raw.get("target"))
        if target is None or not (root / target).is_dir():
            skipped_profiles += 1
            continue
        raw_controls = raw.get("controls", [])
        if not isinstance(raw_controls, list):
            raw_controls = []
        controls: list[str] = []
        for item in raw_controls:
            relative = relative_path(item)
            if relative is None or not (root / relative).is_dir():
                stale_control_paths += 1
                continue
            controls.append(relative)
        profiles.append(
            {
                "name": name,
                "target_directory": target,
                "control_directories": list(dict.fromkeys(controls)),
                "mode": "stem" if raw_controls else "time",
                "threshold_seconds": 50,
                "add_ids": not bool(raw_controls),
            }
        )
    return profiles, skipped_profiles, stale_control_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import saved/ignored state from the legacy PornPics app"
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "pornpic",
        help="Legacy pornpic directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report only; do not change the new database",
    )
    args = parser.parse_args()
    legacy_dir = args.legacy_dir.expanduser().resolve()
    saved, ignored, invalid = collect(legacy_dir)
    sort_profiles, skipped_sort_profiles, stale_sort_controls = collect_sort_profiles(
        legacy_dir, config.sort_root_path
    )

    print(f"Legacy source: {legacy_dir}")
    print(f"Saved galleries: {sum(len(items) for items in saved.values())}")
    for profile, items in sorted(saved.items()):
        print(f"  {profile}: {len(items)}")
    print(f"Ignored galleries: {len(ignored)}")
    print(f"Invalid/skipped lines: {len(invalid)}")
    print(f"Sorter profiles ready: {len(sort_profiles)}")
    print(
        f"Sorter profiles invalid/outside configured sort root: {skipped_sort_profiles}"
    )
    print(f"Stale sorter control folders ignored: {stale_sort_controls}")
    if args.dry_run:
        print("Dry run: destination was not changed.")
        return

    config.ensure_directories()
    database = Database(config.db_path, config.sqlite_vfs)
    database.initialize()
    sorter = SorterService(config, database)
    sorter.ensure_schema()
    for profile, items in saved.items():
        if not database.get_profile(profile):
            database.create_profile(profile, safe_folder_name(profile))
        for url in items.values():
            database.add_history(url, profile, "Legacy import", "", 0)
    for url in ignored.values():
        database.set_ignored(url, True, "Legacy import")
    for profile in sort_profiles:
        sorter.save_profile(profile)
    print(f"Import complete: {config.db_path}")


if __name__ == "__main__":
    main()
