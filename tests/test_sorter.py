from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from app.config import AppConfig
from app.db import Database
from app.sorter import SortConflict, SorterService


def image(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 10), (120, 70, 180)).save(path)
    os.utime(path, (mtime, mtime))
    return path


def service_for(tmp_path: Path) -> tuple[SorterService, AppConfig, Database]:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sort_root=tmp_path / "sort-root",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    database = Database(config.db_path)
    database.initialize()
    service = SorterService(config, database)
    service.ensure_schema()
    return service, config, database


def test_time_matches_actions_collision_and_persistent_undo(tmp_path: Path) -> None:
    service, config, database = service_for(tmp_path)
    target = image(config.sort_root_path / "library/target/target.jpg", 1_000)
    nearest = image(config.sort_root_path / "library/control-a/nearest.jpg", 990)
    boundary = image(config.sort_root_path / "library/control-b/boundary.jpg", 1_050)
    image(config.sort_root_path / "library/control-b/outside.jpg", 1_050.01)

    profile = service.save_profile(
        {
            "name": "Auto siblings",
            "target_directory": "library/target",
            "control_directories": [],
            "mode": "time",
            "threshold_seconds": 50,
            "add_ids": True,
        }
    )
    assert profile["control_directories"] == []

    session = service.start_session(profile)
    assert [item["path"] for item in session["matches"]] == [
        "library/control-a/nearest.jpg",
        "library/control-b/boundary.jpg",
    ]
    assert [item["delta_seconds"] for item in session["matches"]] == [10.0, 50.0]

    collision = target.parent / "selected_target/id001_target.jpg"
    image(collision, 800)
    result = service.apply_action(
        session["id"],
        "match",
        session["current"]["path"],
        session["matches"][0]["path"],
    )
    moved = target.parent / "selected_target/id001_target_copy1.jpg"
    copied = target.parent / "selected_control/id001_nearest.jpg"
    assert result["status"] == "completed"
    assert moved.is_file() and copied.is_file() and not target.exists()

    with pytest.raises(SortConflict):
        service.apply_action(
            session["id"],
            "match",
            session["current"]["path"],
            session["matches"][0]["path"],
        )

    restarted = SorterService(config, database)
    restarted.ensure_schema()
    restored = restarted.undo(session["id"])
    assert restored["current"]["path"] == "library/target/target.jpg"
    assert target.is_file() and not moved.exists() and not copied.exists()
    assert restored["match_counter"] == 1
    assert nearest.is_file() and boundary.is_file()


def test_stem_mode_solo_no_match_skip_and_path_boundary(tmp_path: Path) -> None:
    service, config, _ = service_for(tmp_path)
    target_dir = config.sort_root_path / "work/targets"
    control_dir = config.sort_root_path / "work/controls"
    abc = image(target_dir / "ABC_target.jpg", 100)
    define = image(target_dir / "DEF_target.jpg", 200)
    ghi = image(target_dir / "GHI.jpg", 300)
    abc_control = image(control_dir / "ABC_reference.jpg", 900)
    image(control_dir / "abc_wrong-case.jpg", 100)
    image(control_dir / "DEF_reference.jpg", 100)

    session = service.start_session(
        {
            "target_directory": "work/targets",
            "control_directories": ["work/controls"],
            "mode": "stem",
            "threshold_seconds": 50,
            "add_ids": False,
        }
    )
    assert [item["name"] for item in session["matches"]] == ["ABC_reference.jpg"]
    session = service.apply_action(
        session["id"], "solo", session["current"]["path"], session["matches"][0]["path"]
    )
    assert (target_dir / "selected_target_solo_woman/ABC_target.jpg").is_file()
    assert (target_dir / "control_selected_solo_woman/ABC_reference.jpg").is_file()
    assert not abc.exists() and abc_control.exists()

    session = service.apply_action(
        session["id"], "no_control", session["current"]["path"]
    )
    assert not define.exists()
    assert (target_dir / "selected_target_no_control/DEF_target.jpg").is_file()

    session = service.apply_action(session["id"], "skip", session["current"]["path"])
    assert session["status"] == "completed"
    assert not ghi.exists() and (target_dir / "skipped_target/GHI.jpg").is_file()

    undone = service.undo(session["id"])
    assert undone["current"]["name"] == "GHI.jpg" and ghi.is_file()

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError):
        service.start_session({"target_directory": "../outside", "mode": "time"})
    link = config.sort_root_path / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        service.start_session({"target_directory": "escape", "mode": "time"})
    assert all(item["path"] != "escape" for item in service.folders()["items"])


def test_rescan_supersedes_old_session_and_skips_disappeared_target(
    tmp_path: Path,
) -> None:
    service, config, _ = service_for(tmp_path)
    first_image = image(config.sort_root_path / "targets/a.jpg", 100)
    image(config.sort_root_path / "targets/b.jpg", 101)
    options = {
        "target_directory": "targets",
        "control_directories": [],
        "mode": "time",
    }

    old = service.start_session(options)
    current = service.start_session(options)

    assert service.get_session(old["id"])["status"] == "superseded"
    with pytest.raises(SortConflict):
        service.apply_action(old["id"], "skip", old["current"]["path"])

    first_image.unlink()
    reconciled = service.get_session(current["id"])
    assert reconciled["missing"] == 1
    assert reconciled["current"]["name"] == "b.jpg"
    finished = service.apply_action(
        current["id"], "skip", reconciled["current"]["path"]
    )
    assert finished["status"] == "completed"
    assert finished["processed"] == 1 and finished["missing"] == 1


def test_incomplete_move_and_undo_are_reconciled_from_durable_journal(
    tmp_path: Path,
) -> None:
    service, config, database = service_for(tmp_path)
    target = image(config.sort_root_path / "work/targets/target.jpg", 1_000)
    control = image(config.sort_root_path / "work/controls/control.jpg", 1_000)
    session = service.start_session(
        {
            "target_directory": "work/targets",
            "control_directories": ["work/controls"],
            "mode": "time",
            "add_ids": True,
        }
    )
    target_destination = target.parent / "selected_target/id001_target.jpg"
    control_copy = target.parent / "selected_control/id001_control.jpg"
    target_destination.parent.mkdir()
    control_copy.parent.mkdir()
    with database.connect() as db:
        action = db.execute(
            """INSERT INTO sort_actions(
                   session_id, kind, target_source, target_destination,
                   control_source, control_copy, renamed, created_at
               ) VALUES (?, 'match', ?, ?, ?, ?, 1, 'interrupted')""",
            (
                session["id"],
                "work/targets/target.jpg",
                "work/targets/selected_target/id001_target.jpg",
                "work/controls/control.jpg",
                "work/targets/selected_control/id001_control.jpg",
            ),
        )
        action_id = action.lastrowid
        db.execute(
            """UPDATE sort_target_files SET status = 'applying'
               WHERE session_id = ? AND relative_path = ?""",
            (session["id"], "work/targets/target.jpg"),
        )
    shutil.copy2(control, control_copy)
    shutil.move(target, target_destination)

    restarted = SorterService(config, database)
    restarted.ensure_schema()
    applied = restarted.get_session(session["id"])
    assert applied["status"] == "completed" and applied["match_counter"] == 2
    assert applied["can_undo"] is True

    with database.connect() as db:
        db.execute(
            """UPDATE sort_target_files SET status = 'undoing'
               WHERE session_id = ? AND relative_path = ?""",
            (session["id"], "work/targets/target.jpg"),
        )
    shutil.move(target_destination, target)
    control_copy.unlink()

    restarted.ensure_schema()
    undone = restarted.get_session(session["id"])
    assert undone["status"] == "active" and undone["current"] is not None
    assert undone["match_counter"] == 1 and undone["can_undo"] is False
    with database.connect() as db:
        row = db.execute(
            "SELECT undone_at FROM sort_actions WHERE id = ?", (action_id,)
        ).fetchone()
    assert row["undone_at"] is not None


def test_completed_sort_session_retention_is_bounded(tmp_path: Path) -> None:
    service, config, database = service_for(tmp_path)
    (config.sort_root_path / "empty").mkdir()
    for _ in range(205):
        service.start_session({"target_directory": "empty", "mode": "time"})
    with database.connect() as db:
        count = db.execute("SELECT COUNT(*) FROM sort_sessions").fetchone()[0]
    assert count == 200


def test_undo_sql_failure_restores_files_and_all_database_state(tmp_path: Path) -> None:
    service, config, database = service_for(tmp_path)
    target = image(config.sort_root_path / "work/targets/target.jpg", 100)
    image(config.sort_root_path / "work/controls/control.jpg", 100)
    session = service.start_session(
        {
            "target_directory": "work/targets",
            "control_directories": ["work/controls"],
            "mode": "time",
            "add_ids": True,
        }
    )
    applied = service.apply_action(
        session["id"],
        "match",
        session["current"]["path"],
        session["matches"][0]["path"],
    )
    destination = target.parent / "selected_target/id001_target.jpg"
    control_copy = target.parent / "selected_control/id001_control.jpg"
    with database.connect() as db:
        db.executescript(
            """CREATE TRIGGER fail_undo_counter
               BEFORE UPDATE OF match_counter ON sort_sessions
               WHEN NEW.match_counter < OLD.match_counter
               BEGIN
                   SELECT RAISE(FAIL, 'injected undo failure');
               END;"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected undo failure"):
        service.undo(applied["id"])

    assert not target.exists()
    assert destination.is_file() and control_copy.is_file()
    restored = service.get_session(applied["id"])
    assert restored["match_counter"] == 2 and restored["can_undo"] is True
    with database.connect() as db:
        action = db.execute(
            "SELECT undone_at FROM sort_actions WHERE session_id = ?", (applied["id"],)
        ).fetchone()
        target_row = db.execute(
            """SELECT status FROM sort_target_files
               WHERE session_id = ? AND relative_path = ?""",
            (applied["id"], "work/targets/target.jpg"),
        ).fetchone()
    assert action["undone_at"] is None and target_row["status"] == "processed"


def test_ambiguous_recovery_preserves_both_files_and_blocks_rescan(
    tmp_path: Path,
) -> None:
    service, config, database = service_for(tmp_path)
    source = image(config.sort_root_path / "targets/source.jpg", 100)
    destination = image(
        config.sort_root_path / "targets/skipped_target/source.jpg", 101
    )
    session = service.start_session({"target_directory": "targets", "mode": "time"})
    with database.connect() as db:
        db.execute(
            """INSERT INTO sort_actions(
                   session_id, kind, target_source, target_destination,
                   control_source, control_copy, renamed, created_at
               ) VALUES (?, 'skip', ?, ?, '', '', 0, 'interrupted')""",
            (
                session["id"],
                "targets/source.jpg",
                "targets/skipped_target/source.jpg",
            ),
        )
        db.execute(
            """UPDATE sort_target_files SET status = 'applying'
               WHERE session_id = ? AND relative_path = ?""",
            (session["id"], "targets/source.jpg"),
        )

    recovering = service.get_session(session["id"])
    assert source.is_file() and destination.is_file()
    assert recovering["status"] == "recovering"
    assert recovering["recovering"] == 1 and recovering["remaining"] == 1
    with pytest.raises(SortConflict, match="interrupted sorter operation"):
        service.start_session({"target_directory": "targets", "mode": "time"})

    with database.connect() as db:
        db.execute(
            "UPDATE sort_sessions SET status = 'superseded' WHERE id = ?",
            (session["id"],),
        )
    empty = config.sort_root_path / "empty"
    empty.mkdir()
    for _ in range(205):
        service.start_session({"target_directory": "empty", "mode": "time"})
    with database.connect() as db:
        retained = db.execute(
            "SELECT 1 FROM sort_sessions WHERE id = ?", (session["id"],)
        ).fetchone()
    assert retained is not None


def test_apply_race_preserves_source_and_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, config, database = service_for(tmp_path)
    source = image(config.sort_root_path / "targets/source.jpg", 100)
    session = service.start_session({"target_directory": "targets", "mode": "time"})
    destination = source.parent / "skipped_target/source.jpg"

    def racing_move(_source: str, target: str) -> None:
        image(Path(target), 101)
        raise OSError("injected destination race")

    monkeypatch.setattr(shutil, "move", racing_move)
    with pytest.raises(OSError, match="injected destination race"):
        service.apply_action(session["id"], "skip", session["current"]["path"])

    assert source.is_file() and destination.is_file()
    with database.connect() as db:
        target = db.execute(
            """SELECT status FROM sort_target_files
               WHERE session_id = ? AND relative_path = ?""",
            (session["id"], "targets/source.jpg"),
        ).fetchone()
        action = db.execute(
            "SELECT undone_at FROM sort_actions WHERE session_id = ?",
            (session["id"],),
        ).fetchone()
    assert target["status"] == "applying" and action["undone_at"] is None
    assert service.get_session(session["id"])["status"] == "recovering"


def test_undo_race_preserves_restored_source_and_new_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, config, database = service_for(tmp_path)
    source = image(config.sort_root_path / "targets/source.jpg", 100)
    session = service.start_session({"target_directory": "targets", "mode": "time"})
    applied = service.apply_action(session["id"], "skip", session["current"]["path"])
    destination = source.parent / "skipped_target/source.jpg"
    real_move = shutil.move

    def racing_move(current: str, target: str) -> None:
        real_move(current, target)
        image(Path(current), 101)
        raise OSError("injected undo destination race")

    monkeypatch.setattr(shutil, "move", racing_move)
    with pytest.raises(OSError, match="injected undo destination race"):
        service.undo(applied["id"])

    assert source.is_file() and destination.is_file()
    with database.connect() as db:
        target = db.execute(
            """SELECT status FROM sort_target_files
               WHERE session_id = ? AND relative_path = ?""",
            (session["id"], "targets/source.jpg"),
        ).fetchone()
        action = db.execute(
            "SELECT undone_at FROM sort_actions WHERE session_id = ?",
            (session["id"],),
        ).fetchone()
    assert target["status"] == "undoing" and action["undone_at"] is None
    assert service.get_session(session["id"])["status"] == "recovering"
