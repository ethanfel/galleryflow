from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import app.db as db_module
from app.db import Database, DatabaseStartupLockedError, utc_now


GALLERY = "https://www.pornpics.com/galleries/sample-gallery-79186222/"
RENAMED = "https://www.pornpics.com/galleries/a-new-slug-79186222/"


def make_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "state.sqlite3")
    db.initialize()
    db.create_profile("POV", "POV")
    return db


def test_share_safe_sqlite_vfs_initializes(tmp_path: Path) -> None:
    db = Database(tmp_path / "share-safe.sqlite3", vfs="unix-dotfile")
    db.initialize()
    assert db.get_profile("Default")["directory"] == "Default"
    with db.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_local_sqlite_uses_wal(tmp_path: Path) -> None:
    db = Database(tmp_path / "local.sqlite3")
    db.initialize()
    with db.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_initialize_retries_a_transient_share_safe_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "transient-lock.sqlite3"
    db = Database(path, vfs="unix-dotfile")
    db.initialize()
    real_connect = sqlite3.connect
    blocker = real_connect(
        f"file:{path.resolve()}?vfs=unix-dotfile", uri=True, timeout=0
    )
    blocker.execute("BEGIN EXCLUSIVE")
    sleeps: list[float] = []
    timeouts: list[float] = []

    def immediate_connect(*args, **kwargs):
        timeouts.append(float(kwargs["timeout"]))
        kwargs["timeout"] = 0
        return real_connect(*args, **kwargs)

    def release_after_second_retry(delay: float) -> None:
        sleeps.append(delay)
        if len(sleeps) == 2:
            blocker.rollback()
            blocker.close()

    monkeypatch.setattr(db_module.sqlite3, "connect", immediate_connect)
    monkeypatch.setattr(db_module, "DATABASE_INITIALIZE_RETRY_DELAYS", (0.1, 0.2, 0.4))
    monkeypatch.setattr(db_module.time, "sleep", release_after_second_retry)

    db.initialize()

    assert sleeps == [0.1, 0.2]
    assert timeouts[:3] == [db_module.DATABASE_STARTUP_CONNECTION_TIMEOUT] * 3
    assert db.get_profile("Default")["directory"] == "Default"
    assert timeouts[-1] == 30.0


def test_initialize_stops_after_bounded_share_safe_lock_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "persistent-lock.sqlite3"
    db = Database(path, vfs="unix-dotfile")
    db.initialize()
    real_connect = sqlite3.connect
    blocker = real_connect(
        f"file:{path.resolve()}?vfs=unix-dotfile", uri=True, timeout=0
    )
    blocker.execute("BEGIN EXCLUSIVE")
    sleeps: list[float] = []
    timeouts: list[float] = []

    def immediate_connect(*args, **kwargs):
        timeouts.append(float(kwargs["timeout"]))
        kwargs["timeout"] = 0
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(db_module.sqlite3, "connect", immediate_connect)
    monkeypatch.setattr(db_module, "DATABASE_INITIALIZE_RETRY_DELAYS", (0.1, 0.2, 0.4))
    monkeypatch.setattr(db_module.time, "sleep", sleeps.append)

    try:
        with pytest.raises(
            DatabaseStartupLockedError,
            match=r"Stop every GalleryFlow container.*persistent-lock.sqlite3.lock",
        ) as raised:
            db.initialize()
    finally:
        blocker.rollback()
        blocker.close()

    assert sleeps == [0.1, 0.2, 0.4]
    assert timeouts == [db_module.DATABASE_STARTUP_CONNECTION_TIMEOUT] * 4
    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)


def test_initialize_never_removes_an_orphaned_dotfile_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "orphaned.sqlite3"
    db = Database(path, vfs="unix-dotfile")
    db.initialize()
    lock_directory = Path(f"{path}.lock")
    lock_directory.mkdir()

    monkeypatch.setattr(db_module, "DATABASE_INITIALIZE_RETRY_DELAYS", ())
    monkeypatch.setattr(db_module, "DATABASE_STARTUP_CONNECTION_TIMEOUT", 0)

    with pytest.raises(DatabaseStartupLockedError, match="orphaned.sqlite3.lock"):
        db.initialize()

    assert lock_directory.is_dir()


def test_initialize_retries_the_complete_startup_schema_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "startup-group.sqlite3")
    calls = 0
    sleeps: list[float] = []

    def initialize_additional_schema() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise sqlite3.OperationalError("database is locked")
        with db.connect() as connection:
            connection.execute("CREATE TABLE extra_startup_schema(id INTEGER)")

    monkeypatch.setattr(db_module, "DATABASE_INITIALIZE_RETRY_DELAYS", (0.1,))
    monkeypatch.setattr(db_module.time, "sleep", sleeps.append)

    db.initialize(initialize_additional_schema)

    assert calls == 2
    assert sleeps == [0.1]
    with db.connect() as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'extra_startup_schema'"
        ).fetchone()


def test_initialize_does_not_reassign_an_existing_journal_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal-mode.sqlite3"
    db = Database(path)
    real_connect = sqlite3.connect
    statements: list[str] = []

    def traced_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(db_module.sqlite3, "connect", traced_connect)

    db.initialize()
    db.initialize()

    assignments = [
        statement
        for statement in statements
        if statement.casefold().startswith("pragma journal_mode =")
    ]
    assert len(assignments) == 1


def test_initialize_does_not_retry_non_lock_operational_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = Database(tmp_path / "non-lock.sqlite3")
    sleeps: list[float] = []

    def fail_once() -> None:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(db, "_initialize_once", fail_once)
    monkeypatch.setattr(db_module.time, "sleep", sleeps.append)

    with pytest.raises(sqlite3.OperationalError, match="unable to open"):
        db.initialize()

    assert sleeps == []


def test_connect_closes_when_connection_setup_is_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LockedSetupConnection:
        row_factory = None
        closed = False

        def execute(self, statement: str):
            if statement == "PRAGMA synchronous = NORMAL":
                raise sqlite3.OperationalError("database is locked")
            return None

        def close(self) -> None:
            self.closed = True

    connection = LockedSetupConnection()
    monkeypatch.setattr(
        db_module.sqlite3, "connect", lambda *args, **kwargs: connection
    )
    db = Database(tmp_path / "setup-lock.sqlite3")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        with db.connect():
            pass

    assert connection.closed is True


def test_saved_is_per_profile_but_ignore_is_global(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.add_history(GALLERY, "POV", "Sample", "POV/sample", 10)
    db.set_ignored(GALLERY, True, "Sample")

    pov = db.status_for_urls([RENAMED], "POV")[RENAMED]
    default = db.status_for_urls([RENAMED], "Default")[RENAMED]
    assert pov["saved"] is True and pov["ignored"] is True
    assert default["saved"] is False and default["ignored"] is True

    db.set_ignored(RENAMED, False)
    assert db.status_for_urls([GALLERY], "POV")[GALLERY]["saved"] is True
    assert db.status_for_urls([GALLERY], "POV")[GALLERY]["ignored"] is False


def test_per_image_partial_state(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    images = [
        {"url": f"https://cdni.pornpics.com/1280/x/{index}.jpg", "ordinal": index}
        for index in range(1, 4)
    ]
    db.register_gallery_images(GALLERY, images)
    db.add_profile_image("POV", GALLERY, images[0]["url"], "POV/1.jpg", 100)
    state = db.status_for_urls([GALLERY], "POV")[GALLERY]
    assert state["state"] == "partial"
    assert state["downloaded_images"] == 1
    assert state["total_images"] == 3

    db.add_history(GALLERY, "POV", "Sample", "POV/sample", 3)
    assert db.status_for_urls([GALLERY], "POV")[GALLERY]["state"] == "complete"


def test_job_item_result_commits_image_and_progress_atomically(
    tmp_path: Path,
) -> None:
    db = make_db(tmp_path)
    image_url = "https://cdni.pornpics.com/1280/x/1.jpg"
    db.create_job(
        {
            "id": "atomic-result",
            "gallery_url": GALLERY,
            "profile": "POV",
            "created_at": utc_now(),
        }
    )
    db.create_job_items("atomic-result", [{"url": image_url, "ordinal": 1}])
    db.update_job("atomic-result", status="downloading", total=1)

    summary, committed = db.record_job_item_result(
        "atomic-result",
        image_url,
        status="completed",
        completed_delta=1,
        byte_count=321,
        relative_path="POV/sample/0001.jpg",
        profile="POV",
        gallery_url=GALLERY,
    )

    assert committed is True
    assert summary and summary["completed"] == 1
    assert summary["failed"] == 0
    assert db.list_job_items("atomic-result")[0] == {
        "job_id": "atomic-result",
        "image_url": image_url,
        "ordinal": 1,
        "status": "completed",
        "attempts": 0,
        "byte_count": 321,
        "relative_path": "POV/sample/0001.jpg",
        "error": "",
    }
    assert db.image_statuses("POV", GALLERY) == {image_url}


def test_job_item_result_rolls_back_all_progress_on_failure(
    tmp_path: Path,
) -> None:
    db = make_db(tmp_path)
    image_url = "https://cdni.pornpics.com/1280/x/2.jpg"
    db.create_job(
        {
            "id": "rollback-result",
            "gallery_url": GALLERY,
            "profile": "POV",
            "created_at": utc_now(),
        }
    )
    db.create_job_items("rollback-result", [{"url": image_url, "ordinal": 1}])
    db.update_job("rollback-result", status="downloading", total=1)
    with db.connect() as connection:
        connection.executescript(
            """
            CREATE TRIGGER reject_profile_image
            AFTER INSERT ON profile_images
            BEGIN
                SELECT RAISE(ABORT, 'test rollback');
            END;
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="test rollback"):
        db.record_job_item_result(
            "rollback-result",
            image_url,
            status="completed",
            completed_delta=1,
            byte_count=321,
            relative_path="POV/sample/0002.jpg",
            profile="POV",
            gallery_url=GALLERY,
        )

    assert db.get_job_summary("rollback-result")["completed"] == 0
    assert db.list_job_items("rollback-result")[0]["status"] == "pending"
    assert db.image_statuses("POV", GALLERY) == set()


def test_job_result_does_not_commit_after_cancellation(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    image_url = "https://cdni.pornpics.com/1280/x/3.jpg"
    db.create_job(
        {
            "id": "canceled-result",
            "gallery_url": GALLERY,
            "profile": "POV",
            "created_at": utc_now(),
        }
    )
    db.create_job_items("canceled-result", [{"url": image_url, "ordinal": 1}])
    db.update_job(
        "canceled-result",
        status="canceling",
        total=1,
        cancel_requested=1,
    )

    summary, committed = db.record_job_item_result(
        "canceled-result",
        image_url,
        status="completed",
        completed_delta=1,
        byte_count=321,
        relative_path="POV/sample/0003.jpg",
        profile="POV",
        gallery_url=GALLERY,
    )

    assert committed is False
    assert summary and summary["cancel_requested"] is True
    assert summary["completed"] == 0
    assert db.list_job_items("canceled-result")[0]["status"] == "pending"
    assert db.image_statuses("POV", GALLERY) == set()
    assert db.finish_job("canceled-result", "completed")["status"] == "canceled"


def test_restart_requeues_work_but_finishes_interrupted_cancellation(
    tmp_path: Path,
) -> None:
    db = make_db(tmp_path)
    for job_id, cancel_requested in (("resume", 0), ("cancel", 1)):
        db.create_job(
            {
                "id": job_id,
                "gallery_url": GALLERY.replace(
                    "79186222", f"7918622{cancel_requested}"
                ),
                "profile": "POV",
                "created_at": utc_now(),
            }
        )
        db.update_job(
            job_id,
            status="downloading" if not cancel_requested else "canceling",
            cancel_requested=cancel_requested,
        )

    db.initialize()

    assert db.get_job("resume")["status"] == "queued"
    assert db.get_job("cancel")["status"] == "canceled"
    assert db.queued_job_ids() == ["resume"]
