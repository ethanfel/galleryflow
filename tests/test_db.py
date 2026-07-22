from __future__ import annotations

from pathlib import Path

from app.db import Database, utc_now


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
