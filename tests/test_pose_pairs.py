from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest

from app.config import AppConfig
from app.db import Database, PoseRevisionConflict, utc_now
from app.downloader import DownloadManager, EventBroker, PoseExportCanceled
from app.main import create_app
from app.security import encode_gallery_id


GALLERY = "https://www.pornpics.com/galleries/sample-gallery-79186222/"
CONTROL = "https://cdni.pornpics.com/1280/set/control.jpg"
TARGET_A = "https://cdni.pornpics.com/1280/set/target-a.jpg"
TARGET_B = "https://cdni.pornpics.com/1280/set/target-b.jpg"


def make_database(tmp_path: Path) -> tuple[AppConfig, Database]:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sort_root=tmp_path / "library",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    database = Database(config.db_path)
    database.initialize()
    return config, database


def test_existing_job_schema_is_migrated_for_pose_exports(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                gallery_url TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                profile TEXT NOT NULL,
                requested_images TEXT,
                status TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                destination TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
    database = Database(path)
    database.initialize()
    with database.connect() as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
    assert {"kind", "payload", "pair_count", "pose_revision"} <= columns


def test_pose_tags_keep_slug_and_drafts_use_optimistic_revisions(
    tmp_path: Path,
) -> None:
    _, database = make_database(tmp_path)
    database.create_profile("Training", "Training")
    tag = database.create_pose_tag("Standing pose", "solo")
    assert tag["slug"] == "standing-pose"

    renamed = database.update_pose_tag(
        tag["id"], label="Upright pose", default_role="couple"
    )
    assert renamed and renamed["slug"] == "standing-pose"
    assert renamed["label"] == "Upright pose"
    with pytest.raises(sqlite3.IntegrityError):
        database.create_pose_tag("Upright pose", "group")

    saved = database.save_pose_draft(
        GALLERY,
        "Training",
        0,
        {"solo": CONTROL, "couple": None, "group": None},
        [
            {
                "image_url": TARGET_A,
                "ordinal": 2,
                "pose_tag_id": tag["id"],
                "role": "solo",
            }
        ],
    )
    assert saved["revision"] == 1
    assert saved["targets"][0] == {
        "image_url": TARGET_A,
        "ordinal": 2,
        "pose_tag_id": tag["id"],
        "role": "solo",
        "pose_slug": "standing-pose",
        "pose_label": "Upright pose",
    }

    with pytest.raises(PoseRevisionConflict) as conflict:
        database.save_pose_draft(
            GALLERY,
            "Training",
            0,
            {"solo": None, "couple": None, "group": None},
            [],
        )
    assert conflict.value.current["revision"] == 1

    database.delete_profile("Training")
    assert database.get_pose_draft(GALLERY, "Training")["revision"] == 0


@pytest.mark.asyncio
async def test_pose_export_fetches_unique_sources_and_is_idempotent(
    tmp_path: Path,
) -> None:
    config, database = make_database(tmp_path)
    tag = database.create_pose_tag("Standing", "solo")
    draft = database.save_pose_draft(
        GALLERY,
        "Default",
        0,
        {"solo": CONTROL, "couple": None, "group": None},
        [
            {
                "image_url": TARGET_A,
                "ordinal": 2,
                "pose_tag_id": tag["id"],
                "role": "solo",
            },
            {
                "image_url": TARGET_B,
                "ordinal": 3,
                "pose_tag_id": tag["id"],
                "role": "solo",
            },
        ],
    )
    manager = DownloadManager(config, database, object(), EventBroker())  # type: ignore[arg-type]
    fetched: list[str] = []

    async def fake_download(
        url: str, destination: Path, position: int, *, referer: str
    ) -> Path:
        assert referer == GALLERY
        fetched.append(url)
        path = destination / f"{position:04d}.jpg"
        path.write_bytes(f"stable:{url}".encode())
        return path

    manager._download_image = fake_download  # type: ignore[method-assign]
    await manager.start()
    try:
        first = manager.enqueue_pose_export(
            gallery_url=GALLERY, profile="Default", draft=draft
        )
        await asyncio.wait_for(manager.queue.join(), 5)
        assert database.get_job(first["id"])["status"] == "completed"
        assert len(fetched) == 3
        assert len(set(fetched)) == 3

        root = config.pose_root_path / "standing"
        expected = {
            root / "selected_target/g79186222-0002_target.jpg",
            root / "selected_control/g79186222-0002_control.jpg",
            root / "selected_target/g79186222-0003_target.jpg",
            root / "selected_control/g79186222-0003_control.jpg",
        }
        assert all(path.is_file() for path in expected)
        assert not list(config.pose_root_path.rglob("*.part"))
        first_control = root / "selected_control/g79186222-0002_control.jpg"
        second_control = root / "selected_control/g79186222-0003_control.jpg"
        assert first_control.stat().st_ino != second_control.stat().st_ino
        original_contents = {path: path.read_bytes() for path in expected}

        second = manager.enqueue_pose_export(
            gallery_url=GALLERY, profile="Default", draft=draft
        )
        await asyncio.wait_for(manager.queue.join(), 5)
        assert database.get_job(second["id"])["status"] == "completed"
        assert len(fetched) == 6
        assert {path: path.read_bytes() for path in expected} == original_contents
        assert database.list_history(None) == []
        assert database.image_statuses("Default", GALLERY) == set()
    finally:
        await manager.stop()


def test_pose_export_preflights_both_sides_and_pose_jobs_restart(
    tmp_path: Path, monkeypatch
) -> None:
    config, database = make_database(tmp_path)
    manager = DownloadManager(config, database, object(), EventBroker())  # type: ignore[arg-type]
    target_source = config.pose_root_path / ".source-target.jpg"
    target_b_source = config.pose_root_path / ".source-target-b.jpg"
    control_source = config.pose_root_path / ".source-control.jpg"
    target_source.write_bytes(b"target")
    target_b_source.write_bytes(b"target-b")
    control_source.write_bytes(b"control")
    control_output = (
        config.pose_root_path
        / "standing/selected_control/g79186222-0002_control.jpg"
    )
    control_output.parent.mkdir(parents=True)
    control_output.write_bytes(b"different")

    with pytest.raises(FileExistsError):
        manager._materialize_pose_pairs(
            GALLERY,
            config.pose_root_path,
            {"solo": CONTROL},
            [
                {
                    "image_url": TARGET_A,
                    "ordinal": 2,
                    "pose_slug": "standing",
                    "role": "solo",
                }
            ],
            {CONTROL: control_source, TARGET_A: target_source},
        )
    assert not (
        config.pose_root_path
        / "standing/selected_target/g79186222-0002_target.jpg"
    ).exists()

    control_output.unlink()
    original_copy = manager._copy_without_overwrite
    copy_calls = 0

    def fail_second_copy(
        source: Path, target: Path, should_cancel=None
    ) -> bool:
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 4:
            raise OSError("target disk error")
        return original_copy(source, target, should_cancel)

    monkeypatch.setattr(manager, "_copy_without_overwrite", fail_second_copy)
    with pytest.raises(OSError, match="target disk error"):
        manager._materialize_pose_pairs(
            GALLERY,
            config.pose_root_path,
            {"solo": CONTROL},
            [
                {
                    "image_url": TARGET_A,
                    "ordinal": 2,
                    "pose_slug": "standing",
                    "role": "solo",
                },
                {
                    "image_url": TARGET_B,
                    "ordinal": 3,
                    "pose_slug": "standing",
                    "role": "solo",
                },
            ],
            {
                CONTROL: control_source,
                TARGET_A: target_source,
                TARGET_B: target_b_source,
            },
        )
    assert not control_output.exists()
    assert not list((config.pose_root_path / "standing").rglob("*.jpg"))

    old_target = (
        config.pose_root_path
        / "old-pose/selected_target/g79186222-0003_target.jpg"
    )
    old_target.parent.mkdir(parents=True)
    old_target.write_bytes(target_source.read_bytes())
    with pytest.raises(FileExistsError, match="already exported under pose 'old-pose'"):
        manager._materialize_pose_pairs(
            GALLERY,
            config.pose_root_path,
            {"solo": CONTROL},
            [
                {
                    "image_url": TARGET_A,
                    "ordinal": 2,
                    "pose_slug": "standing",
                    "role": "solo",
                },
                {
                    "image_url": TARGET_B,
                    "ordinal": 3,
                    "pose_slug": "standing",
                    "role": "solo",
                },
            ],
            {
                CONTROL: control_source,
                TARGET_A: target_source,
                TARGET_B: target_b_source,
            },
        )
    assert not (
        config.pose_root_path
        / "standing/selected_target/g79186222-0002_target.jpg"
    ).exists()
    old_target.unlink()

    monkeypatch.setattr(manager, "_copy_without_overwrite", original_copy)
    cancel_checks = 0

    def cancel_after_control() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks >= 5

    with pytest.raises(PoseExportCanceled, match="Pose export canceled"):
        manager._materialize_pose_pairs(
            GALLERY,
            config.pose_root_path,
            {"solo": CONTROL},
            [
                {
                    "image_url": TARGET_A,
                    "ordinal": 2,
                    "pose_slug": "standing",
                    "role": "solo",
                }
            ],
            {CONTROL: control_source, TARGET_A: target_source},
            cancel_after_control,
        )
    assert not control_output.exists()

    database.create_job(
        {
            "id": "resume-pose",
            "gallery_url": GALLERY,
            "profile": "Default",
            "kind": "pose_export",
            "payload": {"revision": 1, "controls": {}, "targets": []},
            "created_at": utc_now(),
        }
    )
    database.update_job("resume-pose", status="downloading")
    database.initialize()
    resumed = database.get_job("resume-pose")
    assert resumed and resumed["kind"] == "pose_export"
    assert resumed["status"] == "queued"
    assert "resume-pose" in database.queued_job_ids()
    public = manager.public_job(resumed)
    assert public and public["pair_count"] == 0
    assert "payload" not in public and "image_urls" not in public

    keep = config.pose_root_path / ".galleryflow-tmp/resume-pose"
    remove_terminal = config.pose_root_path / ".galleryflow-tmp/finished-pose"
    remove_missing = config.pose_root_path / ".galleryflow-tmp/missing-pose"
    for path in (keep, remove_terminal, remove_missing):
        path.mkdir(parents=True)
        (path / "image.jpg").write_bytes(b"cached")
    database.create_job(
        {
            "id": "finished-pose",
            "gallery_url": GALLERY,
            "profile": "Default",
            "kind": "pose_export",
            "payload": {"revision": 1, "controls": {}, "targets": []},
            "created_at": utc_now(),
        }
    )
    database.update_job("finished-pose", status="canceled", cancel_requested=1)
    orphan_part = (
        config.pose_root_path
        / "standing/selected_target/.g79186222-0002_target.jpg.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.part"
    )
    orphan_part.parent.mkdir(parents=True, exist_ok=True)
    orphan_part.write_bytes(b"partial")
    outside = tmp_path / "outside-pose-output"
    outside.mkdir()
    external_part = (
        outside
        / ".g79186222-0009_target.jpg.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.part"
    )
    external_part.write_bytes(b"must survive")
    linked_pose = config.pose_root_path / "linked-pose"
    linked_pose.mkdir()
    (linked_pose / "selected_target").symlink_to(outside, target_is_directory=True)
    manager._cleanup_orphan_pose_staging()
    assert keep.is_dir()
    assert not remove_terminal.exists()
    assert not remove_missing.exists()
    assert not orphan_part.exists()
    assert external_part.read_bytes() == b"must survive"


def test_atomic_pose_copy_never_leaves_partial_final(
    tmp_path: Path, monkeypatch
) -> None:
    config, database = make_database(tmp_path)
    manager = DownloadManager(config, database, object(), EventBroker())  # type: ignore[arg-type]
    source = config.pose_root_path / "source.jpg"
    target = config.pose_root_path / "pose/selected_target/pair_target.jpg"
    source.write_bytes(b"complete image")
    target.parent.mkdir(parents=True)

    def fail_publish(_: Path, __: Path) -> None:
        raise OSError("publish interrupted")

    monkeypatch.setattr("app.downloader.os.link", fail_publish)
    with pytest.raises(OSError, match="publish interrupted"):
        manager._copy_without_overwrite(source, target)
    assert not target.exists()
    assert not list(target.parent.glob(f".{target.name}.*.part"))


@pytest.mark.asyncio
async def test_pose_api_contract_and_validation(tmp_path: Path, monkeypatch) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    app = create_app(config)
    images = [
        {"url": CONTROL, "ordinal": 1, "filename": "control.jpg"},
        {"url": TARGET_A, "ordinal": 2, "filename": "target-a.jpg"},
    ]

    async def fake_gallery(url: str) -> dict:
        return {"url": url, "title": "Sample", "images": images}

    monkeypatch.setattr(app.state.scraper, "gallery", fake_gallery)
    gallery_id = encode_gallery_id(GALLERY)
    captured: dict = {}

    def fake_enqueue_pose_export(**values):
        captured.update(values)
        return {"id": "pose-job", "kind": "pose_export", "pair_count": 1}

    monkeypatch.setattr(
        app.state.downloads, "enqueue_pose_export", fake_enqueue_pose_export
    )
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            created = await client.post(
                "/api/pose-tags",
                json={"label": "Standing", "default_role": "solo"},
            )
            assert created.status_code == 201
            tag = created.json()["tag"]
            assert (await client.get("/api/pose-tags")).json()["items"] == [tag]

            empty = (
                await client.get(
                    f"/api/galleries/{gallery_id}/pose-draft",
                    params={"profile": "Default"},
                )
            ).json()["draft"]
            assert empty["revision"] == 0
            assert empty["controls"] == {
                "solo": None,
                "couple": None,
                "group": None,
            }

            invalid = await client.put(
                f"/api/galleries/{gallery_id}/pose-draft",
                json={
                    "expected_revision": 0,
                    "controls": {},
                    "targets": [
                        {
                            "image_url": TARGET_A,
                            "pose_tag_id": tag["id"],
                            "role": "solo",
                        }
                    ],
                },
            )
            assert invalid.status_code == 400

            body = {
                "expected_revision": 0,
                "controls": {"solo": CONTROL},
                "targets": [
                    {
                        "image_url": TARGET_A,
                        "pose_tag_id": tag["id"],
                        "role": "solo",
                    }
                ],
            }
            pose_events = app.state.events.subscribe()
            saved = await client.put(
                f"/api/galleries/{gallery_id}/pose-draft", json=body
            )
            assert saved.status_code == 200
            draft = saved.json()["draft"]
            assert draft["revision"] == 1
            assert draft["targets"][0]["pose_slug"] == "standing"
            draft_event = pose_events.get_nowait()
            app.state.events.unsubscribe(pose_events)
            assert draft_event == {
                "type": "pose",
                "action": "draft",
                "gallery_id": gallery_id,
                "profile": "Default",
                "revision": 1,
            }

            stale = await client.put(
                f"/api/galleries/{gallery_id}/pose-draft", json=body
            )
            assert stale.status_code == 409
            assert stale.json()["draft"]["revision"] == 1

            changed = await client.patch(
                f"/api/pose-tags/{tag['id']}",
                json={"label": "Standing upright", "default_role": "group"},
            )
            assert changed.status_code == 200
            assert changed.json()["tag"]["slug"] == "standing"

            exported = await client.post(
                "/api/pose-exports",
                json={
                    "gallery_id": gallery_id,
                    "profile": "Default",
                    "expected_revision": 1,
                },
            )
            assert exported.status_code == 202
            assert exported.json()["job"]["kind"] == "pose_export"
            assert captured["draft"]["targets"][0]["pose_slug"] == "standing"

            stale_export = await client.post(
                "/api/pose-exports",
                json={
                    "gallery_id": gallery_id,
                    "profile": "Default",
                    "expected_revision": 2,
                },
            )
            assert stale_export.status_code == 409
            assert stale_export.json()["draft"]["revision"] == 1

            app.state.db.create_job(
                {
                    "id": "public-pose-job",
                    "gallery_url": GALLERY,
                    "profile": "Default",
                    "kind": "pose_export",
                    "image_urls": [CONTROL, TARGET_A],
                    "payload": {
                        "revision": 1,
                        "controls": {"solo": CONTROL},
                        "targets": draft["targets"],
                    },
                    "pair_count": 1,
                    "pose_revision": 1,
                    "created_at": utc_now(),
                }
            )
            jobs = (await client.get("/api/downloads")).json()["items"]
            public_job = next(item for item in jobs if item["id"] == "public-pose-job")
            assert public_job["pair_count"] == 1
            assert public_job["pose_revision"] == 1
            assert "payload" not in public_job and "image_urls" not in public_job
