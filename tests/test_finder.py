from __future__ import annotations

import asyncio
import io
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

import app.finder as finder_module
from app.config import AppConfig
from app.db import Database
from app.downloader import EventBroker
from app.finder import FinderService
from app.main import create_app
from app.security import verify_media_signature


ROOT = "https://www.pornpics.com/"
GALLERY_A = "https://www.pornpics.com/galleries/alpha-79186222/"
GALLERY_B = "https://www.pornpics.com/galleries/beta-79186223/"
GALLERY_C = "https://www.pornpics.com/galleries/broken-79186224/"


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 18), color).save(output, format="PNG")
    return output.getvalue()


class FakeEncoder:
    model_key = "fake-dinov2-v1"

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.embed_calls = 0

    async def prepare(self) -> Path:
        self.prepare_calls += 1
        return Path("fake.onnx")

    def embed_bytes(self, data: bytes, *, include_mirror: bool = False) -> np.ndarray:
        self.embed_calls += 1
        with Image.open(io.BytesIO(data)) as image:
            red, green, blue = image.convert("RGB").getpixel((0, 0))
        if red < 20 and green < 20 and blue < 20:
            vector = np.asarray([-1.0, 0.0], dtype=np.float32)
        elif blue > red and blue > green:
            vector = np.asarray([0.0, 1.0], dtype=np.float32)
        elif green > red and green > blue:
            vector = np.asarray([0.6, 0.8], dtype=np.float32)
        else:
            vector = np.asarray([1.0, 0.0], dtype=np.float32)
        rows = [vector]
        if include_mirror:
            rows.append(np.asarray([0.0, 1.0], dtype=np.float32))
        return np.stack(rows)


class FakeScraper:
    async def browse(self, **_: object) -> dict:
        return {
            "items": [
                {
                    "url": GALLERY_A,
                    "title": "Alpha",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/a.png",
                },
                {
                    "url": GALLERY_B,
                    "title": "Beta",
                    "thumbnail_remote_url": "https://cdni.pornpics.com/t/b.png",
                },
            ],
            "next_url": None,
        }

    async def gallery(self, url: str) -> dict:
        if url == GALLERY_A:
            images = [
                {
                    "url": "https://cdni.pornpics.com/full/blue.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/blue.png",
                    "ordinal": 7,
                },
                {
                    "url": "https://cdni.pornpics.com/full/black.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/black.png",
                    "ordinal": 8,
                },
            ]
        else:
            images = [
                {
                    "url": "https://cdni.pornpics.com/full/green.png",
                    "preview_remote_url": "https://cdni.pornpics.com/p/green.png",
                    "ordinal": 3,
                }
            ]
        return {"url": url, "images": images}


async def fake_media(url: str, _: str) -> bytes:
    if "blue" in url:
        return image_bytes("blue")
    if "green" in url:
        return image_bytes("lime")
    return image_bytes("black")


def configured(tmp_path: Path) -> tuple[AppConfig, Database, int]:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        finder_examples_root=tmp_path / "references",
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        finder_request_delay=0,
        sqlite_vfs=None,
    )
    config.ensure_directories()
    (config.finder_examples_root / "pose").mkdir()
    (config.finder_examples_root / "pose" / "example.png").write_bytes(
        image_bytes("red")
    )
    database = Database(config.db_path)
    database.initialize()
    tag = database.create_pose_tag("Standing", "solo")
    return config, database, int(tag["id"])


@pytest.mark.asyncio
async def test_finder_max_score_ranking_review_and_persistent_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    encoder = FakeEncoder()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=encoder,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        first = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 5)
        completed = service.get_scan(first["id"])
        assert completed and completed["status"] == "completed"
        assert completed["progress"] == 100
        assert completed["pose_tag"]["label"] == "Standing"

        service._save_result(
            first["id"],
            {"url": GALLERY_C, "title": "Broken"},
            order=3,
            score=0,
            images_scored=0,
            best=None,
            status="failed",
            error="preview failed",
        )

        results, total = service.results(
            first["id"], review="all", min_score=0, limit=20, offset=0
        )
        assert total == 2
        assert [item["title"] for item in results] == ["Alpha", "Beta"]
        # Alpha is 1.0 because gallery aggregation is max, not an average with black.
        assert results[0]["score"] == pytest.approx(1.0)
        assert results[0]["images_scored"] == 2
        assert results[0]["best_ordinal"] == 7
        # The blue candidate only matches the reference's mirrored fake embedding.
        assert results[0]["best_image_url"].endswith("blue.png")
        assert results[1]["score"] == pytest.approx(0.8)

        service.set_review(first["id"], results[0]["id"], "accepted")
        pending, _ = service.results(
            first["id"], review="pending", min_score=0, limit=20, offset=0
        )
        assert [item["title"] for item in pending] == ["Beta"]

        calls_after_first_scan = encoder.embed_calls
        second = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 5)
        assert service.get_scan(second["id"])["status"] == "completed"
        assert encoder.embed_calls == calls_after_first_scan
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_model_prepare_is_lazy_and_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    # Even a present-but-corrupt artifact must never trigger prepare/download at startup.
    config.finder_model_path.write_bytes(b"corrupt existing model")

    class FlakyEncoder(FakeEncoder):
        async def prepare(self) -> Path:
            self.prepare_calls += 1
            if self.prepare_calls == 1:
                raise RuntimeError("temporary model mirror failure")
            return Path("fake.onnx")

    encoder = FlakyEncoder()
    service = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=encoder,
        media_fetcher=fake_media,
    )
    await service.start()
    try:
        assert service.status()["available"] is True
        assert service.status()["model_ready"] is False
        assert encoder.prepare_calls == 0
        failed = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 5)
        assert service.get_scan(failed["id"])["status"] == "failed"
        assert service.status()["available"] is True

        retried = service.create_scan(
            example_directory="pose",
            pose_tag_id=tag_id,
            source_url=ROOT,
            page_limit=1,
            minimum_score=0,
        )
        await asyncio.wait_for(service.queue.join(), 5)
        assert service.get_scan(retried["id"])["status"] == "completed"
        assert service.status()["model_ready"] is True
        assert encoder.prepare_calls == 2
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_finder_pause_resume_and_restart_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config, database, tag_id = configured(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingScraper(FakeScraper):
        async def browse(self, **kwargs: object) -> dict:
            started.set()
            await release.wait()
            return await super().browse(**kwargs)

    service = FinderService(
        config,
        database,
        BlockingScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await service.start()
    scan = service.create_scan(
        example_directory="pose",
        pose_tag_id=tag_id,
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    await asyncio.wait_for(started.wait(), 5)
    assert service.pause(scan["id"])["status"] == "pausing"
    release.set()
    await asyncio.wait_for(service.queue.join(), 5)
    assert service.get_scan(scan["id"])["status"] == "paused"
    service.resume(scan["id"])
    await asyncio.wait_for(service.queue.join(), 5)
    assert service.get_scan(scan["id"])["status"] == "completed"

    # Simulate an interrupted process after the immutable references were prepared.
    with database.connect() as connection:
        connection.execute(
            "UPDATE finder_scans SET status = 'scanning', finished_at = NULL WHERE id = ?",
            (scan["id"],),
        )
    await service.stop()
    restarted = FinderService(
        config,
        database,
        FakeScraper(),
        EventBroker(),
        encoder=FakeEncoder(),
        media_fetcher=fake_media,
    )
    await restarted.start()
    try:
        await asyncio.wait_for(restarted.queue.join(), 5)
        assert restarted.get_scan(scan["id"])["status"] == "completed"
    finally:
        await restarted.stop()


def test_finder_folder_confinement_and_symlink_rejection(
    tmp_path: Path,
) -> None:
    config, database, _ = configured(tmp_path)
    root = config.finder_examples_root
    (root / "direct.png").write_bytes(image_bytes("blue"))
    (root / "nested").mkdir()
    (root / "nested" / "sample.png").write_bytes(image_bytes("green"))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(image_bytes("black"))
    (root / "linked").symlink_to(outside, target_is_directory=True)
    service = FinderService(config, database, FakeScraper(), EventBroker())

    listing = service.folders()
    assert listing["root"] == str(root)
    assert listing["current"]["path"] == "."
    assert listing["current"]["image_count"] == 1
    assert {item["path"] for item in listing["items"]} == {"nested", "pose"}
    nested = service.folders("nested")
    assert nested["path"] == "nested"
    assert nested["parent"] == "."
    assert nested["current"]["image_count"] == 1
    with pytest.raises(ValueError, match="cannot contain"):
        service._resolve_example_directory("../outside")
    with pytest.raises(ValueError, match="Symlinked"):
        service._resolve_example_directory("linked")
    (root / "linked-image.png").symlink_to(outside / "secret.png")
    with pytest.raises(ValueError, match="Symlinked"):
        service._example_files(root)


def test_finder_accepts_free_relative_and_confined_absolute_library_paths(
    tmp_path: Path,
) -> None:
    library = tmp_path / "library"
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=library,
        sort_root=library,
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        sqlite_vfs=None,
    )
    config.ensure_directories()
    relative = "sorted_outpaint/mating press - backview/selected_target_upscaled"
    target = library / relative
    target.mkdir(parents=True)
    (target / "example.png").write_bytes(image_bytes("red"))
    database = Database(config.db_path)
    database.initialize()
    tag = database.create_pose_tag("Mating press backview", "couple")
    service = FinderService(config, database, FakeScraper(), EventBroker())
    service.ensure_schema()

    assert config.finder_examples_root == library.resolve()
    assert service.status()["folder_root"] == str(library.resolve())
    assert service._resolve_example_directory(relative) == (target, relative)
    assert service._resolve_example_directory(str(target)) == (target, relative)
    assert service._resolve_example_directory(str(library)) == (library, ".")

    first = service.create_scan(
        example_directory=relative,
        pose_tag_id=int(tag["id"]),
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    second = service.create_scan(
        example_directory=str(target),
        pose_tag_id=int(tag["id"]),
        source_url=ROOT,
        page_limit=1,
        minimum_score=0,
    )
    assert first["example_directory"] == relative
    assert second["example_directory"] == relative

    sibling_prefix = tmp_path / "library2"
    sibling_prefix.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    for forbidden in (
        str(sibling_prefix),
        str(outside),
        "/mnt/user/Davinci/Qwen_edit_lora/pornpic",
    ):
        with pytest.raises(ValueError, match="inside"):
            service._resolve_example_directory(forbidden)
    with pytest.raises(ValueError, match="cannot contain"):
        service._resolve_example_directory("sorted_outpaint/../outside")

    linked = library / "linked"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinked"):
        service._resolve_example_directory("linked")


@pytest.mark.asyncio
async def test_finder_api_signs_best_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(finder_module, "validate_public_media_url", lambda value: value)
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        finder_examples_root=tmp_path / "references",
        finder_model_path=tmp_path / "models" / "dinov2.onnx",
        finder_request_delay=0,
        sqlite_vfs=None,
    )
    config.ensure_directories()
    (config.finder_examples_root / "example.png").write_bytes(image_bytes("red"))
    app = create_app(config)
    app.state.finder.encoder = FakeEncoder()
    app.state.finder.media_fetcher = fake_media
    app.state.finder.scraper = FakeScraper()
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            folder_response = await client.get("/api/finder/folders")
            assert folder_response.status_code == 200
            folders = folder_response.json()
            assert folders["root"] == str(config.finder_examples_root)
            assert folders["path"] == "."
            assert folders["current"]["image_count"] == 1
            assert folders["items"] == []
            finder_status = (await client.get("/api/finder/status")).json()
            assert finder_status["folder_root"] == str(config.finder_examples_root)
            assert finder_status["examples_root"] == finder_status["folder_root"]
            outside_response = await client.get(
                "/api/finder/folders", params={"path": str(tmp_path)}
            )
            assert outside_response.status_code == 400
            empty_scans = await client.get("/api/finder/scans")
            assert empty_scans.status_code == 200
            assert empty_scans.json()["items"] == []
            tag = (
                await client.post(
                    "/api/pose-tags",
                    json={"label": "Finder pose", "default_role": "solo"},
                )
            ).json()["tag"]
            response = await client.post(
                "/api/finder/scans",
                json={
                    "example_directory": ".",
                    "pose_tag_id": tag["id"],
                    "source_url": ROOT,
                    "page_limit": 1,
                    "minimum_score": 0,
                },
            )
            assert response.status_code == 202
            scan_id = response.json()["scan"]["id"]
            await asyncio.wait_for(app.state.finder.queue.join(), 5)
            listed_scans = await client.get("/api/finder/scans", params={"limit": 1})
            assert listed_scans.status_code == 200
            assert listed_scans.json()["items"][0]["id"] == scan_id
            result_response = await client.get(
                f"/api/finder/scans/{scan_id}/results",
                params={"review": "all", "min_score": 0},
            )
            assert result_response.status_code == 200
            result = result_response.json()["items"][0]
            assert "best_preview_remote_url" not in result
            assert result["best_preview_url"].startswith("/api/media?")
            request = httpx.URL(result["best_preview_url"])
            assert verify_media_signature(
                request.params["url"],
                request.params["token"],
                config.media_signing_key,
            )
