from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from app.config import AppConfig
from app.main import create_app


@pytest.mark.asyncio
async def test_reference_analysis_and_joytag_scan_api_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig(
        data_dir=tmp_path / "data",
        download_root=tmp_path / "downloads",
        sqlite_vfs=None,
    )
    app = create_app(config)
    analyzed: dict[str, object] = {}

    async def fake_analysis(
        directory: str, *, top_tags: int
    ) -> dict[str, object]:
        analyzed.update(directory=directory, top_tags=top_tags)
        return {
            "directory": directory,
            "fingerprint": "a" * 64,
            "image_count": 1,
            "tag_catalog": ["target_tag", "pose_tag", "blocked_tag"],
            "tags": [{"tag": "target_tag"}],
            "images": [{"name": "one.jpg"}],
        }

    created: dict[str, object] = {}

    def fake_create_scan(**values: object) -> dict[str, object]:
        created.update(values)
        required = list(values.get("joytag_required_tags") or [])
        if not required and values.get("joytag_tag"):
            required = [str(values["joytag_tag"])]
        return {
            "id": "tag-scan",
            "search_mode": values["mode"],
            "joytag_tag": required[0] if required else "",
            "joytag_required_tags": required,
            "joytag_excluded_tags": values["joytag_excluded_tags"],
            "joytag_reject_threshold": values["joytag_reject_threshold"],
        }

    monkeypatch.setattr(
        app.state.finder,
        "analyze_reference_directory",
        fake_analysis,
    )
    monkeypatch.setattr(app.state.finder, "create_scan", fake_create_scan)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/finder/reference-analysis",
            json={"example_directory": "references/one", "top_tags": 24},
        )
        assert response.status_code == 200
        assert response.json()["analysis"]["fingerprint"] == "a" * 64
        assert response.json()["analysis"]["tag_catalog"] == [
            "target_tag",
            "pose_tag",
            "blocked_tag",
        ]
        assert analyzed == {"directory": "references/one", "top_tags": 24}

        invalid = await client.post(
            "/api/finder/reference-analysis",
            json={"example_directory": "references/one", "top_tags": 4},
        )
        assert invalid.status_code == 422

        scan = await client.post(
            "/api/finder/scans",
            json={
                "example_directory": "references/one",
                "pose_tag_id": 7,
                "source_url": "https://www.pornpics.com/",
                "page_limit": 5,
                "minimum_score": 0.4,
                "mode": "joytag",
                "joytag_tag": "target_tag",
                "reference_fingerprint": "a" * 64,
            },
        )
        assert scan.status_code == 202
        assert scan.json()["scan"]["search_mode"] == "joytag"
        assert created["mode"] == "joytag"
        assert created["joytag_tag"] == "target_tag"
        assert created["joytag_required_tags"] == []
        assert created["joytag_excluded_tags"] == []
        assert created["joytag_reject_threshold"] == 0.4
        assert created["reference_fingerprint"] == "a" * 64

        multi = await client.post(
            "/api/finder/scans",
            json={
                "example_directory": "references/one",
                "pose_tag_id": 7,
                "source_url": "https://www.pornpics.com/",
                "page_limit": 5,
                "minimum_score": 0.55,
                "mode": "joytag",
                "joytag_required_tags": ["target_tag", "pose_tag"],
                "joytag_excluded_tags": ["blocked_tag"],
                "joytag_reject_threshold": 0.65,
                "reference_fingerprint": "a" * 64,
            },
        )
        assert multi.status_code == 202
        assert multi.json()["scan"]["joytag_tag"] == "target_tag"
        assert multi.json()["scan"]["joytag_required_tags"] == [
            "target_tag",
            "pose_tag",
        ]
        assert multi.json()["scan"]["joytag_excluded_tags"] == ["blocked_tag"]
        assert multi.json()["scan"]["joytag_reject_threshold"] == 0.65
        assert created["joytag_tag"] is None
        assert created["joytag_required_tags"] == ["target_tag", "pose_tag"]
        assert created["joytag_excluded_tags"] == ["blocked_tag"]
        assert created["joytag_reject_threshold"] == 0.65

        too_many = await client.post(
            "/api/finder/scans",
            json={
                "example_directory": "references/one",
                "pose_tag_id": 7,
                "source_url": "https://www.pornpics.com/",
                "mode": "joytag",
                "joytag_required_tags": [f"tag-{index}" for index in range(17)],
                "reference_fingerprint": "a" * 64,
            },
        )
        assert too_many.status_code == 422
