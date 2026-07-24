"""Manual browser screenshot helper; not collected by pytest."""

from __future__ import annotations

import subprocess
import shutil
import argparse
import asyncio
import json
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import HTTPException
from fastapi.responses import Response
from PIL import Image

from app.config import AppConfig
from app.main import create_app
from app.security import encode_gallery_id, gallery_key


def build_visual_app(
    temp_root: Path,
    *,
    load_more: bool = False,
    open_gallery: bool = False,
    open_lightbox: bool = False,
    open_pose: bool = False,
    open_finder: bool = False,
    open_finder_feedback: bool = False,
    open_finder_pose_flow: bool = False,
    finder_direct_assign: bool = False,
    finder_race: bool = False,
    finder_unusable_save: bool = False,
    finder_exhausted: bool = False,
    finder_continue: bool = False,
    finder_switch_source: bool = False,
    finder_pagination: bool = False,
    finder_tag: bool = False,
    finder_retry: bool = False,
    finder_modal_review: bool = False,
):
    finder_source_exhausted = finder_exhausted or finder_continue
    finder_pose_mode = open_finder_pose_flow or finder_direct_assign
    finder_review_mode = (
        open_finder_feedback or finder_pose_mode or finder_unusable_save
    )
    config = AppConfig(
        data_dir=temp_root / "data",
        download_root=temp_root / "downloads",
        sqlite_vfs=None,
    )
    app = create_app(config)
    db = app.state.db
    config.ensure_directories()
    db.initialize()
    pose_tag = db.create_pose_tag("mating press - backview", "couple")
    target_dir = config.sort_root_path / "visual-sort/targets"
    control_dir = config.sort_root_path / "visual-sort/references"
    target_dir.mkdir(parents=True)
    control_dir.mkdir(parents=True)
    Image.new("RGB", (900, 1200), "#453a76").save(target_dir / "sample_target.jpg")
    Image.new("RGB", (900, 1200), "#7d4c72").save(control_dir / "sample_reference.jpg")
    app.state.sorter.ensure_schema()
    sort_session = app.state.sorter.start_session(
        {
            "target_directory": "visual-sort/targets",
            "control_directories": ["visual-sort/references"],
            "mode": "time",
            "threshold_seconds": 50,
            "add_ids": True,
        }
    )
    sample_image = "https://cdni.pornpics.com/460/7/343/79186222/79186222_001_d325.jpg"
    galleries = []
    for index in range(12):
        remote_id = 79186222 + index
        url = f"https://www.pornpics.com/galleries/sample-gallery-{remote_id}/"
        galleries.append(
            {
                "id": encode_gallery_id(url),
                "key": gallery_key(url),
                "url": url,
                "title": [
                    "Midnight studio portrait collection",
                    "Natural light editorial series",
                    "After-hours city gallery",
                    "Soft focus summer collection",
                    "Classic monochrome session",
                    "Warm sunset portrait set",
                ][index % 6],
                "thumbnail_remote_url": sample_image,
                "image_count": 20 + index,
            }
        )
    db.add_history(
        galleries[0]["url"], "Default", galleries[0]["title"], "Default/sample", 20
    )
    db.register_gallery_images(
        galleries[1]["url"],
        [
            {"url": f"https://cdni.pornpics.com/1280/demo/{n}.jpg", "ordinal": n}
            for n in range(1, 11)
        ],
    )
    db.add_profile_image(
        "Default",
        galleries[1]["url"],
        "https://cdni.pornpics.com/1280/demo/1.jpg",
        "Default/demo/0001.jpg",
        1234,
    )

    async def fake_browse(**kwargs: object) -> dict:
        page = int(kwargs.get("page", 1))
        source = str(kwargs.get("url") or "")
        if "offset=20" in source:
            page = 2
        start = 6 if page > 1 else 0
        return {
            "items": [dict(item) for item in galleries[start : start + 6]],
            "source_url": "https://www.pornpics.com/",
            "next_url": (
                "https://www.pornpics.com/?offset=20&limit=20" if page == 1 else None
            ),
            "previous_url": "https://www.pornpics.com/" if page > 1 else None,
        }

    app.state.scraper.browse = fake_browse
    gallery_detail_calls: dict[str, int] = {}
    gallery_preview_calls: dict[str, int] = {}
    gallery_detail_concurrency = {"active": 0, "peak": 0}
    prefetch_neighbor_urls = {
        galleries[2]["url"],
        galleries[4]["url"],
    }

    async def fake_gallery(url: str) -> dict:
        gallery_detail_calls[url] = gallery_detail_calls.get(url, 0) + 1
        background_prefetch = finder_modal_review and url in prefetch_neighbor_urls
        if background_prefetch:
            gallery_detail_concurrency["active"] += 1
            gallery_detail_concurrency["peak"] = max(
                gallery_detail_concurrency["peak"],
                gallery_detail_concurrency["active"],
            )
            await asyncio.sleep(0.04)
        boundary_marker = "/galleries/manual-boundary-"
        if boundary_marker in url:
            boundary_name = url.split(boundary_marker, 1)[1].split("/", 1)[0]
            if boundary_name.isdigit():
                title = f"Boundary candidate {int(boundary_name):03d}"
            else:
                title = f"Higher-ranked insert {boundary_name.rsplit('-', 1)[-1].upper()}"
            gallery = {
                "id": encode_gallery_id(url),
                "key": gallery_key(url),
                "url": url,
                "title": title,
                "thumbnail_remote_url": sample_image,
                "image_count": 21,
            }
        else:
            gallery = next(
                (item for item in galleries if item["url"] == url), galleries[0]
            )
        preview_marker = gallery["url"].rstrip("/").rsplit("/", 1)[-1]
        detail = {
            **gallery,
            "images": [
                {
                    "url": f"https://cdni.pornpics.com/1280/manual/{index:03d}.jpg",
                    "preview_remote_url": (
                        "https://cdni.pornpics.com/460/manual/"
                        f"{preview_marker}/{index:03d}.jpg"
                    ),
                    "filename": f"manual-{index:03d}.jpg",
                    "ordinal": index,
                }
                for index in range(1, 22)
            ],
        }
        if background_prefetch:
            gallery_detail_concurrency["active"] -= 1
        return detail

    app.state.scraper.gallery = fake_gallery

    async def fake_media(request=None, url: str = "", token: str = "") -> Response:
        gallery_preview_calls[url] = gallery_preview_calls.get(url, 0) + 1
        if "overlay" in url:
            svg = b"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1100'><g fill='none' stroke='#63f2bd' stroke-width='18' stroke-linecap='round' stroke-linejoin='round'><circle cx='400' cy='190' r='58'/><path d='m400 250-20 245m20-180-150 145m150-145 145 110M380 495 245 760m135-265 190 245'/></g><g fill='#ffcf67' stroke='#101017' stroke-width='7'><circle cx='400' cy='250' r='17'/><circle cx='400' cy='315' r='17'/><circle cx='250' cy='460' r='17'/><circle cx='545' cy='425' r='17'/><circle cx='380' cy='495' r='17'/><circle cx='245' cy='760' r='17'/><circle cx='570' cy='740' r='17'/></g></svg>"""
        else:
            color = (
                "#284b63"
                if "candidate-2" in url
                else "#57406e"
                if "candidate-3" in url
                else "#31295a"
            )
            svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='800' height='1100'><defs><linearGradient id='g' x2='1' y2='1'><stop stop-color='{color}'/><stop offset='1' stop-color='#121825'/></linearGradient></defs><rect width='800' height='1100' fill='url(#g)'/><circle cx='570' cy='310' r='190' fill='#9b7bfa' opacity='.18'/><path d='M90 860 330 540l150 170 105-125 140 275Z' fill='#ffffff' opacity='.13'/></svg>""".encode()
        return Response(svg, media_type="image/svg+xml")

    finder_scan = {
        "id": "visual-finder",
        "status": "completed",
        "example_directory": "sorted_outpaint/mating press - backview/selected_target_upscaled",
        "pose_tag_id": pose_tag["id"],
        "pose_tag_label": "mating press - backview",
        "source_url": "https://www.pornpics.com/",
        "next_url": (
            None if finder_source_exhausted else "https://www.pornpics.com/?page=6"
        ),
        "page_limit": 500 if finder_continue else 5,
        "pages_completed": 3 if finder_source_exhausted else 5,
        "processed_galleries": 64,
        "processed_images": 1280,
        "corpus_search_complete": True,
        "corpus_galleries_scored": 418,
        "corpus_images_scored": 8156,
        "candidate_count": 2,
        "minimum_score": 0.65,
        "ranking_version": "pose-precision-v2",
        "ranking_current": True,
        "continuable": finder_source_exhausted,
        "progress_percent": 100,
    }
    finder_race_state = {"review": "pending", "results_calls": 0, "selected": []}
    finder_continue_state = {"scan_calls": 0, "continue_calls": 0}
    finder_switch_review_counts = {
        "pending": 119,
        "accepted": 3,
        "maybe": 1,
        "rejected": 1,
        "total": 124,
    }
    finder_switch_source_state: dict[str, object] = {
        "cancel_calls": 0,
        "continue_calls": 0,
        "submitted_source_url": "",
        "submitted_additional_pages": 0,
        "cancelled_next_url": "",
    }
    finder_retry_state = {"retry_calls": 0}
    finder_tag_state: dict[str, object] = {"created": False, "payload": None}
    finder_tag_index_state = {
        "status": "idle",
        "post_calls": 0,
        "delete_calls": 0,
    }
    finder_modal_review_state: dict[str, object] = {
        "reviews": {
            "visual-result-1": "pending",
            "visual-result-2": "pending",
            "visual-result-3": "pending",
        },
        "selections": {},
        "review_history": [],
        "boundary_mode": False,
        "higher_ranked_inserts": False,
        "boundary_backward_probe_seen": False,
        "pose_export_calls": [],
    }
    if finder_race:
        finder_scan["status"] = "running"
    if finder_switch_source:
        finder_scan.update(
            {
                "status": "running",
                "search_mode": "joytag",
                "joytag_tag": "pov",
                "joytag_required_tags": ["pov", "footjob"],
                "joytag_excluded_tags": ["solo"],
                "reference_fingerprint": "c" * 64,
                "ranking_version": "joytag-v1",
                "source_url": "https://www.pornpics.com/?q=pov+footjob",
                "next_url": "https://www.pornpics.com/?q=pov+footjob&page=2",
                "page_limit": 50,
                "pages_completed": 1,
                "processed_galleries": 20,
                "processed_images": 393,
                "corpus_search_complete": True,
                "corpus_galleries_scored": 418,
                "corpus_images_scored": 8156,
                "candidate_count": 124,
                "minimum_score": 0.4,
                "continuable": False,
                "progress_percent": 2,
                "review_counts": dict(finder_switch_review_counts),
            }
        )
    if finder_retry:
        finder_scan.update(
            {
                "status": "failed",
                "search_mode": "joytag",
                "joytag_tag": "1girl",
                "joytag_required_tags": ["1girl", "pov"],
                "joytag_excluded_tags": ["solo"],
                "reference_fingerprint": "b" * 64,
                "ranking_version": "joytag-v1",
                "source_url": "https://www.pornpics.com/?q=pov+footjob",
                "next_url": "https://www.pornpics.com/?q=pov+footjob",
                "page_limit": 50,
                "pages_completed": 0,
                "processed_galleries": 0,
                "processed_images": 0,
                "corpus_search_complete": True,
                "corpus_galleries_scored": 7,
                "corpus_images_scored": 124,
                "candidate_count": 124,
                "minimum_score": 0.4,
                "progress_percent": 0,
                "error": "Could not reach PornPics: temporary upstream failure",
            }
        )
    if finder_modal_review:
        finder_scan["status"] = "running"
        finder_scan["candidate_count"] = 3

    async def fake_finder_status(**kwargs: object) -> dict:
        return {
            "available": True,
            "model_ready": True,
            "model_name": "RTMO-L + visual verifier",
            "device": "CUDA",
            "folder_root": "/library",
            "inference_batch": {
                "configured": 8,
                "appearance": 8,
                "pose": 8,
            },
            "joytag": {
                "available": True,
                "ready": True,
                "error": "",
                "model_key": "joytag-visual-smoke",
                "provider": {"active": "CUDAExecutionProvider"},
            },
        }

    def fake_finder_joytag_coverage() -> dict:
        cached = 1790 if finder_tag_index_state["status"] != "idle" else 1200
        return {
            "model_key": "joytag-visual-smoke",
            "total_images": 8420,
            "cached_images": cached,
            "missing_images": 8420 - cached,
            "percent": round(cached / 8420 * 100, 1),
            "cache_entries": cached,
            "cache_bytes": cached * 5813,
        }

    def fake_finder_joytag_index_job() -> dict | None:
        status = str(finder_tag_index_state["status"])
        if status == "idle":
            return None
        return {
            "id": "visual-joytag-index",
            "status": status,
            "model_key": "joytag-visual-smoke",
            "total_images": 8420,
            "cached_images_at_start": 1200,
            "processed_images": 600,
            "indexed_images": 590,
            "failed_images": 10,
            "remaining_images": 6620,
            "progress": 21.4,
            "cancel_requested": status in {"canceling", "canceled"},
            "error": "",
            "errors": [],
            "created_at": "2026-07-24T12:00:00+00:00",
            "updated_at": "2026-07-24T12:00:01+00:00",
            "finished_at": (
                "2026-07-24T12:00:02+00:00" if status == "canceled" else None
            ),
        }

    def fake_finder_joytag_index_response() -> dict:
        return {
            "job": fake_finder_joytag_index_job(),
            "coverage": fake_finder_joytag_coverage(),
        }

    async def fake_finder_corpus(**kwargs: object) -> dict:
        result = {
            "galleries": 418,
            "images": 8420,
            "complete": 374,
            "partial": 44,
            "ready": 8156,
            "cache_entries": 10820,
            "cache_bytes": 367001600,
            "max_cache_entries": 50000,
            "max_cache_bytes": 2147483648,
            "joytag": fake_finder_joytag_coverage(),
            "joytag_index_job": fake_finder_joytag_index_job(),
        }
        return result

    async def fake_finder_joytag_index_get(**kwargs: object) -> dict:
        return fake_finder_joytag_index_response()

    async def fake_finder_joytag_index_start(**kwargs: object) -> dict:
        finder_tag_index_state["post_calls"] += 1
        finder_tag_index_state["status"] = "running"
        return fake_finder_joytag_index_response()

    async def fake_finder_joytag_index_cancel(**kwargs: object) -> dict:
        finder_tag_index_state["delete_calls"] += 1
        finder_tag_index_state["status"] = "canceled"
        return fake_finder_joytag_index_response()

    async def fake_finder_feedback(**kwargs: object) -> dict:
        return {
            "pose_tag_id": pose_tag["id"],
            "revision": 19,
            "accepted_galleries": 4,
            "rejected_galleries": 3,
            "usable_accepted_galleries": 4,
            "usable_rejected_galleries": 3,
            "accepted_samples": 12,
            "rejected_samples": 7,
            "usable_accepted_samples": 12,
            "usable_rejected_samples": 7,
            "active": True,
            "min_galleries_per_state": 2,
            "max_galleries_per_state": 8,
            "max_samples_per_state": 8,
            "max_adjustment": 0.08,
            "applies_to": "future_scans",
        }

    async def fake_finder_folders(**kwargs: object) -> dict:
        return {
            "folders": [
                {
                    "path": "sorted_outpaint/mating press - backview/selected_target_upscaled",
                    "image_count": 25,
                }
            ]
        }

    async def fake_pose_tags(**kwargs: object) -> dict:
        return {
            "items": [
                {
                    "id": pose_tag["id"],
                    "label": "mating press - backview",
                    "slug": "mating-press-backview",
                    "default_role": "couple",
                }
            ]
        }

    async def fake_finder_scans(**kwargs: object) -> dict:
        if finder_tag:
            return {"scans": [finder_scan] if finder_tag_state["created"] else []}
        return {"scans": [finder_scan]}

    async def fake_finder_reference_analysis(**kwargs: object) -> dict:
        preview = (
            "/api/media?url=https%3A%2F%2Fexample.test%2Freference-{index}.jpg"
            "&token=visual"
        )
        return {
            "analysis": {
                "directory": (
                    "sorted_outpaint/mating press - backview/"
                    "selected_target_upscaled"
                ),
                "fingerprint": "a" * 64,
                "image_count": 2,
                "model_key": "joytag-visual-smoke",
                "provider": "CUDAExecutionProvider",
                "quantization": "uint8",
                "bytes_per_cached_image": 5813,
                "tag_catalog": [
                    "mating_press",
                    "lying",
                    "solo",
                    "1girl",
                    "indoors",
                ],
                "tags": [
                    {
                        "tag": "mating_press",
                        "index": 214,
                        "average": 0.82,
                        "minimum": 0.74,
                        "maximum": 0.90,
                        "median": 0.82,
                        "hits_at_0_4": 2,
                        "image_count": 2,
                    },
                    {
                        "tag": "lying",
                        "index": 18,
                        "average": 0.67,
                        "minimum": 0.54,
                        "maximum": 0.80,
                        "median": 0.67,
                        "hits_at_0_4": 2,
                        "image_count": 2,
                    },
                ],
                "images": [
                    {
                        "name": "reference-01.jpg",
                        "preview_url": preview.format(index=1),
                        "scores": {"mating_press": 0.90, "lying": 0.54},
                    },
                    {
                        "name": "reference-02.jpg",
                        "preview_url": preview.format(index=2),
                        "scores": {"mating_press": 0.74, "lying": 0.80},
                    },
                ],
            }
        }

    async def fake_finder_create(payload: object, **kwargs: object) -> dict:
        values = (
            payload.model_dump()
            if callable(getattr(payload, "model_dump", None))
            else dict(payload)
            if isinstance(payload, dict)
            else {}
        )
        finder_tag_state["payload"] = values
        valid = (
            values.get("mode") == "joytag"
            and values.get("joytag_tag") == "mating_press"
            and values.get("joytag_required_tags")
            == ["mating_press", "lying"]
            and values.get("joytag_excluded_tags") == ["solo"]
            and abs(
                float(values.get("joytag_reject_threshold") or 0) - 0.55
            )
            < 1e-9
            and finder_tag_index_state["post_calls"] == 1
            and finder_tag_index_state["delete_calls"] == 1
            and finder_tag_index_state["status"] == "canceled"
            and values.get("reference_fingerprint") == "a" * 64
            and abs(float(values.get("minimum_score") or 0) - 0.35) < 1e-9
        )
        if not valid:
            raise HTTPException(
                status_code=422,
                detail="Tag Finder sent an invalid scan payload",
            )
        finder_tag_state["created"] = True
        finder_scan.update(
            {
                "id": "visual-tag-finder",
                "status": "completed",
                "pose_tag_id": pose_tag["id"],
                "pose_tag_label": "mating press - backview",
                "search_mode": "joytag",
                "joytag_tag": "mating_press",
                "joytag_required_tags": ["mating_press", "lying"],
                "joytag_excluded_tags": ["solo"],
                "joytag_reject_threshold": 0.55,
                "reference_fingerprint": "a" * 64,
                "minimum_score": 0.35,
                "ranking_version": "joytag-v1",
                "candidate_count": 1,
                "processed_galleries": 1,
                "processed_images": 21,
            }
        )
        return {"scan": dict(finder_scan)}

    async def fake_finder_scan(**kwargs: object) -> dict:
        finder_continue_state["scan_calls"] += 1
        snapshot = dict(finder_scan)
        if finder_continue and finder_continue_state["scan_calls"] == 2:
            await asyncio.sleep(1.2)
        return snapshot

    async def fake_finder_cancel(scan_id: str, **kwargs: object) -> dict:
        if (
            scan_id != finder_scan["id"]
            or finder_scan["status"] != "running"
        ):
            raise HTTPException(
                status_code=409,
                detail="Only the running visual Finder scan can be cancelled",
            )
        finder_switch_source_state["cancel_calls"] = (
            int(finder_switch_source_state["cancel_calls"]) + 1
        )
        finder_switch_source_state["cancelled_next_url"] = finder_scan["next_url"]
        finder_scan["status"] = "cancelled"
        return {"scan": dict(finder_scan)}

    async def fake_finder_continue(
        scan_id: str, payload: object, **kwargs: object
    ) -> dict:
        finder_continue_state["continue_calls"] += 1
        source_url = str(
            payload.get("source_url")
            if isinstance(payload, dict)
            else getattr(payload, "source_url", "")
        )
        additional_pages = int(
            payload.get("additional_pages", 5)
            if isinstance(payload, dict)
            else getattr(payload, "additional_pages", 5)
        )
        if finder_switch_source:
            if (
                scan_id != finder_scan["id"]
                or finder_scan["status"] != "cancelled"
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Cancel the visual Finder scan before switching source",
                )
            finder_switch_source_state["continue_calls"] = (
                int(finder_switch_source_state["continue_calls"]) + 1
            )
            finder_switch_source_state["submitted_source_url"] = source_url
            finder_switch_source_state["submitted_additional_pages"] = (
                additional_pages
            )
            finder_scan.update(
                {
                    "status": "queued",
                    "source_url": source_url,
                    "next_url": source_url,
                    "page_limit": (
                        int(finder_scan["pages_completed"]) + additional_pages
                    ),
                    "continuable": False,
                    "progress_percent": (
                        int(finder_scan["pages_completed"])
                        / (
                            int(finder_scan["pages_completed"])
                            + additional_pages
                        )
                        * 100
                    ),
                }
            )
            return {"scan": dict(finder_scan)}
        await asyncio.sleep(0.15)
        if "blocked-source" in source_url:
            raise HTTPException(
                status_code=409,
                detail="This visual smoke source is intentionally unavailable",
            )
        finder_scan.update(
            {
                "status": "running",
                "source_url": source_url,
                "next_url": source_url,
                "page_limit": int(finder_scan["pages_completed"]) + additional_pages,
                "continuable": False,
                "progress_percent": (
                    int(finder_scan["pages_completed"])
                    / (int(finder_scan["pages_completed"]) + additional_pages)
                    * 100
                ),
            }
        )
        return {"scan": finder_scan}

    async def fake_finder_switch_source_state() -> dict:
        return {
            **finder_switch_source_state,
            "scan_id": finder_scan["id"],
            "status": finder_scan["status"],
            "source_url": finder_scan["source_url"],
            "next_url": finder_scan["next_url"],
            "page_limit": finder_scan["page_limit"],
            "pages_completed": finder_scan["pages_completed"],
            "candidate_count": finder_scan["candidate_count"],
            "review_counts": dict(finder_switch_review_counts),
        }

    async def fake_finder_retry(scan_id: str, **kwargs: object) -> dict:
        if scan_id != finder_scan["id"] or finder_scan["status"] != "failed":
            raise HTTPException(
                status_code=409,
                detail="Only the failed visual Finder scan can be retried",
            )
        finder_retry_state["retry_calls"] += 1
        finder_scan.update(
            {
                "status": "queued",
                "error": "",
                "progress_percent": 0,
            }
        )
        return {"scan": dict(finder_scan)}

    async def fake_finder_retry_state() -> dict:
        return {
            "retry_calls": finder_retry_state["retry_calls"],
            "scan_id": finder_scan["id"],
            "source_url": finder_scan["source_url"],
            "next_url": finder_scan["next_url"],
            "candidate_count": finder_scan["candidate_count"],
            "corpus_search_complete": finder_scan["corpus_search_complete"],
        }

    def finder_modal_review_result(result_id: str) -> dict:
        result_number = int(result_id.rsplit("-", 1)[-1])
        gallery = galleries[result_number + 1]
        first_ordinal = 1 + ((result_number - 1) * 3)
        selected = list(
            finder_modal_review_state["selections"].get(result_id, [])
        )
        review = str(finder_modal_review_state["reviews"][result_id])
        return {
            "id": result_id,
            "gallery_id": gallery["id"],
            "gallery_url": gallery["url"],
            "title": f"Modal review candidate {result_number}",
            "rank": result_number,
            "score": 0.97 - (result_number * 0.02),
            "base_score": 0.93 - (result_number * 0.02),
            "feedback_adjustment": 0.02,
            "feedback_applied": True,
            "feedback_revision": 19,
            "ranking_tier": 2,
            "online_scanned": False,
            "review": review,
            "feedback_image_urls": selected,
            "feedback_usable_image_urls": selected,
            "feedback_pending_image_urls": [],
            "images_scored": 24,
            "image_count": 21,
            "person_count": 2,
            "score_breakdown": {
                "exact": 0.31,
                "pose": 0.97 - (result_number * 0.02),
                "appearance": 0.72,
            },
            "top_matches": [
                {
                    "rank": offset + 1,
                    "image_url": (
                        "https://cdni.pornpics.com/1280/manual/"
                        f"{first_ordinal + offset:03d}.jpg"
                    ),
                    "preview_url": (
                        "/api/media?url=https%3A%2F%2Fexample.test%2F"
                        f"modal-candidate-{result_number}-{offset + 1}.jpg"
                        "&token=visual"
                    ),
                    "ordinal": first_ordinal + offset,
                    "score": 0.97 - (result_number * 0.02) - (offset * 0.03),
                    "pose_score": 0.97
                    - (result_number * 0.02)
                    - (offset * 0.03),
                    "pose_reliable": True,
                    "match_type": "pose",
                    "ranking_tier": 2,
                    "person_count": 2,
                }
                for offset in range(3)
            ],
        }

    def finder_modal_boundary_result(
        result_id: str,
        *,
        rank: int,
        label: str,
        score: float,
    ) -> dict:
        gallery_url = (
            f"https://www.pornpics.com/galleries/manual-boundary-{label}/"
        )
        image_number = max(1, min(999, rank))
        return {
            "id": result_id,
            "gallery_id": encode_gallery_id(gallery_url),
            "gallery_url": gallery_url,
            "title": (
                f"Boundary candidate {int(label):03d}"
                if label.isdigit()
                else f"Higher-ranked insert {label.rsplit('-', 1)[-1].upper()}"
            ),
            "rank": rank,
            "score": score,
            "ranking_tier": 2,
            "online_scanned": True,
            "review": "pending",
            "feedback_image_urls": [],
            "feedback_usable_image_urls": [],
            "feedback_pending_image_urls": [],
            "images_scored": 21,
            "image_count": 21,
            "person_count": 2,
            "top_matches": [
                {
                    "rank": 1,
                    "image_url": (
                        "https://cdni.pornpics.com/1280/manual/"
                        f"{image_number:03d}.jpg"
                    ),
                    "preview_url": (
                        "/api/media?url=https%3A%2F%2Fexample.test%2F"
                        f"boundary-{label}.jpg&token=visual"
                    ),
                    "ordinal": image_number,
                    "score": score,
                    "pose_score": score,
                    "pose_reliable": True,
                    "match_type": "pose",
                    "ranking_tier": 2,
                    "person_count": 2,
                }
            ],
        }

    async def fake_finder_results(**kwargs: object) -> dict:
        def media(name: str) -> str:
            return (
                f"/api/media?url=https%3A%2F%2Fexample.test%2F{name}.jpg&token=visual"
            )

        if finder_modal_review:
            if finder_modal_review_state["boundary_mode"]:
                originals = [
                    finder_modal_boundary_result(
                        f"boundary-result-{index:03d}",
                        rank=index,
                        label=f"{index:03d}",
                        score=0.99 - (index / 1000),
                    )
                    for index in range(1, 51)
                ]
                inserts = (
                    [
                        finder_modal_boundary_result(
                            f"boundary-insert-{suffix}",
                            rank=index,
                            label=f"insert-{suffix}",
                            score=0.999 - (index / 10000),
                        )
                        for index, suffix in enumerate(("a", "b"), 1)
                    ]
                    if finder_modal_review_state["higher_ranked_inserts"]
                    else []
                )
                all_results = [*inserts, *originals]
                review_filter = str(kwargs.get("review") or "pending")
                offset = max(0, int(kwargs.get("offset") or 0))
                limit = max(1, int(kwargs.get("limit") or 24))
                if (
                    finder_modal_review_state["higher_ranked_inserts"]
                    and offset < 24
                ):
                    if review_filter != "all":
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "Active backward queue probes must use review=all "
                                "to retain the stable predecessor anchor"
                            ),
                        )
                    finder_modal_review_state["boundary_backward_probe_seen"] = (
                        True
                    )
                filtered = [
                    result
                    for result in all_results
                    if review_filter == "all" or result["review"] == review_filter
                ]
                page_items = filtered[offset : offset + limit]
                return {
                    "results": page_items,
                    "total": len(filtered),
                    "counts": {
                        "pending": len(filtered),
                        "accepted": 0,
                        "maybe": 0,
                        "rejected": 0,
                        "total": len(filtered),
                    },
                    "limit": limit,
                    "offset": offset,
                    "page": offset // limit + 1,
                    "page_size": limit,
                    "page_count": (len(filtered) + limit - 1) // limit,
                    "has_previous": offset > 0,
                    "has_next": offset + len(page_items) < len(filtered),
                }
            all_results = [
                finder_modal_review_result(f"visual-result-{index}")
                for index in range(1, 4)
            ]
            review_filter = str(kwargs.get("review") or "pending")
            filtered = [
                result
                for result in all_results
                if review_filter == "all" or result["review"] == review_filter
            ]
            counts = {
                review: sum(result["review"] == review for result in all_results)
                for review in ("pending", "accepted", "maybe", "rejected")
            }
            counts["total"] = len(all_results)
            return {
                "results": filtered,
                "counts": counts,
                "limit": 24,
                "offset": 0,
                "page": 1,
                "page_size": 24,
                "page_count": 1,
                "has_previous": False,
                "has_next": False,
            }

        if finder_tag:
            result = {
                "id": "visual-tag-result",
                "gallery_id": galleries[2]["id"],
                "gallery_url": galleries[2]["url"],
                "title": "Single qualifying JoyTag candidate",
                "rank": 1,
                "score": 0.69,
                "tag": "mating_press",
                "tag_score": 0.69,
                "tag_scores": {
                    "mating_press": 0.73,
                    "lying": 0.69,
                    "solo": 0.08,
                },
                "match_type": "tag",
                "ranking_tier": 1,
                "online_scanned": True,
                "review": "pending",
                "feedback_image_urls": [],
                "images_scored": 21,
                "image_count": 21,
                "top_matches": [
                    {
                        "rank": 1,
                        "image_url": "https://example.test/tag-candidate-1.jpg",
                        "preview_url": media("tag-candidate-1"),
                        "ordinal": 9,
                        "score": 0.69,
                        "tag": "mating_press",
                        "tag_score": 0.69,
                        "tag_scores": {
                            "mating_press": 0.73,
                            "lying": 0.69,
                            "solo": 0.08,
                        },
                        "match_type": "tag",
                        "ranking_tier": 1,
                    }
                ],
            }
            return {
                "items": [result],
                "total": 1,
                "counts": {
                    "pending": 1,
                    "accepted": 0,
                    "maybe": 0,
                    "rejected": 0,
                    "total": 1,
                },
                "limit": 24,
                "offset": 0,
                "page_count": 1,
            }

        if finder_race:
            finder_race_state["results_calls"] += 1
            results_call = finder_race_state["results_calls"]
            first_review = finder_race_state["review"]
            if results_call == 2:
                await asyncio.sleep(1.2)
            elif results_call >= 3:
                await asyncio.sleep(0.1)
        elif finder_unusable_save and finder_race_state["selected"]:
            first_review = finder_race_state["review"]
        else:
            first_review = (
                "accepted" if finder_review_mode or finder_continue else "pending"
            )
        first_image = (
            "https://cdni.pornpics.com/1280/manual/001.jpg"
            if finder_review_mode or finder_continue
            else "https://example.test/candidate-1.jpg"
        )
        feedback_urls = (
            list(finder_race_state["selected"])
            if finder_race or (finder_unusable_save and finder_race_state["selected"])
            else [first_image]
            if finder_review_mode or finder_continue
            else []
        )
        feedback_usable_urls = (
            []
            if finder_unusable_save and finder_race_state["selected"]
            else feedback_urls
        )
        feedback_pending_urls = (
            feedback_urls
            if finder_unusable_save and finder_race_state["selected"]
            else []
        )
        second_review = "rejected" if finder_continue else "pending"
        results = [
            {
                "id": "visual-result-1",
                "gallery_id": galleries[2]["id"],
                "gallery_url": galleries[2]["url"],
                "title": "High-confidence multi-person pose candidate",
                "rank": 1,
                "score": 0.96,
                "base_score": 0.94,
                "feedback_adjustment": 0.02,
                "feedback_applied": True,
                "feedback_revision": 19,
                "ranking_tier": 2,
                "online_scanned": False,
                "review": first_review,
                "feedback_image_urls": feedback_urls,
                "feedback_usable_image_urls": feedback_usable_urls,
                "feedback_pending_image_urls": feedback_pending_urls,
                "images_scored": 24,
                "image_count": 24,
                "person_count": 2,
                "score_breakdown": {
                    "exact": 0.31,
                    "pose": 0.96,
                    "appearance": 0.72,
                },
                "top_matches": [
                    {
                        "rank": 1,
                        "image_url": first_image,
                        "preview_url": media("candidate-1"),
                        "ordinal": 1 if finder_review_mode else 12,
                        "score": 0.96,
                        "base_score": 0.94,
                        "feedback_adjustment": 0.02,
                        "feedback_applied": True,
                        "feedback_revision": 19,
                        "ranking_tier": 2,
                        "pose_score": 0.96,
                        "pose_reliable": True,
                        "match_type": "pose",
                        "person_count": 2,
                        "skeleton_overlay_url": media("overlay-1"),
                    },
                    {
                        "rank": 2,
                        "image_url": "https://example.test/candidate-2.jpg",
                        "preview_url": media("candidate-2"),
                        "ordinal": 8,
                        "score": 0.88,
                        "ranking_tier": 2,
                        "pose_score": 0.88,
                        "pose_reliable": True,
                        "match_type": "pose",
                        "person_count": 2,
                        "skeleton_overlay_url": media("overlay-2"),
                    },
                    {
                        "rank": 3,
                        "image_url": "https://example.test/candidate-3.jpg",
                        "preview_url": media("candidate-3"),
                        "ordinal": 19,
                        "score": 0.84,
                        "ranking_tier": 1,
                        "appearance_score": 0.84,
                        "match_type": "visual_fallback",
                        "person_count": 2,
                    },
                ],
            },
            {
                "id": "visual-result-2",
                "gallery_id": galleries[3]["id"],
                "gallery_url": galleries[3]["url"],
                "title": "Exact source image found in gallery",
                "rank": 2,
                "score": 1.0,
                "ranking_tier": 3,
                "online_scanned": True,
                "review": second_review,
                "feedback_image_urls": [],
                "images_scored": 21,
                "image_count": 21,
                "is_exact": True,
                "exact_score": 1.0,
                "best_image_url": "https://example.test/candidate-2.jpg",
                "best_preview_url": media("candidate-2"),
                "best_ordinal": 7,
            },
        ]
        if finder_switch_source:
            all_results = []
            for index in range(124):
                if index < 119:
                    review = "pending"
                elif index < 122:
                    review = "accepted"
                elif index == 122:
                    review = "maybe"
                else:
                    review = "rejected"
                base = results[index % len(results)]
                all_results.append(
                    {
                        **base,
                        "id": f"visual-switch-result-{index + 1}",
                        "gallery_id": galleries[index % len(galleries)]["id"],
                        "gallery_url": galleries[index % len(galleries)]["url"],
                        "title": f"Retained JoyTag candidate {index + 1:03d}",
                        "rank": index + 1,
                        "score": 0.96 - index / 1000,
                        "review": review,
                    }
                )
            review_filter = str(kwargs.get("review", "pending"))
            minimum_score = float(kwargs.get("min_score") or 0)
            filtered = [
                item
                for item in all_results
                if (review_filter == "all" or item["review"] == review_filter)
                and item["score"] >= minimum_score
            ]
            offset = max(0, int(kwargs.get("offset") or 0))
            limit = max(1, int(kwargs.get("limit") or 24))
            page_items = filtered[offset : offset + limit]
            return {
                "items": page_items,
                "total": len(filtered),
                "counts": dict(finder_switch_review_counts),
                "limit": limit,
                "offset": offset,
                "page": offset // limit + 1,
                "page_size": limit,
                "page_count": (len(filtered) + limit - 1) // limit,
                "has_previous": offset > 0,
                "has_next": offset + len(page_items) < len(filtered),
            }
        if finder_pagination:
            all_results = [
                {
                    **results[0],
                    "id": f"visual-result-{index + 1}",
                    "gallery_id": galleries[index % len(galleries)]["id"],
                    "gallery_url": galleries[index % len(galleries)]["url"],
                    "title": f"Paged pose candidate {index + 1:02d}",
                    "rank": index + 1,
                    "score": 0.96 - index / 1000,
                    "review": "pending",
                    "feedback_image_urls": [],
                }
                for index in range(49)
            ]
            review_filter = str(kwargs.get("review", "pending"))
            minimum_score = float(kwargs.get("min_score") or 0)
            filtered = [
                item
                for item in all_results
                if (review_filter == "all" or item["review"] == review_filter)
                and item["score"] >= minimum_score
            ]
            offset = max(0, int(kwargs.get("offset") or 0))
            limit = max(1, int(kwargs.get("limit") or 24))
            page_items = filtered[offset : offset + limit]
            return {
                "items": page_items,
                "total": len(filtered),
                "counts": {
                    "pending": len(filtered),
                    "accepted": 0,
                    "maybe": 0,
                    "rejected": 0,
                    "total": len(filtered),
                },
                "limit": limit,
                "offset": offset,
                "page": offset // limit + 1,
                "page_size": limit,
                "page_count": (len(filtered) + limit - 1) // limit,
                "has_previous": offset > 0,
                "has_next": offset + len(page_items) < len(filtered),
            }
        reviews = [first_review, second_review]
        counts = {
            review: reviews.count(review)
            for review in ("pending", "accepted", "maybe", "rejected")
        }
        counts["total"] = len(reviews)
        return {
            "counts": counts,
            "results": [
                *results,
            ],
        }

    async def fake_finder_review(
        scan_id: str, result_id: str, payload, **kwargs: object
    ) -> dict:
        if finder_modal_review:
            review = str(payload.review)
            selected = list(payload.feedback_image_urls or [])
            finder_modal_review_state["reviews"][result_id] = review
            finder_modal_review_state["selections"][result_id] = selected
            finder_modal_review_state["review_history"].append(
                {
                    "result_id": result_id,
                    "review": review,
                    "feedback_image_urls": selected,
                }
            )
            return {"result": finder_modal_review_result(result_id)}
        await asyncio.sleep(0.6 if finder_race else 0)
        finder_race_state["review"] = payload.review
        selected = list(payload.feedback_image_urls or [])
        finder_race_state["selected"] = selected
        usable = [] if finder_unusable_save else selected
        pending = selected if finder_unusable_save else []
        return {
            "result": {
                "id": result_id,
                "gallery_id": galleries[2]["id"],
                "gallery_url": galleries[2]["url"],
                "title": "High-confidence multi-person pose candidate",
                "rank": 1,
                "score": 0.96,
                "ranking_tier": 2,
                "review": payload.review,
                "feedback_image_urls": selected,
                "feedback_usable_image_urls": usable,
                "feedback_pending_image_urls": pending,
            }
        }

    async def fake_finder_modal_boundary_start() -> dict:
        finder_modal_review_state["boundary_mode"] = True
        finder_modal_review_state["higher_ranked_inserts"] = False
        finder_scan.update(
            {
                "status": "running",
                "candidate_count": 50,
                "processed_galleries": 150,
                "processed_images": 3150,
                "progress_percent": 72,
            }
        )
        return {"active": True, "candidates": 50}

    async def fake_finder_modal_boundary_insert() -> dict:
        finder_modal_review_state["higher_ranked_inserts"] = True
        finder_scan["candidate_count"] = 52
        return {"inserted": 2, "candidates": 52}

    async def fake_finder_modal_boundary_state() -> dict:
        return {
            "backward_probe_seen": bool(
                finder_modal_review_state["boundary_backward_probe_seen"]
            ),
            "pose_export_calls": len(
                finder_modal_review_state["pose_export_calls"]
            ),
        }

    async def fake_finder_modal_prefetch_state() -> dict:
        adjacent = {
            f"result_{index}": galleries[index + 1]["url"]
            for index in range(1, 4)
        }
        return {
            "detail_calls": {
                key: gallery_detail_calls.get(url, 0)
                for key, url in adjacent.items()
            },
            "preview_calls": {
                key: sum(
                    count
                    for url, count in gallery_preview_calls.items()
                    if url.rstrip("/").rsplit("/", 2)[-2]
                    == gallery_url.rstrip("/").rsplit("/", 1)[-1]
                )
                for key, gallery_url in adjacent.items()
            },
            "preview_urls": dict(gallery_preview_calls),
            "preview_markers": {
                key: gallery_url.rstrip("/").rsplit("/", 1)[-1]
                for key, gallery_url in adjacent.items()
            },
            "background_detail_peak": gallery_detail_concurrency["peak"],
        }

    def finder_modal_pose_export_job() -> dict:
        calls = finder_modal_review_state["pose_export_calls"]
        pose_revision = int(calls[0].get("expected_revision") or 0) if calls else 0
        return {
            "id": "visual-pose-export",
            "kind": "pose_export",
            "status": "queued",
            "gallery_id": galleries[2]["id"],
            "gallery_url": galleries[2]["url"],
            "title": galleries[2]["title"],
            "profile": "Default",
            "pair_count": 1,
            "pose_revision": pose_revision,
            "total_images": 2,
            "completed_images": 0,
            "progress": 0,
            "created_at": "2026-07-24T12:00:00+00:00",
        }

    async def fake_finder_modal_pose_export(
        payload: object, **kwargs: object
    ) -> dict:
        values = (
            payload.model_dump()
            if callable(getattr(payload, "model_dump", None))
            else dict(payload)
            if isinstance(payload, dict)
            else {}
        )
        finder_modal_review_state["pose_export_calls"].append(values)
        if len(finder_modal_review_state["pose_export_calls"]) > 1:
            raise HTTPException(
                status_code=409,
                detail="The visual pose draft was already queued",
            )
        if (
            values.get("gallery_id") != galleries[2]["id"]
            or values.get("profile") != "Default"
            or int(values.get("expected_revision") or 0) < 1
        ):
            raise HTTPException(
                status_code=422,
                detail="The visual pose export payload was invalid",
            )
        return {"job": finder_modal_pose_export_job()}

    async def fake_finder_modal_downloads(**kwargs: object) -> dict:
        jobs = (
            [finder_modal_pose_export_job()]
            if finder_modal_review_state["pose_export_calls"]
            else []
        )
        return {"downloads": jobs}

    async def fake_finder_modal_pose_exports(**kwargs: object) -> dict:
        jobs = (
            [finder_modal_pose_export_job()]
            if finder_modal_review_state["pose_export_calls"]
            else []
        )
        return {"items": jobs}

    async def fake_events(request=None) -> Response:
        return Response(status_code=204)

    async def fake_bootstrap() -> Response:
        session_id = json.dumps(sort_session["id"])
        script = f"localStorage.setItem('galleryflow:sort-session', JSON.stringify({session_id}));"
        if load_more:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('#page-next');if(button&&!button.hidden&&!button.disabled){button.click();clearInterval(poll)}},50)});"
        if open_gallery:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.gallery-open');if(button){button.click();clearInterval(poll)}},50)});"
        if open_lightbox:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.image-preview-button');if(button){button.click();clearInterval(poll)}},50)});"
        if open_pose:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const modal=document.querySelector('#gallery-modal');const button=document.querySelector('[data-gallery-mode=pose]');const image=document.querySelector('.image-option:not(.skeleton-image)');if(modal?.open&&button&&image){button.click();clearInterval(poll)}},50)});"
        if open_finder and not finder_tag:
            script += "localStorage.setItem('galleryflow:finder-scan', JSON.stringify('visual-finder'));window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.finder-overlay-toggle:not([hidden])');if(button){button.click();clearInterval(poll)}},50)});"
        if finder_retry:
            script += """
window.addEventListener('load', () => {
  let phase = 'failed';
  let statePending = false;
  const poll = setInterval(() => {
    document.documentElement.dataset.finderRetryPhase = phase;
    const button = document.querySelector('#finder-resume');
    const session = document.querySelector('#finder-session-label')?.textContent || '';
    const error = document.querySelector('#finder-scan-error');
    const candidates = document.querySelector('#finder-candidates-found')?.textContent || '';
    const localState = document.querySelector('#finder-local-progress-state')?.textContent || '';
    const cards = document.querySelectorAll('#finder-result-grid .finder-card').length;
    document.documentElement.dataset.finderRetryDebug = [
      button?.hidden,
      button?.disabled,
      button?.textContent?.trim(),
      session,
      error?.hidden,
      candidates,
      localState,
      cards,
    ].join('|');
    if (
      phase === 'failed'
      && button
      && !button.hidden
      && !button.disabled
      && button.textContent.includes('Retry search')
      && session.includes('JoyTag')
      && session.includes('failed')
      && error
      && !error.hidden
      && error.textContent.includes('Could not reach PornPics')
      && candidates === '124'
      && localState === 'Done'
      && cards > 0
    ) {
      button.click();
      phase = 'retried';
      return;
    }
    if (
      phase === 'retried'
      && button?.hidden
      && session.includes('queued')
      && error?.hidden
      && candidates === '124'
      && localState === 'Done'
      && cards > 0
      && !statePending
    ) {
      statePending = true;
      fetch('/manual/finder-retry/state')
        .then(response => response.json())
        .then(data => {
          const passed = data.retry_calls === 1
            && data.scan_id === 'visual-finder'
            && data.source_url === 'https://www.pornpics.com/?q=pov+footjob'
            && data.next_url === data.source_url
            && data.candidate_count === 124
            && data.corpus_search_complete === true;
          document.documentElement.dataset.finderRetry = passed ? 'pass' : 'fail';
          phase = passed ? 'complete' : 'state-invalid';
          clearInterval(poll);
        })
        .catch(() => {
          document.documentElement.dataset.finderRetry = 'fail';
          phase = 'state-failed';
          clearInterval(poll);
        });
    }
  }, 50);
  setTimeout(() => {
    if (!document.documentElement.dataset.finderRetry) {
      document.documentElement.dataset.finderRetry = 'fail';
      clearInterval(poll);
    }
  }, 5500);
});
"""
        if finder_tag:
            script += """
localStorage.setItem('galleryflow:finder-mode', JSON.stringify('joytag'));
localStorage.setItem('galleryflow:finder-scan', JSON.stringify(''));
window.addEventListener('load', () => {
  let phase = 'configure';
  let indexOk = false;
  let resultOk = false;
  const folderValue = 'sorted_outpaint/mating press - backview/selected_target_upscaled';
  const requiredChips = () => document.querySelectorAll(
    '#finder-joytag-required-tags .finder-joytag-query-chip'
  );
  const excludedChips = () => document.querySelectorAll(
    '#finder-joytag-excluded-tags .finder-joytag-query-chip'
  );
  const poll = setInterval(() => {
    document.documentElement.dataset.finderTagPhase = phase;
    const folder = document.querySelector('#finder-folder');
    const analyze = document.querySelector('#finder-analyze-references');
    if (phase === 'configure' && folder && analyze && !analyze.hidden) {
      if (folder.value !== folderValue) {
        folder.value = folderValue;
        folder.dispatchEvent(new Event('input', { bubbles: true }));
      }
      const panel = document.querySelector('#finder-corpus-joytag');
      const start = document.querySelector('#finder-corpus-index-start');
      const cached = Number(
        document.querySelector('#finder-corpus-joytag-cached')
          ?.textContent.replace(/\\D/g, '')
      );
      const total = Number(
        document.querySelector('#finder-corpus-joytag-total')
          ?.textContent.replace(/\\D/g, '')
      );
      const policy = document.querySelector(
        '.finder-corpus-index-policy'
      )?.textContent || '';
      const ordinaryCopy = document.querySelector(
        '#finder-corpus-copy'
      )?.textContent || '';
      document.documentElement.dataset.finderTagDebug = [
        panel?.hidden,
        start?.disabled,
        cached,
        total,
        policy.includes('Ordinary Tag searches do not index'),
        policy.includes('cached local images only'),
        policy.includes('go straight to the Source URL'),
        ordinaryCopy.includes('never fill missing cache entries'),
      ].join('|');
      if (
        panel
        && !panel.hidden
        && start
        && !start.disabled
        && cached === 1200
        && total === 8420
        && policy.includes('Ordinary Tag searches do not index')
        && policy.includes('cached local images only')
        && policy.includes('go straight to the Source URL')
        && ordinaryCopy.includes('never fill missing cache entries')
      ) {
        start.click();
        phase = 'index-running';
      }
    } else if (phase === 'index-running') {
      const cancel = document.querySelector('#finder-corpus-index-cancel');
      const progress = Number(
        document.querySelector('#finder-corpus-index-progress')
          ?.getAttribute('aria-valuenow')
      );
      const progressCopy = document.querySelector(
        '#finder-corpus-index-progress-copy'
      )?.textContent || '';
      const state = document.querySelector(
        '#finder-corpus-joytag-state'
      )?.textContent;
      document.documentElement.dataset.finderTagDebug = [
        cancel?.hidden,
        cancel?.disabled,
        progress,
        state,
        progressCopy.includes('600 processed'),
        progressCopy.includes('590 cached'),
        progressCopy.includes('10 failed'),
        progressCopy.includes('6,620 remaining'),
      ].join('|');
      if (
        cancel
        && !cancel.hidden
        && !cancel.disabled
        && progress === 21.4
        && state === 'Indexing'
        && progressCopy.includes('600 processed')
        && progressCopy.includes('590 cached')
        && progressCopy.includes('10 failed')
        && progressCopy.includes('6,620 remaining')
      ) {
        indexOk = true;
        cancel.click();
        phase = 'index-canceled';
      }
    } else if (phase === 'index-canceled') {
      const start = document.querySelector('#finder-corpus-index-start');
      const state = document.querySelector(
        '#finder-corpus-joytag-state'
      )?.textContent;
      const cached = Number(
        document.querySelector('#finder-corpus-joytag-cached')
          ?.textContent.replace(/\\D/g, '')
      );
      const progressCopy = document.querySelector(
        '#finder-corpus-index-progress-copy'
      )?.textContent || '';
      if (
        indexOk
        && state === 'Canceled'
        && start
        && !start.hidden
        && !start.disabled
        && cached === 1790
        && progressCopy.includes('stopped safely')
        && progressCopy.includes('6,630')
        && !analyze.disabled
      ) {
        analyze.click();
        phase = 'analysis';
      }
    } else if (phase === 'analysis') {
      const references = document.querySelectorAll(
        '#finder-joytag-reference-grid .finder-joytag-reference'
      );
      if (references.length !== 2) return;
      const requiredCopy = [...requiredChips()]
        .map(item => item.textContent)
        .join(' ');
      const excludedCopy = [...excludedChips()]
        .map(item => item.textContent)
        .join(' ');
      if (!requiredCopy.includes('Mating press')) {
        document.querySelector(
          '[data-finder-joytag-inspect="mating_press"]'
        )?.click();
        return;
      }
      if (!requiredCopy.includes('Lying')) {
        document.querySelector(
          '[data-finder-joytag-role="required"][data-finder-joytag-tag="lying"]'
        )?.click();
        return;
      }
      const filter = document.querySelector('#finder-joytag-tag-filter');
      if (!excludedCopy.includes('Solo')) {
        if (filter?.value !== 'solo') {
          filter.value = 'solo';
          filter.dispatchEvent(new Event('input', { bubbles: true }));
          return;
        }
        document.querySelector(
          '[data-finder-joytag-role="excluded"][data-finder-joytag-tag="solo"]'
        )?.click();
        return;
      }
      if (filter?.value) {
        filter.value = '';
        filter.dispatchEvent(new Event('input', { bubbles: true }));
        return;
      }
      if (
        !document.querySelector('#finder-joytag-selected-tag')
          ?.textContent.startsWith('Mating press')
      ) {
        document.querySelector(
          '[data-finder-joytag-inspect="mating_press"]'
        )?.click();
        return;
      }
      const threshold = document.querySelector('#finder-joytag-threshold');
      threshold.value = '0.35';
      threshold.dispatchEvent(new Event('input', { bubbles: true }));
      const rejectThreshold = document.querySelector(
        '#finder-joytag-reject-threshold'
      );
      rejectThreshold.value = '0.55';
      rejectThreshold.dispatchEvent(new Event('input', { bubbles: true }));
      const label = document.querySelector('#finder-joytag-dataset-label');
      label.value = 'mating press - backview';
      label.dispatchEvent(new Event('input', { bubbles: true }));
      const oldThreshold = document.querySelector('#finder-min-similarity');
      const statsOk = (
        document.querySelector('#finder-joytag-average')?.textContent
          === '0.820'
        && document.querySelector('#finder-joytag-reference-coverage')
          ?.textContent === '2 / 2 pass'
      );
      const queryOk = (
        requiredChips().length === 2
        && excludedChips().length === 1
        && document.querySelector('#finder-joytag-query-count')?.textContent
          === '2 required · 1 excluded'
      );
      const oldHidden = (
        oldThreshold?.disabled
        && oldThreshold.closest('.field')?.hidden
      );
      if (
        statsOk
        && queryOk
        && oldHidden
        && !document.querySelector('#finder-start')?.disabled
      ) {
        document.querySelector('#finder-start').click();
        phase = 'results';
      }
    } else if (phase === 'results') {
      const card = document.querySelector('#finder-result-grid .finder-card');
      if (!card) return;
      const threshold = document.querySelector('#finder-result-threshold');
      const copy = [...card.querySelectorAll(
        '.finder-similarity,.finder-match-kind,.finder-match-score,'
        + '.finder-feedback-selection-copy,.finder-score-breakdown'
      )].map(item => item.textContent).join(' ');
      const oneMatch = card.querySelectorAll('.finder-match').length === 1;
      const rangeOk = (
        threshold?.min === '0.05'
        && threshold?.value === '0.35'
      );
      const queryEvidence = (
        copy.includes('ALL 2')
        && copy.includes('weakest')
        && copy.includes('Mating press')
        && copy.includes('Lying')
        && copy.includes('Solo')
        && copy.includes('required')
        && copy.includes('excluded')
        && copy.includes('73%')
        && copy.includes('69%')
        && copy.includes('8%')
      );
      const scanLabelOk = [
        ...document.querySelector('#finder-scan-select')?.options || []
      ].some(option => option.textContent.includes('Mating press +1 · −1'));
      const joytagCopy = (
        copy.includes('JoyTag')
        && !copy.includes('Visual fallback')
        && !copy.includes('Pose match')
      );
      resultOk = (
        indexOk
        &&
        oneMatch
        && rangeOk
        && queryEvidence
        && scanLabelOk
        && joytagCopy
      );
      if (!resultOk) {
        document.documentElement.dataset.finderTag = 'fail';
        clearInterval(poll);
        return;
      }
      document.querySelector(
        '[data-finder-joytag-role="required"][data-finder-joytag-tag="lying"]'
      )?.click();
      phase = 'remove-excluded';
    } else if (phase === 'remove-excluded') {
      if (requiredChips().length !== 1) return;
      document.querySelector(
        '[data-finder-joytag-role="excluded"][data-finder-joytag-tag="solo"]'
      )?.click();
      phase = 'restore';
    } else if (phase === 'restore') {
      if (excludedChips().length !== 0) return;
      const requiredThreshold = document.querySelector(
        '#finder-joytag-threshold'
      );
      requiredThreshold.value = '0.90';
      requiredThreshold.dispatchEvent(new Event('input', { bubbles: true }));
      const rejectThreshold = document.querySelector(
        '#finder-joytag-reject-threshold'
      );
      rejectThreshold.value = '0.20';
      rejectThreshold.dispatchEvent(new Event('input', { bubbles: true }));
      const select = document.querySelector('#finder-scan-select');
      if (select?.value !== 'visual-tag-finder') return;
      select.dispatchEvent(new Event('change', { bubbles: true }));
      phase = 'restored';
    } else if (phase === 'restored') {
      const restored = (
        document.querySelector('#finder-scan-select')?.value
          === 'visual-tag-finder'
        && requiredChips().length === 2
        && excludedChips().length === 1
        && document.querySelector('#finder-joytag-threshold')?.value
          === '0.35'
        && document.querySelector('#finder-joytag-reject-threshold')?.value
          === '0.55'
      );
      if (restored) {
        document.querySelector('#finder-joytag-analysis')?.removeAttribute(
          'open'
        );
        document.documentElement.dataset.finderTag = resultOk
          ? 'pass'
          : 'fail';
        clearInterval(poll);
      }
    }
  }, 50);
  setTimeout(() => {
    if (!document.documentElement.dataset.finderTag) {
      document.documentElement.dataset.finderTag = 'fail';
    }
  }, 7500);
});
"""
        if finder_pagination:
            script += "window.addEventListener('load',()=>{let clicked=false;const poll=setInterval(()=>{const status=document.querySelector('#finder-page-status')?.textContent;const cards=document.querySelectorAll('#finder-result-grid .finder-card');if(!clicked&&status==='Page 1 of 3'&&cards.length===24){document.querySelector('#finder-page-next')?.click();clicked=true}else if(clicked&&status==='Page 2 of 3'&&cards.length===24&&cards[0]?.querySelector('.finder-rank')?.textContent==='#25'){document.documentElement.dataset.finderPagination='pass';clearInterval(poll)}},50);setTimeout(()=>{if(!document.documentElement.dataset.finderPagination)document.documentElement.dataset.finderPagination='fail'},4500)});"
        if finder_modal_review:
            script += """
window.addEventListener('load', () => {
  let phase = 'open-first';
  let refreshAt = 0;
  let prefetchRequestPending = false;
  let poseExportStatePending = false;
  let previewBaseline = {};
  let prefetchDiagnostics = '';
  const optionAt = index =>
    document.querySelectorAll('#image-grid .image-option')[index];
  const optionInput = index => optionAt(index)?.querySelector('input');
  const checkedCount = () => document.querySelectorAll(
    '#image-grid .image-option input:checked'
  ).length;
  const reviewButton = review => document.querySelector(
    `[data-gallery-finder-review="${review}"]`
  );
  const setPhase = value => {
    phase = value;
    document.documentElement.dataset.finderModalReviewPhase = value;
  };
  const poll = setInterval(() => {
    const modal = document.querySelector('#gallery-modal');
    const title = document.querySelector('#gallery-modal-title')?.textContent || '';
    const position = document.querySelector(
      '#gallery-review-position'
    )?.textContent || '';
    const status = document.querySelector(
      '#gallery-review-status'
    )?.textContent || '';
    const feedbackCount = document.querySelector(
      '#gallery-review-feedback-count'
    )?.textContent || '';
    const poseStatus = document.querySelector(
      '#pose-save-status'
    )?.textContent || '';
    const counts = [
      document.querySelector('#finder-accepted-count')?.textContent,
      document.querySelector('#finder-maybe-count')?.textContent,
      document.querySelector('#finder-rejected-count')?.textContent,
    ].join('/');
    document.documentElement.dataset.finderModalReviewDebug = [
      phase,
      title,
      position,
      status,
      feedbackCount,
      modal?.className,
      modal?.open,
      checkedCount(),
      poseStatus,
      counts,
      document.querySelectorAll('#finder-result-grid .finder-card').length,
      document.querySelector('#finder-pending-count')?.textContent,
      document.querySelector('#finder-page-status')?.textContent,
      document.querySelector('#finder-session-label')?.textContent,
      document.querySelector('#finder-scan-select')?.value,
      document.querySelector('#finder-scan-select')?.disabled,
      document.querySelector('#finder-result-grid')?.getAttribute('aria-busy'),
      document.querySelector('#finder-result-threshold')?.value,
      document.querySelector('#pose-export')?.disabled,
      document.querySelector('#pose-export')?.textContent?.trim(),
      prefetchDiagnostics,
    ].join('|');

    if (phase === 'open-first') {
      const cards = document.querySelectorAll(
        '#finder-result-grid .finder-card'
      );
      if (
        cards.length === 3
        && document.querySelector('#finder-pending-count')?.textContent === '3'
        && !prefetchRequestPending
      ) {
        prefetchRequestPending = true;
        setPhase('prefetch-snapshotting');
        fetch('/manual/finder-modal-review/prefetch-state')
          .then(response => response.json())
          .then(data => {
            previewBaseline = { ...(data.preview_urls || {}) };
            prefetchRequestPending = false;
            cards[1].querySelector('.finder-open')?.click();
            setPhase('prefetch-middle');
          })
          .catch(() => {
            prefetchRequestPending = false;
            setPhase('prefetch-baseline-failed');
          });
      }
      return;
    }

    if (
      phase === 'prefetch-middle'
      && modal?.open
      && title === 'Soft focus summer collection'
      && position === '2 of 3'
      && document.querySelector('#image-grid')?.getAttribute('aria-busy')
        === 'false'
      && optionAt(20)
      && !prefetchRequestPending
    ) {
      prefetchRequestPending = true;
      fetch('/manual/finder-modal-review/prefetch-state')
        .then(response => response.json())
        .then(data => {
          prefetchRequestPending = false;
          const calls = data.detail_calls || {};
          const previewUrls = data.preview_urls || {};
          const markers = data.preview_markers || {};
          const deltasFor = key => Object.entries(previewUrls)
            .filter(([url]) => url.includes(
              `/460/manual/${markers[key]}/`
            ))
            .map(([url, count]) => ({
              url,
              count: Math.max(
                0,
                Number(count || 0) - Number(previewBaseline[url] || 0)
              ),
            }))
            .filter(item => item.count > 0);
          const previousPreviewDeltas = deltasFor('result_1');
          const nextPreviewDeltas = deltasFor('result_3');
          const previousPreviewCount = previousPreviewDeltas.reduce(
            (total, item) => total + item.count,
            0
          );
          const nextPreviewCount = nextPreviewDeltas.reduce(
            (total, item) => total + item.count,
            0
          );
          const previewCount = previousPreviewCount + nextPreviewCount;
          const previewsReady = (
            previousPreviewDeltas.length > 0
            && nextPreviewDeltas.length > 0
          );
          const previewBoundsOk = (
            previousPreviewDeltas.length <= 3
            && nextPreviewDeltas.length <= 3
            && previousPreviewCount <= 3
            && nextPreviewCount <= 3
            && previewCount <= 6
            && [...previousPreviewDeltas, ...nextPreviewDeltas]
              .every(item => item.count === 1)
          );
          const backgroundPeak = Number(data.background_detail_peak || 0);
          prefetchDiagnostics = [
            `detail:${calls.result_1}/${calls.result_2}/${calls.result_3}`,
            `preview:${previousPreviewCount}/${nextPreviewCount}`,
            `urls:${previousPreviewDeltas.length}/${nextPreviewDeltas.length}`,
            `peak:${backgroundPeak}`,
          ].join(',');
          if (Object.values(calls).some(count => count > 1)) {
            setPhase('prefetch-detail-duplicated-before-navigation');
            return;
          }
          if (!previewsReady) return;
          if (!previewBoundsOk) {
            setPhase('prefetch-preview-bounds-failed');
            return;
          }
          if (backgroundPeak < 1 || backgroundPeak > 2) {
            setPhase('prefetch-concurrency-bound-failed');
            return;
          }
          if (
            calls.result_1 === 1
            && calls.result_2 === 1
            && calls.result_3 === 1
          ) {
            const next = document.querySelector('#gallery-review-next');
            if (next && !next.disabled) {
              next.click();
              setPhase('prefetch-next');
            }
          }
        })
        .catch(() => {
          prefetchRequestPending = false;
          setPhase('prefetch-state-failed');
        });
      return;
    }

    if (
      phase === 'prefetch-next'
      && title === 'Classic monochrome session'
      && position === '3 of 3'
      && !prefetchRequestPending
    ) {
      prefetchRequestPending = true;
      fetch('/manual/finder-modal-review/prefetch-state')
        .then(response => response.json())
        .then(data => {
          prefetchRequestPending = false;
          if (data.detail_calls?.result_3 !== 1) {
            setPhase('prefetch-next-refetched-detail');
            return;
          }
          const previous = document.querySelector('#gallery-review-previous');
          if (previous && !previous.disabled) {
            previous.click();
            setPhase('prefetch-middle-return');
          }
        })
        .catch(() => {
          prefetchRequestPending = false;
          setPhase('prefetch-state-failed');
        });
      return;
    }

    if (
      phase === 'prefetch-middle-return'
      && title === 'Soft focus summer collection'
      && position === '2 of 3'
    ) {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('prefetch-first');
      }
      return;
    }

    if (
      phase === 'prefetch-first'
      && title === 'After-hours city gallery'
      && position === '1 of 3'
      && !prefetchRequestPending
    ) {
      prefetchRequestPending = true;
      fetch('/manual/finder-modal-review/prefetch-state')
        .then(response => response.json())
        .then(data => {
          prefetchRequestPending = false;
          const calls = data.detail_calls || {};
          if (
            calls.result_1 === 1
            && calls.result_2 === 1
            && calls.result_3 === 1
          ) {
            setPhase('open-feedback');
          } else {
            setPhase('prefetch-previous-refetched-detail');
          }
        })
        .catch(() => {
          prefetchRequestPending = false;
          setPhase('prefetch-state-failed');
        });
      return;
    }

    if (
      phase === 'open-feedback'
      && modal?.open
      && !document.querySelector('#gallery-review-rail')?.hidden
      && title === 'After-hours city gallery'
      && position === '1 of 3'
      && document.querySelector('#image-grid')?.getAttribute('aria-busy')
        === 'false'
      && optionAt(20)
    ) {
      if (!modal.classList.contains('is-feedback-mode')) {
        document.querySelector('[data-gallery-mode="feedback"]')?.click();
        return;
      }
      setPhase('clear-feedback');
      return;
    }

    if (phase === 'clear-feedback' && modal.classList.contains(
      'is-feedback-mode'
    )) {
      if (checkedCount()) {
        document.querySelector('#select-none')?.click();
        return;
      }
      setPhase('select-feedback');
      return;
    }

    if (phase === 'select-feedback') {
      const target = optionInput(9);
      if (target && !target.checked) {
        target.click();
        return;
      }
      if (
        target?.checked
        && feedbackCount === '1'
        && !reviewButton('accepted')?.disabled
      ) {
        reviewButton('accepted').click();
        setPhase('accepted');
      }
      return;
    }

    if (
      phase === 'accepted'
      && status === 'Accepted'
      && feedbackCount === '1'
      && document.querySelector('#finder-accepted-count')?.textContent === '1'
    ) {
      const prepare = document.querySelector('#finder-feedback-prepare-pose');
      if (prepare && !prepare.hidden && !prepare.disabled) {
        prepare.click();
        setPhase('set-control');
      }
      return;
    }

    if (
      phase === 'set-control'
      && modal.classList.contains('is-pose-mode')
      && !poseStatus.includes('Loading')
      && optionAt(20)
    ) {
      document.querySelectorAll(
        '#image-grid .image-option input:checked'
      ).forEach(input => input.click());
      const target = optionInput(9);
      if (target && !target.checked) target.click();
      const applyTarget = document.querySelector(
        '[data-pose-assignment="target"]'
      );
      if (target?.checked && applyTarget && !applyTarget.disabled) {
        applyTarget.click();
        setPhase('target-before-control');
      }
      return;
    }

    if (
      phase === 'target-before-control'
      && optionAt(9)?.classList.contains('has-pose-target')
      && poseStatus.includes('Draft saved')
    ) {
      const exportButton = document.querySelector('#pose-export');
      if (!exportButton?.disabled) {
        setPhase('target-without-control-export-enabled');
        return;
      }
      document.querySelectorAll(
        '#image-grid .image-option input:checked'
      ).forEach(input => input.click());
      const control = optionInput(0);
      if (control && !control.checked) control.click();
      const setControl = document.querySelector(
        '[data-pose-assignment="couple"]'
      );
      if (control?.checked && setControl && !setControl.disabled) {
        setControl.click();
        setPhase('draft-saved');
      }
      return;
    }

    if (
      phase === 'draft-saved'
      && optionAt(0)?.classList.contains('has-pose-control')
      && optionAt(9)?.classList.contains('has-pose-target')
      && poseStatus.includes('Draft saved')
      && feedbackCount === '1'
    ) {
      const exportButton = document.querySelector('#pose-export');
      if (
        exportButton
        && !exportButton.disabled
        && exportButton.textContent.includes('Download & organize')
      ) {
        exportButton.click();
        setPhase('pose-export-queued');
      }
      return;
    }

    if (
      phase === 'pose-export-queued'
      && modal.open
      && title === 'After-hours city gallery'
      && position === '1 of 3'
      && !poseExportStatePending
    ) {
      const exportButton = document.querySelector('#pose-export');
      const next = document.querySelector('#gallery-review-next');
      if (
        exportButton?.disabled
        && exportButton.textContent.includes('Queued')
        && next
        && !next.disabled
      ) {
        poseExportStatePending = true;
        fetch('/manual/finder-modal-review/state')
          .then(response => response.json())
          .then(data => {
            poseExportStatePending = false;
            if (data.pose_export_calls !== 1) {
              setPhase('pose-export-count-invalid');
              return;
            }
            next.click();
            setPhase('maybe-second');
          })
          .catch(() => {
            poseExportStatePending = false;
            setPhase('pose-export-state-failed');
          });
      }
      return;
    }

    if (
      phase === 'maybe-second'
      && title === 'Soft focus summer collection'
      && position === '2 of 3'
      && !reviewButton('maybe')?.disabled
    ) {
      reviewButton('maybe').click();
      setPhase('next-after-maybe');
      return;
    }

    if (
      phase === 'next-after-maybe'
      && status === 'Maybe'
      && document.querySelector('#finder-maybe-count')?.textContent === '1'
    ) {
      const next = document.querySelector('#gallery-review-next');
      if (next && !next.disabled) {
        next.click();
        setPhase('reject-third');
      }
      return;
    }

    if (
      phase === 'reject-third'
      && title === 'Classic monochrome session'
      && position === '3 of 3'
      && !reviewButton('rejected')?.disabled
    ) {
      reviewButton('rejected').click();
      setPhase('previous-to-second');
      return;
    }

    if (
      phase === 'previous-to-second'
      && status === 'Rejected'
      && document.querySelector('#finder-rejected-count')?.textContent === '1'
    ) {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('previous-to-first');
      }
      return;
    }

    if (
      phase === 'previous-to-first'
      && title === 'Soft focus summer collection'
      && position === '2 of 3'
      && status === 'Maybe'
    ) {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('verify-feedback');
      }
      return;
    }

    if (
      phase === 'verify-feedback'
      && title === 'After-hours city gallery'
      && position === '1 of 3'
      && status === 'Accepted'
      && feedbackCount === '1'
      && document.querySelector('#gallery-review-previous')?.disabled
    ) {
      if (!modal.classList.contains('is-feedback-mode')) {
        document.querySelector('[data-gallery-mode="feedback"]')?.click();
        return;
      }
      if (checkedCount() !== 1 || !optionInput(9)?.checked) return;
      document.querySelector('[data-gallery-mode="pose"]')?.click();
      setPhase('verify-draft');
      return;
    }

    if (
      phase === 'verify-draft'
      && modal.classList.contains('is-pose-mode')
      && optionAt(0)?.classList.contains('has-pose-control')
      && optionAt(9)?.classList.contains('has-pose-target')
      && poseStatus.includes('Draft saved')
      && feedbackCount === '1'
      && counts === '1/1/1'
    ) {
      const exportButton = document.querySelector('#pose-export');
      if (
        exportButton?.disabled
        && exportButton.textContent.includes('Queued')
      ) {
        exportButton.click();
        setPhase('verify-pose-export-idempotent');
      }
      return;
    }

    if (
      phase === 'verify-pose-export-idempotent'
      && modal.open
      && !poseExportStatePending
    ) {
      const exportButton = document.querySelector('#pose-export');
      if (
        !exportButton?.disabled
        || !exportButton.textContent.includes('Queued')
      ) {
        setPhase('returned-pose-export-enabled');
        return;
      }
      poseExportStatePending = true;
      fetch('/manual/finder-modal-review/state')
        .then(response => response.json())
        .then(data => {
          poseExportStatePending = false;
          if (data.pose_export_calls !== 1) {
            setPhase('pose-export-duplicated');
            return;
          }
          document.querySelector('#select-none')?.click();
          setPhase('prepare-busy-enter');
        })
        .catch(() => {
          poseExportStatePending = false;
          setPhase('pose-export-state-failed');
        });
      return;
    }

    if (
      phase === 'prepare-busy-enter'
      && modal.classList.contains('is-pose-mode')
      && checkedCount() === 0
      && optionAt(0)?.classList.contains('has-pose-control')
      && optionAt(9)?.classList.contains('has-pose-target')
    ) {
      const untouchedTarget = optionInput(3);
      if (untouchedTarget && !untouchedTarget.checked) {
        untouchedTarget.click();
        setPhase('trigger-busy-enter');
        return;
      }
    }

    if (
      phase === 'trigger-busy-enter'
      && modal.classList.contains('is-pose-mode')
    ) {
      const untouchedTarget = optionInput(3);
      const apply = document.querySelector('#pose-apply-checked');
      const next = document.querySelector('#gallery-review-next');
      if (untouchedTarget?.checked && apply && !apply.disabled && next && !next.disabled) {
        next.click();
        document.body.dispatchEvent(new KeyboardEvent('keydown', {
          key: 'Enter',
          bubbles: true,
        }));
        setPhase('verify-busy-enter');
      }
      return;
    }

    if (
      phase === 'verify-busy-enter'
      && title === 'Soft focus summer collection'
      && position === '2 of 3'
    ) {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('verify-busy-enter-return');
      }
      return;
    }

    if (
      phase === 'verify-busy-enter-return'
      && title === 'After-hours city gallery'
      && position === '1 of 3'
      && modal.classList.contains('is-pose-mode')
      && optionAt(0)?.classList.contains('has-pose-control')
      && optionAt(9)?.classList.contains('has-pose-target')
      && !optionAt(3)?.classList.contains('has-pose-target')
      && document.querySelector('#pose-target-count')?.textContent === '1'
      && poseStatus.includes('Draft saved')
    ) {
      modal.querySelector('.modal-header [data-close-modal]')?.click();
      setPhase('boundary-starting');
      fetch('/manual/finder-modal-review/boundary', {
        method: 'POST',
      }).then(response => {
        if (!response.ok) {
          setPhase(`boundary-start-http-${response.status}`);
          return;
        }
        setPhase('boundary-refresh');
      }).catch(() => {
        setPhase('boundary-start-failed');
      });
      return;
    }

    if (phase === 'boundary-refresh' && !modal.open) {
      const acceptedTab = document.querySelector(
        '[data-finder-review="accepted"]'
      );
      if (acceptedTab) {
        acceptedTab.click();
        refreshAt = Date.now();
        setPhase('boundary-select-pending');
      }
      return;
    }

    if (
      phase === 'boundary-select-pending'
      && document.querySelector('[data-finder-review="accepted"]')
        ?.getAttribute('aria-selected') === 'true'
    ) {
      document.querySelector('[data-finder-review="pending"]')?.click();
      refreshAt = Date.now();
      setPhase('boundary-page-one');
      return;
    }

    if (phase === 'boundary-page-one') {
      const cards = document.querySelectorAll(
        '#finder-result-grid .finder-card'
      );
      if (
        cards.length === 24
        && document.querySelector('#finder-pending-count')?.textContent === '50'
        && document.querySelector('#finder-page-status')?.textContent
          === 'Page 1 of 3'
        && document.querySelector('#finder-session-label')?.textContent
          .includes('running')
      ) {
        document.querySelector('#finder-page-next')?.click();
        setPhase('boundary-page-two');
      } else if (Date.now() - refreshAt > 600) {
        const pendingTab = document.querySelector(
          '[data-finder-review="pending"]'
        );
        if (pendingTab?.getAttribute('aria-selected') !== 'true') {
          pendingTab?.click();
        } else {
          document.querySelector('#finder-result-threshold')?.dispatchEvent(
            new Event('change', { bubbles: true })
          );
        }
        refreshAt = Date.now();
      }
      return;
    }

    if (phase === 'boundary-page-two') {
      const cards = document.querySelectorAll(
        '#finder-result-grid .finder-card'
      );
      if (
        cards.length === 24
        && document.querySelector('#finder-page-status')?.textContent
          === 'Page 2 of 3'
        && cards[0]?.querySelector('.finder-rank')?.textContent === '#25'
        && cards[0]?.querySelector('.finder-card-title')?.textContent
          === 'Boundary candidate 025'
      ) {
        cards[0].querySelector('.finder-open')?.click();
        setPhase('boundary-open');
      }
      return;
    }

    if (
      phase === 'boundary-open'
      && modal.open
      && title === 'Boundary candidate 025'
      && position === '25 of 50'
      && document.querySelector('#image-grid')?.getAttribute('aria-busy')
        === 'false'
    ) {
      setPhase('boundary-inserting');
      fetch('/manual/finder-modal-review/insert', {
        method: 'POST',
      }).then(response => {
        if (!response.ok) {
          setPhase(`boundary-insert-http-${response.status}`);
          return;
        }
        setPhase('boundary-inserted');
      }).catch(() => {
        setPhase('boundary-insert-failed');
      });
      return;
    }

    if (phase === 'boundary-inserted') {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('boundary-previous-24');
      }
      return;
    }

    if (
      phase === 'boundary-previous-24'
      && title === 'Boundary candidate 024'
    ) {
      const previous = document.querySelector('#gallery-review-previous');
      if (previous && !previous.disabled) {
        previous.click();
        setPhase('boundary-previous-23');
      }
      return;
    }

    if (
      phase === 'boundary-previous-23'
      && title === 'Boundary candidate 023'
    ) {
      setPhase('boundary-probe-check');
      fetch('/manual/finder-modal-review/state')
        .then(response => response.json())
        .then(data => {
          if (data.backward_probe_seen) {
            document.documentElement.dataset.finderModalReview = 'pass';
            clearInterval(poll);
          } else {
            setPhase('boundary-probe-missing');
          }
        })
        .catch(() => setPhase('boundary-probe-check-failed'));
    }
  }, 60);
  setTimeout(() => {
    if (!document.documentElement.dataset.finderModalReview) {
      document.documentElement.dataset.finderModalReview = 'fail';
    }
  }, 16500);
});
"""
        if finder_switch_source:
            script += """
window.confirm = () => true;
window.addEventListener('load', () => {
  const oldCursor = 'https://www.pornpics.com/?q=pov+footjob&page=2';
  const nextUrl = 'https://www.pornpics.com/?q=pov+paizuri';
  let phase = 'waiting-running';
  let initialTitle = '';
  let initialCards = 0;
  let statePending = false;
  const countsOk = () => (
    document.querySelector('#finder-pending-count')?.textContent === '119'
    && document.querySelector('#finder-accepted-count')?.textContent === '3'
    && document.querySelector('#finder-maybe-count')?.textContent === '1'
    && document.querySelector('#finder-rejected-count')?.textContent === '1'
  );
  const resultsOk = () => {
    const cards = document.querySelectorAll('#finder-result-grid .finder-card');
    const title = document.querySelector('#finder-result-grid .finder-card-title')
      ?.textContent || '';
    return cards.length === initialCards && title === initialTitle;
  };
  const poll = setInterval(() => {
    document.documentElement.dataset.finderSwitchSourcePhase = phase;
    const session = document.querySelector('#finder-session-label')?.textContent || '';
    const cancel = document.querySelector('#finder-cancel');
    const form = document.querySelector('#finder-continue');
    const title = document.querySelector('#finder-continue-title')?.textContent || '';
    const source = document.querySelector('#finder-continue-source');
    const pages = document.querySelector('#finder-continue-pages');
    const button = document.querySelector('#finder-continue-button');
    const pagesScanned = document.querySelector('#finder-pages-scanned')?.textContent || '';
    const pagesTotal = document.querySelector('#finder-pages-total')?.textContent || '';
    const candidates = document.querySelector('#finder-candidates-found')?.textContent || '';
    const cardCount = document.querySelectorAll('#finder-result-grid .finder-card').length;
    const firstTitle = document.querySelector('#finder-result-grid .finder-card-title')
      ?.textContent || '';
    document.documentElement.dataset.finderSwitchSourceDebug = [
      session,
      cancel?.hidden,
      form?.hidden,
      title,
      button?.disabled,
      source?.value,
      pagesScanned,
      pagesTotal,
      candidates,
      cardCount,
      firstTitle,
    ].join('|');
    if (
      phase === 'waiting-running'
      && session.includes('JoyTag')
      && session.includes('running')
      && cancel
      && !cancel.hidden
      && !cancel.disabled
      && pagesScanned === '1'
      && pagesTotal === '50'
      && candidates === '124'
      && countsOk()
      && cardCount === 24
      && firstTitle
    ) {
      initialTitle = firstTitle;
      initialCards = cardCount;
      phase = 'cancel-sent';
      cancel.click();
      return;
    }
    if (
      phase === 'cancel-sent'
      && session.includes('cancelled')
      && cancel?.hidden
      && form
      && !form.hidden
      && title === 'Switch source'
      && button?.textContent.includes('Switch source')
      && !button.disabled
      && pagesScanned === '1'
      && pagesTotal === '50'
      && candidates === '124'
      && countsOk()
      && resultsOk()
    ) {
      source.value = nextUrl;
      source.dispatchEvent(new Event('input', { bubbles: true }));
      pages.value = '9';
      pages.dispatchEvent(new Event('input', { bubbles: true }));
      const summary = document.querySelector('#finder-continue-summary')
        ?.textContent || '';
      if (
        !summary.includes('Up to 9 new pages')
        || !summary.includes('keep 124 candidates and all reviews')
      ) return;
      phase = 'switch-sent';
      button.click();
      return;
    }
    if (
      phase === 'switch-sent'
      && session.includes('queued')
      && form?.hidden
      && document.querySelector('#finder-source')?.value === nextUrl
      && document.querySelector('#finder-scan-select')?.value === 'visual-finder'
      && pagesScanned === '1'
      && pagesTotal === '10'
      && candidates === '124'
      && countsOk()
      && resultsOk()
      && !statePending
    ) {
      statePending = true;
      phase = 'checking-state';
      fetch('/manual/finder-switch-source/state')
        .then(response => response.json())
        .then(data => {
          const reviews = data.review_counts || {};
          const passed = (
            data.cancel_calls === 1
            && data.continue_calls === 1
            && data.submitted_source_url === nextUrl
            && data.submitted_additional_pages === 9
            && data.cancelled_next_url === oldCursor
            && data.scan_id === 'visual-finder'
            && data.status === 'queued'
            && data.source_url === nextUrl
            && data.next_url === nextUrl
            && data.page_limit === 10
            && data.pages_completed === 1
            && data.candidate_count === 124
            && reviews.pending === 119
            && reviews.accepted === 3
            && reviews.maybe === 1
            && reviews.rejected === 1
            && reviews.total === 124
          );
          document.documentElement.dataset.finderSwitchSource = passed
            ? 'pass'
            : 'fail';
          phase = passed ? 'complete' : 'state-invalid';
          clearInterval(poll);
        })
        .catch(() => {
          phase = 'state-request-failed';
        });
    }
  }, 60);
  setTimeout(() => {
    if (!document.documentElement.dataset.finderSwitchSource) {
      document.documentElement.dataset.finderSwitchSource = 'fail';
    }
  }, 7500);
});
"""
        if finder_continue:
            script += """window.addEventListener('load',()=>{const failedUrl='https://www.pornpics.com/blocked-source/';const nextUrl='https://www.pornpics.com/new-category/';let phase='waiting';let successAt=0;const countsOk=()=>document.querySelector('#finder-accepted-count')?.textContent==='1'&&document.querySelector('#finder-rejected-count')?.textContent==='1';const cardOk=()=>document.querySelectorAll('#finder-result-grid .finder-card').length===1;const poll=setInterval(()=>{const form=document.querySelector('#finder-continue');const source=document.querySelector('#finder-continue-source');const pages=document.querySelector('#finder-continue-pages');const button=document.querySelector('#finder-continue-button');const toasts=[...document.querySelectorAll('.toast')].map(item=>item.textContent).join(' ');if(phase==='waiting'&&form&&!form.hidden&&source&&pages&&button&&!button.disabled&&countsOk()){document.querySelector('[data-finder-review=accepted]')?.click();pages.value='7';pages.dispatchEvent(new Event('input',{bubbles:true}));if(!document.querySelector('#finder-continue-summary')?.textContent.includes('Up to 7 new pages'))return;source.value=failedUrl;button.click();phase='failure-sent'}else if(phase==='failure-sent'&&form&&!form.hidden&&!button?.disabled&&source?.value===failedUrl&&countsOk()&&cardOk()&&toasts.includes('Could not continue Finder search')){document.documentElement.dataset.finderContinueFailure='pass';document.querySelector('#finder-refresh')?.click();setTimeout(()=>{source.value=nextUrl;button.click();successAt=Date.now();phase='success-sent'},180)}else if(phase==='success-sent'&&Date.now()-successAt>1800){const passed=form?.hidden&&document.querySelector('#finder-session-label')?.textContent.includes('running')&&document.querySelector('#finder-source')?.value===nextUrl&&document.querySelector('#finder-scan-select')?.value==='visual-finder'&&countsOk()&&cardOk();document.documentElement.dataset.finderContinue=passed?'pass':'fail';clearInterval(poll)}},50);setTimeout(()=>{if(!document.documentElement.dataset.finderContinue)document.documentElement.dataset.finderContinue='fail';if(!document.documentElement.dataset.finderContinueFailure)document.documentElement.dataset.finderContinueFailure='fail'},6500)});"""
        if finder_review_mode:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const tab=document.querySelector('[data-finder-review=accepted]');if(tab&&document.querySelector('#finder-accepted-count')?.textContent==='1'){tab.click();requestAnimationFrame(()=>document.querySelector('.finder-open')?.click());clearInterval(poll)}},50)});"
        if finder_pose_mode:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('#finder-feedback-prepare-pose:not([hidden])');const grid=document.querySelector('#image-grid');if(button&&document.querySelector('#gallery-modal')?.open&&grid?.getAttribute('aria-busy')==='false'&&grid.querySelector('.image-option')){button.click();clearInterval(poll)}},50)});"
        if finder_direct_assign:
            script += "window.addEventListener('load',()=>{let phase=0;const poll=setInterval(()=>{const modal=document.querySelector('#gallery-modal');const options=[...document.querySelectorAll('#image-grid .image-option')];if(!modal?.classList.contains('is-pose-mode')||options.length<2||document.querySelector('#pose-save-status')?.textContent.includes('Loading'))return;if(phase===0){const target=options[0].querySelector('input');const control=options[1].querySelector('input');if(target?.checked)target.click();if(control&&!control.checked)control.click();document.querySelector('[data-pose-assignment=couple]')?.click();phase=1}else if(phase===1&&options[1]?.classList.contains('has-pose-control')){const target=options[0].querySelector('input');if(target&&!target.checked)target.click();document.querySelector('[data-pose-assignment=target]')?.click();phase=2}else if(phase===2&&options[0]?.classList.contains('has-pose-target')){document.documentElement.dataset.finderDirect='pass';clearInterval(poll)}},80);setTimeout(()=>{if(!document.documentElement.dataset.finderDirect)document.documentElement.dataset.finderDirect='fail'},5200)});"
        if finder_race:
            script += "window.addEventListener('load',()=>{const poll=setInterval(()=>{const button=document.querySelector('.finder-accept:not(:disabled)');if(button){setTimeout(()=>button.click(),1400);clearInterval(poll)}},50);setTimeout(()=>{const passed=document.querySelector('#finder-accepted-count')?.textContent==='1'&&document.querySelector('#finder-pending-count')?.textContent==='1';document.documentElement.dataset.finderRace=passed?'pass':'fail'},3800)});"
        if finder_unusable_save:
            script += "window.addEventListener('load',()=>{let submitted=false;const poll=setInterval(()=>{const modal=document.querySelector('#gallery-modal');const grid=document.querySelector('#image-grid');const save=document.querySelector('#finder-feedback-gallery-save');const options=[...document.querySelectorAll('#image-grid .image-option input')];if(!modal?.open||!modal.classList.contains('is-feedback-mode')||grid?.getAttribute('aria-busy')!=='false'||options.length<2)return;if(!submitted){for(const input of options){if(input.checked)input.click()}options[1].click();if(!save?.disabled){save.click();submitted=true}}else{const summary=document.querySelector('#selection-summary')?.textContent||'';const toast=[...document.querySelectorAll('.toast')].map(item=>item.textContent).join(' ');const kept=options[1].checked;const passed=kept&&summary.includes('selection saved')&&summary.includes('not currently pose-usable')&&toast.includes('remains selected for review')&&toast.includes('does not currently affect automatic ranking')&&!toast.includes('Could not save review');if(passed){document.documentElement.dataset.finderUnusableSave='pass';clearInterval(poll)}}},80);setTimeout(()=>{if(!document.documentElement.dataset.finderUnusableSave)document.documentElement.dataset.finderUnusableSave='fail'},6200)});"
        return Response(script, media_type="application/javascript")

    async def fake_index() -> Response:
        markup = config.static_dir.joinpath("index.html").read_text(encoding="utf-8")
        markup = markup.replace(
            '<script src="/static/app.js" defer></script>',
            '<script src="/manual-bootstrap.js"></script><script src="/static/app.js" defer></script>',
        )
        return Response(markup, media_type="text/html")

    app.add_api_route("/manual-bootstrap.js", fake_bootstrap, methods=["GET"])

    for route in app.routes:
        if getattr(route, "path", None) == "/":
            route.endpoint = fake_index
            route.dependant.call = fake_index
        elif getattr(route, "path", None) == "/api/media":
            route.endpoint = fake_media
            route.dependant.call = fake_media
        elif getattr(route, "path", None) == "/api/events":
            route.endpoint = fake_events
            route.dependant.call = fake_events
        elif (
            finder_modal_review
            and getattr(route, "path", None) == "/api/pose-exports"
            and "POST" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_modal_pose_export
            route.dependant.call = fake_finder_modal_pose_export
        elif (
            finder_modal_review
            and getattr(route, "path", None) == "/api/pose-exports"
            and "GET" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_modal_pose_exports
            route.dependant.call = fake_finder_modal_pose_exports
        elif (
            finder_modal_review
            and getattr(route, "path", None) == "/api/downloads"
            and "GET" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_modal_downloads
            route.dependant.call = fake_finder_modal_downloads
        elif open_finder and getattr(route, "path", None) == "/api/finder/status":
            route.endpoint = fake_finder_status
            route.dependant.call = fake_finder_status
        elif (
            finder_tag
            and getattr(route, "path", None)
            == "/api/finder/corpus/joytag-index"
            and "POST" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_joytag_index_start
            route.dependant.call = fake_finder_joytag_index_start
        elif (
            finder_tag
            and getattr(route, "path", None)
            == "/api/finder/corpus/joytag-index"
            and "DELETE" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_joytag_index_cancel
            route.dependant.call = fake_finder_joytag_index_cancel
        elif (
            finder_tag
            and getattr(route, "path", None)
            == "/api/finder/corpus/joytag-index"
            and "GET" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_joytag_index_get
            route.dependant.call = fake_finder_joytag_index_get
        elif open_finder and getattr(route, "path", None) == "/api/finder/corpus":
            route.endpoint = fake_finder_corpus
            route.dependant.call = fake_finder_corpus
        elif (
            open_finder
            and getattr(route, "path", None) == "/api/finder/feedback/{pose_tag_id}"
        ):
            route.endpoint = fake_finder_feedback
            route.dependant.call = fake_finder_feedback
        elif open_finder and getattr(route, "path", None) == "/api/finder/folders":
            route.endpoint = fake_finder_folders
            route.dependant.call = fake_finder_folders
        elif open_finder and getattr(route, "path", None) == "/api/pose-tags":
            route.endpoint = fake_pose_tags
            route.dependant.call = fake_pose_tags
        elif (
            finder_tag
            and getattr(route, "path", None) == "/api/finder/reference-analysis"
        ):
            route.endpoint = fake_finder_reference_analysis
            route.dependant.call = fake_finder_reference_analysis
        elif (
            finder_tag
            and getattr(route, "path", None) == "/api/finder/scans"
            and "POST" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_create
            route.dependant.call = fake_finder_create
        elif open_finder and getattr(route, "path", None) == "/api/finder/scans":
            route.endpoint = fake_finder_scans
            route.dependant.call = fake_finder_scans
        elif (
            finder_switch_source
            and getattr(route, "path", None) == "/api/finder/scans/{scan_id}"
            and "DELETE" in getattr(route, "methods", set())
        ):
            route.endpoint = fake_finder_cancel
            route.dependant.call = fake_finder_cancel
        elif (
            open_finder
            and getattr(route, "path", None) == "/api/finder/scans/{scan_id}"
        ):
            route.endpoint = fake_finder_scan
            route.dependant.call = fake_finder_scan
        elif (
            open_finder
            and getattr(route, "path", None) == "/api/finder/scans/{scan_id}/continue"
        ):
            route.endpoint = fake_finder_continue
            route.dependant.call = fake_finder_continue
        elif (
            finder_retry
            and getattr(route, "path", None) == "/api/finder/scans/{scan_id}/retry"
        ):
            route.endpoint = fake_finder_retry
            route.dependant.call = fake_finder_retry
        elif (
            open_finder
            and getattr(route, "path", None) == "/api/finder/scans/{scan_id}/results"
        ):
            route.endpoint = fake_finder_results
            route.dependant.call = fake_finder_results
        elif (
            open_finder
            and getattr(route, "path", None)
            == "/api/finder/scans/{scan_id}/results/{result_id}"
        ):
            route.endpoint = fake_finder_review
            route.dependant.call = fake_finder_review
    if open_finder and not any(
        getattr(route, "path", None) == "/api/finder/feedback/{pose_tag_id}"
        for route in app.routes
    ):
        app.add_api_route(
            "/api/finder/feedback/{pose_tag_id}",
            fake_finder_feedback,
            methods=["GET", "DELETE"],
        )
    if open_finder and not any(
        getattr(route, "path", None) == "/api/finder/scans/{scan_id}/continue"
        for route in app.routes
    ):
        app.add_api_route(
            "/api/finder/scans/{scan_id}/continue",
            fake_finder_continue,
            methods=["POST"],
        )
    if finder_tag and not any(
        getattr(route, "path", None) == "/api/finder/corpus/joytag-index"
        for route in app.routes
    ):
        app.add_api_route(
            "/api/finder/corpus/joytag-index",
            fake_finder_joytag_index_get,
            methods=["GET"],
        )
        app.add_api_route(
            "/api/finder/corpus/joytag-index",
            fake_finder_joytag_index_start,
            methods=["POST"],
        )
        app.add_api_route(
            "/api/finder/corpus/joytag-index",
            fake_finder_joytag_index_cancel,
            methods=["DELETE"],
        )
    if finder_retry:
        app.add_api_route(
            "/manual/finder-retry/state",
            fake_finder_retry_state,
            methods=["GET"],
        )
    if finder_switch_source:
        app.add_api_route(
            "/manual/finder-switch-source/state",
            fake_finder_switch_source_state,
            methods=["GET"],
        )
    if finder_modal_review:
        app.add_api_route(
            "/manual/finder-modal-review/boundary",
            fake_finder_modal_boundary_start,
            methods=["POST"],
        )
        app.add_api_route(
            "/manual/finder-modal-review/insert",
            fake_finder_modal_boundary_insert,
            methods=["POST"],
        )
        app.add_api_route(
            "/manual/finder-modal-review/state",
            fake_finder_modal_boundary_state,
            methods=["GET"],
        )
        app.add_api_route(
            "/manual/finder-modal-review/prefetch-state",
            fake_finder_modal_prefetch_state,
            methods=["GET"],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mobile", action="store_true")
    parser.add_argument("--sort", action="store_true")
    parser.add_argument("--load-more", action="store_true")
    parser.add_argument("--gallery", action="store_true")
    parser.add_argument("--lightbox", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--finder", action="store_true")
    parser.add_argument("--finder-feedback", action="store_true")
    parser.add_argument("--finder-pose-flow", action="store_true")
    parser.add_argument("--finder-direct-assign", action="store_true")
    parser.add_argument("--finder-race", action="store_true")
    parser.add_argument("--finder-unusable-save", action="store_true")
    parser.add_argument("--finder-exhausted", action="store_true")
    parser.add_argument("--finder-continue", action="store_true")
    parser.add_argument("--finder-switch-source", action="store_true")
    parser.add_argument("--finder-pagination", action="store_true")
    parser.add_argument("--finder-tag", action="store_true")
    parser.add_argument("--finder-retry", action="store_true")
    parser.add_argument("--finder-modal-review", action="store_true")
    args = parser.parse_args()
    finder_mode = (
        args.finder
        or args.finder_feedback
        or args.finder_pose_flow
        or args.finder_direct_assign
        or args.finder_race
        or args.finder_unusable_save
        or args.finder_exhausted
        or args.finder_continue
        or args.finder_switch_source
        or args.finder_pagination
        or args.finder_tag
        or args.finder_retry
        or args.finder_modal_review
    )
    suffix = (
        "finder-modal-review-mobile"
        if args.finder_modal_review and args.mobile
        else "finder-modal-review"
        if args.finder_modal_review
        else "finder-tag-mobile"
        if args.finder_tag and args.mobile
        else "finder-tag"
        if args.finder_tag
        else "finder-retry-mobile"
        if args.finder_retry and args.mobile
        else "finder-retry"
        if args.finder_retry
        else "finder-switch-source-mobile"
        if args.finder_switch_source and args.mobile
        else "finder-switch-source"
        if args.finder_switch_source
        else "finder-pagination"
        if args.finder_pagination
        else "finder-continue-mobile"
        if args.finder_continue and args.mobile
        else "finder-continue"
        if args.finder_continue
        else "finder-unusable-save"
        if args.finder_unusable_save
        else "finder-direct-assign"
        if args.finder_direct_assign
        else "finder-race"
        if args.finder_race
        else "finder-pose-flow-mobile"
        if args.finder_pose_flow and args.mobile
        else "finder-pose-flow"
        if args.finder_pose_flow
        else "finder-feedback-mobile"
        if args.finder_feedback and args.mobile
        else "finder-feedback"
        if args.finder_feedback
        else "finder-exhausted-mobile"
        if args.finder_exhausted and args.mobile
        else "finder-exhausted"
        if args.finder_exhausted
        else "finder-mobile"
        if finder_mode and args.mobile
        else "finder"
        if finder_mode
        else "pose-mobile"
        if args.pose and args.mobile
        else "pose"
        if args.pose
        else "lightbox-mobile"
        if args.lightbox and args.mobile
        else "lightbox"
        if args.lightbox
        else "gallery"
        if args.gallery
        else "load-more"
        if args.load_more
        else "sort-mobile"
        if args.sort and args.mobile
        else "sort"
        if args.sort
        else "mobile"
        if args.mobile
        else "smoke"
    )
    output = Path(f"/tmp/pornpic-webui-{suffix}.png")
    viewport = (
        "390,1800"
        if args.mobile and finder_mode
        else "390,844"
        if args.mobile
        else "1440,1350"
        if args.finder_tag
        else "1920,969"
        if args.gallery or args.lightbox or args.pose
        else "1440,1100"
    )
    with tempfile.TemporaryDirectory(prefix="pornpic-webui-") as directory:
        server = uvicorn.Server(
            uvicorn.Config(
                build_visual_app(
                    Path(directory),
                    load_more=args.load_more,
                    open_gallery=args.gallery or args.lightbox or args.pose,
                    open_lightbox=args.lightbox,
                    open_pose=args.pose,
                    open_finder=finder_mode,
                    open_finder_feedback=args.finder_feedback,
                    open_finder_pose_flow=args.finder_pose_flow,
                    finder_direct_assign=args.finder_direct_assign,
                    finder_race=args.finder_race,
                    finder_unusable_save=args.finder_unusable_save,
                    finder_exhausted=args.finder_exhausted,
                    finder_continue=args.finder_continue,
                    finder_switch_source=args.finder_switch_source,
                    finder_pagination=args.finder_pagination,
                    finder_tag=args.finder_tag,
                    finder_retry=args.finder_retry,
                    finder_modal_review=args.finder_modal_review,
                ),
                host="127.0.0.1",
                port=18101,
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:18101/api/health", timeout=0.2)
                break
            except Exception:
                time.sleep(0.05)
        browser = shutil.which("google-chrome-stable")
        if not browser:
            raise RuntimeError(
                "google-chrome-stable is required for this manual smoke check"
            )
        command = [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-sync",
            "--force-prefers-reduced-motion",
            "--no-first-run",
            f"--virtual-time-budget={18000 if args.finder_modal_review else 8500 if args.finder_tag or args.finder_switch_source else 7500 if args.finder_unusable_save or args.finder_continue else 6500 if args.finder_direct_assign else 6000 if args.finder_retry else 4500 if args.finder_race else 4000 if args.finder_pose_flow else 3000 if args.lightbox or args.pose else 2000 if finder_mode else 1000}",
            f"--user-data-dir={Path(directory) / 'chrome-profile'}",
            f"--window-size={viewport}",
            f"--screenshot={output}",
            f"http://127.0.0.1:18101/{'#finder' if finder_mode else '#sort' if args.sort else ''}",
        ]
        if (
            args.finder_race
            or args.finder_direct_assign
            or args.finder_unusable_save
            or args.finder_continue
            or args.finder_switch_source
            or args.finder_pagination
            or args.finder_tag
            or args.finder_retry
            or args.finder_modal_review
        ):
            command.insert(1, "--dump-dom")
        completed = subprocess.run(
            command,
            check=True,
            timeout=45,
            capture_output=(
                args.finder_race
                or args.finder_direct_assign
                or args.finder_unusable_save
                or args.finder_continue
                or args.finder_switch_source
                or args.finder_pagination
                or args.finder_tag
                or args.finder_retry
                or args.finder_modal_review
            ),
            text=(
                args.finder_race
                or args.finder_direct_assign
                or args.finder_unusable_save
                or args.finder_continue
                or args.finder_switch_source
                or args.finder_pagination
                or args.finder_tag
                or args.finder_retry
                or args.finder_modal_review
            ),
        )
        if args.finder_race and 'data-finder-race="pass"' not in completed.stdout:
            raise AssertionError("stale Finder results overwrote the completed review")
        if (
            args.finder_direct_assign
            and 'data-finder-direct="pass"' not in completed.stdout
        ):
            raise AssertionError("direct gallery-grid pose assignment did not complete")
        if (
            args.finder_unusable_save
            and 'data-finder-unusable-save="pass"' not in completed.stdout
        ):
            raise AssertionError(
                "pose-unusable Finder selection was not retained as saved feedback"
            )
        if args.finder_continue:
            if 'data-finder-continue-failure="pass"' not in completed.stdout:
                raise AssertionError(
                    "failed Finder continuation did not retain its URL and results"
                )
            if 'data-finder-continue="pass"' not in completed.stdout:
                raise AssertionError(
                    "continued Finder scan lost its new source, results, or review counts"
                )
        if (
            args.finder_switch_source
            and 'data-finder-switch-source="pass"' not in completed.stdout
        ):
            phase = completed.stdout.partition(
                'data-finder-switch-source-phase="'
            )[2].partition('"')[0]
            debug = completed.stdout.partition(
                'data-finder-switch-source-debug="'
            )[2].partition('"')[0]
            raise AssertionError(
                "cancelled JoyTag Finder scan did not switch source in place "
                "while retaining its cursor, candidates, and reviews "
                f"(phase={phase or 'unknown'}, debug={debug or 'none'})"
            )
        if (
            args.finder_pagination
            and 'data-finder-pagination="pass"' not in completed.stdout
        ):
            raise AssertionError(
                "Finder pagination did not load page two with global result ranks"
            )
        if args.finder_tag and 'data-finder-tag="pass"' not in completed.stdout:
            phase = completed.stdout.partition(
                'data-finder-tag-phase="'
            )[2].partition('"')[0]
            debug = completed.stdout.partition(
                'data-finder-tag-debug="'
            )[2].partition('"')[0]
            raise AssertionError(
                "Tag Finder corpus indexing, analysis, scan payload, or result "
                f"rendering failed (phase={phase or 'unknown'}, debug={debug or 'none'})"
            )
        if args.finder_retry and 'data-finder-retry="pass"' not in completed.stdout:
            phase = completed.stdout.partition(
                'data-finder-retry-phase="'
            )[2].partition('"')[0]
            debug = completed.stdout.partition(
                'data-finder-retry-debug="'
            )[2].partition('"')[0]
            raise AssertionError(
                "failed Finder retry did not preserve its scan, cursor, local "
                f"candidates, or UI state (phase={phase or 'unknown'}, debug={debug or 'none'})"
            )
        if (
            args.finder_modal_review
            and 'data-finder-modal-review="pass"' not in completed.stdout
        ):
            phase = completed.stdout.partition(
                'data-finder-modal-review-phase="'
            )[2].partition('"')[0]
            debug = completed.stdout.partition(
                'data-finder-modal-review-debug="'
            )[2].partition('"')[0]
            raise AssertionError(
                "in-modal Finder decisions, result navigation, selected-image "
                "feedback, or pose-draft persistence failed "
                f"(phase={phase or 'unknown'}, debug={debug or 'none'})"
            )
        server.should_exit = True
        thread.join(timeout=5)
    print(output)


if __name__ == "__main__":
    main()
