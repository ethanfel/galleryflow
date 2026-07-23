from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class ProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)


class ProfilePatch(BaseModel):
    new_name: str = Field(min_length=1, max_length=64)


class GalleryPatch(BaseModel):
    ignored: bool
    title: str | None = Field(default=None, max_length=300)


class DownloadCreate(BaseModel):
    gallery_id: str | None = None
    gallery_url: str | None = None
    title: str | None = Field(default=None, max_length=300)
    profile: str = Field(default="Default", min_length=1, max_length=64)
    image_urls: list[str] | None = None

    @field_validator("image_urls")
    @classmethod
    def limit_images(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and len(value) > 2_000:
            raise ValueError("A job may contain at most 2,000 selected images")
        return value


PoseRole = Literal["solo", "couple", "group"]


class PoseTagCreate(BaseModel):
    label: str = Field(min_length=1, max_length=80)
    default_role: PoseRole = "solo"


class PoseTagPatch(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=80)
    default_role: PoseRole | None = None


class PoseControls(BaseModel):
    solo: str | None = Field(default=None, max_length=2_000)
    couple: str | None = Field(default=None, max_length=2_000)
    group: str | None = Field(default=None, max_length=2_000)


class PoseTarget(BaseModel):
    image_url: str = Field(min_length=1, max_length=2_000)
    pose_tag_id: int = Field(gt=0)
    role: PoseRole


class PoseDraftPut(BaseModel):
    expected_revision: int = Field(ge=0)
    controls: PoseControls = Field(default_factory=PoseControls)
    targets: list[PoseTarget] = Field(default_factory=list, max_length=2_000)


class PoseExportCreate(BaseModel):
    gallery_id: str = Field(min_length=1, max_length=4_000)
    profile: str = Field(default="Default", min_length=1, max_length=64)
    expected_revision: int = Field(ge=1)


class FinderScanCreate(BaseModel):
    example_directory: str = Field(min_length=1, max_length=500)
    pose_tag_id: int = Field(gt=0)
    source_url: str = Field(min_length=1, max_length=2_000)
    page_limit: int = Field(default=5, ge=1, le=50)
    minimum_score: float = Field(default=0.7, ge=0, le=1)


class FinderScanExtend(BaseModel):
    additional_pages: int = Field(ge=1, le=50, strict=True)


class FinderScanContinue(BaseModel):
    source_url: str = Field(min_length=1, max_length=2_000)
    additional_pages: int = Field(ge=1, le=50, strict=True)


class FinderReviewPatch(BaseModel):
    review: Literal["pending", "maybe", "accepted", "rejected"]
    feedback_image_urls: (
        list[Annotated[str, Field(min_length=1, max_length=2_000)]] | None
    ) = Field(default=None, max_length=3)


class SettingsPatch(BaseModel):
    request_timeout: float | None = Field(default=None, ge=5, le=120)
    image_workers: int | None = Field(default=None, ge=1, le=24)
    job_workers: int | None = Field(default=None, ge=1, le=8)
    theme: Literal["dark", "light", "system"] | None = None


class SortSessionCreate(BaseModel):
    target_directory: str = Field(min_length=1, max_length=500)
    control_directories: list[str] = Field(default_factory=list, max_length=1_000)
    mode: Literal["time", "stem"] = "time"
    threshold_seconds: float = Field(default=50, ge=0, le=3_600)
    add_ids: bool = True


class SortProfileCreate(SortSessionCreate):
    name: str = Field(min_length=1, max_length=64)


class SortActionCreate(BaseModel):
    kind: Literal["match", "solo", "no_control", "skip"]
    expected_target: str = Field(min_length=1, max_length=1_000)
    control_path: str | None = Field(default=None, max_length=1_000)


class LegacyIgnoreRequest(BaseModel):
    url: str


class LegacyDownloadRequest(BaseModel):
    folder_name: str | None = None
    urls: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    origin_url: str
    profile: str = "Default"
