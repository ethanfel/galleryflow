from __future__ import annotations

from typing import Literal

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
