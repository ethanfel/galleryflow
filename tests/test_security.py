from __future__ import annotations

from pathlib import Path

import pytest

from app.security import (
    UnsafeUrl,
    clean_profile_name,
    confined_path,
    decode_gallery_id,
    encode_gallery_id,
    gallery_key,
    validate_public_media_url,
    validate_source_url,
)


def test_gallery_identity_survives_slug_and_url_variants() -> None:
    variants = [
        "http://www.pornpics.com/galleries/old-title-79186222/?from=search#top",
        "https://pornpics.com/galleries/new-title-79186222/",
        "https://www.pornpics.com/galleries/79186222/",
    ]
    assert {gallery_key(url) for url in variants} == {"pornpics:gallery:79186222"}


def test_gallery_id_round_trip() -> None:
    url = "https://www.pornpics.com/galleries/example-12345678/"
    assert decode_gallery_id(encode_gallery_id(url)) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/galleries/x-12345/",
        "https://evil.example/galleries/x-12345/",
        "file:///etc/passwd",
        "https://user:pass@www.pornpics.com/galleries/x-12345/",
        "https://www.pornpics.com:8443/galleries/x-12345/",
    ],
)
def test_source_url_boundary(url: str) -> None:
    with pytest.raises((UnsafeUrl, ValueError)):
        validate_source_url(url)


def test_media_host_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.security.socket.getaddrinfo",
        lambda *_: [(None, None, None, None, ("93.184.216.34", 0))],
    )
    assert validate_public_media_url("https://cdni.pornpics.com/1280/a.jpg").startswith(
        "https://"
    )
    with pytest.raises(UnsafeUrl):
        validate_public_media_url("https://www.pornpics.com/a.jpg")


def test_profile_and_path_validation(tmp_path: Path) -> None:
    assert clean_profile_name("POV collection") == "POV collection"
    assert confined_path(tmp_path, "profile", "gallery").is_relative_to(tmp_path)
    with pytest.raises(ValueError):
        confined_path(tmp_path, "..", "escape")
