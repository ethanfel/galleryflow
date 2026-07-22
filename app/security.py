from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


PROFILE_RE = re.compile(r"[^A-Za-z0-9 _.-]+")
FOLDER_RE = re.compile(r"[^\w .-]+", flags=re.UNICODE)


class UnsafeUrl(ValueError):
    pass


def canonicalize_url(value: str) -> str:
    value = (value or "").strip()
    if not value:
        raise UnsafeUrl("URL is required")
    if "://" not in value:
        value = "https://" + value
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"}:
        raise UnsafeUrl("Only HTTP and HTTPS URLs are supported")
    if not parts.hostname or parts.username or parts.password:
        raise UnsafeUrl("Invalid URL")
    host = parts.hostname.lower().rstrip(".")
    port = f":{parts.port}" if parts.port else ""
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), host + port, path, parts.query, ""))


def validate_source_url(value: str) -> str:
    url = canonicalize_url(value)
    host = (urlsplit(url).hostname or "").lower()
    if host not in {"pornpics.com", "www.pornpics.com"}:
        raise UnsafeUrl("Browsing is restricted to pornpics.com")
    parts = urlsplit(url)
    if parts.port not in {None, 443}:
        raise UnsafeUrl("PornPics URLs may not use a custom port")
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))


def _is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_media_url(value: str) -> str:
    url = canonicalize_url(value)
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host != "cdni.pornpics.com":
        raise UnsafeUrl("Media URL is not from the approved PornPics image host")
    if parts.port not in {None, 443}:
        raise UnsafeUrl("Media URLs may not use a custom port")
    try:
        if not all(
            _is_public_ip(item[4][0]) for item in socket.getaddrinfo(host, None)
        ):
            raise UnsafeUrl("Media host resolves to a non-public address")
    except socket.gaierror as exc:
        raise UnsafeUrl("Media host could not be resolved") from exc
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))


def gallery_key(value: str) -> str:
    """Return an identity stable across PornPics slug redirects and URL variants."""
    url = validate_source_url(value)
    path = urlsplit(url).path
    match = re.search(r"/galleries/(?:[^/?]*-)?(\d{5,})/?$", path, flags=re.I)
    if match:
        return f"pornpics:gallery:{match.group(1)}"
    normalized_path = "/" + "/".join(part for part in path.split("/") if part)
    if normalized_path != "/":
        normalized_path += "/"
    return urlunsplit(("https", "pornpics.com", normalized_path, "", ""))


def encode_gallery_id(url: str) -> str:
    raw = canonicalize_url(url).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_gallery_id(value: str) -> str:
    try:
        padding = "=" * (-len(value) % 4)
        return canonicalize_url(
            base64.urlsafe_b64decode(value + padding).decode("utf-8")
        )
    except Exception as exc:
        raise ValueError("Invalid gallery id") from exc


def sign_media_url(url: str, key: str) -> str:
    return hmac.new(key.encode(), url.encode(), hashlib.sha256).hexdigest()


def verify_media_signature(url: str, token: str, key: str) -> bool:
    return hmac.compare_digest(sign_media_url(url, key), token)


def clean_profile_name(value: str) -> str:
    raw = (value or "").strip()
    if PROFILE_RE.search(raw):
        raise ValueError(
            "Profile names may only contain letters, numbers, spaces, dots, dashes, and underscores"
        )
    name = raw.strip(" .")
    if not name or name in {".", ".."}:
        raise ValueError("Profile name must contain letters or numbers")
    return name[:64]


def safe_folder_name(value: str, fallback: str = "Untitled_Gallery") -> str:
    cleaned = FOLDER_RE.sub("", value or "").strip(" .")
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned or fallback)[:120]


def confined_path(root: Path, *parts: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path escapes the configured download directory")
    return candidate
