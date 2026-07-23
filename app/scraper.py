from __future__ import annotations

import html as html_module
import hashlib
import json
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, quote_plus, urlencode, urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup, Tag

from .config import AppConfig
from .security import (
    canonicalize_url,
    encode_gallery_id,
    gallery_key,
    validate_source_url,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")


class ScrapeError(RuntimeError):
    pass


@dataclass(slots=True)
class ScrapedPage:
    url: str
    html: str


class PornPicsScraper:
    def __init__(self, app_config: AppConfig):
        self.config = app_config

    async def _get_html(self, url: str) -> ScrapedPage:
        current = validate_source_url(url)
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
        }
        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            for _ in range(6):
                try:
                    response = await client.get(
                        current, headers=headers, follow_redirects=False
                    )
                except httpx.HTTPError as exc:
                    raise ScrapeError(f"Could not reach PornPics: {exc}") from exc
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ScrapeError("Source returned an empty redirect")
                    current = validate_source_url(urljoin(current, location))
                    continue
                if response.status_code >= 400:
                    raise ScrapeError(f"PornPics returned HTTP {response.status_code}")
                if len(response.content) > 12 * 1024 * 1024:
                    raise ScrapeError("Source page exceeds the 12 MB safety limit")
                content_type = response.headers.get("content-type", "")
                leading = response.text.lstrip()[:1]
                if (
                    "html" not in content_type.lower()
                    and "json" not in content_type.lower()
                    and leading not in {"<", "[", "{"}
                ):
                    raise ScrapeError("Source did not return HTML or JSON")
                return ScrapedPage(str(response.url), response.text)
        raise ScrapeError("Too many redirects from source")

    @staticmethod
    def _src_from_image(image: Tag | None, base_url: str) -> str | None:
        if not image:
            return None
        srcset = image.get("data-srcset") or image.get("srcset")
        if isinstance(srcset, str) and srcset.strip():
            candidates = [
                part.strip().split()[0] for part in srcset.split(",") if part.strip()
            ]
            if candidates:
                return urljoin(base_url, html_module.unescape(candidates[-1]))
        for attr in ("data-original", "data-src", "data-lazy-src", "data-image", "src"):
            value = image.get(attr)
            if isinstance(value, str) and value and not value.startswith("data:"):
                return urljoin(base_url, html_module.unescape(value))
        return None

    @staticmethod
    def _approved_media_url(value: str | None, base_url: str) -> str | None:
        if not value:
            return None
        try:
            url = canonicalize_url(urljoin(base_url, html_module.unescape(value)))
        except ValueError:
            return None
        parts = urlsplit(url)
        if parts.hostname != "cdni.pornpics.com" or parts.port not in {None, 443}:
            return None
        return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))

    @staticmethod
    def _clean_title(value: str | None) -> str:
        if not value:
            return "Untitled gallery"
        value = re.sub(r"\s+", " ", html_module.unescape(value)).strip()
        value = re.sub(r"\s*[-–|]\s*PornPics(?:\.com)?\s*$", "", value, flags=re.I)
        return value[:300] or "Untitled gallery"

    @staticmethod
    def _navigation_url(
        soup: BeautifulSoup, base_url: str, direction: str
    ) -> str | None:
        node = soup.find("link", rel=lambda rel: rel and direction in rel)
        if not node:
            selectors = (
                f"a[rel='{direction}']",
                f"a.{direction}",
                f".pagination a.{direction}",
                f"a[aria-label*='{direction.title()}']",
            )
            for selector in selectors:
                node = soup.select_one(selector)
                if node:
                    break
        href = node.get("href") if isinstance(node, Tag) else None
        if not isinstance(href, str):
            return None
        try:
            return validate_source_url(urljoin(base_url, href))
        except ValueError:
            return None

    @staticmethod
    def _infinite_scroll_url(
        soup: BeautifulSoup, base_url: str, item_count: int
    ) -> str | None:
        """Build the cursor used by PornPics category/tag infinite scrolling.

        Those pages render their first 20 galleries in HTML and fetch later
        batches from the same path as JSON using ``offset`` and ``limit``.
        They do not expose a conventional ``rel=next`` link.
        """
        if item_count <= 0:
            return None
        scripts = "\n".join(
            script.get_text(" ", strip=False) for script in soup.select("script")
        )
        page_type = re.search(
            r"\bPP_PAGE_TYPE\s*=\s*['\"]"
            r"(category_rotator_maps|tag_rotator_maps)['\"]",
            scripts,
        )
        if not page_type:
            return None
        parts = urlsplit(base_url)
        # PornPics' category strategy starts its JSON cursor at 20 even when
        # one rendered card is filtered locally (for example, an ad or a
        # thumbnail from an unapproved media host). It also deliberately drops
        # the original page query and sends only its cursor parameters.
        params = {"offset": "20", "limit": "20"}
        try:
            return validate_source_url(
                urlunsplit(
                    (parts.scheme, parts.netloc, parts.path, urlencode(params), "")
                )
            )
        except ValueError:
            return None

    async def browse(
        self, *, url: str | None = None, query: str | None = None, page: int = 1
    ) -> dict:
        if url:
            source_url = validate_source_url(url)
            if "/galleries/" in urlsplit(source_url).path.lower():
                detail = await self.gallery(source_url)
                first_image = detail["images"][0]
                return {
                    "items": [
                        {
                            "id": detail["id"],
                            "key": detail["key"],
                            "url": detail["url"],
                            "title": detail["title"],
                            "thumbnail_remote_url": first_image["preview_remote_url"],
                            "image_count": len(detail["images"]),
                        }
                    ],
                    "source_url": detail["url"],
                    "next_url": None,
                    "previous_url": None,
                }
        elif query:
            offset = (page - 1) * 20
            source_url = (
                f"{self.config.source_home.rstrip('/')}/search/srch.php?"
                f"q={quote_plus(query)}&lang=en&offset={offset}&limit=20"
            )
        else:
            source_url = self.config.source_home
            if page > 1:
                source_url = f"{source_url.rstrip('/')}/page/{page}/"

        fetched = await self._get_html(source_url)
        stripped = fetched.html.lstrip()
        if stripped.startswith("["):
            try:
                payload = json.loads(fetched.html)
            except json.JSONDecodeError as exc:
                raise ScrapeError("PornPics returned invalid browse JSON") from exc
            if not isinstance(payload, list):
                raise ScrapeError("Unexpected browse response")
            items: list[dict] = []
            seen_keys: set[str] = set()
            for raw in payload:
                if not isinstance(raw, dict):
                    continue
                href = raw.get("g_url")
                thumbnail = raw.get("t_url_460") or raw.get("t_url")
                if not isinstance(href, str) or not isinstance(thumbnail, str):
                    continue
                try:
                    gallery_url = validate_source_url(urljoin(fetched.url, href))
                    key = gallery_key(gallery_url)
                except ValueError:
                    continue
                media = self._approved_media_url(thumbnail, fetched.url)
                if not media or key in seen_keys:
                    continue
                seen_keys.add(key)
                items.append(
                    {
                        "id": encode_gallery_id(gallery_url),
                        "key": key,
                        "url": gallery_url,
                        "title": self._clean_title(str(raw.get("desc") or "")),
                        "thumbnail_remote_url": media,
                        "image_count": None,
                    }
                )
            parts = urlsplit(fetched.url)
            params = parse_qs(parts.query)
            offset = int(params.get("offset", [0])[0] or 0)
            limit = int(params.get("limit", [20])[0] or 20)

            def with_offset(value: int) -> str:
                updated = {key: values[-1] for key, values in params.items()}
                updated["offset"] = str(max(0, value))
                updated["limit"] = str(limit)
                return urlunsplit(
                    (parts.scheme, parts.netloc, parts.path, urlencode(updated), "")
                )

            return {
                "items": items,
                "source_url": fetched.url,
                # The site's rotator advances fixed-size windows and declares
                # exhaustion only when the endpoint returns an empty array.
                "next_url": with_offset(offset + limit) if payload else None,
                "previous_url": with_offset(offset - limit) if offset > 0 else None,
            }

        soup = BeautifulSoup(fetched.html, "html.parser")
        items: list[dict] = []
        seen: set[str] = set()

        selectors = (
            "#tiles > li.thumbwook:not(.r2-frame) > a.rel-link[href*='/galleries/']",
            "#tiles a.rel-link[href*='/galleries/']",
            "a[href*='/galleries/']",
        )
        anchors: list[Tag] = []
        for selector in selectors:
            anchors = list(soup.select(selector))
            if anchors:
                break
        for anchor in anchors:
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            try:
                gallery_url = validate_source_url(urljoin(fetched.url, href))
            except ValueError:
                continue
            key = gallery_key(gallery_url)
            if key in seen:
                continue
            image = anchor.find("img")
            thumbnail = self._src_from_image(image, fetched.url)
            title = None
            if image:
                title = image.get("alt") or image.get("title")
            title = title or anchor.get("title")
            if not title:
                caption = anchor.select_one(".title, .name, figcaption")
                title = (
                    caption.get_text(" ", strip=True)
                    if caption
                    else anchor.get_text(" ", strip=True)
                )
            if not thumbnail:
                container = anchor.find_parent(["li", "article", "figure", "div"])
                thumbnail = self._src_from_image(
                    container.find("img") if container else None, fetched.url
                )
            thumbnail = self._approved_media_url(thumbnail, fetched.url)
            if not thumbnail:
                continue
            seen.add(key)
            items.append(
                {
                    "id": encode_gallery_id(gallery_url),
                    "key": key,
                    "url": gallery_url,
                    "title": self._clean_title(str(title) if title else None),
                    "thumbnail_remote_url": thumbnail,
                    "image_count": None,
                }
            )

        if not items:
            # JSON-LD fallback for layout changes.
            for script in soup.select("script[type='application/ld+json']"):
                try:
                    data = json.loads(script.string or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                entries = (
                    data.get("itemListElement", []) if isinstance(data, dict) else []
                )
                for entry in entries:
                    raw = entry.get("item", entry) if isinstance(entry, dict) else {}
                    if not isinstance(raw, dict) or not raw.get("url"):
                        continue
                    try:
                        gallery_url = validate_source_url(
                            urljoin(fetched.url, raw["url"])
                        )
                    except ValueError:
                        continue
                    thumbnail = raw.get("image") or raw.get("thumbnailUrl")
                    key = gallery_key(gallery_url)
                    thumbnail = self._approved_media_url(
                        thumbnail if isinstance(thumbnail, str) else None, fetched.url
                    )
                    if key in seen or not thumbnail:
                        continue
                    seen.add(key)
                    items.append(
                        {
                            "id": encode_gallery_id(gallery_url),
                            "key": key,
                            "url": gallery_url,
                            "title": self._clean_title(raw.get("name")),
                            "thumbnail_remote_url": thumbnail,
                            "image_count": None,
                        }
                    )

        next_url = self._navigation_url(soup, fetched.url, "next")
        if not next_url:
            next_url = self._infinite_scroll_url(soup, fetched.url, len(items))

        return {
            "items": items,
            "source_url": fetched.url,
            "next_url": next_url,
            "previous_url": self._navigation_url(soup, fetched.url, "prev"),
        }

    @staticmethod
    def _looks_like_image(url: str) -> bool:
        path = urlsplit(url).path.lower()
        return path.endswith(IMAGE_EXTENSIONS)

    async def gallery(self, url: str) -> dict:
        gallery_url = validate_source_url(url)
        fetched = await self._get_html(gallery_url)
        gallery_url = validate_source_url(fetched.url)
        soup = BeautifulSoup(fetched.html, "html.parser")
        title_node = soup.select_one("h1")
        og_title = soup.select_one("meta[property='og:title']")
        title = (
            title_node.get_text(" ", strip=True)
            if title_node
            else (
                og_title.get("content")
                if og_title
                else (soup.title.string if soup.title else None)
            )
        )

        images: list[dict] = []
        seen: set[str] = set()
        containers = soup.select("#tiles > li.thumbwook, #tiles li")
        scan_roots: list[Tag | BeautifulSoup] = containers or [
            soup.select_one("#tiles") or soup
        ]
        candidates: list[tuple[str, str | None]] = []
        for root in scan_roots:
            found_link = False
            for anchor in root.select("a[href]"):
                href = anchor.get("href")
                if not isinstance(href, str):
                    continue
                absolute = self._approved_media_url(href, fetched.url)
                preview = self._src_from_image(anchor.find("img"), fetched.url)
                preview = self._approved_media_url(preview, fetched.url)
                if absolute and self._looks_like_image(absolute):
                    candidates.append((absolute, preview))
                    found_link = True
            if not found_link:
                for image in root.select("img"):
                    source = self._src_from_image(image, fetched.url)
                    source = self._approved_media_url(source, fetched.url)
                    if source and self._looks_like_image(source):
                        candidates.append((source, source))

        for image_url, preview_url in candidates:
            try:
                image_url = canonicalize_url(image_url)
                preview_url = canonicalize_url(preview_url or image_url)
            except ValueError:
                continue
            if image_url in seen:
                continue
            seen.add(image_url)
            suffix = (
                urlsplit(image_url).path.rsplit("/", 1)[-1]
                or f"image-{len(images) + 1}.jpg"
            )
            images.append(
                {
                    "id": hashlib.sha256(image_url.encode()).hexdigest()[:20],
                    "ordinal": len(images) + 1,
                    "url": image_url,
                    "preview_remote_url": preview_url,
                    "filename": suffix[:180],
                }
            )

        if not images:
            for meta in soup.select(
                "meta[property='og:image'], meta[name='twitter:image']"
            ):
                content = meta.get("content")
                if isinstance(content, str):
                    absolute = self._approved_media_url(content, fetched.url)
                    if not absolute:
                        continue
                    if absolute not in seen:
                        seen.add(absolute)
                        images.append(
                            {
                                "id": hashlib.sha256(absolute.encode()).hexdigest()[
                                    :20
                                ],
                                "ordinal": 1,
                                "url": absolute,
                                "preview_remote_url": absolute,
                                "filename": "0001.jpg",
                            }
                        )

        if not images:
            raise ScrapeError("No downloadable images were found in this gallery")

        return {
            "id": encode_gallery_id(gallery_url),
            "key": gallery_key(gallery_url),
            "url": gallery_url,
            "title": self._clean_title(str(title) if title else None),
            "images": images,
        }
