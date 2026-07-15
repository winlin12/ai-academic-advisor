"""Disk-based page cache to avoid re-fetching unchanged pages."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CachedPage:
    url: str
    html: str
    http_status: int
    fetched_at: datetime
    content_hash: str


def _url_to_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


class DiskCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.meta.json"

    def _html_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.html"

    def get(self, url: str) -> CachedPage | None:
        key = _url_to_key(url)
        meta_path = self._meta_path(key)
        html_path = self._html_path(key)
        if not meta_path.exists() or not html_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            html = html_path.read_text(encoding="utf-8")
            return CachedPage(
                url=meta["url"],
                html=html,
                http_status=meta["http_status"],
                fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                content_hash=meta["content_hash"],
            )
        except Exception as exc:
            logger.warning("Cache read error for %s: %s", url, exc)
            return None

    def put(self, page: CachedPage) -> None:
        key = _url_to_key(page.url)
        try:
            self._html_path(key).write_text(page.html, encoding="utf-8")
            self._meta_path(key).write_text(
                json.dumps(
                    {
                        "url": page.url,
                        "http_status": page.http_status,
                        "fetched_at": page.fetched_at.isoformat(),
                        "content_hash": page.content_hash,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            logger.warning("Cache write error for %s: %s", page.url, exc)

    def has(self, url: str) -> bool:
        key = _url_to_key(url)
        return self._meta_path(key).exists() and self._html_path(key).exists()

    def invalidate(self, url: str) -> None:
        key = _url_to_key(url)
        for path in (self._meta_path(key), self._html_path(key)):
            if path.exists():
                path.unlink()

    def stats(self) -> dict[str, int]:
        html_files = list(self.cache_dir.glob("*.html"))
        return {"cached_pages": len(html_files)}


def content_hash(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8", errors="replace")).hexdigest()
