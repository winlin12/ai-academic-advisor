"""Extract navigation links from catalog pages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlparse


@dataclass(frozen=True)
class NavLink:
    text: str
    url: str
    catoid: int | None
    navoid: int | None


ANCHOR_RE = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    text = TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _query_int(url: str, key: str) -> int | None:
    vals = parse_qs(urlparse(url).query).get(key)
    if vals:
        try:
            return int(vals[0])
        except ValueError:
            pass
    return None


def extract_nav_links(html: str, base_url: str) -> list[NavLink]:
    links: list[NavLink] = []
    seen: set[str] = set()
    for match in ANCHOR_RE.finditer(html):
        href = match.group(1).strip()
        text = _clean(match.group(2))
        if not href or href.startswith(("#", "javascript:", "mailto:")):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        # Only keep same-host catalog links
        base_parsed = urlparse(base_url)
        if parsed.netloc != base_parsed.netloc:
            continue
        norm = parsed._replace(fragment="").geturl()
        if norm in seen:
            continue
        seen.add(norm)
        links.append(
            NavLink(
                text=text,
                url=norm,
                catoid=_query_int(norm, "catoid"),
                navoid=_query_int(norm, "navoid"),
            )
        )
    return links


def filter_program_links(links: list[NavLink]) -> list[NavLink]:
    return [l for l in links if "preview_program.php" in l.url]


def filter_course_links(links: list[NavLink]) -> list[NavLink]:
    return [l for l in links if "preview_course_nopop.php" in l.url or "coid=" in l.url]


def filter_content_links(links: list[NavLink], catoid: int) -> list[NavLink]:
    return [
        l for l in links
        if "content.php" in l.url and l.catoid == catoid
    ]
