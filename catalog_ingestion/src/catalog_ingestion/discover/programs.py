"""Discover program page URLs from the catalog.

Program pages follow the pattern:
  /preview_program.php?catoid=N&poid=XXXXX

They are linked from the "Graduate and Undergraduate Programs Lists" page
and from individual college/department pages.
"""

from __future__ import annotations

import html as _html
import logging
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urljoin, urlparse

logger = logging.getLogger(__name__)

PREVIEW_PROGRAM_RE = re.compile(
    r'href=["\']([^"\']*preview_program\.php[^"\']*)["\']',
    re.IGNORECASE,
)
POID_RE = re.compile(r'[?&]poid=(\d+)')

# navoid for "Graduate and Undergraduate Programs Lists" per known catoid
# navoid for the "Undergraduate Programs List" page (flat list of preview_program.php
# links) per known catoid. NOTE: 25586 (for catoid 19) is the *hub* page and has no
# direct program links; 25484 is the actual list (959 links, verified live).
# Other years are resolved dynamically (see resolve_programs_navoid) since Acalog
# assigns a different navoid per catalog year.
KNOWN_PROGRAMS_NAVOIDS: dict[int, int] = {
    19: 25484,  # 2026-2027 (Undergraduate Programs List)
}

# Anchor text Acalog uses for the flat program list in the catalog navigation.
PROGRAMS_LIST_LINK_TEXT_RE = re.compile(
    r"(undergraduate\s+programs?\s+list|programs?\s+a-?z|all\s+programs?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProgramLink:
    url: str
    poid: int
    catoid: int
    text: str


def get_programs_navoid(catoid: int) -> int | None:
    return KNOWN_PROGRAMS_NAVOIDS.get(catoid)


def discover_program_links(html: str, base_url: str, catoid: int) -> list[ProgramLink]:
    """Extract all preview_program.php links from a catalog page."""
    seen_poids: set[int] = set()
    links: list[ProgramLink] = []

    anchor_re = re.compile(
        r'<a\s[^>]*href=["\']([^"\']*preview_program\.php[^"\']*)["\'][^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    tag_re = re.compile(r"<[^>]+>")

    for match in anchor_re.finditer(html):
        # Catalog hrefs are HTML-escaped (e.g. ?catoid=19&amp;poid=34694). Unescape
        # so parse_qs sees `poid`, not `amp;poid`, and the fetched URL is well-formed.
        href = _html.unescape(match.group(1).strip())
        text = tag_re.sub("", match.group(2)).strip()
        text = re.sub(r"\s+", " ", text)

        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        qs = parse_qs(parsed.query)

        poid_vals = qs.get("poid", [])
        catoid_vals = qs.get("catoid", [])
        if not poid_vals:
            continue

        try:
            poid = int(poid_vals[0])
            link_catoid = int(catoid_vals[0]) if catoid_vals else catoid
        except ValueError:
            continue

        if link_catoid != catoid:
            continue
        if poid in seen_poids:
            continue

        seen_poids.add(poid)
        norm = parsed._replace(fragment="").geturl()
        links.append(ProgramLink(url=norm, poid=poid, catoid=catoid, text=text))

    links.sort(key=lambda l: l.poid)
    logger.debug("Found %d program links (catoid=%d)", len(links), catoid)
    return links


def build_programs_list_url(base_url: str, catoid: int) -> str | None:
    navoid = get_programs_navoid(catoid)
    if navoid is None:
        return None
    return urljoin(base_url, f"/content.php?catoid={catoid}&navoid={navoid}")


_NAV_ANCHOR_RE = re.compile(
    r'<a\s[^>]*href=["\']([^"\']*content\.php[^"\']*navoid=(\d+)[^"\']*)["\'][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def resolve_programs_navoid(hub_html: str, catoid: int) -> int | None:
    """Find the program-list navoid from a catalog hub/nav page by anchor text.

    Acalog assigns a different navoid per catalog year, so for years not in
    KNOWN_PROGRAMS_NAVOIDS we discover it from a fetched navigation page (e.g. the
    "Graduate and Undergraduate Programs Lists" hub) by matching the link whose text
    is "Undergraduate Programs List".
    """
    for match in _NAV_ANCHOR_RE.finditer(hub_html):
        href = _html.unescape(match.group(1))
        navoid = int(match.group(2))
        text = re.sub(r"\s+", " ", _TAG_RE.sub("", match.group(3))).strip()
        if PROGRAMS_LIST_LINK_TEXT_RE.search(text):
            qs = parse_qs(urlparse(href).query)
            link_catoid = qs.get("catoid", [str(catoid)])[0]
            if str(link_catoid) == str(catoid):
                logger.info("Resolved programs navoid=%d (%r) for catoid=%d", navoid, text, catoid)
                return navoid
    return None
