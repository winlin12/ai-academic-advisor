"""Discover catalog years from the Purdue catalog index page.

The catalog index (index.php) is served from CDN without WAF challenge.
It contains a <select> dropdown listing all catalog years with their catoid values.
This is the only page we can reliably fetch with a simple HTTP client.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

from catalog_ingestion.fetch.client import FetchedPage

logger = logging.getLogger(__name__)

CATALOG_YEAR_LABEL_RE = re.compile(
    r"(\d{4})-(\d{4}|\d{2})\s+University\s+Catalog"
)
SELECT_OPTION_RE = re.compile(
    r'<option[^>]+value=["\'](\d+)["\'][^>]*>(.*?)</option>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class CatalogYearInfo:
    label: str
    catoid: int
    start_year: int
    end_year: int
    is_archived: bool
    catalog_url: str


def discover_catalog_years(
    page: FetchedPage,
    base_url: str,
) -> list[CatalogYearInfo]:
    """Parse the catalog index page and return all available catalog years."""
    results: list[CatalogYearInfo] = []

    for match in SELECT_OPTION_RE.finditer(page.html):
        catoid_str = match.group(1).strip()
        label_html = match.group(2).strip()
        label = re.sub(r"<[^>]+>", "", label_html).strip()
        label = re.sub(r"\s+", " ", label)

        year_match = CATALOG_YEAR_LABEL_RE.search(label)
        if not year_match:
            continue

        try:
            catoid = int(catoid_str)
        except ValueError:
            continue

        start_year = int(year_match.group(1))
        end_raw = year_match.group(2)
        end_year = int(end_raw) if len(end_raw) == 4 else start_year + 1

        is_archived = "archived" in label.lower()
        catalog_url = urljoin(base_url, f"/index.php?catoid={catoid}")

        results.append(
            CatalogYearInfo(
                label=f"{start_year}-{end_year}",
                catoid=catoid,
                start_year=start_year,
                end_year=end_year,
                is_archived=is_archived,
                catalog_url=catalog_url,
            )
        )
        logger.debug("Found catalog year: %s (catoid=%d, archived=%s)", label, catoid, is_archived)

    results.sort(key=lambda y: y.start_year, reverse=True)
    logger.info("Discovered %d catalog years", len(results))
    return results


def find_current_year(years: list[CatalogYearInfo]) -> CatalogYearInfo | None:
    """Return the most recent non-archived year."""
    for year in years:
        if not year.is_archived:
            return year
    return years[0] if years else None
