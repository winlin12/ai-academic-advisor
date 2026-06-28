"""Tests for catalog year discovery."""

from catalog_ingestion.fetch.client import FetchedPage
from catalog_ingestion.discover.catalog_years import discover_catalog_years, find_current_year

SAMPLE_INDEX_HTML = """
<html><body>
<select name="catalog">
  <option value="19" selected>2026-2027 University Catalog</option>
  <option value="18">2025-2026 University Catalog [ARCHIVED CATALOG]</option>
  <option value="17">2024-2025 University Catalog [ARCHIVED CATALOG]</option>
  <option value="4">2014-2015 University Catalog [ARCHIVED CATALOG]</option>
</select>
</body></html>
"""


def test_discovers_all_years():
    from datetime import datetime
    page = FetchedPage(url="https://catalog.purdue.edu/index.php", html=SAMPLE_INDEX_HTML, http_status=200)
    years = discover_catalog_years(page, "https://catalog.purdue.edu")
    assert len(years) == 4
    labels = [y.label for y in years]
    assert "2026-2027" in labels
    assert "2014-2015" in labels


def test_finds_current_year():
    page = FetchedPage(url="https://catalog.purdue.edu/index.php", html=SAMPLE_INDEX_HTML, http_status=200)
    years = discover_catalog_years(page, "https://catalog.purdue.edu")
    current = find_current_year(years)
    assert current is not None
    assert current.label == "2026-2027"
    assert not current.is_archived


def test_archived_flags_correctly():
    page = FetchedPage(url="https://catalog.purdue.edu/index.php", html=SAMPLE_INDEX_HTML, http_status=200)
    years = discover_catalog_years(page, "https://catalog.purdue.edu")
    by_label = {y.label: y for y in years}
    assert not by_label["2026-2027"].is_archived
    assert by_label["2025-2026"].is_archived


def test_catoid_extracted():
    page = FetchedPage(url="https://catalog.purdue.edu/index.php", html=SAMPLE_INDEX_HTML, http_status=200)
    years = discover_catalog_years(page, "https://catalog.purdue.edu")
    by_label = {y.label: y for y in years}
    assert by_label["2026-2027"].catoid == 19
    assert by_label["2014-2015"].catoid == 4
