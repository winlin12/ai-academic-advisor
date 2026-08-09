"""Tests for the requirement-group parser and its link resolver.

Fixtures are REAL Acalog HTML, not hand-written snippets, because the behavior under test
(a block linking out to another program page instead of listing courses inline) depends on
exact real-world markup that would be easy to get subtly wrong by hand:

  - program_machine_intelligence.html: the actual "Computer Science: Machine Intelligence,
    BS" program page (catoid=19, poid=36006), pulled from this box's already-crawled
    Postgres `source_pages.raw_html`.
  - program_composition_presentation.html: one of the pages it links to instead of listing
    courses inline (catoid=19, poid=35114, "Composition and Presentation"), fetched live
    once during development and cached by the crawler's own disk cache from then on.
"""

from pathlib import Path

from catalog_ingestion.parse.requirements import (
    RequirementGroupData,
    parse_requirement_sections,
    resolve_requirement_links,
)

FIXTURES = Path(__file__).parent / "fixtures"
MI_PROGRAM_HTML = (FIXTURES / "program_machine_intelligence.html").read_text()
COMPOSITION_HTML = (FIXTURES / "program_composition_presentation.html").read_text()


def _find_or_none(groups: list[RequirementGroupData], name: str) -> RequirementGroupData | None:
    for g in groups:
        if g.name == name:
            return g
        found = _find_or_none(g.children, name)
        if found is not None:
            return found
    return None


def _find(groups: list[RequirementGroupData], name: str) -> RequirementGroupData:
    found = _find_or_none(groups, name)
    if found is None:
        raise AssertionError(f"{name!r} not found in the parsed tree")
    return found


class _FakeFetcher:
    """Serves one canned page for any poid, and records every URL it was asked for."""

    def __init__(self, html: str):
        self.html = html
        self.requested: list[str] = []

    def fetch(self, url: str):
        self.requested.append(url)
        return type("Page", (), {"html": self.html})()


def test_direct_link_captured_on_the_block_that_has_it():
    """'College of Science Core Requirements' has no inline courses, but links out to
    several sibling program pages — those links must survive parsing as `linked_refs`."""
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    core = _find(groups, "College of Science Core Requirements")
    assert core.options == []
    assert core.children == []
    assert (19, 35114) in core.linked_refs  # "Composition and Presentation"


def test_link_not_captured_across_a_nested_block_boundary():
    """A link inside a CHILD block must not leak onto its ancestor's own `linked_refs` —
    same scoping `_direct_text`/`_parse_course_options` already rely on."""
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    additional = _find(groups, "College of Science Additional Requirements")
    # Its child "Laboratory Science (6-8 credits)" carries the poid=35118 link; the parent
    # itself has no link markup of its own in that block's direct text.
    assert (19, 35118) not in additional.linked_refs


def test_unrelated_link_type_is_never_resolved():
    """"About the Program" (type `overview`) links to this program's own CODO
    (change-of-major) page — real markup, same mechanism, but not a requirement list, and
    must never be fetched."""
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    about = _find(groups, "About the Program")
    assert about.linked_refs  # the CODO link is captured...
    fetcher = _FakeFetcher(COMPOSITION_HTML)
    resolve_requirement_links(groups, fetcher=fetcher, catoid=19)
    assert about.children == []  # ...but never resolved, because `overview` isn't in scope
    assert not any("35513" in u for u in fetcher.requested)


def test_resolve_requirement_links_fills_in_real_courses():
    """End to end: a group with a linked poid and no options gains real course options
    after resolution, sourced from the linked page's own parsed content.

    The winning parent is "Curriculum and Degree Requirements for College of Science", NOT
    the more specifically-named "College of Science Core Requirements" — both link to the
    same ten poids (a table-of-contents-style overview and a detailed narrative covering the
    same ground), and the overview comes first in document order. See
    `resolve_requirement_links`'s docstring for why first-occurrence-wins is an accepted
    simplification: the resolved group's OWN name/options are correct either way, only its
    parent is less specific than it could be.
    """
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    fetcher = _FakeFetcher(COMPOSITION_HTML)
    resolve_requirement_links(groups, fetcher=fetcher, catoid=19)

    overview = _find(groups, "Curriculum and Degree Requirements for College of Science")
    assert overview.children, "expected linked pages to be attached as children"

    # The linked page's own parse (independently verified against the real HTML) surfaces
    # a real ENGL course under a nested "Option 1" group.
    option_1 = _find(overview.children, "Option 1")
    codes = {opt.course_code_raw for opt in option_1.options}
    assert "ENGL 30400" in codes

    # The more specifically-named sibling is left unresolved this pass — its poids were
    # already claimed by the earlier, more generic block.
    core = _find(groups, "College of Science Core Requirements")
    assert core.children == []


def test_resolve_requirement_links_fetches_each_poid_at_most_once():
    """The same poid is linked from more than one place on a real program page (a
    table-of-contents-style overview, then again inline in the detailed narrative). The
    resolver must not fetch — or attach — it twice."""
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    fetcher = _FakeFetcher(COMPOSITION_HTML)
    resolve_requirement_links(groups, fetcher=fetcher, catoid=19)

    urls = fetcher.requested
    assert len(urls) == len(set(urls)), f"duplicate fetches: {urls}"


def test_off_domain_and_stale_catalog_year_links_are_ignored():
    """Only same-domain, same-`catoid` `preview_program.php` links are followed. A
    different catalog year's link (catoid=14, an archived year) must never be fetched, and
    the off-domain University Senate link is never even captured (doesn't match the
    `preview_program.php` pattern at all)."""
    groups = parse_requirement_sections(MI_PROGRAM_HTML)
    fetcher = _FakeFetcher(COMPOSITION_HTML)
    resolve_requirement_links(groups, fetcher=fetcher, catoid=19)
    assert not any("catoid=14" in u for u in fetcher.requested)
    assert not any("purdue.edu/senate" in u for u in fetcher.requested)
