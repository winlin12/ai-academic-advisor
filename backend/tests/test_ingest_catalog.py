"""Tests for the catalog -> academic_rules chunk builders.

These cover the pure formatting functions (no DB, no model): given a course row or a program's
requirement block, they must produce a self-contained, tagged chunk. The DB-backed iterators
and the embedding driver are exercised by the manual ``--dry-run`` ingestion, not here.
"""

from app.models.schemas import (
    AcademicProgramDetail,
    RequirementBlockDetail,
    RequirementCourseOption,
    RequirementRuleDetail,
    RequirementRuleOption,
)
from app.services.rag import ingest_catalog


def _course_row(**overrides):
    row = {
        "course_code": "CS 25100",
        "subject_code": "CS",
        "title": "Data Structures and Algorithms",
        "description": "  Trees, graphs,   sorting.  ",
        "credit_hours_min": 3.0,
        "credit_hours_max": 3.0,
        "prerequisites_raw": "CS 18200",
    }
    row.update(overrides)
    return row


def test_build_course_chunk_is_self_contained_and_tagged():
    content, metadata = ingest_catalog.build_course_chunk(_course_row())
    # Code, title, credits, description and prereqs all land in one retrievable string.
    assert "CS 25100" in content
    assert "Data Structures and Algorithms" in content
    assert "(3 cr)" in content
    assert "Trees, graphs, sorting." in content  # whitespace collapsed
    assert "Prerequisites: CS 18200" in content
    assert metadata == {"type": "course", "code": "CS 25100", "subject": "CS"}


def test_build_course_chunk_handles_missing_fields():
    content, metadata = ingest_catalog.build_course_chunk(
        _course_row(description=None, prerequisites_raw=None, subject_code=None)
    )
    assert "No description on file." in content
    assert "Prerequisites: not listed in the catalog." in content
    assert metadata["subject"] is None


def _program() -> AcademicProgramDetail:
    return AcademicProgramDetail(
        id="prog-1",
        catalog_year=2026,
        school="College of Science",
        program_title="Computer Science",
        degree_code="BS",
        variant=None,
        source_url="",
        parser_status="parsed",
    )


def _block_with_courses() -> RequirementBlockDetail:
    course = RequirementCourseOption(
        id="opt-1",
        sort_order=0,
        course_code_text="CS 25100",
        course_title="Data Structures",
        credits_text="3",
    )
    rule = RequirementRuleDetail(
        id="rule-1",
        sort_order=0,
        rule_type="requirement",
        options=[RequirementRuleOption(id="ro-1", option_index=0, sort_order=0, courses=[course])],
    )
    return RequirementBlockDetail(id="blk-1", sort_order=0, title="Major Core", rules=[rule])


def test_build_block_chunk_includes_program_and_courses():
    chunk = ingest_catalog.build_block_chunk(_program(), _block_with_courses())
    assert chunk is not None
    content, metadata = chunk
    assert "PROGRAM: Computer Science (BS)" in content
    assert "College of Science" in content
    assert "REQUIREMENT BLOCK: Major Core" in content
    assert "CS 25100 — Data Structures (3 cr)" in content
    assert metadata["type"] == "requirement"
    assert metadata["program"] == "Computer Science"
    assert metadata["program_id"] == "prog-1"
    assert metadata["block"] == "Major Core"


def test_build_block_chunk_skips_empty_blocks():
    empty = RequirementBlockDetail(id="blk-0", sort_order=0, title="Placeholder", rules=[])
    assert ingest_catalog.build_block_chunk(_program(), empty) is None


def test_build_block_chunk_keeps_narrative_rules():
    rule = RequirementRuleDetail(
        id="rule-n",
        sort_order=0,
        rule_type="policy",
        raw_text="A minimum GPA of 2.0 is required.",
    )
    block = RequirementBlockDetail(id="blk-n", sort_order=0, title="Academic Standing", rules=[rule])
    chunk = ingest_catalog.build_block_chunk(_program(), block)
    assert chunk is not None
    assert "minimum GPA of 2.0" in chunk[0]


def test_build_block_chunk_clips_oversized_content():
    many = [
        RequirementCourseOption(
            id=f"opt-{i}", sort_order=i, course_code_text=f"CS {10000 + i}", course_title="X" * 50
        )
        for i in range(400)
    ]
    rule = RequirementRuleDetail(
        id="rule-big",
        sort_order=0,
        rule_type="requirement",
        options=[RequirementRuleOption(id="ro-big", option_index=0, sort_order=0, courses=many)],
    )
    block = RequirementBlockDetail(id="blk-big", sort_order=0, title="Huge", rules=[rule])
    content, _ = ingest_catalog.build_block_chunk(_program(), block)
    assert len(content) <= ingest_catalog.MAX_CHUNK_CHARS
    assert content.endswith("…")
