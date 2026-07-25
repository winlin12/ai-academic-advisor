"""Tests for academic_db._build_blocks — the pure (group, option) row -> block/rule/course
folding used by GET /v1/academic/programs/{id}. No DB: rows are hand-built in the exact shape
the real SQL query in fetch_program_detail returns."""

from app.services.academic_db import _build_blocks, _credits_text


def _row(
    group_id: str,
    *,
    parent_group_id: str | None = None,
    group_name: str | None = "Group",
    requirement_type: str | None = "core",
    credits_min: float | None = None,
    credits_max: float | None = None,
    group_raw_text: str | None = None,
    group_order: int = 0,
    option_id: str | None = None,
    option_order: int | None = 0,
    course_code_raw: str | None = None,
    resolved_course_id: str | None = None,
    course_title_raw: str | None = None,
    option_text: str | None = None,
    credits: float | None = None,
    option_credits_max: float | None = None,
) -> dict:
    return {
        "group_id": group_id,
        "parent_group_id": parent_group_id,
        "group_name": group_name,
        "requirement_type": requirement_type,
        "credits_min": credits_min,
        "credits_max": credits_max,
        "group_raw_text": group_raw_text,
        "group_order": group_order,
        "option_id": option_id,
        "option_order": option_order,
        "course_code_raw": course_code_raw,
        "resolved_course_id": resolved_course_id,
        "course_title_raw": course_title_raw,
        "option_text": option_text,
        "credits": credits,
        "option_credits_max": option_credits_max,
        "is_selective_option": False,
        "minimum_grade": None,
    }


def test_credits_text_formats_single_value_and_range():
    assert _credits_text(3, 3) == "3"
    assert _credits_text(3, 4) == "3-4"
    assert _credits_text(None, None) is None
    assert _credits_text(3, None) == "3"


def test_build_blocks_groups_a_required_rule_with_its_courses():
    rows = [
        _row(
            "core",
            group_name="Core Requirements",
            requirement_type="core",
            credits_min=7,
            credits_max=7,
            option_id="opt-1",
            course_code_raw="CS 18000",
            course_title_raw="Problem Solving",
            credits=4,
        ),
        _row(
            "core",
            group_name="Core Requirements",
            requirement_type="core",
            credits_min=7,
            credits_max=7,
            option_id="opt-2",
            option_order=1,
            course_code_raw="CS 18200",
            course_title_raw="Discrete Math",
            credits=3,
        ),
    ]
    blocks = _build_blocks(rows)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.title == "Core Requirements"
    assert block.credits_text == "7"
    assert len(block.rules) == 1
    rule = block.rules[0]
    assert rule.rule_type == "core"  # not selective -> passes the raw requirement_type through
    assert rule.credits_min == 7.0
    courses = rule.options[0].courses
    assert [c.course_code_text for c in courses] == ["CS 18000", "CS 18200"]


def test_build_blocks_marks_selective_groups_with_courses_as_choose():
    rows = [
        _row(
            "sel",
            group_name="Statistics Elective",
            requirement_type="selective",
            credits_min=3,
            option_id="opt-1",
            course_code_raw="STAT 35000",
            credits=3,
        ),
        _row(
            "sel",
            group_name="Statistics Elective",
            requirement_type="selective",
            credits_min=3,
            option_id="opt-2",
            option_order=1,
            course_code_raw="STAT 51100",
            credits=3,
        ),
    ]
    blocks = _build_blocks(rows)
    rule = blocks[0].rules[0]
    assert rule.rule_type == "choose"
    assert rule.credits_min == 3.0


def test_build_blocks_keeps_narrative_groups_with_no_courses():
    rows = [_row("policy", group_name="Policy", requirement_type="policy", group_raw_text="Keep a 2.0 GPA.")]
    blocks = _build_blocks(rows)
    rule = blocks[0].rules[0]
    assert rule.raw_text == "Keep a 2.0 GPA."
    assert rule.options == []


def test_build_blocks_nests_child_groups_as_rules_within_the_parent_block():
    rows = [
        _row("block", group_name="Major Requirements", requirement_type="block", group_order=0),
        _row(
            "child",
            parent_group_id="block",
            group_name="Choose one",
            requirement_type="selective",
            group_order=1,
            option_id="opt-1",
            course_code_raw="PHIL 11000",
            credits=None,
        ),
    ]
    blocks = _build_blocks(rows)
    assert len(blocks) == 1  # only the top-level (parentless) group becomes its own block
    assert [rule.id for rule in blocks[0].rules] == ["block", "child"]
