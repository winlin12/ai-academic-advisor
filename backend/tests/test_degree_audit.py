"""Tests for the degree-audit cross-referencing logic (services/degree_audit.py). Pure
function, no DB — builds AcademicProgramDetail fixtures by hand, mirroring how
academic_db.fetch_program_detail would shape real rows."""

from app.models.schemas import (
    AcademicProgramDetail,
    RequirementBlockDetail,
    RequirementCourseOption,
    RequirementRuleDetail,
    RequirementRuleOption,
)
from app.services.degree_audit import build_program_audit


def _course(code: str, credits: str | None = "3") -> RequirementCourseOption:
    return RequirementCourseOption(
        id=f"opt-{code}",
        sort_order=0,
        course_code_text=code,
        course_title=f"{code} title",
        credits_text=credits,
    )


def _rule(
    rule_id: str,
    rule_type: str,
    courses: list[RequirementCourseOption],
    *,
    credits_min: float | None = None,
    raw_text: str | None = None,
) -> RequirementRuleDetail:
    options = (
        [RequirementRuleOption(id=f"{rule_id}-opt", option_index=0, sort_order=0, courses=courses)]
        if courses
        else []
    )
    return RequirementRuleDetail(
        id=rule_id,
        sort_order=0,
        rule_type=rule_type,
        credits_min=credits_min,
        raw_text=raw_text,
        options=options,
    )


def _program(blocks: list[RequirementBlockDetail]) -> AcademicProgramDetail:
    return AcademicProgramDetail(
        id="prog-1",
        catalog_year=2026,
        school="College of Science",
        program_title="BS Computer Science",
        degree_code="BS",
        variant=None,
        source_url="https://example.test",
        parser_status="parsed",
        blocks=blocks,
    )


# --- "requirement" (take-all) rules ---------------------------------------------------------


def test_requirement_rule_satisfied_only_when_every_course_is_done():
    rule = _rule("core", "requirement", [_course("CS 18000"), _course("CS 18200")])
    block = RequirementBlockDetail(id="b1", sort_order=0, title="Core", rules=[rule])
    program = _program([block])

    partial = build_program_audit(program, completed_courses=["CS 18000"])
    assert partial.blocks[0].rules[0].satisfied is False
    assert partial.blocks[0].satisfied is False

    full = build_program_audit(program, completed_courses=["CS 18000", "CS 18200"])
    assert full.blocks[0].rules[0].satisfied is True
    assert full.blocks[0].satisfied is True


def test_requirement_rule_accepts_planned_courses_as_well_as_completed():
    rule = _rule("core", "requirement", [_course("CS 18000"), _course("CS 18200")])
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[rule])])

    audit = build_program_audit(
        program, completed_courses=["CS 18000"], planned_courses=["CS 18200"]
    )
    courses = audit.blocks[0].rules[0].options[0].courses
    assert {c.course_code_text: c.satisfied_by for c in courses} == {
        "CS 18000": "completed",
        "CS 18200": "planned",
    }
    assert audit.blocks[0].rules[0].satisfied is True


def test_code_matching_is_normalized():
    rule = _rule("core", "requirement", [_course("CS 18000")])
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[rule])])
    audit = build_program_audit(program, completed_courses=["cs18000"])
    assert audit.blocks[0].rules[0].satisfied is True


def test_completed_takes_precedence_over_planned_for_the_same_course():
    rule = _rule("core", "requirement", [_course("CS 18000")])
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[rule])])
    audit = build_program_audit(
        program, completed_courses=["CS 18000"], planned_courses=["CS 18000"]
    )
    assert audit.blocks[0].rules[0].options[0].courses[0].satisfied_by == "completed"


# --- "choose" (selective) rules -------------------------------------------------------------


def test_choose_rule_with_credit_target_needs_enough_credits_not_just_one_course():
    rule = _rule(
        "sel",
        "choose",
        [_course("STAT 35000", "3"), _course("STAT 51100", "3"), _course("STAT 41600", "3")],
        credits_min=6.0,
    )
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[rule])])

    one_done = build_program_audit(program, completed_courses=["STAT 35000"])
    assert one_done.blocks[0].rules[0].satisfied is False

    two_done = build_program_audit(program, completed_courses=["STAT 35000", "STAT 51100"])
    assert two_done.blocks[0].rules[0].satisfied is True


def test_choose_rule_without_credit_target_falls_back_to_choose_one():
    rule = _rule("sel", "choose", [_course("PHIL 11000", None), _course("PHIL 12000", None)])
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[rule])])

    none_done = build_program_audit(program, completed_courses=[])
    assert none_done.blocks[0].rules[0].satisfied is False

    one_done = build_program_audit(program, completed_courses=["PHIL 12000"])
    assert one_done.blocks[0].rules[0].satisfied is True


# --- narrative rules (no courses — GPA/policy text) -----------------------------------------


def test_narrative_rule_is_not_applicable_rather_than_unsatisfied():
    narrative = _rule("gpa", "requirement", [], raw_text="Maintain a 2.0 GPA overall.")
    program = _program([RequirementBlockDetail(id="b1", sort_order=0, rules=[narrative])])

    audit = build_program_audit(program, completed_courses=[])
    assert audit.blocks[0].rules[0].satisfied is None
    # A block made up only of narrative rules has nothing to check either.
    assert audit.blocks[0].satisfied is None
    # And it doesn't get counted in the top-line rollup.
    assert audit.total_requirements == 0
    assert audit.satisfied_requirements == 0


def test_narrative_rule_does_not_block_the_rest_of_its_block():
    narrative = _rule("gpa", "requirement", [], raw_text="Maintain a 2.0 GPA overall.")
    real = _rule("core", "requirement", [_course("CS 18000")])
    block = RequirementBlockDetail(id="b1", sort_order=0, rules=[narrative, real])
    program = _program([block])

    audit = build_program_audit(program, completed_courses=["CS 18000"])
    assert audit.blocks[0].satisfied is True  # the narrative rule doesn't drag it down
    assert audit.total_requirements == 1
    assert audit.satisfied_requirements == 1


# --- top-level rollup ------------------------------------------------------------------------


def test_summary_rollup_counts_across_multiple_blocks():
    block_a = RequirementBlockDetail(
        id="a",
        sort_order=0,
        rules=[_rule("a1", "requirement", [_course("CS 18000")])],
    )
    block_b = RequirementBlockDetail(
        id="b",
        sort_order=1,
        rules=[
            _rule("b1", "requirement", [_course("CS 18200")]),
            _rule("b2", "choose", [_course("PHIL 11000", None)]),
        ],
    )
    program = _program([block_a, block_b])

    audit = build_program_audit(program, completed_courses=["CS 18000"])
    assert audit.total_requirements == 3
    assert audit.satisfied_requirements == 1
