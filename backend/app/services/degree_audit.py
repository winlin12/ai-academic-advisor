"""Degree audit: cross-references a program's requirement tree against a student's completed
and planned courses — the MyPurduePlan-style "what's satisfied, what's left" view.

Pure function, no DB/model I/O — the caller (``api/routers/academic.py``) supplies an
already-fetched :class:`AcademicProgramDetail` (from ``academic_db.fetch_program_detail``,
which the frontend also uses for the plain read-only requirement view) plus the student's own
course lists. Matching reuses ``planner_catalog.normalize_course_code`` and
``DEFAULT_OPTION_CREDITS`` so a course counts as "done" here under exactly the same rules the
planner already uses to decide what's *remaining* — one notion of "satisfied", not two that can
quietly drift apart.

Satisfaction rules, mirroring ``planner_catalog.select_remaining_courses``:

* A "requirement" rule (take-all) is satisfied once every course in it is completed or planned.
* A "choose" rule (selective/elective/free_elective) is satisfied once enough of its courses are
  completed/planned to cover the rule's ``credits_min`` target — or, when the catalog recorded
  no credit target at all, once at least one course in it is done (matches the planner's
  "choose one" fallback).
* A narrative rule (GPA minimums, policy text — no courses at all) has nothing to check, so it
  reports ``satisfied=None`` rather than counting as met or unmet either way.
"""

from __future__ import annotations

from app.models.schemas import (
    AcademicProgramDetail,
    ProgramAuditResponse,
    RequirementBlockDetail,
    RequirementBlockProgress,
    RequirementCourseOption,
    RequirementCourseProgress,
    RequirementRuleDetail,
    RequirementRuleOptionProgress,
    RequirementRuleProgress,
)
from app.services.planner_catalog import DEFAULT_OPTION_CREDITS, normalize_course_code


def _course_progress(
    course: RequirementCourseOption,
    completed: set[str],
    planned: set[str],
) -> RequirementCourseProgress:
    code = normalize_course_code(course.course_code_text) if course.course_code_text else None
    satisfied_by: str | None = None
    if code and code in completed:
        satisfied_by = "completed"
    elif code and code in planned:
        satisfied_by = "planned"

    return RequirementCourseProgress(
        **course.model_dump(),
        satisfied=satisfied_by is not None,
        satisfied_by=satisfied_by,
    )


def _credits_of(course: RequirementCourseProgress) -> float:
    """Best-effort numeric credits for one option, for summing toward a rule's credit target.

    ``credits_text`` is display text (``"3"``, ``"3-4"``) — take the low end of a range, which
    matches using ``credits_min`` everywhere else in the app. Falls back to
    ``DEFAULT_OPTION_CREDITS`` (mirrors ``planner_catalog``) when the option has no credits on
    file at all, so a missing number never silently zeroes out progress toward the target.
    """
    if course.credits_text:
        try:
            return float(course.credits_text.split("-")[0])
        except ValueError:
            pass
    return DEFAULT_OPTION_CREDITS


def _option_progress(
    option_id: str,
    option_index: int,
    sort_order: int,
    label: str | None,
    courses: list[RequirementCourseOption],
    *,
    is_choose: bool,
    credits_min: float | None,
    completed: set[str],
    planned: set[str],
) -> RequirementRuleOptionProgress:
    course_progress = [_course_progress(c, completed, planned) for c in courses]

    satisfied: bool | None
    if not course_progress:
        satisfied = None
    elif is_choose:
        if credits_min is None:
            satisfied = any(c.satisfied for c in course_progress)
        else:
            have = sum(_credits_of(c) for c in course_progress if c.satisfied)
            satisfied = have >= credits_min
    else:
        satisfied = all(c.satisfied for c in course_progress)

    return RequirementRuleOptionProgress(
        id=option_id,
        option_index=option_index,
        sort_order=sort_order,
        label=label,
        courses=course_progress,
        satisfied=satisfied,
    )


def _rule_progress(
    rule: RequirementRuleDetail, completed: set[str], planned: set[str]
) -> RequirementRuleProgress:
    is_choose = rule.rule_type == "choose"
    options = [
        _option_progress(
            option.id,
            option.option_index,
            option.sort_order,
            option.label,
            option.courses,
            is_choose=is_choose,
            credits_min=rule.credits_min,
            completed=completed,
            planned=planned,
        )
        for option in rule.options
    ]

    # A rule is satisfied if any one of its options is (today there's usually exactly one
    # option per rule, so this reduces to that option's own status; the "any" keeps the door
    # open for a future rule with genuine alternative option-sets without changing semantics).
    applicable = [o.satisfied for o in options if o.satisfied is not None]
    rule_satisfied = any(applicable) if applicable else None

    return RequirementRuleProgress(
        id=rule.id,
        sort_order=rule.sort_order,
        rule_type=rule.rule_type,
        choose_count=rule.choose_count,
        credits_min=rule.credits_min,
        raw_text=rule.raw_text,
        options=options,
        satisfied=rule_satisfied,
    )


def _block_progress(
    block: RequirementBlockDetail, completed: set[str], planned: set[str]
) -> RequirementBlockProgress:
    rules = [_rule_progress(rule, completed, planned) for rule in block.rules]
    applicable = [r.satisfied for r in rules if r.satisfied is not None]
    block_satisfied = all(applicable) if applicable else None

    return RequirementBlockProgress(
        id=block.id,
        sort_order=block.sort_order,
        title=block.title,
        credits_text=block.credits_text,
        rules=rules,
        satisfied=block_satisfied,
    )


def build_program_audit(
    program: AcademicProgramDetail,
    completed_courses: list[str],
    planned_courses: list[str] | None = None,
) -> ProgramAuditResponse:
    """The full audit: every block/rule/course in ``program`` annotated with satisfaction.

    A course completed for real always wins over merely being planned — ``planned_courses``
    is normalized with the completed set subtracted out before matching, so a course can't be
    reported as both.
    """
    completed = {normalize_course_code(c) for c in completed_courses if c.strip()}
    planned = {normalize_course_code(c) for c in (planned_courses or []) if c.strip()} - completed

    blocks = [_block_progress(block, completed, planned) for block in program.blocks]

    all_rule_statuses = [
        rule.satisfied for block in blocks for rule in block.rules if rule.satisfied is not None
    ]

    return ProgramAuditResponse(
        id=program.id,
        catalog_year=program.catalog_year,
        school=program.school,
        program_title=program.program_title,
        degree_code=program.degree_code,
        variant=program.variant,
        blocks=blocks,
        total_requirements=len(all_rule_statuses),
        satisfied_requirements=sum(1 for satisfied in all_rule_statuses if satisfied),
    )
