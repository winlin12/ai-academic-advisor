"""Tests for the planner ↔ academic-DB bridge (TODO §4). No database required:

* pure helpers (code normalization, row→Course mapping, choose-N selection) run as-is;
* resolve_profile_and_catalog is tested with the DB fetches monkeypatched, including the
  fall-back to the bundled courses.json fixture when the DB is down or knows nothing.
"""

import psycopg
import pytest

from app.models.schemas import Course, StudentProfile
from app.services import planner_catalog
from app.services.planner import TERM_ORDER, generate_plan
from app.services.planner_catalog import (
    ProgramNotFoundError,
    _course_from_row,
    derive_remaining_courses,
    normalize_course_code,
    resolve_profile_and_catalog,
    select_remaining_courses,
)
from app.services.planner_db import ProgramCatalog, RequirementGroup


def test_normalize_course_code_variants():
    assert normalize_course_code("cs18000") == "CS 18000"
    assert normalize_course_code("CS  18000") == "CS 18000"
    assert normalize_course_code("CS 18000") == "CS 18000"
    assert normalize_course_code("ENGL") == "ENGL"  # no digits — untouched
    assert normalize_course_code("18000") == "18000"  # no subject prefix


def test_course_from_row_maps_catalog_gaps_to_permissive_defaults():
    course = _course_from_row(
        {"course_code": "CS 18000", "title": " Problem Solving ", "credit_hours_min": 4.0}
    )
    assert course.code == "CS 18000"
    assert course.title == "Problem Solving"
    assert course.credits == 4
    # The catalog publishes neither prereqs nor term offerings; the planner must not block.
    # `TERM_ORDER` is the terms the planner may schedule INTO, which excludes summer by policy
    # (PLANNER_INCLUDE_SUMMER) — so "offered every term we plan into" is the permissive default
    # here, not "offered in all three seasons".
    assert course.prereqs == []
    assert set(course.offered_terms) == set(TERM_ORDER)
    assert "summer" not in course.offered_terms


def test_course_from_row_handles_null_credits():
    course = _course_from_row({"course_code": "CS 09200", "title": None, "credit_hours_min": None})
    assert course.credits == 0
    assert course.title == "Untitled course"


def _row(group_id, req_type, group_credits, code, option_credits, order=0):
    return {
        "group_id": group_id,
        "requirement_type": req_type,
        "group_credits_min": group_credits,
        "group_order": order,
        "course_code": code,
        "option_credits": option_credits,
        "option_order": 0,
    }


def test_select_remaining_required_blocks_before_selectives():
    rows = [
        _row("g-selective", "selective", 3.0, "STAT 35000", 3.0),
        _row("g-selective", "selective", 3.0, "STAT 51100", 3.0),
        _row("g-core", "core", None, "CS 18000", 4.0),
        _row("g-core", "core", None, "CS 18200", 3.0),
    ]
    remaining = select_remaining_courses(rows, completed=set())
    assert remaining == ["CS 18000", "CS 18200", "STAT 35000"]


def test_select_remaining_skips_completed_and_counts_them_toward_selectives():
    rows = [
        _row("g-core", "core", None, "CS 18000", 4.0),
        _row("g-core", "core", None, "CS 18200", 3.0),
        # 6-credit selective: one 3-credit option already completed → only one more needed.
        _row("g-sel", "selective", 6.0, "STAT 35000", 3.0),
        _row("g-sel", "selective", 6.0, "STAT 51100", 3.0),
        _row("g-sel", "selective", 6.0, "STAT 41600", 3.0),
    ]
    remaining = select_remaining_courses(rows, completed={"CS 18000", "STAT 35000"})
    assert remaining == ["CS 18200", "STAT 51100"]


def test_select_remaining_counts_a_required_course_toward_a_selective_that_lists_it():
    """A course counts for every list it appears in. CS 18000 is a core requirement AND one of
    the selective's options, so the selective is already paid for and must not pull in a second
    course for credits the student is taking anyway."""
    rows = [
        _row("g-core", "core", None, "CS 18000", 4.0),
        _row("g-sel", "selective", 3.0, "CS 18000", 4.0),
        _row("g-sel", "selective", 3.0, "STAT 35000", 3.0),
    ]
    assert select_remaining_courses(rows, completed=set()) == ["CS 18000"]


def test_select_remaining_counts_one_shared_course_once_per_group():
    """Paying the target down twice from one course would leave a 6-credit group half-empty."""
    rows = [
        _row("g-core", "core", None, "CS 18000", 4.0),
        # The same option crawled twice under one group, plus a genuine second option.
        _row("g-sel", "selective", 6.0, "CS 18000", 4.0),
        _row("g-sel", "selective", 6.0, "CS 18000", 4.0),
        _row("g-sel", "selective", 6.0, "STAT 35000", 3.0),
    ]
    assert select_remaining_courses(rows, completed=set()) == ["CS 18000", "STAT 35000"]


def test_select_remaining_selective_without_credit_target_chooses_one():
    rows = [
        _row("g-sel", "elective", None, "PHIL 11000", None),
        _row("g-sel", "elective", None, "PHIL 12000", None),
    ]
    assert select_remaining_courses(rows, completed=set()) == ["PHIL 11000"]
    # Any completed option satisfies the choose-one group.
    assert select_remaining_courses(rows, completed={"PHIL 12000"}) == []


def test_select_remaining_normalizes_completed_codes():
    rows = [_row("g-core", "core", None, "CS 18000", 4.0)]
    assert select_remaining_courses(rows, completed={"cs18000"}) == []


def test_derive_remaining_rejects_non_uuid_program_id():
    with pytest.raises(ProgramNotFoundError):
        derive_remaining_courses("not-a-uuid", set())


def _fake_catalog_fetch(known: dict[str, tuple[str, float]]):
    def fetch(codes: set[str]):
        return {
            normalize_course_code(code): _course_from_row(
                {
                    "course_code": normalize_course_code(code),
                    "title": known[normalize_course_code(code)][0],
                    "credit_hours_min": known[normalize_course_code(code)][1],
                }
            )
            for code in codes
            if normalize_course_code(code) in known
        }

    return fetch


def test_resolve_uses_db_catalog_and_canonicalizes_codes(monkeypatch):
    monkeypatch.setattr(
        planner_catalog,
        "fetch_courses_by_codes",
        _fake_catalog_fetch({"CS 18000": ("Problem Solving", 4.0), "CS 18200": ("Discrete", 3.0)}),
    )
    profile = StudentProfile(remaining_courses=["cs18000", "CS 18200", "FAKE 999"])
    resolved, catalog = resolve_profile_and_catalog(profile)

    # Matched codes are canonicalized; unknown ones stay so the planner can warn by name.
    assert resolved.remaining_courses == ["CS 18000", "CS 18200", "FAKE 999"]
    codes = {course.code for course in catalog}
    assert codes == {"CS 18000", "CS 18200"}

    plan = generate_plan(resolved, catalog)
    assert any("FAKE 999" in warning for warning in plan.warnings)
    planned = {c.code for s in plan.semesters for c in s.courses}
    assert planned == {"CS 18000", "CS 18200"}


def test_resolve_falls_back_to_fixture_when_db_down(monkeypatch):
    def boom(codes):
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(planner_catalog, "fetch_courses_by_codes", boom)
    profile = StudentProfile(remaining_courses=["CS381"])
    resolved, catalog = resolve_profile_and_catalog(profile)

    assert resolved is profile  # untouched
    assert any(course.code == "CS381" for course in catalog)  # bundled fixture answered


def test_resolve_falls_back_to_fixture_when_no_codes_match(monkeypatch):
    monkeypatch.setattr(planner_catalog, "fetch_courses_by_codes", _fake_catalog_fetch({}))
    profile = StudentProfile(remaining_courses=["CS381", "CS502"])
    resolved, catalog = resolve_profile_and_catalog(profile)

    assert resolved is profile
    assert any(course.code == "CS381" for course in catalog)


def test_resolve_with_program_id_derives_remaining(monkeypatch):
    """A chosen major decides the courses; whatever the client listed is ignored.

    Patched at ``load_program_catalog`` rather than at the old ``derive_remaining_courses``:
    the program path now goes through ``planner_db``, which returns requirement groups and a
    catalog carrying real prereq edges and offerings, not a bare list of codes.
    """
    catalog_stub = ProgramCatalog(
        program_id="4d1f34a2-0000-0000-0000-000000000000",
        name="BS Computer Science",
        catalog_year="2026-2027",
        groups=[
            RequirementGroup(id="core", name="Core", selective=False, credits_min=9.0,
                             options=[["CS 18000"], ["CS 18200"], ["STAT 35000"]]),
        ],
        courses=[
            Course(code="CS 18000", title="Problem Solving", credits=4),
            Course(code="CS 18200", title="Discrete", credits=3),
            Course(code="STAT 35000", title="Statistics", credits=3),
        ],
    )
    monkeypatch.setattr(planner_catalog, "load_program_catalog",
                        lambda program_id, extra_codes=None, profile=None: catalog_stub)

    profile = StudentProfile(
        program_id="4d1f34a2-0000-0000-0000-000000000000",
        completed_courses=["CS 18000"],
        remaining_courses=["IGNORED 101"],  # client-listed codes lose to the program derivation
    )
    resolved, catalog = resolve_profile_and_catalog(profile)

    assert resolved.remaining_courses == ["CS 18200", "STAT 35000"]
    assert {course.code for course in catalog} == {"CS 18000", "CS 18200", "STAT 35000"}
    # The major subject is derived from the program's own required courses, never asked for.
    assert resolved.major_subject == "CS"
