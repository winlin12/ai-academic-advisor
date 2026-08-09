"""Tests for the checker and repairer that stand between the model and the student.

No database and no model: schedules are plain lists of course codes and the catalog is built by
hand, so every case here is a statement about the RULES rather than about the data.
"""

import pytest

from app.models.schemas import Course, StudentProfile
from app.services.plan_validation import (
    HARD_CREDIT_CAP,
    normalize_code,
    repair_plan,
    validate_semesters,
)


def _course(code, credits=3, prereqs=(), groups=(), coreqs=(), terms=("fall", "spring")):
    return Course(
        code=code, title=f"{code} title", credits=credits,
        prereqs=list(prereqs), prereq_groups=[list(g) for g in groups],
        coreqs=list(coreqs), offered_terms=list(terms),
    )


CATALOG = [
    _course("CS 18000", credits=4),
    _course("CS 18200", groups=[["CS 18000"]]),
    _course("CS 24000", groups=[["CS 18000"]]),
    _course("CS 25100", groups=[["CS 18200"], ["CS 24000"]]),
    _course("CS 37300", groups=[["CS 25100"]], coreqs=["STAT 35000"]),
    _course("STAT 35000"),
    _course("CS 43900", terms=("fall",)),          # fall only — the trap the eval kept hitting
    _course("MA 16100", credits=5),
    _course("MA 16500", credits=5),                # approved substitute for MA 16100
    _course("ENGL 10600"),
]

# MA 16500 counts as MA 16100 everywhere: for the requirement, for the prerequisite, and for
# "you already took this".
CANONICAL = {c.code: c.code for c in CATALOG} | {"MA 16500": "MA 16100"}


def _profile(**overrides) -> StudentProfile:
    base = {
        "profile_label": "Test", "completed_courses": [], "remaining_courses": [],
        "start_term": "fall", "start_year": 2026, "semesters_to_plan": 4,
        "max_credits_per_semester": 15, "major_subject": "CS",
        "max_major_courses_per_semester": 3, "preferred_major_courses_per_semester": 3,
    }
    return StudentProfile(**{**base, **overrides})


def test_a_prerequisite_in_the_same_semester_is_not_early_enough():
    check = validate_semesters([["CS 18000", "CS 18200"]], _profile(), CATALOG)
    kinds = [v.kind for v in check.hard]
    assert kinds == ["prereq_violation"]
    assert "CS 18000" in check.hard[0].detail


def test_an_or_group_is_satisfied_by_either_option():
    catalog = [*CATALOG, _course("CS 49000", groups=[["CS 25100", "CS 37300"]])]
    check = validate_semesters(
        [["CS 18000"], ["CS 18200", "CS 24000"], ["CS 25100"], ["CS 49000"]],
        _profile(), catalog,
    )
    assert check.viable, [str(v) for v in check.hard]


def test_a_corequisite_may_share_the_semester_but_a_prerequisite_may_not():
    plan = [["CS 18000"], ["CS 18200", "CS 24000"], ["CS 25100"], ["CS 37300", "STAT 35000"]]
    assert validate_semesters(plan, _profile(), CATALOG).viable

    plan[3] = ["CS 37300"]        # STAT 35000 nowhere — the coreq is now unmet
    kinds = [v.kind for v in validate_semesters(plan, _profile(), CATALOG).hard]
    assert kinds == ["coreq_violation"]


def test_an_approved_substitute_is_the_same_course_for_every_check():
    profile = _profile(completed_courses=["MA 16500"])
    # Scheduling the primary as well is scheduling one course twice.
    check = validate_semesters([["MA 16100"]], profile, CATALOG, canonical=CANONICAL)
    assert [v.kind for v in check.hard] == ["duplicate_course"]


def test_the_students_own_cap_is_soft_and_the_registrars_is_hard():
    # 15 credits with a 9-credit request: over what they asked for, under what the university
    # allows. A plan they can register for and merely did not ask for is not a broken plan.
    profile = _profile(max_credits_per_semester=9)
    check = validate_semesters([["MA 16100", "MA 16500", "ENGL 10600", "CS 18200"]],
                               profile, CATALOG)
    assert not any(v.kind == "credit_cap_violation" for v in check.violations)
    assert any(v.kind == "over_requested_credits" and not v.hard for v in check.violations)

    over_the_registrar = [["MA 16100", "MA 16500", "ENGL 10600", "CS 18200", "CS 24000",
                           "STAT 35000"]]
    check = validate_semesters(over_the_registrar, _profile(), CATALOG)
    assert sum(check.semester_credits) > HARD_CREDIT_CAP
    assert any(v.kind == "credit_cap_violation" and v.hard for v in check.violations)


def test_a_term_restricted_course_cannot_be_scheduled_in_the_wrong_season():
    profile = _profile(start_term="spring", start_year=2027)
    check = validate_semesters([["CS 43900"]], profile, CATALOG)
    assert [v.kind for v in check.hard] == ["term_offering_violation"]


# --- repair --------------------------------------------------------------------------------


def test_repair_places_a_forgotten_prerequisite_instead_of_deleting_the_chain():
    """The failure that motivated move-before-delete.

    A draft that opens with CS 18200/CS 24000 and never schedules CS 18000 is one mistake, not
    five. A delete-only repair took the whole computer-science chain with it; this asserts the
    chain survives and the missing course is added.
    """
    draft = [["CS 18200", "CS 24000"], ["CS 25100"], [], []]
    profile = _profile(remaining_courses=["CS 18000", "CS 18200", "CS 24000", "CS 25100"])

    plan, removed, added, check = repair_plan(draft, profile, CATALOG)

    assert check.viable, [str(v) for v in check.hard]
    assert removed == [], "nothing should have been deleted — everything was rescuable"
    assert "CS 18000" in added
    scheduled = [code for term in plan for code in term]
    assert {"CS 18000", "CS 18200", "CS 24000", "CS 25100"} <= set(scheduled)

    # CS 18000 lands strictly before the courses that need it.
    def semester_of(code):
        return next(i for i, term in enumerate(plan) if code in term)

    assert semester_of("CS 18000") < semester_of("CS 18200")
    assert semester_of("CS 18000") < semester_of("CS 24000")


def test_repair_moves_a_course_out_of_a_term_it_is_not_offered_in():
    profile = _profile(start_term="fall", start_year=2026,
                       remaining_courses=["CS 43900"])
    # Semester 2 is a spring; CS 43900 is fall-only, so it belongs in semester 3.
    plan, removed, _added, check = repair_plan([[], ["CS 43900"], [], []], profile, CATALOG)
    assert check.viable
    assert removed == []
    assert "CS 43900" in plan[2]


def test_a_duplicate_keeps_the_EARLIER_placement_and_drops_the_later_one():
    """MA 26100 listed twice leaves the student with one MA 26100, in the first slot it had.

    Which copy survives is not arbitrary. The earlier one is the one the rest of the plan was
    built around — anything scheduled after it may depend on it — so dropping the earlier copy
    would turn one duplicate into a cascade of prerequisite violations. Keeping the first
    occurrence makes the removal a no-op for every other course.
    """
    catalog = [*CATALOG, _course("MA 26100", credits=4)]
    profile = _profile(remaining_courses=["MA 26100", "CS 18000", "ENGL 10600"])

    plan, removed, _added, check = repair_plan(
        [["MA 26100", "CS 18000"], ["ENGL 10600"], ["MA 26100"], []], profile, catalog
    )

    assert removed == ["MA 26100"]
    assert plan[0] == ["MA 26100", "CS 18000"], "the first placement is untouched"
    assert plan[2] == [], "the later copy is the one that goes"
    assert sum(term.count("MA 26100") for term in plan) == 1
    assert check.viable


def test_a_duplicate_inside_one_semester_leaves_a_single_copy():
    catalog = [*CATALOG, _course("MA 26100", credits=4)]
    profile = _profile(remaining_courses=["MA 26100", "CS 18000"])
    plan, removed, _added, check = repair_plan(
        [["MA 26100", "MA 26100", "CS 18000"], [], [], []], profile, catalog
    )
    assert plan[0] == ["MA 26100", "CS 18000"]
    assert removed == ["MA 26100"]
    assert check.viable


def test_a_course_the_student_already_completed_is_dropped_from_the_plan():
    """Same class of removal: scheduling it again wastes a seat they are paying for."""
    profile = _profile(completed_courses=["CS 18000"], remaining_courses=["CS 18200"])
    plan, removed, _added, check = repair_plan(
        [["CS 18000"], ["CS 18200"], [], []], profile, CATALOG
    )
    assert removed == ["CS 18000"]
    assert plan[0] == []
    assert plan[1] == ["CS 18200"], "the completed course still satisfies the prerequisite"
    assert check.viable


def test_repair_deletes_what_it_cannot_rescue():
    # A code the catalog has never heard of is wrong in every term, so there is nowhere to move
    # it to and deletion is the only honest outcome.
    plan, removed, _added, check = repair_plan([["CS 99999"], [], [], []], _profile(), CATALOG)
    assert removed == ["CS 99999"]
    assert check.viable
    assert plan == [[], [], [], []]


def test_repair_never_leaves_hard_violations_it_could_fix():
    messy = [["CS 25100", "CS 37300"], ["CS 18000"], ["CS 18200", "CS 18200"], []]
    profile = _profile(remaining_courses=["CS 18000", "CS 18200", "CS 24000", "CS 25100",
                                          "CS 37300", "STAT 35000"])
    _plan, _removed, _added, check = repair_plan(messy, profile, CATALOG)
    assert check.viable, [str(v) for v in check.hard]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("cs25100", "CS 25100"), ("CS-25100", "CS 25100"), ("CS  25100", "CS 25100"),
     ("ENGL", "ENGL"), ("25100", "25100")],
)
def test_normalize_code(raw, expected):
    assert normalize_code(raw) == expected
