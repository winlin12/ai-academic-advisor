from app.models.schemas import Course, StudentProfile
from app.services.catalog import load_catalog
from app.services.planner import generate_plan


def test_generate_plan_has_semesters():
    profile = StudentProfile(
        completed_courses=["CS251"],
        remaining_courses=["CS381", "CS502", "CS536"],
        start_term="fall",
        start_year=2026,
        semesters_to_plan=3,
        max_credits_per_semester=6,
    )

    plan = generate_plan(profile, load_catalog())

    assert len(plan.semesters) >= 1
    planned_codes = {
        course.code
        for semester in plan.semesters
        for course in semester.courses
    }
    assert "CS381" in planned_codes
    assert "CS536" in planned_codes


def test_next_term_crosses_calendar_year_at_spring():
    from app.services.planner import next_term

    assert next_term("fall", 2026) == ("spring", 2027)
    assert next_term("spring", 2027) == ("summer", 2027)
    assert next_term("summer", 2027) == ("fall", 2027)


def test_unknown_course_warning():
    profile = StudentProfile(
        completed_courses=[],
        remaining_courses=["FAKE101"],
        start_term="fall",
        start_year=2026,
        semesters_to_plan=1,
        max_credits_per_semester=6,
    )

    plan = generate_plan(profile, load_catalog())

    assert any("Unknown courses" in warning for warning in plan.warnings)


# --- major-course load cap ---------------------------------------------------------------
#
# A credit cap alone cannot say "not four CS courses at once": four 4-credit CS courses and
# one CS course plus three gen-eds are both 16 credits and nothing like the same semester.
# These pin the two-limit behaviour, because the failure is silent — a plan that breaks it
# still looks complete and still passes every other check.


def _load_catalog(codes: list[str]) -> list[Course]:
    """A flat catalog: everything 3 credits, no prereqs, offered every term.

    Deliberately unconstrained so the ONLY thing that can shape a semester is the two load
    caps — with prereqs or term offerings in play a passing test would not tell you which
    rule did the work.
    """
    return [
        Course(code=code, title=code, credits=3, prereqs=[],
               offered_terms=["fall", "spring", "summer"], requirement_tags=["required"])
        for code in codes
    ]


def _profile(codes: list[str], **overrides) -> StudentProfile:
    return StudentProfile(
        completed_courses=[],
        remaining_courses=codes,
        start_term="fall",
        start_year=2026,
        semesters_to_plan=overrides.pop("semesters_to_plan", 4),
        max_credits_per_semester=overrides.pop("max_credits_per_semester", 18),
        **overrides,
    )


def _codes_by_semester(plan) -> list[list[str]]:
    return [[course.code for course in semester.courses] for semester in plan.semesters]


def test_major_courses_stop_at_the_preferred_limit_when_others_can_fill_the_term():
    # target pinned to the cap so the MAJOR cap is the only thing shaping the term — otherwise
    # the credit target spreads these 8 courses out and the assertion below would be measuring
    # distribution rather than the major limit.
    codes = ["CS101", "CS102", "CS103", "CS104", "MA101", "MA102", "ENGL101", "PHYS101"]
    plan = generate_plan(_profile(codes, target_credits_per_semester=18),
                         _load_catalog(codes))

    first = _codes_by_semester(plan)[0]
    assert len([c for c in first if c.startswith("CS")]) == 2, first
    # ...and the room the third CS course would have taken went to a requirement, rather than
    # the semester simply coming up short.
    assert len(first) == 6, first


def test_major_courses_reach_the_hard_limit_only_when_nothing_else_fits():
    # CS-only palette: there is nothing non-major to fill the term with, so the planner is
    # allowed to go past the preferred 2 — but not past the hard 3.
    codes = ["CS101", "CS102", "CS103", "CS104", "CS105", "CS106"]
    plan = generate_plan(_profile(codes, target_credits_per_semester=18),
                         _load_catalog(codes))

    per_semester = _codes_by_semester(plan)
    assert per_semester[0] == ["CS101", "CS102", "CS103"], per_semester
    assert all(len(s) <= 3 for s in per_semester), per_semester


def test_major_cap_is_never_exceeded_even_with_credits_to_spare():
    # 18 credits of room, six 3-credit CS courses available: the credit cap alone would take
    # all six. The major cap is what stops it at three.
    codes = ["CS101", "CS102", "CS103", "CS104", "CS105", "CS106"]
    plan = generate_plan(_profile(codes, max_credits_per_semester=18), _load_catalog(codes))

    for semester in plan.semesters:
        assert semester.total_credits <= 9, _codes_by_semester(plan)


def test_major_cap_is_per_student_and_can_be_lowered():
    codes = ["CS101", "CS102", "CS103", "MA101", "MA102", "MA103"]
    profile = _profile(codes, max_major_courses_per_semester=1,
                       preferred_major_courses_per_semester=1)
    plan = generate_plan(profile, _load_catalog(codes))

    for semester in _codes_by_semester(plan):
        assert len([c for c in semester if c.startswith("CS")]) <= 1, semester


def test_non_major_subjects_are_not_capped():
    codes = ["MA101", "MA102", "MA103", "MA104", "MA105"]
    plan = generate_plan(_profile(codes, max_credits_per_semester=18,
                                  target_credits_per_semester=18), _load_catalog(codes))

    assert len(_codes_by_semester(plan)[0]) == 5, _codes_by_semester(plan)


def test_is_major_course_matches_both_code_spellings():
    from app.services.planner import is_major_course

    assert is_major_course("CS 25100", "CS")
    assert is_major_course("CS25100", "CS")
    assert is_major_course("cs 25100", "cs")
    assert not is_major_course("MA 26100", "CS")
    # A subject that merely starts with the same letters is a different department.
    assert not is_major_course("CSR 30000", "CS")


# --- credit target vs credit cap ----------------------------------------------------------
#
# The cap says what a semester may never exceed; the target says what it should aim for.
# Filling to the cap is what front-loads a plan and leaves the student's last terms empty.


def test_credits_spread_across_the_whole_horizon_by_default():
    # 8 courses x 3 credits = 24 credits, 4 semesters, an 18-credit cap. Filling to the cap
    # would finish in 2 semesters and leave 2 empty; the derived target is 6 a term.
    codes = [f"MA10{i}" for i in range(8)]
    plan = generate_plan(_profile(codes, semesters_to_plan=4, max_credits_per_semester=18),
                         _load_catalog(codes))

    per_semester = [s.total_credits for s in plan.semesters]
    assert per_semester == [6, 6, 6, 6], per_semester
    assert all(s.total_credits > 0 for s in plan.semesters), "no semester should be left empty"


def test_explicit_target_is_honoured_and_clamped_by_the_cap():
    codes = [f"MA10{i}" for i in range(8)]
    catalog = _load_catalog(codes)

    pinned = generate_plan(
        _profile(codes, semesters_to_plan=8, max_credits_per_semester=18,
                 target_credits_per_semester=9), catalog)
    assert [s.total_credits for s in pinned.semesters][:2] == [9, 9]

    # A target above the cap must not raise the effective limit.
    clamped = generate_plan(
        _profile(codes, semesters_to_plan=4, max_credits_per_semester=6,
                 target_credits_per_semester=18), catalog)
    assert all(s.total_credits <= 6 for s in clamped.semesters)


def test_target_equal_to_cap_restores_front_loading():
    codes = [f"MA10{i}" for i in range(8)]
    plan = generate_plan(
        _profile(codes, semesters_to_plan=4, max_credits_per_semester=18,
                 target_credits_per_semester=18), _load_catalog(codes))

    per_semester = [s.total_credits for s in plan.semesters]
    assert per_semester[0] == 18, per_semester      # packed, not spread
    assert len(plan.semesters) < 4, per_semester    # and it finishes early


def test_target_self_corrects_when_a_term_cannot_be_filled():
    # CS200 is spring-only and needs CS100, so the fall terms cannot reach an even split.
    # The target must rise for the later terms rather than stranding courses unplanned.
    catalog = [
        Course(code="CS100", title="a", credits=3, offered_terms=["fall", "spring"]),
        Course(code="CS200", title="b", credits=3, prereqs=["CS100"], offered_terms=["spring"]),
        Course(code="MA100", title="c", credits=3, offered_terms=["fall", "spring"]),
    ]
    plan = generate_plan(
        _profile(["CS100", "CS200", "MA100"], semesters_to_plan=2,
                 max_credits_per_semester=12), catalog)

    assert not plan.unplanned_courses, plan.unplanned_courses


def test_spreading_never_costs_coverage():
    """A reorder must not be able to strand a course that would otherwise have been planned.

    The credit target throttles early terms, and that capacity does not come back — so when a
    prerequisite chain opens up late there can be nowhere left to put what it was holding. This
    is the regression that took model_eval's Mode A viability from 100% to 50%, and it only
    appeared once ``remaining_courses`` was REORDERED, which is precisely what the revise-plan
    agent does to it.
    """
    # A 3-long chain plus loose electives. Reordered so the chain comes first, which is what
    # starves the early terms: only the head of the chain is eligible in semester one.
    catalog = [
        Course(code="CS100", title="a", credits=4, offered_terms=["fall", "spring"]),
        Course(code="CS200", title="b", credits=4, prereqs=["CS100"],
               offered_terms=["fall", "spring"]),
        Course(code="CS300", title="c", credits=4, prereqs=["CS200"],
               offered_terms=["fall", "spring"]),
        *[Course(code=f"MA{i}0", title="m", credits=4, offered_terms=["fall", "spring"])
          for i in range(1, 5)],
    ]
    codes = ["CS100", "CS200", "CS300", "MA10", "MA20", "MA30", "MA40"]
    profile = _profile(codes, semesters_to_plan=3, max_credits_per_semester=12)

    plan = generate_plan(profile, catalog)

    assert not plan.unplanned_courses, (
        f"spreading stranded {plan.unplanned_courses}; the packed fallback should have won"
    )


def test_spread_is_kept_when_it_does_not_strand_anything():
    # The fallback must not fire gratuitously — a plan that fits while spread stays spread.
    codes = [f"MA10{i}" for i in range(8)]
    plan = generate_plan(_profile(codes, semesters_to_plan=4, max_credits_per_semester=18),
                         _load_catalog(codes))

    assert [s.total_credits for s in plan.semesters] == [6, 6, 6, 6]
