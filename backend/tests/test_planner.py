from app.models.schemas import StudentProfile
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
