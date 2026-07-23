from app.core.config import settings
from app.models.schemas import Course, PlannedCourse, PlanResponse, SemesterPlan, StudentProfile

# Every term the academic calendar has. Order matters: it defines what "the next term" is.
ALL_TERMS = ["fall", "spring", "summer"]


def planning_terms() -> list[str]:
    """The terms the planner is allowed to schedule into.

    Summer is excluded by default (``PLANNER_INCLUDE_SUMMER=false``). It is a real term and
    the catalog offers courses in it, but summer enrolment carries cost, aid and residency
    consequences a course planner has no way to reason about, so recommending it unprompted
    is worse than omitting it. The switch exists because summer IS planned for later; when it
    comes back it should come back as a deliberate student choice, not a silent default.

    Note this only constrains SCHEDULING. Offerings data still records summer availability,
    and a course whose only offering is summer will simply go unplanned with a warning —
    which is the honest outcome, not a hidden one.
    """
    if settings.planner_include_summer:
        return list(ALL_TERMS)
    return [term for term in ALL_TERMS if term != "summer"]


# Kept as a module-level name because callers import it (planner_catalog, plan_editor).
TERM_ORDER = planning_terms()


def next_term(term: str, year: int) -> tuple[str, int]:
    """Advance one academic term within the planning calendar.

    With summer excluded: fall 2026 -> spring 2027 -> fall 2027.
    With summer included: fall 2026 -> spring 2027 -> summer 2027 -> fall 2027.
    The year rolls at spring, which is what makes fall the start of an academic year.
    """
    terms = planning_terms()
    idx = terms.index(term) if term in terms else -1
    nxt = terms[(idx + 1) % len(terms)]
    return nxt, year + 1 if nxt == "spring" else year


def first_planning_term(term: str, year: int) -> tuple[str, int]:
    """Snap a start term onto the planning calendar.

    A student whose profile says they start in summer still has to be given a plan; with
    summer excluded they simply start that fall. Silently scheduling INTO summer would be the
    bug, so the snap happens once, here, rather than in each caller."""
    return (term, year) if term in planning_terms() else next_term(term, year)


def prereqs_satisfied(course: Course, completed: set[str]) -> bool:
    return all(prereq in completed for prereq in course.prereqs)


def generate_plan(profile: StudentProfile, catalog: list[Course]) -> PlanResponse:
    catalog_by_code = {course.code: course for course in catalog}
    completed = set(profile.completed_courses)
    remaining = list(profile.remaining_courses)
    warnings: list[str] = []
    semesters: list[SemesterPlan] = []

    unknown_courses = [code for code in remaining if code not in catalog_by_code]
    if unknown_courses:
        warnings.append(f"Unknown courses ignored: {', '.join(unknown_courses)}")
        remaining = [code for code in remaining if code in catalog_by_code]

    # Treat the order of ``remaining_courses`` as the student's (or the advisor agent's)
    # preference order: courses listed earlier are scheduled earlier when several are legal in
    # the same term. This is the knob the LFM2 revise-plan agent turns via reorder/defer — the
    # planner still guarantees legality (prereqs, term offerings, credit cap) regardless.
    priority = {code: index for index, code in enumerate(remaining)}

    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)

    for _ in range(profile.semesters_to_plan):
        selected: list[Course] = []
        selected_credits = 0
        semester_warnings: list[str] = []

        candidates = []
        for code in remaining:
            course = catalog_by_code[code]
            if term not in course.offered_terms:
                continue
            if not prereqs_satisfied(course, completed):
                continue
            candidates.append(course)

        candidates.sort(
            key=lambda c: (
                priority.get(c.code, len(priority)),
                "required" not in c.requirement_tags,
                c.workload_score,
                c.code,
            )
        )

        for course in candidates:
            if selected_credits + course.credits <= profile.max_credits_per_semester:
                selected.append(course)
                selected_credits += course.credits

        if not selected and remaining:
            blocked_reasons = []
            for code in remaining:
                course = catalog_by_code[code]
                reasons = []
                missing = [p for p in course.prereqs if p not in completed]
                if missing:
                    reasons.append(f"missing prereqs {missing}")
                if term not in course.offered_terms:
                    reasons.append(f"not offered in {term}")
                if reasons:
                    blocked_reasons.append(f"{code}: {', '.join(reasons)}")
            if blocked_reasons:
                semester_warnings.append(
                    "No courses selected this term. Blocked courses: " + "; ".join(blocked_reasons)
                )

        planned_codes = {course.code for course in selected}
        completed.update(planned_codes)
        remaining = [code for code in remaining if code not in planned_codes]

        avg_workload = (
            round(sum(course.workload_score for course in selected) / len(selected), 2)
            if selected
            else 0.0
        )

        semesters.append(
            SemesterPlan(
                term=term,
                year=year,
                courses=[
                    PlannedCourse(
                        code=course.code,
                        title=course.title,
                        credits=course.credits,
                        workload_score=course.workload_score,
                    )
                    for course in selected
                ],
                total_credits=selected_credits,
                average_workload=avg_workload,
                warnings=semester_warnings,
            )
        )

        if not remaining:
            break

        term, year = next_term(term, year)

    if remaining:
        warnings.append(
            "Some courses could not be planned within the requested number of semesters: "
            + ", ".join(remaining)
        )

    return PlanResponse(
        student_name=profile.name,
        degree_program=profile.degree_program,
        semesters=semesters,
        unplanned_courses=remaining,
        warnings=warnings,
    )
