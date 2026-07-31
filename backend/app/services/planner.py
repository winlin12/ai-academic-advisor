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
    """Conservative on purpose: corequisites are required UP FRONT here.

    A coreq may legally be taken alongside its course, but making the greedy planner exploit
    that needs it to reason about pairs it has not selected yet. Requiring it earlier is always
    legal — never a registration wall — at the cost of occasionally scheduling a term later than
    strictly necessary. Validation of a plan the student built themselves applies the true
    same-semester rule, so a hand-edited plan is not flagged for something legal.
    """
    required = set(course.prereqs) | set(course.coreqs)
    return all(code in completed for code in required)


def is_major_course(code: str, subject: str) -> bool:
    """Does this course code belong to the major's subject ("CS 25100" -> CS)?

    Subject prefix rather than requirement tags: ``required`` is carried by MA 26100 and
    PHYS 17200 as much as by CS 25100, and it is specifically the department's own courses
    that stack into an unmanageable term. Matched on the leading alphabetic run, so "CS25100",
    "CS 25100" and "cs 25100" all count.
    """
    if not subject:
        return False
    letters = "".join(ch for ch in str(code) if not ch.isdigit()).strip().upper()
    return letters.replace("-", "").replace(" ", "") == subject.strip().upper()


def semester_credit_target(
    profile: StudentProfile,
    remaining: list[str],
    catalog_by_code: dict[str, Course],
    semester_index: int,
) -> int:
    """Credits to aim for in THIS semester. Never above ``max_credits_per_semester``.

    An explicit ``target_credits_per_semester`` is honoured as-is (clamped). Otherwise the
    target is the even split of what is left over the terms that are left, which is what makes
    a plan spread across the student's horizon instead of front-loading to the cap and leaving
    the tail empty. Recomputing it every term is what lets it self-correct: a term that cannot
    reach the target raises the bar for the terms after it, so the work still lands inside the
    horizon rather than piling into a stub semester at the end.

    Both degenerate cases fall back to the cap, i.e. the pre-2026-07-29 behaviour: no semesters
    left to divide by, and nothing left to place.
    """
    explicit = profile.target_credits_per_semester
    if explicit:
        return min(int(explicit), profile.max_credits_per_semester)
    semesters_left = profile.semesters_to_plan - semester_index
    if semesters_left <= 0:
        return profile.max_credits_per_semester
    credits_left = sum(
        catalog_by_code[code].credits for code in remaining if code in catalog_by_code
    )
    if credits_left <= 0:
        return profile.max_credits_per_semester
    even_split = -(-credits_left // semesters_left)  # integer ceiling
    return min(even_split, profile.max_credits_per_semester)


def generate_plan(
    profile: StudentProfile, catalog: list[Course], *, _spread: bool = True
) -> PlanResponse:
    """Build a legal plan, spreading the load across the horizon where that is free.

    SPREAD FIRST, PACK IF IT DOES NOT FIT. Aiming at ``target_credits_per_semester`` instead of
    the cap leaves capacity unused in early terms, and that capacity does not come back: when a
    prerequisite chain opens up late, the courses it was holding have nowhere to go and end up
    unplanned. MEASURED in model_eval — a sophomore profile with the CS track reordered to the
    front placed 55 credits as [11, 9, 10, 9, 9, 7] under a 15-credit cap and stranded two
    required electives; the same profile filling to the cap placed 61 as [14, 13, 15, 13, 6]
    and stranded nothing.

    It bites specifically when ``remaining_courses`` has been REORDERED, which is what the
    revise-plan agent does — so without this fallback a student's stated preference could
    silently drop a required course from their plan. Coverage is never the price of a nicer
    distribution: a spread plan that strands more than the packed one is discarded.
    ``_spread`` is the internal re-entry for that second attempt, not part of the API.
    """
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

    for semester_index in range(profile.semesters_to_plan):
        selected: list[Course] = []
        selected_credits = 0
        semester_warnings: list[str] = []
        # Recomputed EVERY term, not once: `remaining` shrinks as courses are placed and the
        # horizon shrinks with it, so a term that came up short automatically raises the bar
        # for the terms after it. Clamped by the cap, which is still the only hard bound.
        credit_target = (
            semester_credit_target(profile, remaining, catalog_by_code, semester_index)
            if _spread
            else profile.max_credits_per_semester
        )

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

        # TWO PASSES over the same candidate list, differing only in the major-course ceiling
        # they allow. Pass one holds the soft limit (``preferred_major_courses_per_semester``),
        # so everything else the student could be taking gets first claim on the room; pass two
        # raises it to the hard limit and admits a further major course only into room nothing
        # else filled. The result is "at most `preferred` CS courses unless the semester would
        # otherwise go short, and never more than `max`" — with no second sort, so the priority
        # order the revise-plan agent controls still decides which courses those are.
        #
        # A credit cap alone cannot express this: four 4-credit CS courses and one CS course
        # plus three gen-eds are both 16 credits and nothing like the same term. Before this
        # existed the planner produced 4- and 5-CS semesters, which is not a plan a student
        # would follow.
        chosen: set[str] = set()
        selected_majors = 0
        soft_major_limit = min(
            profile.preferred_major_courses_per_semester,
            profile.max_major_courses_per_semester,
        )
        for major_limit in (soft_major_limit, profile.max_major_courses_per_semester):
            for course in candidates:
                if course.code in chosen:
                    continue
                # `credit_target`, not `max_credits_per_semester`: this is the line that
                # turns a front-loaded plan into a spread one. The cap still exists — the
                # target is clamped to it — but it is no longer what the greedy fill aims at.
                if selected_credits + course.credits > credit_target:
                    continue
                if is_major_course(course.code, profile.major_subject):
                    if selected_majors >= major_limit:
                        continue
                    selected_majors += 1
                chosen.add(course.code)
                selected_credits += course.credits
        # Emitted in candidate (priority) order rather than the order the passes admitted them,
        # so a semester reads the same way it always has.
        selected = [course for course in candidates if course.code in chosen]

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

    plan = PlanResponse(
        student_name=profile.name,
        degree_program=profile.degree_program,
        semesters=semesters,
        unplanned_courses=remaining,
        warnings=warnings,
    )
    # The fallback. Only ever taken when spreading actually cost coverage, so a plan can never
    # come out worse than it would have before the credit target existed.
    if _spread and plan.unplanned_courses:
        packed = generate_plan(profile, catalog, _spread=False)
        if len(packed.unplanned_courses) < len(plan.unplanned_courses):
            return packed
    return plan
