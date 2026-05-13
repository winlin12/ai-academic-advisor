from app.models.schemas import Course, PlannedCourse, PlanResponse, SemesterPlan, StudentProfile

TERM_ORDER = ["fall", "spring", "summer"]


def next_term(term: str, year: int) -> tuple[str, int]:
    idx = TERM_ORDER.index(term)
    if idx == len(TERM_ORDER) - 1:
        return TERM_ORDER[0], year + 1
    return TERM_ORDER[idx + 1], year


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

    term = profile.start_term.lower()
    year = profile.start_year

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
