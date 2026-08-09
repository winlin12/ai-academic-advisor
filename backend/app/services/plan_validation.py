"""Is this schedule one a student could actually register for?

PORT OF ``model_eval/harness/plan_scorers.score_plan``, which is the harness's headline
instrument, onto the serving path. The eval needed it to score models; the app needs it for a
harder reason — Mode B lets the model emit the schedule directly, so this is the only thing
standing between a plausible-looking JSON object and a student registering for a course whose
prerequisite they have not taken.

The split between hard and soft is the eval's and is deliberate:

    HARD — a registration wall or a degree that cannot finish. Any one of these means the plan
    is not viable, and ``repair_plan`` deletes the placement rather than shipping it.

        hallucinated_course      a code the program's catalog does not contain
        prereq_violation         a prerequisite not complete BEFORE this semester
        coreq_violation          a corequisite neither complete nor in the same semester
        term_offering_violation  scheduled in a term the course is not taught
        credit_cap_violation     over the REGISTRAR's ceiling (18), not the student's ask
        duplicate_course         the same course twice, or one already completed
        major_overload_violation more major-subject courses in a term than the cap allows

    SOFT — recorded, never repaired. Over what the student ASKED for but under what the
    university allows: a term they can register for and merely did not request. Charging these
    as hard would let the repair step delete a legal course to satisfy a preference.

WHY REPAIR RATHER THAN RETRY. The eval's Mode C measured both (harness/convergence.py) and the
repair variant is the one that works: deleting a violating placement is decidable without a
model, costs no GPU time, and never makes the plan worse — while asking the model to fix its
own plan costs a full regeneration and, on the smaller models, often traded one violation for
another. So the model gets one shot at the schedule, the checker deletes what is illegal, and
what survives is legal by construction. Whatever was deleted comes back as `unplanned_courses`
with the reason, which is the honest outcome and the one the student can act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.schemas import Course, StudentProfile
from app.services.planner import first_planning_term, is_major_course, next_term

# The registrar's ceiling. A term above the student's own number but under this is a plan they
# CAN register for and merely did not ask for — scored soft, never repaired.
HARD_CREDIT_CAP = 18

_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?-?\s?(\d{3,5})\b")


def normalize_code(code: str) -> str:
    """'cs25100', 'CS-25100' -> 'CS 25100'. Purely lexical.

    Models are not graded on spacing. A three-digit legacy code is left as-is and will simply
    fail the catalog lookup, which IS the right outcome — 'CS 251' is not a code the current
    catalog can register anyone for.
    """
    compact = "".join(str(code).split()).upper().replace("-", "")
    for index, char in enumerate(compact):
        if char.isdigit():
            return compact if index == 0 else f"{compact[:index]} {compact[index:]}"
    return compact


@dataclass
class Violation:
    kind: str
    detail: str
    semester_index: int
    course_code: str | None = None
    # False for the soft classes, which are reported to the student but never repaired.
    hard: bool = True

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclass
class PlanValidation:
    violations: list[Violation] = field(default_factory=list)
    semester_credits: list[int] = field(default_factory=list)
    major_courses_per_semester: list[int] = field(default_factory=list)
    planned_credits: int = 0

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.hard]

    @property
    def soft(self) -> list[Violation]:
        return [v for v in self.violations if not v.hard]

    @property
    def viable(self) -> bool:
        """Zero HARD violations. Requirement coverage is a separate question, answered by
        ``planner_db.requirement_coverage`` — a legal plan that omits a required course is
        legal and incomplete, and conflating the two hides which one went wrong."""
        return not self.hard

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for violation in self.violations:
            out[violation.kind] = out.get(violation.kind, 0) + 1
        return out


def term_sequence(profile: StudentProfile, count: int) -> list[tuple[str, int]]:
    """The calendar a plan's Nth semester actually falls in.

    Snaps the start term exactly as ``planner.generate_plan`` does. When these two disagree the
    checker grades semester N against the wrong term and reports term-offering violations for a
    schedule that is perfectly legal — which is what happened in the eval the first time summer
    was switched off.
    """
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(count):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def validate_semesters(
    semesters: list[list[str]],
    profile: StudentProfile,
    catalog: list[Course],
    *,
    canonical: dict[str, str] | None = None,
) -> PlanValidation:
    """Check a schedule expressed as one course-code list per semester.

    ``canonical`` collapses approved substitutes onto the course they stand in for BEFORE any
    check runs, so all three checks that care agree by construction: MA 16500 satisfies a
    prerequisite naming MA 16100, counts toward the group listing MA 16100, and scheduling both
    is scheduling one course twice. Doing it in only one of them is how you get a plan that
    covers the degree but still reports a prereq violation.
    """
    canonical = canonical or {}
    by_code = {course.code: course for course in catalog}
    result = PlanValidation()

    def canon(code: str) -> str:
        return canonical.get(code, code)

    completed = {canon(normalize_code(c)) for c in profile.completed_courses}
    seen: set[str] = set()
    terms = term_sequence(profile, len(semesters))
    soft_major_cap = min(profile.preferred_major_courses_per_semester,
                         profile.max_major_courses_per_semester)

    for index, (raw_codes, (term, year)) in enumerate(zip(semesters, terms)):
        codes = [normalize_code(code) for code in raw_codes]
        this_semester = {canon(code) for code in codes}
        credits = 0
        major_courses = 0
        # Prerequisites are evaluated against courses complete BEFORE this semester: a course
        # taken alongside its own prerequisite is a scheduling error, not a corequisite.
        available = set(completed)

        for code in codes:
            course = by_code.get(code)
            if course is None:
                result.violations.append(Violation(
                    "hallucinated_course",
                    f"{code} (semester {index + 1}) is not in this program's catalog",
                    index, code))
                continue

            key = canon(code)
            alias = f" (counts as {key})" if key != code else ""
            if key in seen:
                result.violations.append(Violation(
                    "duplicate_course", f"{code}{alias} appears more than once",
                    index, code))
            elif key in completed:
                result.violations.append(Violation(
                    "duplicate_course",
                    f"{code}{alias} was already completed before the plan starts",
                    index, code))
            seen.add(key)

            credits += course.credits
            if is_major_course(code, profile.major_subject):
                major_courses += 1

            if course.offered_terms and term not in course.offered_terms:
                result.violations.append(Violation(
                    "term_offering_violation",
                    f"{code} is placed in {term} {year} but is offered in "
                    f"{'/'.join(course.offered_terms)}",
                    index, code))

            # AND-of-ORs when the catalog has it; the flat list otherwise. A group is satisfied
            # by ANY of its members, so reading the flat union instead would demand every
            # alternative and charge a violation for a legal plan.
            groups = course.prereq_groups or [[code] for code in course.prereqs]
            unmet = [group for group in groups
                     if not any(canon(normalize_code(p)) in available for p in group)]
            if unmet:
                needs = ", ".join(" or ".join(group) for group in unmet)
                result.violations.append(Violation(
                    "prereq_violation",
                    f"{code} in semester {index + 1} needs {needs} in an earlier semester",
                    index, code))

            # COREQUISITES take the looser reading: same semester counts. The checker's job is
            # to not charge a violation for a schedule the registrar would accept, and a coreq
            # alongside its course is exactly that. In the eval, getting this wrong made
            # CS 37300/STAT 35000 15% of every prereq violation in the corpus.
            missing_co = [c for c in course.coreqs
                          if canon(normalize_code(c)) not in available | this_semester]
            if missing_co:
                result.violations.append(Violation(
                    "coreq_violation",
                    f"{code} in semester {index + 1} needs {', '.join(missing_co)} in the same "
                    f"semester or earlier",
                    index, code))

        result.semester_credits.append(credits)
        result.major_courses_per_semester.append(major_courses)

        if credits > HARD_CREDIT_CAP:
            result.violations.append(Violation(
                "credit_cap_violation",
                f"semester {index + 1} has {credits} credits; the university limit is "
                f"{HARD_CREDIT_CAP}",
                index))
        elif credits > profile.max_credits_per_semester:
            result.violations.append(Violation(
                "over_requested_credits",
                f"semester {index + 1} has {credits} credits; you asked for at most "
                f"{profile.max_credits_per_semester}",
                index, hard=False))

        if major_courses > profile.max_major_courses_per_semester:
            result.violations.append(Violation(
                "major_overload_violation",
                f"semester {index + 1} has {major_courses} {profile.major_subject} courses; "
                f"the limit is {profile.max_major_courses_per_semester}",
                index))
        elif major_courses > soft_major_cap:
            result.violations.append(Violation(
                "heavy_major_load",
                f"semester {index + 1} has {major_courses} {profile.major_subject} courses, "
                f"more than the {soft_major_cap} that is comfortable",
                index, hard=False))

        completed |= {canon(code) for code in codes if code in by_code}

    result.planned_credits = sum(result.semester_credits)
    return result


def _fits(
    semester: list[str],
    course: Course,
    profile: StudentProfile,
    by_code: dict[str, Course],
    term: str,
) -> bool:
    """Is there room for ``course`` in this term, and is it taught then?"""
    if course.offered_terms and term not in course.offered_terms:
        return False
    credits = sum(by_code[c].credits for c in semester if c in by_code)
    if credits + course.credits > profile.max_credits_per_semester:
        return False
    if is_major_course(course.code, profile.major_subject):
        majors = sum(1 for c in semester if is_major_course(c, profile.major_subject))
        if majors + 1 > profile.max_major_courses_per_semester:
            return False
    return True


def _place_missing_prerequisite(
    current: list[list[str]],
    course: Course,
    profile: StudentProfile,
    by_code: dict[str, Course],
    canonical: dict[str, str],
    completed: set[str],
    terms: list[tuple[str, int]],
) -> str | None:
    """Schedule a prerequisite of ``course`` that the plan never contained at all.

    MUTATES ``current`` and returns the code placed, or None when there was nothing missing to
    place or nowhere legal to put it. Only ever adds a course the student genuinely still needs
    — it is a prerequisite of something already in their plan — so this can never invent work.

    Goes in the EARLIEST term that has the room, the offering, and its own prerequisites
    already behind it. Earliest rather than "just before the course that needs it" because the
    dependent still has to move afterwards, and every term of slack helps.
    """
    scheduled = {canonical.get(c, c) for term in current for c in term}
    groups = course.prereq_groups or [[code] for code in course.prereqs]

    for group in groups:
        if any(canonical.get(normalize_code(c), normalize_code(c)) in scheduled | completed
               for c in group):
            continue
        # First option of the unmet group that the catalog actually has — the same "first
        # listed wins" rule requirement selection uses for alternatives.
        missing = next((by_code[normalize_code(c)] for c in group
                        if normalize_code(c) in by_code), None)
        if missing is None:
            continue
        for index in range(len(current)):
            behind = {canonical.get(c, c) for t in current[:index] for c in t} | completed
            own = missing.prereq_groups or [[c] for c in missing.prereqs]
            if not all(any(canonical.get(normalize_code(c), normalize_code(c)) in behind
                           for c in grp) for grp in own):
                continue
            if not _fits(current[index], missing, profile, by_code, terms[index][0]):
                continue
            current[index].append(missing.code)
            return missing.code
    return None


def _relocate_target(
    current: list[list[str]],
    index: int,
    course: Course,
    profile: StudentProfile,
    by_code: dict[str, Course],
    canonical: dict[str, str],
    completed: set[str],
    terms: list[tuple[str, int]],
) -> int | None:
    """The earliest LATER semester where ``course`` would be legal, or None.

    Only looks forward. Moving a course earlier can never fix a prerequisite problem, and a
    course that is legal earlier was not the thing violating anything.
    """
    for target in range(index + 1, len(current)):
        if course.offered_terms and terms[target][0] not in course.offered_terms:
            continue
        behind = {canonical.get(c, c) for t in current[:target] for c in t if t is not
                  current[index]}
        behind |= {canonical.get(c, c) for c in current[index]
                   if canonical.get(c, c) != canonical.get(course.code, course.code)}
        behind |= completed
        groups = course.prereq_groups or [[c] for c in course.prereqs]
        if not all(any(canonical.get(normalize_code(p), normalize_code(p)) in behind
                       for p in group) for group in groups):
            continue
        same_term = {canonical.get(c, c) for c in current[target]}
        if any(canonical.get(normalize_code(c), normalize_code(c)) not in behind | same_term
               for c in course.coreqs):
            continue
        if not _fits(current[target], course, profile, by_code, terms[target][0]):
            continue
        return target
    return None


def repair_plan(
    semesters: list[list[str]],
    profile: StudentProfile,
    catalog: list[Course],
    *,
    canonical: dict[str, str] | None = None,
    max_operations: int = 60,
) -> tuple[list[list[str]], list[str], list[str], PlanValidation]:
    """Make the schedule legal, moving courses where possible and deleting only as a last
    resort. Returns (plan, removed, added, check).

    MOVE BEFORE DELETE, and the difference is not cosmetic. The first Gemma draft measured
    against a real Machine Intelligence student omitted CS 18000 and opened with CS 18200 and
    CS 24000, which both need it. A delete-only repair charged that one omission fifteen times:
    the two openers went, then everything that needed THEM went, and the entire computer-science
    chain disappeared from the plan. The student would have been shown a degree with no major
    in it because the model forgot one first-semester course.

    Moving fixes it the way an advisor would — CS 18000 goes in first, CS 18200 and CS 24000
    slide to the second semester — and it is fully decidable: a course is legal in a term or it
    is not. Deletion remains for the cases where nothing can be moved (a hallucinated code, a
    duplicate, a course with no legal term left at all).

    ITERATIVE, because repairs interact: sliding a course out of semester 2 frees credits that
    may fix semester 2's cap violation, and lands credits in semester 3 that may break its own.
    One operation per pass, re-checked from scratch, bounded by ``max_operations`` so a
    pathological plan cannot loop. Hitting the bound leaves the remaining violations REPORTED
    rather than silently accepted — the caller surfaces them as warnings.
    """
    canonical = canonical or {}
    by_code = {course.code: course for course in catalog}
    current = [list(codes) for codes in semesters]
    removed: list[str] = []
    inserted: list[str] = []
    completed = {canonical.get(normalize_code(c), normalize_code(c))
                 for c in profile.completed_courses}
    terms = term_sequence(profile, len(current))

    def drop(index: int, code: str) -> bool:
        for position, existing in enumerate(current[index]):
            if normalize_code(existing) == normalize_code(code):
                current[index].pop(position)
                removed.append(normalize_code(code))
                return True
        return False

    def move(index: int, code: str, target: int) -> bool:
        for position, existing in enumerate(current[index]):
            if normalize_code(existing) == normalize_code(code):
                current[index].pop(position)
                current[target].append(normalize_code(code))
                return True
        return False

    check = validate_semesters(current, profile, catalog, canonical=canonical)
    for _ in range(max_operations):
        hard = check.hard
        if not hard:
            return current, removed, inserted, check

        violation = hard[0]
        index = violation.semester_index
        code = violation.course_code

        # Cap and overload violations name no single course: take the LAST one in the term, so
        # an earlier, deliberately-placed course is never the one sacrificed.
        if code is None:
            code = current[index][-1] if current[index] else None
        if code is None:
            return current, removed, inserted, check

        course = by_code.get(normalize_code(code))

        # STEP ONE FOR A PREREQ GAP: schedule the prerequisite, if the plan simply never had
        # it. Tried before relocation because relocation cannot succeed while the prerequisite
        # is missing — there is no term in which CS 18200 is legal if CS 18000 is nowhere in
        # the plan, so a relocate-first repair deletes the whole chain and calls it legal.
        # Placing it first is also the only repair that can help a violation in semester 1,
        # where by definition there is no earlier term to move anything into.
        if course is not None and violation.kind == "prereq_violation":
            placed = _place_missing_prerequisite(
                current, course, profile, by_code, canonical, completed, terms
            )
            if placed is not None:
                inserted.append(placed)
                check = validate_semesters(current, profile, catalog, canonical=canonical)
                continue
        # A duplicate or a hallucination cannot be relocated into legality — the second copy is
        # wrong wherever it sits, and an unknown code is wrong everywhere.
        relocatable = course is not None and violation.kind not in {
            "duplicate_course", "hallucinated_course"
        }
        target = (
            _relocate_target(current, index, course, profile, by_code, canonical, completed,
                             terms)
            if relocatable else None
        )
        moved = move(index, code, target) if target is not None else False
        if not moved and not drop(index, code):
            return current, removed, inserted, check

        check = validate_semesters(current, profile, catalog, canonical=canonical)

    return current, removed, inserted, check


def unmet_candidates(
    semesters: list[list[str]],
    profile: StudentProfile,
    *,
    canonical: dict[str, str] | None = None,
) -> list[str]:
    """Courses the student still needs that are nowhere in the plan, in preference order."""
    canonical = canonical or {}
    placed = {canonical.get(normalize_code(c), normalize_code(c))
              for term in semesters for c in term}
    placed |= {canonical.get(normalize_code(c), normalize_code(c))
               for c in profile.completed_courses}
    return [code for code in profile.remaining_courses
            if canonical.get(normalize_code(code), normalize_code(code)) not in placed]


def legal_slots(
    semesters: list[list[str]],
    profile: StudentProfile,
    catalog: list[Course],
    course_code: str,
    *,
    canonical: dict[str, str] | None = None,
) -> list[int]:
    """Which semesters ``course_code`` could be added to without creating a violation.

    COMPUTED, NOT GUESSED, and that is the whole point of handing it to the model. With a late
    prerequisite these lists are often a single semester, and a small model asked to search
    eight terms for it simply fails — the eval watched one answer "fall 2026" when told
    "semester 3 or 4", which is semester 1, where the course busts the credit cap. Anything
    decidable without a model should not be spent on one.

    Tries the placement and re-checks: no second copy of the prerequisite/credit/term rules to
    drift from ``validate_semesters``.
    """
    baseline = len(validate_semesters(semesters, profile, catalog, canonical=canonical).hard)
    out: list[int] = []
    for index in range(len(semesters)):
        trial = [list(term) for term in semesters]
        trial[index].append(normalize_code(course_code))
        if len(validate_semesters(trial, profile, catalog,
                                  canonical=canonical).hard) <= baseline:
            out.append(index)
    return out


def cited_course_flags(text: str, planned: set[str], known: set[str]) -> list[str]:
    """Course codes a rationale claims about that are not in the plan or the catalog.

    Triage only, exactly as in the eval: it catches entity invention, not relational fiction
    ("CS 38100 is easier in the spring"). Cheap enough to run on every generated rationale.
    """
    flags: list[str] = []
    for subject, number in _COURSE_CODE_RE.findall(text.upper()):
        code = normalize_code(f"{subject} {number}")
        if code not in known:
            flags.append(f"mentions a course that is not in the catalog: {code}")
        elif code not in planned:
            flags.append(f"mentions a course that is not in the plan: {code}")
    return flags
