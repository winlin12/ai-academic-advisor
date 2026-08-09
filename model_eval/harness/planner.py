"""Vendored copy of the app's deterministic planner — stdlib only.

WHY A COPY. The harness's standing rule is that it depends on nothing in the app (no
pydantic, no FastAPI, no database), so it can run on a bare Python install on whichever
box has the GPU. But Mode A of the plan eval measures the app's REAL revise-plan path,
which means the model's ``PlanEditProposal`` has to be folded into a profile and re-planned
by the same algorithm production uses. Reimplementing it "roughly" would make Mode A
measure the harness instead of the app.

So this file is a deliberate, faithful port of:

    backend/app/services/planner.py            -> generate_plan / prereqs_satisfied / next_term
    backend/app/services/advisor_agent.py      -> _apply_proposal / _severity

DRIFT IS THE COST. If someone changes the app's planner and not this file, Mode A silently
stops measuring production. ``python run.py parity`` diffs the two implementations on a
generated battery of profiles whenever the backend happens to be importable, and the report
prints whether that check has been run. Keep the two in sync or the Mode A column is a lie.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Mirrors backend/app/services/planner.py. Summer is excluded from PLANNING by default
# (backend setting ``planner_include_summer``) — it is a real term with real offerings, but
# summer enrolment carries cost/aid/residency consequences a planner cannot reason about.
# The harness sets this from config.yaml's ``run.planning_terms`` at load time so the eval and
# the app never silently disagree about what a valid semester is.
ALL_TERMS = ["fall", "spring", "summer"]
TERM_ORDER = ["fall", "spring"]


def set_planning_terms(terms: list[str]) -> None:
    """Set the terms the planner may schedule into. Called once from config; the module-level
    list is mutated in place so callers holding a reference see the change."""
    TERM_ORDER[:] = [t for t in ALL_TERMS if t in terms]


@dataclass(frozen=True)
class Course:
    code: str
    title: str
    credits: int
    prereqs: tuple[str, ...] = ()
    # COREQUISITES: satisfied by taking the course EARLIER **or in the same semester**. Purdue
    # lists several of these and the fixture modelled them all as hard prerequisites, which
    # charged models a violation for a schedule a student could actually register for — 15% of
    # every prereq violation in the corpus was one pair, CS 37300 needing STAT 35000.
    #
    # The PLANNER treats a coreq as if it were a prereq (see prereqs_satisfied): scheduling the
    # pair strictly in order is always legal, so the deterministic path stays conservative and
    # can never emit something the registrar would bounce. Only the SCORER takes the looser
    # reading, because its job is to not penalise a legal plan. The asymmetry is deliberate.
    coreqs: tuple[str, ...] = ()
    offered_terms: tuple[str, ...] = ()
    requirement_tags: tuple[str, ...] = ()
    # The course this one is an approved substitute for ("MA 16500" -> "MA 16100"). Carried
    # here only so the fixture row round-trips; NOTHING in this module reads it, and that is
    # deliberate. Equivalence is a SCORING question ("has the degree requirement been met?"),
    # handled in plan_scorers.py. The app's planner has no such field, so acting on it here
    # would break `run.py parity` and quietly stop Mode A from measuring production.
    # Equivalents are in no requirement group, so select_remaining_courses never offers one to
    # generate_plan and the two implementations cannot diverge on this.
    equivalent_to: str = ""


@dataclass
class Profile:
    name: str = "Student"
    degree_program: str = "Computer Science"
    completed_courses: list[str] = field(default_factory=list)
    remaining_courses: list[str] = field(default_factory=list)
    start_term: str = "fall"
    start_year: int = 2026
    semesters_to_plan: int = 8
    max_credits_per_semester: int = 16
    # THE LOAD TO AIM FOR, as distinct from the load never to exceed. Filling every semester to
    # `max` is what produces the front-loaded shape the planner had until 2026-07-29:
    # mi-fresh-start packed 85 credits into 6 of the student's 8 semesters as
    # [16, 15, 14, 16, 15, 9] and left the last two empty, which is a worse four years than
    # ~11 credits a term even though both are legal.
    #
    # None = DERIVE IT, per semester, as ceil(credits still to place / semesters still
    # available), clamped by `max`. Deriving beats a fixed number because it self-corrects: a
    # term that cannot reach the target (prereqs not met, nothing offered) raises the target
    # for the terms after it, so the work still lands inside the horizon instead of a stub
    # semester at the end. Set an integer to pin it; set it equal to `max` to get the old
    # fill-to-the-cap behaviour back.
    target_credits_per_semester: int | None = None
    # THE REGISTRAR'S CEILING, as distinct from the two above — added 2026-08-02, and the only
    # one of the three that is a rule about what the UNIVERSITY permits rather than about what
    # this student wants. 18 credits is a normal full-time overload at Purdue; above it you need
    # a dean's signature, which is what makes 19 a different kind of thing from 17.
    #
    # Why it exists: `max_credits_per_semester` was doing both jobs, and scoring one 17-credit
    # semester as `credit_cap_violation` made the whole plan NOT VIABLE — the same word the
    # scorer uses for a course scheduled before its own prerequisite. Those are not the same
    # failure. A plan with a 17-credit term is a plan the registrar accepts and the student
    # merely did not ask for; a plan with a broken prereq chain cannot be registered at all.
    # Charging viability for the first also billed it twice, since the student's own number is
    # ALREADY scored, by name, as a `max_credits_at_most` assertion.
    #
    # So: `max_credits_per_semester` stays the student's number — it drives the planner, the
    # prompt's "hard limit" line and the assertion — and only a semester above THIS one is a
    # violation. Same reasoning as the coreq rule in plan_scorers: do not charge a violation
    # for a schedule the registrar would accept.
    hard_credit_cap: int = 18
    # MAJOR-COURSE LOAD, a second cap alongside credits. A 16-credit semester is not one
    # workload: four 4-credit CS courses and a CS course plus three gen-eds are the same number
    # and nothing like the same term. The credit cap alone cannot tell them apart, and the
    # deterministic planner MEASURABLY produced 4- and 5-CS semesters (mi-fresh-start,
    # mi-tight-horizon, mi-spring-start) before this existed — plans no student would follow.
    #
    #   preferred_...  SOFT. The planner fills the semester with non-major courses first and
    #                  only comes back for a third major course if nothing else fits the room.
    #   max_...        HARD. Never exceeded; the course goes to a later semester instead, even
    #                  if that leaves the semester under its credit cap or the course unplanned.
    major_subject: str = "CS"
    max_major_courses_per_semester: int = 3
    preferred_major_courses_per_semester: int = 2

    def replace(self, **changes) -> "Profile":
        return Profile(**{**self.__dict__, **changes})


@dataclass
class Semester:
    term: str
    year: int
    courses: list[str]
    total_credits: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class Plan:
    semesters: list[Semester]
    unplanned_courses: list[str]
    warnings: list[str]


@dataclass
class Proposal:
    """Harness-side mirror of ``app.models.schemas.PlanEditProposal``."""

    rationale: str = ""
    reorder: list[str] = field(default_factory=list)
    defer: list[str] = field(default_factory=list)
    avoid_tags: list[str] = field(default_factory=list)
    max_credits_per_semester: int | None = None


# A student naming a number is a parsing problem, not a planning problem, and a regex is right
# 100% of the time where the best model measured was right 0/3. model_eval 2026-07-29:
# gemma4-e4b left max_credits_per_semester null on every replicate of the explicit-cap scenario
# while the student had written "cap me at 12 credits a semester".
#
# FALLBACK, NOT AN OVERRIDE. The model's own value wins whenever it set one — this only fills
# the field the model left empty, so a model that understood the request is never second-guessed
# and the eval can still measure whether it understood.
_CREDIT_CAP_PATTERNS = [
    # "cap me at 12 credits", "keep me at/under 12 credits", "limit ... to 12 credits"
    re.compile(r"(?:cap|keep|limit|hold|stay|max(?:imum)?|no more than|at most|under|below)"
               r"[^.\n]{0,40}?(\b\d{1,2}\b)\s*(?:cr\b|credits?\b|credit hours?\b)", re.I),
    # "12 credits a semester" / "12 credit semesters" with no leading verb
    re.compile(r"\b(\d{1,2})\s*(?:cr\b|credits?\b|credit hours?\b)"
               r"[^.\n]{0,20}?(?:a|per|each|every)\s+(?:semester|term)", re.I),
]


def extract_credit_cap(text: str, *, low: int = 1, high: int = 24) -> int | None:
    """The per-semester credit load a student explicitly asked for, or None.

    Deliberately conservative: it fires only on phrasings that name a number next to a credit
    word, and it refuses anything outside a plausible course load, so "I finished 120 credits"
    or "MA 16100 is 5 credits" cannot be mistaken for a cap.
    """
    if not text:
        return None
    for pattern in _CREDIT_CAP_PATTERNS:
        for match in pattern.finditer(text):
            value = int(match.group(1))
            if low <= value <= high:
                return value
    return None


def next_term(term: str, year: int) -> tuple[str, int]:
    """Advance one term. A term that is not schedulable (summer, by default) is treated as
    'before the first schedulable term', so a summer start rolls forward to that fall rather
    than raising."""
    idx = TERM_ORDER.index(term) if term in TERM_ORDER else -1
    nxt = TERM_ORDER[(idx + 1) % len(TERM_ORDER)]
    return nxt, year + 1 if nxt == "spring" else year


def first_planning_term(term: str, year: int) -> tuple[str, int]:
    """Snap a start term onto the planning calendar.

    A student whose record says they start in summer still has to be given a plan; with summer
    excluded they simply start that fall. Silently scheduling INTO the summer term would be the
    bug, so the snap happens once, here, rather than being left to each caller."""
    return (term, year) if term in TERM_ORDER else next_term(term, year)


def prereqs_satisfied(course: Course, completed: set[str]) -> bool:
    """Conservative on purpose: corequisites are required UP FRONT here.

    A coreq may legally be taken alongside its course, but making the greedy planner exploit
    that needs it to reason about pairs it has not selected yet. Requiring it earlier is always
    legal — never a registration wall — at the cost of occasionally scheduling a term later than
    strictly necessary. ``plan_scorers`` applies the true same-semester rule, so a MODEL that
    uses the flexibility is not charged for it.
    """
    required = set(course.prereqs) | set(course.coreqs)
    return all(code in completed for code in required)


def is_major_course(code: str, subject: str) -> bool:
    """Does this course code belong to the major's subject ("CS 25100" -> CS)?

    Subject prefix, not requirement tags: `required` is carried by MA 26100 and PHYS 17200 as
    much as by CS 25100, and it is specifically the department's own courses that stack into
    an unmanageable term. Matched on the leading alphabetic run so "CS25100", "CS 25100" and
    "cs 25100" all count.
    """
    if not subject:
        return False
    letters = "".join(ch for ch in str(code) if not ch.isdigit()).strip().upper()
    return letters.replace("-", "").replace(" ", "") == subject.strip().upper()


def semester_credit_target(
    profile: Profile, remaining: list[str], catalog_by_code: dict[str, Course],
    semester_index: int,
) -> int:
    """Credits to aim for in THIS semester. Never above ``max_credits_per_semester``.

    An explicit ``target_credits_per_semester`` is honoured as-is (clamped). Otherwise the
    target is the even split of what is left over the terms that are left — which is what makes
    the result spread rather than front-load, without ever needing a second balancing pass.

    Degenerate cases both fall back to the cap, which is the old behaviour: no semesters left to
    divide by, and nothing left to place.
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
    even_split = -(-credits_left // semesters_left)      # ceil, ints only
    return min(even_split, profile.max_credits_per_semester)


def generate_plan(profile: Profile, catalog: list[Course], *, _spread: bool = True) -> Plan:
    """Port of ``app.services.planner.generate_plan``. Keep the two byte-for-byte equivalent
    in behaviour; the ordering key in particular is load-bearing for Mode A scores.

    SPREAD FIRST, PACK IF IT DOES NOT FIT. Aiming at the credit target instead of the cap
    leaves capacity unused in early terms, and that capacity does not come back: when a
    prerequisite chain opens up late, the courses it was holding have nowhere to go and end up
    unplanned. MEASURED — mi-sophomore-catchup with a reorder of
    [CS 37300, CS 47100, CS 43900] placed 55 credits as [11, 9, 10, 9, 9, 7] under a 15-credit
    cap and stranded CS 49000 and CS 35400; the same profile filling to the cap placed 61 as
    [14, 13, 15, 13, 6] and stranded nothing.

    That made Mode A viability fall from 100% to 50%, and it bit specifically when the model
    REORDERED — i.e. exactly the knob Mode A exists to give it. Coverage must never be the
    price of a nicer distribution, so a spread plan that strands anything is discarded in
    favour of the packed one. ``_spread`` is the internal re-entry for that second attempt and
    is not part of the API.
    """
    catalog_by_code = {course.code: course for course in catalog}
    completed = set(profile.completed_courses)
    remaining = list(profile.remaining_courses)
    warnings: list[str] = []
    semesters: list[Semester] = []

    unknown = [code for code in remaining if code not in catalog_by_code]
    if unknown:
        warnings.append(f"Unknown courses ignored: {', '.join(unknown)}")
        remaining = [code for code in remaining if code in catalog_by_code]

    priority = {code: index for index, code in enumerate(remaining)}

    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)

    for semester_index in range(profile.semesters_to_plan):
        selected: list[Course] = []
        selected_credits = 0
        semester_warnings: list[str] = []
        # Recomputed EVERY term, not once: `remaining` shrinks as courses are placed and the
        # horizon shrinks with it, so a term that came up short automatically raises the bar
        # for the terms after it. Clamped by `max`, which is still the only hard bound.
        credit_target = (
            semester_credit_target(profile, remaining, catalog_by_code, semester_index)
            if _spread else profile.max_credits_per_semester
        )

        candidates = [
            catalog_by_code[code]
            for code in remaining
            if term in catalog_by_code[code].offered_terms
            and prereqs_satisfied(catalog_by_code[code], completed)
        ]
        candidates.sort(
            key=lambda c: (
                priority.get(c.code, len(priority)),
                "required" not in c.requirement_tags,
                c.code,
            )
        )

        # TWO PASSES over the same candidate list, differing only in the major-course ceiling
        # they allow. Pass one holds the soft limit, so everything else a student could be
        # taking gets first claim on the room; pass two raises it to the hard limit and admits
        # a further major course only into room nothing else filled. The result is "at most
        # `preferred` CS courses unless the semester would otherwise go short, and never more
        # than `max`" — without a second sort, so the priority order the revise-plan agent
        # controls still decides which courses those are.
        chosen: set[str] = set()
        selected_majors = 0
        soft = min(profile.preferred_major_courses_per_semester,
                   profile.max_major_courses_per_semester)
        for major_limit in (soft, profile.max_major_courses_per_semester):
            for course in candidates:
                if course.code in chosen:
                    continue
                # `credit_target`, not `max_credits_per_semester`: this is the line that turns
                # a front-loaded plan into a spread one. The cap still exists — the target is
                # clamped to it — but it is no longer what the greedy fill aims at.
                if selected_credits + course.credits > credit_target:
                    continue
                if is_major_course(course.code, profile.major_subject):
                    if selected_majors >= major_limit:
                        continue
                    selected_majors += 1
                chosen.add(course.code)
                selected_credits += course.credits
        # Emitted in candidate (priority) order, not in the order the passes admitted them, so
        # a semester reads the same way it always has.
        selected = [course for course in candidates if course.code in chosen]

        if not selected and remaining:
            blocked = []
            for code in remaining:
                course = catalog_by_code[code]
                reasons = []
                missing = [p for p in course.prereqs if p not in completed]
                if missing:
                    reasons.append(f"missing prereqs {list(missing)}")
                if term not in course.offered_terms:
                    reasons.append(f"not offered in {term}")
                if reasons:
                    blocked.append(f"{code}: {', '.join(reasons)}")
            if blocked:
                semester_warnings.append(
                    "No courses selected this term. Blocked courses: " + "; ".join(blocked)
                )

        planned = {course.code for course in selected}
        completed.update(planned)
        remaining = [code for code in remaining if code not in planned]

        semesters.append(
            Semester(
                term=term,
                year=year,
                courses=[course.code for course in selected],
                total_credits=selected_credits,
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

    plan = Plan(semesters=semesters, unplanned_courses=remaining, warnings=warnings)
    # The fallback. Only ever taken when spreading actually cost coverage, so a plan can never
    # come out worse than it would have before the credit target existed.
    if _spread and plan.unplanned_courses:
        packed = generate_plan(profile, catalog, _spread=False)
        if len(packed.unplanned_courses) < len(plan.unplanned_courses):
            return packed
    return plan


def apply_proposal(profile: Profile, proposal: Proposal, catalog: list[Course]) -> Profile:
    """Port of ``advisor_agent._apply_proposal``. Everything is a preference, not a command:
    unknown codes drop out, reorder beats defer, and the cap falls back to the profile's."""
    by_code = {course.code: course for course in catalog}
    remaining = list(profile.remaining_courses)
    present = set(remaining)
    avoid = {tag.strip().lower() for tag in proposal.avoid_tags if tag.strip()}

    def is_avoided(code: str) -> bool:
        course = by_code.get(code)
        if course is None or not avoid:
            return False
        return any(tag.lower() in avoid for tag in course.requirement_tags)

    base = [c for c in remaining if not is_avoided(c)] + [c for c in remaining if is_avoided(c)]

    reorder = [c for c in proposal.reorder if c in present]
    reorder_set = set(reorder)
    defer = [c for c in proposal.defer if c in present and c not in reorder_set]
    pinned = reorder_set | set(defer)
    middle = [c for c in base if c not in pinned]

    cap = proposal.max_credits_per_semester or profile.max_credits_per_semester
    return profile.replace(
        remaining_courses=reorder + middle + defer,
        max_credits_per_semester=cap,
    )


def severity(plan: Plan) -> int:
    """Port of ``advisor_agent._severity`` — how incomplete a plan is."""
    return len(plan.unplanned_courses)
