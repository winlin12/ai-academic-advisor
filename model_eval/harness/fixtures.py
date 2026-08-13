"""The harness's planner types, and the students the eval runs against.

THERE ARE NO FIXTURE FILES ANY MORE (2026-08-12). This module used to load
``plan_fixtures/*.yaml`` — one hand-authored file per school, carrying that program's course
catalog, its requirement groups, and a set of hand-written student scenarios. Every one of
those three is now read from, or derived from, the real crawled database instead:

    courses + requirement_groups   `real_db.fixture_from_database`, off the same
                                   `CatalogDatabase` the model is shown
    students                       `synthesize_scenarios` below, derived from that program's
                                   own required courses

WHY. A fixture file was a per-program cost paid by hand, so `--major` only ever worked for the
nineteen programs somebody had written a file for, out of ~950 crawled. Worse, the hand-written
half drifted: fixtures invented course codes for gen-ed categories that were never real, and
every model was scored against them (see `real_scoring.py`'s module docstring for what that cost
in false "missing requirement" reports). What is left is uniform — any program in the catalog is
a valid `--major`, and every program is described the same way, by the crawl.

What this module still owns:

1. :class:`planner.Course`/:class:`Profile` construction (`Scenario`, `Fixture`).
2. ``select_remaining_courses`` — the ordered course list a profile carries, a port of
   ``backend/app/services/planner_catalog.select_remaining_courses`` (required groups first,
   then just enough selective options to cover each group's credit target, counting completed
   courses toward the group). Mode A has to start from the same baseline production would, or
   it isn't measuring production.
3. ``synthesize_scenarios`` — the students themselves.

The DATABASE hash (`CatalogDatabase.db_hash`) now lands in every plan record where the fixture
hash used to, and it carries the same meaning: change what the model was shown and old records
stop being comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .planner import Course, Profile

DEFAULT_OPTION_CREDITS = 3.0


@dataclass
class Scenario:
    id: str
    label: str
    profile: Profile
    feedback: str
    assertions: list[dict[str, Any]]
    # True = no legal plan can cover every requirement in the horizon, on purpose. The
    # reachability check in `run.py check` expects the deterministic planner to fail here,
    # and the report scores the row on violations and honest "unplanned" reporting rather
    # than on PLAN_VIABLE (which is 0% for everyone by construction).
    expect_unsatisfiable: bool = False
    # MODE B/C ONLY — never read by the deterministic planner (Mode A), which is why these
    # live on `Scenario` rather than `Profile`: `Profile` mirrors the production app's own
    # student-state type, and neither field has a production counterpart to mirror. See
    # `harness/real_db.build_scenario_database` for how each is applied.
    #
    # Free text ranked against `real_db._GEN_ED_THEMES`, or one of a handful of "no real
    # preference" phrases (also in real_db.py) that resolve to a hand-picked default list —
    # there being no enrollment data anywhere in either crawled database to rank "popular" by.
    # None behaves exactly like "whatever works".
    gen_ed_preference: str | None = None
    # (subject prefix, level) — e.g. ("SPAN", 3). Narrows every world-language menu to that
    # subject's next three courses in sequence and drops every OTHER language's options
    # entirely; see `real_db._language_window`. None leaves language menus in the general
    # gen_ed_preference ranking, same as any other elective group.
    world_language: tuple[str, int] | None = None
    # SECOND MAJORS AND MINORS this student is also pursuing — `(poid, kind, label)` per extra
    # program, keyed on Purdue's own `programs.poid` rather than the crawl's surrogate UUID so a
    # re-crawl does not orphan them (the failure that stranded 16 of 19 fixtures).
    #
    # MODE B/C ONLY, and scenario-level rather than fixture-level on purpose: "CS student who is
    # also doing a Data Science second major and a Math minor" is a property of one student, not
    # of the program. Every requirement group of every listed program is merged into what the
    # model is shown and scored against; a course satisfies each of them independently (the
    # harness allows unlimited double counting), while its CREDITS count once toward the
    # graduation total. See `real_db.merge_real_db_bases`.
    additional_programs: tuple[tuple[str, str, str], ...] = ()


@dataclass
class Fixture:
    name: str
    path: Path
    fixture_hash: str
    verified: bool
    catalog: list[Course]
    requirement_groups: list[dict[str, Any]]
    scenarios: list[Scenario]
    # Multi-school support. `slug` is the short name `run.py run`'s sweep uses for
    # `results/results_<slug>/`; `real_db_program_id` is the crawled Postgres program every
    # mode reads. Both are set by `real_db.fixture_from_database`, which is now the only thing
    # that builds a `Fixture` at all.
    slug: str = ""
    real_db_program_id: str = ""

    @property
    def by_code(self) -> dict[str, Course]:
        return {course.code: course for course in self.catalog}

    def credits(self, code: str) -> int:
        course = self.by_code.get(code)
        return course.credits if course else 0

    @property
    def canonical(self) -> dict[str, str]:
        """code -> the PRIMARY course it counts as. Identity for everything unpaired.

        "MA 16500" -> "MA 16100": an approved substitute satisfies the requirement, and the
        prerequisite, that its primary appears under, and taking both is taking one course
        twice. Collapsing to the primary before any of those three checks is what makes all
        three agree; doing it in only one of them is how you get a plan that covers the degree
        but still reports a prereq violation.

        Chains are followed (A -> B -> C all collapse to C) with a visited set, so a fixture
        typo that makes two courses equivalent to each other cannot hang the scorer.
        """
        direct = {c.code: c.equivalent_to for c in self.catalog if c.equivalent_to}
        out: dict[str, str] = {}
        for course in self.catalog:
            code, seen = course.code, {course.code}
            while direct.get(code) and direct[code] not in seen:
                code = direct[code]
                seen.add(code)
            out[course.code] = code
        return out


def select_remaining_courses(
    groups: list[dict[str, Any]], completed: set[str], credits_of,
    canonical: dict[str, str] | None = None,
) -> list[str]:
    """Port of ``planner_catalog.select_remaining_courses`` over fixture groups.

    ``kind: all`` mirrors a required requirement_group; ``kind: choose`` mirrors a selective
    one. Required blocks come before all selectives and duplicates keep their first slot.

    A COURSE COUNTS FOR EVERY LIST IT APPEARS IN (2026-08-12) — the same rule
    ``real_db.merge_real_db_bases`` scores against, applied to the input side. A ``choose``
    group's target is paid down by every option already on the list, whether the student
    completed it or an earlier group required it; before this date only completed courses
    counted, so a group whose options were already scheduled elsewhere still pulled in extra
    courses to fill a target that was in fact already met.

    APPROVED SUBSTITUTES COUNT AS ONE COURSE (2026-08-12), which is what `canonical` is for.
    The real catalog states an either/or requirement as SEPARATE options on the same group —
    CS Machine Intelligence's "Mathematics" group lists MA 16100, MA 16200, MA 16500 AND
    MA 16600, where 16500/16600 are the approved substitutes for 16100/16200 — and with no
    collapsing this function put all four on the list. The student was told to take Calculus I
    twice. That was invisible until prerequisites arrived: with ordering unconstrained the two
    extra courses merely wasted semesters, but with a real chain behind them the planner spent
    its horizon on the duplicate track, never reached MA 16200, and every downstream course
    (MA 26100, MA 26500, MA 41600) fell out of the plan — Mode A coverage 91%, viable for
    nobody, for a program whose requirements are entirely satisfiable.

    The APP does the same job with a different mechanism, and the difference is data, not
    intent: `planner_db.select_remaining_courses` receives each requirement as a run of
    alternatives (`[["MA 16100", "MA 16500"]]`) and takes the first of each run. The database
    export this harness reads flattens options into one list, so the alternative structure is
    gone by the time it gets here and `course_aliases` is what is left to reconstruct it. Same
    rule, same result, expressed with what each side actually has.
    """
    canon = (lambda code: (canonical or {}).get(code, code))  # noqa: E731
    required: list[str] = []
    selective: list[str] = []
    seen: set[str] = {canon(c) for c in completed}

    for group in groups:
        if group.get("kind") != "all":
            continue
        for code in group.get("courses", []):
            if canon(code) not in seen:
                required.append(code)
                seen.add(canon(code))

    for group in groups:
        if group.get("kind") != "choose":
            continue
        options = list(group.get("courses", []))
        if not options:
            continue
        target = group.get("choose_credits")
        if target is None:
            target = min((credits_of(c) or DEFAULT_OPTION_CREDITS) for c in options)
        # `seen`, not `completed`: an option another group already put on the list pays into
        # this group's target too. Deduplicated — one course must not pay twice within one
        # group, even if the group lists it twice.
        counted: set[str] = set()
        needed = float(target)
        for code in options:
            if canon(code) in seen and canon(code) not in counted:
                counted.add(canon(code))
                needed -= credits_of(code) or DEFAULT_OPTION_CREDITS
        for code in options:
            if needed <= 0:
                break
            if canon(code) in seen:
                continue
            selective.append(code)
            seen.add(canon(code))
            needed -= credits_of(code) or DEFAULT_OPTION_CREDITS

    return required + selective


# --- students, derived from the program itself --------------------------------------------------
#
# FIVE ARCHETYPES, THE SAME FIVE FOR EVERY PROGRAM. They are what the nineteen hand-written
# fixtures' scenarios all turned out to be variations of, once the program-specific course codes
# came out: a student starting clean, one part-way through, one who wants a light term, one whose
# calendar starts in spring, and one in a hurry. Keeping the SET identical across programs is the
# point — a per-program student list made two schools' numbers incomparable for reasons that had
# nothing to do with the model.
#
# Everything program-specific is derived, never authored: `completed_courses` comes off this
# program's own required courses in the catalog's prerequisite order, and `major_subject` is
# whichever subject prefix dominates its requirement groups.
#
# `feedback` is the second turn of the Mode A/B feedback variants and Mode C's revise loop, so it
# has to be a real constraint a planner can act on while naming nothing program-specific.
_ARCHETYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "fresh-start",
        "label": "First-year student, nothing completed, full eight-semester horizon",
        "completed": 0,
        "profile": {"semesters_to_plan": 8},
        "feedback": "Please keep my first semester lighter — I am adjusting to college and "
                    "would rather not carry a full load right away.",
    },
    {
        "id": "midway-catchup",
        "label": "Part-way through the degree, planning the rest",
        # TWO SEMESTERS OF WORK BEHIND THEM, six ahead — the horizons and the transcripts have
        # to add up to a whole degree or the scenario is unsatisfiable by construction and
        # every model scores zero on it for arithmetic reasons. ~5 courses per completed
        # semester, matching a 15-credit term.
        "completed": 10,
        "profile": {"semesters_to_plan": 6},
        "feedback": "I want to finish on time. If anything has to slip, slip an elective, not "
                    "a course something else depends on.",
    },
    {
        "id": "light-load",
        "label": "Working student who cannot carry a full course load",
        "completed": 0,
        # TEN SEMESTERS, not eight. A 13-credit ceiling over 8 terms is 104 credits against a
        # 120-credit degree — the student cannot graduate in that window no matter who plans
        # it, so an 8-semester horizon here would measure arithmetic, not the model. Taking
        # longer IS the trade a working student makes.
        "profile": {"semesters_to_plan": 10, "max_credits_per_semester": 13,
                    "target_credits_per_semester": 12},
        "feedback": "I work about 25 hours a week. Twelve or thirteen credits a semester is my "
                    "real ceiling — please do not plan a term above it.",
    },
    {
        "id": "spring-start",
        "label": "Spring admit — the calendar starts off-cycle",
        "completed": 0,
        "profile": {"start_term": "spring", "semesters_to_plan": 8},
        "feedback": "I started in the spring, so please do not assume the usual fall-first "
                    "sequence works for me.",
    },
    {
        "id": "accelerate",
        "label": "Wants to finish early, willing to carry a heavy load",
        # Three semesters of credit already banked (AP, dual enrolment, a summer) is what makes
        # "graduate a year early" a plan rather than a wish — same add-up-to-a-degree rule as
        # `midway-catchup`.
        "completed": 15,
        "profile": {"semesters_to_plan": 6, "max_credits_per_semester": 18},
        "feedback": "I would like to graduate a year early. I can handle heavy semesters, but "
                    "not ones that break the prerequisite order.",
    },
)


def dominant_subject(requirement_groups: list[dict[str, Any]]) -> str:
    """The subject prefix that carries this degree — "CS", "NUR", "ME".

    `major_subject` drives the per-semester major-course caps and `is_major_course` in every
    scorer, and it used to be authored per fixture. Counted over the courses in `kind: all`
    groups only: a required course is the program's own, while a selective menu is mostly
    gen-ed and would swamp the count with whatever subject happens to have the most electives.
    Falls back to every group, then to "" (which makes the major-course caps inert rather than
    wrong) for a program whose requirements name no courses at all.
    """
    def _counts(groups: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in groups:
            for code in group.get("courses", []):
                subject = code.split(" ")[0].strip()
                if subject:
                    counts[subject] = counts.get(subject, 0) + 1
        return counts

    for candidates in (_counts([g for g in requirement_groups if g.get("kind") == "all"]),
                       _counts(requirement_groups)):
        if candidates:
            return max(candidates.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return ""


def _completed_prefix(
    catalog: list[Course], requirement_groups: list[dict[str, Any]], count: int,
    canonical: dict[str, str] | None = None,
) -> list[str]:
    """The first `count` courses of this degree, in an order a student could have taken them.

    Built from the SAME list the student would be planning if they were starting fresh — the
    program's own selected courses plus their prerequisite closure — truncated to `count` in
    catalog order. `real_db` builds `catalog` in prerequisite-chain order and never re-sorts,
    so a prefix of it is legal by construction: nothing lands on the transcript before
    something it depends on.

    IT USED TO WALK ONLY `kind: "all"` GROUPS, and stalled almost immediately: a required
    course whose prerequisite lives in a SELECTIVE group (MA 26100 needs MA 16200, which CS
    reaches through University Core: Quantitative Reasoning) failed the "are its prerequisites
    already here" test and stopped the walk. `accelerate` asked for fifteen completed courses
    and got two — a student one year from graduating, handed a transcript with CS 18000 on it,
    then scored on whether a model could fit the whole degree into six semesters. Coverage 91%
    for everyone, no violations, entirely an artifact of this function.
    """
    def credits_of(code: str) -> int:
        course = next((c for c in catalog if c.code == code), None)
        return course.credits if course else 0

    full = _with_prereq_closure(
        select_remaining_courses(requirement_groups, set(), credits_of, canonical),
        catalog, set(), canonical,
    )
    wanted = set(full)
    ordered = [c.code for c in catalog if c.code in wanted]
    return ordered[:count]


def _with_prereq_closure(
    codes: list[str], catalog: list[Course], completed: set[str],
    canonical: dict[str, str] | None = None,
) -> list[str]:
    """`codes` plus every prerequisite they transitively need that nothing else supplies.

    THE DEGREE AUDIT IS NOT A PLAN, and this is where that bites. A requirement group lists
    what COUNTS toward the degree, not everything a student must sit through to get there:
    Mechanical Engineering requires MA 26100, MA 26500 and MA 26600 by name and never mentions
    Calculus I or II, because the audit assumes those arrive via placement, AP credit, or the
    University Core. `select_remaining_courses` faithfully reproduces that list — and the
    deterministic planner then cannot schedule a single one of them, because `generate_plan`
    only ever places courses that are ON the list and MA 26100's prerequisite is not. Measured
    on ME: 57% coverage for all five students, with zero violations, purely from this.

    So the closure is added at SCENARIO-CONSTRUCTION time, not inside
    `select_remaining_courses` — that function is a port of the app's own
    `planner_catalog.select_remaining_courses` and must keep matching it (`run.py parity`).
    "What this student still has to take" is a fact about the student, which is exactly what a
    scenario is for. The app has the same hole and should probably close it the same way, but
    that is a product decision, not a porting one.

    Added courses come FIRST, in catalog order — `real_db` builds `catalog` in
    prerequisite-chain order, so that is already a legal sequence, and the planner reads list
    position as preference. Anything already completed is skipped; a prerequisite naming a
    course outside the catalog is dropped, since nothing could schedule it anyway.
    """
    by_code = {c.code: c for c in catalog}
    canon = (lambda code: (canonical or {}).get(code, code))  # noqa: E731
    have = {canon(c) for c in completed} | {canon(c) for c in codes}
    added: set[str] = set()
    frontier = list(codes)
    while frontier:
        course = by_code.get(frontier.pop())
        if course is None:
            continue
        for need in tuple(course.prereqs) + tuple(course.coreqs):
            if need not in by_code or canon(need) in have:
                continue
            have.add(canon(need))
            added.add(need)
            frontier.append(need)
    if not added:
        return codes
    order = {course.code: index for index, course in enumerate(catalog)}
    return sorted(added, key=lambda c: order.get(c, 0)) + codes


def synthesize_scenarios(
    program_name: str,
    catalog: list[Course],
    requirement_groups: list[dict[str, Any]],
    *,
    major_subject: str = "",
    canonical: dict[str, str] | None = None,
) -> list[Scenario]:
    """The five students every program is evaluated against. See `_ARCHETYPES`.

    Deterministic for a given database state: same program, same catalog order, same students,
    so a re-run compares against the last one. Nothing here is hand-authored per program, which
    is what makes `--major <anything in the catalog>` work.
    """
    subject = major_subject or dominant_subject(requirement_groups)
    by_code = {course.code: course for course in catalog}

    def credits_of(code: str) -> int:
        course = by_code.get(code)
        return course.credits if course else 0

    scenarios: list[Scenario] = []
    for spec in _ARCHETYPES:
        completed = _completed_prefix(catalog, requirement_groups,
                                      int(spec["completed"]), canonical)
        overrides = dict(spec["profile"])
        scenarios.append(Scenario(
            id=str(spec["id"]),
            label=str(spec["label"]),
            profile=Profile(
                name="Student",
                degree_program=program_name,
                completed_courses=completed,
                remaining_courses=_with_prereq_closure(
                    select_remaining_courses(
                        requirement_groups, set(completed), credits_of, canonical),
                    catalog, set(completed), canonical),
                start_term=str(overrides.pop("start_term", "fall")),
                start_year=int(overrides.pop("start_year", 2026)),
                major_subject=subject,
                **overrides,
            ),
            feedback=str(spec["feedback"]),
            assertions=[],
        ))
    return scenarios
