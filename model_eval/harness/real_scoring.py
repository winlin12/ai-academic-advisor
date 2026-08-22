"""Scores a Mode B plan against the REAL requirement structure and REAL course facts of
whichever program `real_db.py` actually loaded — not a hand-authored fixture.

WHY THIS EXISTS, AND WHY IT IS NOW THE ONLY SCORER MODE B USES. `plan_scorers.score_plan()` is
fixture-only: its course universe, credits, prereqs, coreqs, term offerings and requirement
groups all come from `plan_fixtures/cs_machine_intelligence.yaml` regardless of which program
the model was actually SHOWN — and `run.py --major` (or a `real_db.program_id` that drifts from
the fixture's own program) can point Mode B at any program in the catalog. Two ways that showed
up as a correctness bug, not just a display quirk:

  * A model choosing a real, valid course outside the fixture's course list — the WGSS incident
    (`real_db.py`'s module docstring) got scored "hallucinated" because the CS fixture's course
    universe had never heard of the course, even though it was right there in `courses` in the
    same prompt. Confirmed again 2026-08-04: a Mechanical Engineering run (real_db.program_id
    pointed off the fixture's own program) flagged 24 genuinely real ME courses as
    `hallucinated_course`, and a same-program CS run still showed `gen-ed-core`/`science` as
    "missing" because the FIXTURE invented specific gen-ed course codes (SCLA 10100, PHYS
    17200, ...) that were never real, enumerable requirements — the real catalog correctly
    states these in prose (at the time, in an `unresolved_requirement_groups` table; removed
    2026-08-12, see `catalog_export.TABLES`), and the fixture's guess at turning prose into a
    course list is exactly the kind of hardcoding this module replaces.
  * `missing_requirements` read back labels (`cs-elective`, `gen-ed-selective`) that describe
    the FIXTURE's invented taxonomy, not the real program's actual `requirement_groups` names —
    misleading regardless of which program real_db.program_id names.

So `plan_scorers.score_plan` is fixture-only BY DESIGN and stays exactly that for Mode A (the
deterministic planner's own output, "legal by construction" against the SAME fixture it was
built from — nothing about a real database belongs there) — see `runner.run_mode_a`. Mode B is
the one call site that grammar-constrains the model to `ctx.database.course_codes` and shows it
`ctx.database`'s tables as the prompt, so Mode B is the one call site this module scores
instead, not alongside: `runner._run_freeform_plan` no longer calls `plan_scorers.score_plan`
for Mode B at all.

FIELD-COMPATIBLE WITH `plan_scorers.PlanScore` ON PURPOSE. `RealScore` carries the same
attribute names (`viable`, `violations`, `semester_credits`, `major_courses_per_semester`, ...)
so `runner._score_detail` and `plan_scorers.check_assertions` — both written against
`PlanScore` — work on a `RealScore` UNCHANGED; duck typing, not a shared base class, because a
shared base class would tempt someone into making the two interchangeable in the other
direction too, and Mode A must never silently start reading real-DB data it was never shown.
`groups` is the one real addition: per-group detail a flat `PlanScore` has no slot for, used by
`runner._real_score_detail` for the "every requirement group" table.

PROSE-ONLY REQUIREMENTS ARE NOT SCORED AND NOT REPORTED, 2026-08-12. They never counted toward
`requirement_coverage` (nothing to check), but the export showed them to the model and this
scorer listed them back in the report as "cannot be checked at all". Both are gone: the table is
no longer in `catalog_export.TABLES` and the prompt no longer describes it. Every group this
module scores is one with a real course list. See `catalog_export.TABLES` for the removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .catalog_export import CatalogDatabase
from .planner import Profile, first_planning_term, is_major_course, next_term
from .plan_scorers import normalize_code


@dataclass
class RealGroupResult:
    name: str
    selective: bool
    credits_required: float | None
    credits_filled: float
    satisfied: bool
    filled: list[str] = field(default_factory=list)
    missing_label: str | None = None


@dataclass
class RealScore:
    """Everything `plan_scorers.PlanScore` tracks, computed from the real database instead of
    the fixture, plus per-group detail (`groups`) the flat original has no slot for. See the
    module docstring for why the field names match `PlanScore` exactly."""

    structure_ok: bool = True
    viable: bool = False
    violations: list[str] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    # Over the groups that carry a course list (`groups`, below) and nothing else — a
    # requirement the catalog states only in prose is not represented here at all any more (see
    # the module docstring), so nothing uncheckable can push this number down.
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = field(default_factory=list)
    planned_credits: int = 0
    # CREDITS TOWARD THE DEGREE, counted once per distinct course no matter how many
    # requirement lists that course satisfies — see the block that computes these in
    # `score_against_real_db`. `total_credits` = completed + planned; a plan can satisfy every
    # requirement group and still fail `meets_graduation_credits`, which is exactly the case
    # unlimited double-counting would otherwise hide.
    completed_credits: int = 0
    total_credits: int = 0
    graduation_credits_required: float = 120.0
    meets_graduation_credits: bool = False
    # Purdue's Upper Level Requirement: >= 32 credits of junior-level-or-above coursework,
    # counted over distinct courses exactly like `total_credits`. See the block computing it.
    upper_level_credits: int = 0
    upper_level_required: float = 32.0
    meets_upper_level: bool = False
    semester_credits: list[int] = field(default_factory=list)
    credit_spread: int = 0
    idle_credits: int = 0
    semesters_used: int = 0
    semesters_available: int = 0
    major_courses_per_semester: list[int] = field(default_factory=list)
    soft_major_overloads: int = 0
    soft_credit_overages: int = 0
    hallucinated: list[str] = field(default_factory=list)
    assertions_passed: dict[str, bool] = field(default_factory=dict)

    # Real-DB-specific detail, absent from `PlanScore`.
    groups: list[RealGroupResult] = field(default_factory=list)

    @property
    def groups_satisfied(self) -> int:
        return sum(1 for g in self.groups if g.satisfied)

    @property
    def groups_total(self) -> int:
        return len(self.groups)

    @property
    def coverage(self) -> float:
        """Alias for `requirement_coverage` — kept for `runner._real_score_detail`, which reads
        it by this name."""
        return self.requirement_coverage

    def as_record(self) -> dict[str, Any]:
        """`PlanScore.as_record()`'s exact key surface, so `rec.update(real_score.as_record())`
        in `runner._run_freeform_plan` IS Mode B's primary, unprefixed score — no translation
        layer, and report.py needs no changes to read it. `real_groups` is additive: the
        per-group breakdown `PlanScore` cannot express, kept under a distinct name so it never
        collides with anything `plan_scorers` writes."""
        return {
            "structure_ok": self.structure_ok,
            "plan_viable": self.viable,
            "violations": self.violations,
            "violation_counts": self.violation_counts,
            "requirement_coverage": round(self.requirement_coverage, 4),
            "missing_requirements": self.missing_requirements,
            "planned_credits": self.planned_credits,
            # Distinct-course credits: a course pays into this once however many requirement
            # lists it satisfies. See the fields' own comment on `RealScore`.
            "completed_credits": self.completed_credits,
            "total_credits": self.total_credits,
            "graduation_credits_required": self.graduation_credits_required,
            "meets_graduation_credits": self.meets_graduation_credits,
            "upper_level_credits": self.upper_level_credits,
            "upper_level_required": self.upper_level_required,
            "meets_upper_level": self.meets_upper_level,
            "semester_credits": self.semester_credits,
            "credit_spread": self.credit_spread,
            "idle_credits": self.idle_credits,
            "semesters_used": self.semesters_used,
            "semesters_available": self.semesters_available,
            "major_courses_per_semester": self.major_courses_per_semester,
            "soft_major_overloads": self.soft_major_overloads,
            "soft_credit_overages": self.soft_credit_overages,
            "hallucinated_courses": self.hallucinated,
            "assertions_passed": self.assertions_passed,
            "real_groups": [
                {
                    "name": g.name, "selective": g.selective,
                    "credits_required": g.credits_required, "credits_filled": g.credits_filled,
                    "satisfied": g.satisfied, "filled": g.filled, "missing": g.missing_label,
                }
                for g in self.groups
            ],
        }


# The point past which a semester is not merely heavy but impossible. 21 is reachable with a
# dean's approval in exceptional cases; 22 is not, on any account this project has found.
_ABSURD_SEMESTER_CREDITS = 22

# Mirrors `prompts._CREDIT_BAND`: the prompt asks for target +/- this, so the scorer forgives
# the same margin. Kept as two constants rather than an import because `real_scoring` must not
# depend on the prompt module — but if one moves, move the other.
_CREDIT_BAND = 2


def _is_upper_level(code: str) -> bool:
    """Is this a junior-level-or-above course — the Upper Level Requirement's own test?

    THE FIRST DIGIT OF THE COURSE NUMBER, deliberately, rather than a `>= 30000` comparison:
    Purdue writes five digits today (CS 30700) but the same course is CS 307 in older records,
    and both are junior level. Comparing numerically would count the five-digit form and silently
    fail the three-digit one. A code with no trailing number is not upper level (and not a course
    this scorer can price anyway).
    """
    digits = "".join(ch for ch in code.split(" ")[-1] if ch.isdigit())
    return bool(digits) and digits[0] >= "3"


def _terms(profile: Profile, count: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(count):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def _groups_from_tables(database: CatalogDatabase) -> list[dict[str, Any]]:
    """Reconstruct each requirement group's courses and selective/take-all flag from
    `requirement_groups`/`requirement_options` — the same two tables `catalog_export
    .render_context` renders (as one nested section since 2026-08-12; the flat rows read here
    are unchanged), so this scores exactly what the model was shown, nothing else."""
    options_by_group: dict[str, list[str]] = {}
    for row in database.rows("requirement_options"):
        options_by_group.setdefault(row["requirement_group_id"], []).append(row["course_code"])
    return [
        {
            "id": row["id"], "name": row["name"],
            "selective": row["requirement_type"] == "choose_credits",
            "credits_min": row["credits_min"],
            "courses": options_by_group.get(row["id"], []),
        }
        for row in database.rows("requirement_groups")
        if options_by_group.get(row["id"])
    ]


def score_against_real_db(
    database: CatalogDatabase, profile: Profile, semesters: list[list[str]],
    *, assertions: list[dict[str, Any]] | None = None,
) -> RealScore:
    """Score a schedule (list of per-semester course-code lists) against the requirement
    groups and course facts actually loaded for `database`'s program — every check
    `plan_scorers.score_plan` runs, reimplemented against real rows instead of the fixture. See
    the module docstring for why Mode B is scored by this and only this.
    """
    by_code = {row["course_code"]: row for row in database.rows("courses")}
    alias_of = {row["course_code"]: row["alias_of"] for row in database.rows("course_aliases")}
    canon = lambda code: alias_of.get(code, code)  # noqa: E731

    score = RealScore()
    counts: dict[str, int] = {}

    def flag(kind: str, detail: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        score.violations.append(f"{kind}: {detail}")

    completed = {canon(normalize_code(c)) for c in profile.completed_courses}
    seen: set[str] = set()
    terms = _terms(profile, len(semesters))

    soft_major_cap = min(profile.preferred_major_courses_per_semester,
                         profile.max_major_courses_per_semester)

    for index, (raw_codes, (term, year)) in enumerate(zip(semesters, terms)):
        codes = [normalize_code(c) for c in raw_codes]
        this_semester = {canon(c) for c in codes}
        credits = 0
        major_courses = 0
        # Prereqs/coreqs evaluated against courses completed BEFORE this semester, same rule
        # `plan_scorers.score_plan` uses — a course taken alongside its own prerequisite is a
        # scheduling error, not a corequisite.
        available = set(completed)

        for code in codes:
            course = by_code.get(code)
            if course is None:
                score.hallucinated.append(code)
                flag("hallucinated_course", f"{code} (semester {index + 1}) is not in the "
                                            f"real catalog export the model was shown")
                continue
            key = canon(code)
            if key in seen or key in completed:
                # Every real caller runs `plan_scorers.dedupe_semesters` before scoring, so
                # this should never fire in practice — kept as a defensive SKIP, not a
                # violation, matching `plan_scorers.score_plan`'s identical change. See
                # `plan_scorers`'s module docstring for why duplicates left scoring entirely.
                continue
            seen.add(key)

            credits += int(course.get("credit_hours_min") or 0)
            if is_major_course(code, profile.major_subject):
                major_courses += 1

            missing_groups = [
                " or ".join(group) for group in (course.get("prereq_groups") or [])
                if not any(canon(normalize_code(c)) in available for c in group)
            ]
            if missing_groups:
                flag("prereq_violation",
                     f"{code} in semester {index + 1} needs {', '.join(missing_groups)} first")

            # Looser than a prerequisite the same way `plan_scorers.score_plan` reads it: a
            # coreq taken IN this semester or earlier both count. See that function's comment
            # for the CS 37300/STAT 35000 incident this distinction fixes.
            missing_co = [
                c for c in (course.get("coreq_codes") or [])
                if canon(normalize_code(c)) not in available | this_semester
            ]
            if missing_co:
                flag("coreq_violation",
                     f"{code} in semester {index + 1} needs {', '.join(missing_co)} "
                     f"in the same semester or earlier")

        score.semester_credits.append(credits)
        score.major_courses_per_semester.append(major_courses)
        # A SEMESTER CAN BE HEAVY. IT CANNOT BE ABSURD. (2026-08-16)
        #
        # Everything between the student's stated ask and 21 credits stays SOFT — a heavy term
        # is the student's own call, it changes from one advising conversation to the next, and
        # under genuinely exceptional circumstances a student does carry 19, 20 or 21 with a
        # dean's signature. Failing a whole plan for one such term would be the harness
        # inventing a rule stricter than the university's own practice, and the plan a student
        # would actually be handed is not wrong — it is demanding.
        #
        # At 22 the claim stops being defensible under any circumstance, so that is where the
        # hard line sits. Note the deliberate gap between this and the PROMPT, which tells the
        # model 18 is the registrar's maximum: the instruction states the rule, the scorer
        # refuses only what no student could do. A model that lands on 20 is over-heavy and
        # shows up as `soft_credit_overages`; a model that writes a 22-credit semester has
        # produced a schedule nobody can register for.
        if credits >= _ABSURD_SEMESTER_CREDITS:
            flag("credit_cap_violation",
                 f"semester {index + 1} carries {credits} credits, which no student can "
                 f"register for")
        # TOLERANCE OF ONE LAB COURSE. The prompt now asks for a RANGE (see
        # `prompts.credit_load_line`), because course sizes make an exact per-semester figure
        # unreachable, and the scorer matches it: a student who asked for 13 and got 14 has
        # not been badly served, and flagging that as an overload would report a failure the
        # student would not recognise as one. Past the band it is a real overshoot and still
        # counted — softly, because a heavy term remains the student's call to make.
        if credits > profile.max_credits_per_semester + _CREDIT_BAND:
            score.soft_credit_overages += 1
        if major_courses > soft_major_cap:
            score.soft_major_overloads += 1

        completed |= {canon(code) for code in codes if code in by_code}

    score.violation_counts = counts
    score.planned_credits = sum(score.semester_credits)
    score.semesters_available = profile.semesters_to_plan
    score.semesters_used = sum(1 for c in score.semester_credits if c > 0)
    nonempty = [c for c in score.semester_credits if c > 0]
    score.credit_spread = (max(nonempty) - min(nonempty)) if nonempty else 0

    # --- CREDITS TOWARD GRADUATION ------------------------------------------------------------
    #
    # DISTINCT COURSES ONLY, and that is the whole point of computing it separately from
    # requirement coverage. A course may satisfy any number of requirement lists at once — the
    # major's math requirement and the college's and UCC: QR all at the same time — and this
    # harness deliberately allows that without limit (see `prompts.py`'s statement of the rule).
    # What it must NOT do is let one course pay its credits three times toward the 120: a plan
    # that satisfies every list but only carries 90 credits of actual coursework does not
    # graduate anybody. So requirements are counted per list, credits are counted per course.
    #
    # `completed` and the planned semesters are disjoint by construction — `dedupe_semesters`
    # strips every already-completed course from the plan before scoring — so summing the two is
    # already a distinct-course sum, with no set arithmetic needed here.
    # `all_credits` FIRST, `by_code` second. `by_code` is the budget-trimmed course table the
    # model was shown; a completed gen-ed frequently does not survive that trim, and reading
    # credits only from it silently valued those courses at zero — mi-ai-early lost 12 real
    # credits (ENGL 10600, COM 11400, PHIL 11000, HIST 10300) and was then reported short of the
    # graduation minimum it had actually met. What the student HOLDS is not a function of what
    # fits in a prompt.
    def _credits_of(code: str) -> int:
        if database.all_credits.get(code):
            return int(database.all_credits[code])
        return int((by_code.get(code) or {}).get("credit_hours_min") or 0)

    completed_codes = {canon(normalize_code(x)) for x in profile.completed_courses}
    score.completed_credits = sum(_credits_of(c) for c in completed_codes)
    score.total_credits = score.completed_credits + score.planned_credits

    # --- UPPER LEVEL REQUIREMENT --------------------------------------------------------------
    #
    # "at least 32 semester hours of coursework ... expected to be at least junior-level
    # (30000+)" — the crawl states this as prose on a group with no options, so like the
    # University Core it is unscorable from the group tree and entirely computable from the plan
    # itself. Counted the same way as the graduation total: DISTINCT courses, completed plus
    # planned, each contributing its credits once.
    #
    # The test is the FIRST DIGIT of the course number, not `>= 30000`, so it holds for both this
    # catalog's five-digit codes (CS 30700) and any three-digit legacy code (CS 307) that reaches
    # the scorer — both are junior-level and both should count.
    upper = {c for c in ({canon(normalize_code(x)) for x in profile.completed_courses} | seen)
             if c in by_code and _is_upper_level(c)}
    score.upper_level_credits = sum(int(by_code[c].get("credit_hours_min") or 0) for c in upper)
    score.meets_upper_level = score.upper_level_credits >= score.upper_level_required
    # The program's own crawled minimum where it states one; 120 is Purdue's floor for a
    # bachelor's degree and the fallback when the catalog page omits it.
    program_rows = database.rows("programs")
    stated = next((r.get("total_credits_min") for r in program_rows if r.get("total_credits_min")),
                  None)
    score.graduation_credits_required = float(stated or 120)
    score.meets_graduation_credits = score.total_credits >= score.graduation_credits_required
    if not score.meets_graduation_credits:
        short = score.graduation_credits_required - score.total_credits
        missing_graduation = (
            f"{short:g} more credit(s) toward the {score.graduation_credits_required:g}-credit "
            f"graduation minimum ({score.completed_credits} completed + {score.planned_credits} "
            f"planned = {score.total_credits})"
        )
    else:
        missing_graduation = None

    canonical_satisfied = {canon(normalize_code(c)) for c in profile.completed_courses} | seen
    missing_labels: list[str] = []
    for group in _groups_from_tables(database):
        courses = group["courses"]
        if group["selective"]:
            filled = [c for c in courses if canon(c) in canonical_satisfied]
            credits_filled = sum(float(by_code[c].get("credit_hours_min") or 0)
                                 for c in filled if c in by_code)
            # NO STATED MINIMUM MEANS "CHOOSE ONE", NOT "CHOOSE THREE CREDITS". The 3.0 that
            # used to sit here was an invented figure — a standard single-course slot — and it
            # disagreed with `fixtures.select_remaining_courses`, which fills an unquantified
            # group with its cheapest option. A 1-credit option therefore satisfied the
            # selector and left the scorer asking for 2 more credits forever. Both ends now use
            # the same rule, and it is the one the catalog can actually support: if the crawl
            # recorded no minimum, one option from the list is the requirement.
            stated = float(group["credits_min"] or 0)
            option_credits = [float(by_code[c].get("credit_hours_min") or 0)
                              for c in courses if c in by_code]
            cheapest = min((c for c in option_credits if c > 0), default=3.0)
            target = stated or cheapest
            satisfied = credits_filled >= target
            missing_label = (None if satisfied else
                             f"{target - credits_filled:g} more credit(s) from this list")
            score.groups.append(RealGroupResult(
                name=group["name"], selective=True, credits_required=group["credits_min"],
                credits_filled=credits_filled, satisfied=satisfied, filled=filled,
                missing_label=missing_label,
            ))
        else:
            missing = [c for c in courses if canon(c) not in canonical_satisfied]
            filled = [c for c in courses if c not in missing]
            satisfied = not missing
            missing_label = None if satisfied else f"missing {', '.join(missing)}"
            score.groups.append(RealGroupResult(
                name=group["name"], selective=False, credits_required=group["credits_min"],
                credits_filled=float(len(filled)), satisfied=satisfied, filled=filled,
                missing_label=missing_label,
            ))
        if missing_label:
            missing_labels.append(f"{group['name']}: {missing_label}")

    score.requirement_coverage = score.groups_satisfied / score.groups_total \
        if score.groups_total else 1.0
    # LISTED WITH THE REQUIREMENTS, BUT NOT COUNTED IN `requirement_coverage`. Coverage is
    # "what fraction of the requirement groups are satisfied"; the credit floor is not a group
    # and folding it in would quietly change the denominator every existing number was computed
    # against. It appears here so a short plan says so in the one place a reader looks.
    if missing_graduation:
        missing_labels.append(f"Graduation credit minimum: {missing_graduation}")
    if not score.meets_upper_level:
        missing_labels.append(
            f"Upper Level Requirement: {score.upper_level_required - score.upper_level_credits:g} "
            f"more credit(s) from courses numbered 30000+ "
            f"({score.upper_level_credits} of {score.upper_level_required:g})"
        )
    score.missing_requirements = missing_labels
    score.viable = not counts and score.requirement_coverage >= 1.0

    # Same idle-capacity diagnostic as `plan_scorers.score_plan`: credits the student could
    # legally have taken but was not given, counted only while checkable requirements are still
    # unmet. Zero when coverage is complete.
    if score.requirement_coverage < 1.0:
        cap = profile.max_credits_per_semester
        horizon = max(profile.semesters_to_plan, len(score.semester_credits))
        padded = score.semester_credits + [0] * (horizon - len(score.semester_credits))
        score.idle_credits = sum(max(cap - c, 0) for c in padded[:horizon])

    if assertions:
        # Duck-typed: `check_assertions` only reads `.semester_credits` and
        # `.major_courses_per_semester` off `score`, both present here under the same names.
        from .plan_scorers import check_assertions
        score.assertions_passed = check_assertions(
            assertions, semesters, score, canonical={code: canon(code) for code in by_code})

    return score
