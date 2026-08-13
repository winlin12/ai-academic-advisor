"""Plan-of-study scoring. The headline instrument of this harness.

Unlike faithfulness (a heuristic that needs your eyes), everything here is FULLY AUTOMATIC
and decidable: a plan either schedules CS 38100 before its prerequisites or it doesn't. The
only judgment call baked in is the fixture itself — see the provenance header in
the program's own crawled database (`real_db.fixture_from_database`). Wrong rows bias every model the same
direction, which keeps rankings usable and absolute numbers suspect.

WHAT IS MEASURED

  Hard violations (any one of these makes a plan NOT viable — a student following it would
  hit a registration wall or fail to graduate):

    hallucinated_course      a code that does not exist in the catalog fixture
    prereq_violation         scheduled in a term where a prerequisite is not yet complete;
                             same-semester prerequisites do NOT count as satisfied
    coreq_violation          a COREQUISITE is neither already complete nor scheduled in the
                             same term. Corequisites are the exception to the rule above:
                             taking them together is what the catalog actually permits

  NOT a hard violation, not even soft — REMOVED FROM SCORING ENTIRELY, 2026-08-07:
  duplicate_course (the same course twice, or a course already completed). It stays a real,
  registration-wall-shaped mistake — Banner will not enroll a student in a course twice or
  re-enroll them in one already completed — but a plan that is otherwise flawless should not
  be scored the same as one with a genuine, unfixable prerequisite error just because it
  repeated one line. `dedupe_semesters()` below deletes every later occurrence (or the whole
  occurrence if it duplicates a completed course) BEFORE scoring runs at all, the same way
  Mode C's `repair` variant already deleted duplicates deterministically — this generalizes
  that to every mode instead of just one variant. What is removed is tracked, not discarded:
  every caller reports it back as `duplicates_removed`, so a model that keeps re-listing
  courses is still visible in the record, just not folded into PLAN_VIABLE.

  Soft (recorded, NOT part of PLAN_VIABLE):

    soft_major_overloads     semesters holding more major-subject courses than
                             ``preferred_major_courses_per_semester``. Was a hard
                             `major_overload_violation` above ``max_major_courses_per_semester``
                             until 2026-08-07 — a heavy major-course term is the student's own
                             call to adjust, same as a heavy credit term, not a registration wall
    soft_credit_overages     semesters over the student's stated ``max_credits_per_semester``.
                             Was a hard `credit_cap_violation` above ``hard_credit_cap`` until
                             2026-08-07 — removed entirely, not just softened, because the
                             registrar's ceiling is as changeable as the student's own ask, and
                             neither reflects something the student can't fix by asking

  REMOVED ENTIRELY, 2026-08-07 (not even soft): term_offering_violation. Whether a given course
  actually runs in a given future term is not knowable from this data — Purdue's published
  offering pattern is a historical observation, not a guarantee — so scoring against it charged
  models for a fact the harness itself could not verify.

  Completeness:

    requirement_coverage     fraction of the fixture's requirement groups satisfied by
                             completed + planned courses
    missing_requirements     which groups are short, and by what

  PLAN_VIABLE = zero hard violations AND requirement_coverage == 1.0 within the horizon.
  That conjunction is the number to compare models on. ``violations`` is reported alongside
  it so a near-miss (one missed prerequisite) is visibly different from a plan that invented
  six courses.

DELIBERATELY NOT SCORED HERE: whether the plan is *pleasant* — workload balance, spreading
theory courses, summer usage, a heavy major-course term. Those are preferences, not
correctness, and rolling them into the headline number would let a model trade a real
prerequisite violation against a nicer-looking schedule. They are recorded as diagnostics
(``credit_spread``, ``soft_major_overloads``, ``soft_credit_overages``) and left out of
PLAN_VIABLE on purpose — a student adjusts a heavy term by asking; a missed prerequisite
is not adjustable after the fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .fixtures import Fixture
from .planner import Plan, Profile, first_planning_term, is_major_course, next_term

_COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,5})\s?-?\s?(\d{3,5})\b")


def normalize_code(code: str) -> str:
    """'cs25100', 'CS-25100', 'CS 251' -> 'CS 25100'-shaped. Purely lexical.

    Models are not being graded on spacing. A three-digit legacy code is left as-is and will
    simply fail the catalog lookup — that IS the right outcome, since 'CS 251' is not a code
    the current catalog can register you for.
    """
    compact = "".join(str(code).split()).upper().replace("-", "")
    for index, char in enumerate(compact):
        if char.isdigit():
            return compact if index == 0 else f"{compact[:index]} {compact[index:]}"
    return compact


def dedupe_semesters(
    semesters: list[list[str]], *, canon: Callable[[str], str], completed: set[str],
) -> tuple[list[list[str]], list[str]]:
    """Delete every later occurrence of a repeated course, and every occurrence of a course
    already completed, BEFORE any scoring runs. Keeps the FIRST placement of anything else.

    Called once, by every mode, on the model's raw semesters — right after parsing (Mode B) or
    right after `apply_locked_slots` (Mode C), before `score_plan`/`score_against_real_db` ever
    see the plan. Neither scorer has to know this happened: cleaned input simply never contains
    a duplicate to flag. ``canon`` must already fold approved substitutes to the same key the
    caller's own scorer uses (``fixture.canonical`` for `plan_scorers`, the DB's `alias_of` map
    for `real_scoring`) — two spellings of the same course are one course for this purpose.

    Returns (cleaned_semesters, removed) where ``removed`` is a human-readable line per deleted
    occurrence, in semester order — the record this drops the "duplicate_course" violation in
    favour of, see this module's docstring.
    """
    seen: set[str] = set()
    cleaned: list[list[str]] = []
    removed: list[str] = []
    for index, courses in enumerate(semesters):
        kept: list[str] = []
        for raw in courses:
            code = normalize_code(raw)
            key = canon(code)
            if key in seen:
                removed.append(f"{code} (semester {index + 1}) — duplicate, removed before scoring")
                continue
            if key in completed:
                removed.append(
                    f"{code} (semester {index + 1}) — already completed before the plan "
                    f"starts, removed before scoring"
                )
                continue
            seen.add(key)
            kept.append(raw)
        cleaned.append(kept)
    return cleaned, removed


@dataclass
class PlanScore:
    structure_ok: bool = False          # did we get a parseable plan at all?
    viable: bool = False
    violations: list[str] = field(default_factory=list)
    violation_counts: dict[str, int] = field(default_factory=dict)
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = field(default_factory=list)
    planned_credits: int = 0
    semester_credits: list[int] = field(default_factory=list)
    credit_spread: int = 0              # max - min across non-empty semesters (diagnostic)
    idle_credits: int = 0               # unused capacity while requirements went unmet
    semesters_used: int = 0              # non-empty semesters, vs profile.semesters_to_plan
    semesters_available: int = 0
    major_courses_per_semester: list[int] = field(default_factory=list)
    soft_major_overloads: int = 0       # semesters over the preferred cap but within the hard one
    soft_credit_overages: int = 0       # semesters over the STUDENT's cap but within the registrar's
    hallucinated: list[str] = field(default_factory=list)
    assertions_passed: dict[str, bool] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "structure_ok": self.structure_ok,
            "plan_viable": self.viable,
            "violations": self.violations,
            "violation_counts": self.violation_counts,
            "requirement_coverage": round(self.requirement_coverage, 4),
            "missing_requirements": self.missing_requirements,
            "planned_credits": self.planned_credits,
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
        }


def _terms(profile: Profile, count: int) -> list[tuple[str, int]]:
    """The calendar a plan's Nth semester actually falls in.

    Must snap the start term the same way the planner does. When these two disagree the
    scorer grades semester N against the wrong term and reports term-offering violations for
    a schedule that is perfectly legal — which is exactly what happened the first time summer
    was switched off.
    """
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(count):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def requirement_coverage(
    fixture: Fixture, satisfied: set[str]
) -> tuple[float, list[str]]:
    """Fraction of requirement groups met, plus a human-readable list of what's short.

    ``satisfied`` must already be canonicalised (see ``Fixture.canonical``) — a student who
    took MA 16500 has met the group that lists MA 16100. Credits are counted from the code the
    GROUP lists, not the substitute actually taken, since the group is what defines the target.
    """
    groups = fixture.requirement_groups
    if not groups:
        return 1.0, []
    canonical = fixture.canonical
    met = 0
    missing: list[str] = []
    for group in groups:
        courses = group.get("courses", [])
        if group.get("kind") == "all":
            short = [c for c in courses if canonical.get(c, c) not in satisfied]
            if not short:
                met += 1
            else:
                missing.append(f"{group['id']}: missing {', '.join(short)}")
        else:
            have = sum(fixture.credits(c) for c in courses
                       if canonical.get(c, c) in satisfied)
            target = float(group.get("choose_credits") or 0)
            if have >= target:
                met += 1
            else:
                missing.append(
                    f"{group['id']}: {have:g}/{target:g} credits "
                    f"(options: {', '.join(courses)})"
                )
    return met / len(groups), missing


def score_plan(
    fixture: Fixture,
    profile: Profile,
    semesters: list[list[str]],
    *,
    assertions: list[dict[str, Any]] | None = None,
) -> PlanScore:
    """Score a schedule expressed as a list of per-semester course-code lists.

    Works for BOTH modes: Mode A passes the deterministic planner's output (so hard
    violations should be zero by construction — if they aren't, the harness's planner port
    has drifted from the app's and that is worth knowing), Mode B passes whatever JSON the
    model produced.
    """
    score = PlanScore(structure_ok=True)
    by_code = fixture.by_code
    counts: dict[str, int] = {}

    def flag(kind: str, detail: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        score.violations.append(f"{kind}: {detail}")

    # Approved substitutes collapse to the course they stand in for, BEFORE any check runs.
    # All three checks that care — duplicates, prereqs, coverage — then agree by construction:
    # MA 16500 satisfies a prereq naming MA 16100, counts toward the group listing MA 16100,
    # and scheduling both is scheduling one course twice.
    canonical = fixture.canonical
    canon = lambda code: canonical.get(code, code)

    completed = {canon(normalize_code(c)) for c in profile.completed_courses}
    seen: set[str] = set()
    terms = _terms(profile, len(semesters))

    soft_major_cap = min(profile.preferred_major_courses_per_semester,
                         profile.max_major_courses_per_semester)

    for index, (raw_codes, (term, year)) in enumerate(zip(semesters, terms)):
        codes = [normalize_code(c) for c in raw_codes]
        # Everything scheduled in THIS term, canonicalised — what a corequisite may lean on.
        this_semester = {canon(c) for c in codes}
        credits = 0
        # Counted over catalog courses only, exactly like `credits` below: a hallucinated
        # "CS 99999" is already a hard violation on its own and must not also inflate a second
        # one, or one mistake would be billed twice.
        major_courses = 0
        # Prereqs are evaluated against courses completed BEFORE this semester: a course
        # taken alongside its own prerequisite is a scheduling error, not a co-requisite.
        available = set(completed)

        for code in codes:
            course = by_code.get(code)
            if course is None:
                score.hallucinated.append(code)
                flag("hallucinated_course", f"{code} (semester {index + 1}) is not in the catalog")
                continue
            key = canon(code)
            if key in seen or key in completed:
                # Every real caller runs `dedupe_semesters` before scoring, so this should
                # never fire in practice — kept as a defensive SKIP, not a violation, so a
                # caller that forgot to dedupe still cannot fail a plan over a duplicate the
                # way it would over a genuine one. See this module's docstring.
                continue
            seen.add(key)

            credits += course.credits
            if is_major_course(code, profile.major_subject):
                major_courses += 1
            missing = [p for p in course.prereqs if canon(normalize_code(p)) not in available]
            if missing:
                flag(
                    "prereq_violation",
                    f"{code} in semester {index + 1} needs {', '.join(missing)} first",
                )
            # COREQUISITES take the looser reading here than in the planner: same semester
            # counts. The scorer's job is to not charge a violation for a schedule the
            # registrar would accept, and a coreq taken alongside its course is exactly that.
            # Measured cost of getting this wrong: CS 37300 needing STAT 35000 was 15% of every
            # prereq violation in the corpus — models were being marked down for a legal plan.
            missing_co = [
                c for c in course.coreqs
                if canon(normalize_code(c)) not in available | this_semester
            ]
            if missing_co:
                flag(
                    "coreq_violation",
                    f"{code} in semester {index + 1} needs {', '.join(missing_co)} "
                    f"in the same semester or earlier",
                )

        score.semester_credits.append(credits)
        score.major_courses_per_semester.append(major_courses)
        # SOFT ONLY, never gates PLAN_VIABLE — a heavy semester is the student's own call and
        # can change from one advising conversation to the next; it is not a registration wall
        # the way a missed prerequisite is. It IS still scored — via the `max_credits_at_most`
        # assertion, which is where a term the student did not ask for belongs.
        if credits > profile.max_credits_per_semester:
            score.soft_credit_overages += 1
        # SOFT ONLY, same call as the credit cap above and for the same reason: a heavy major
        # course load is the student's own call to adjust, not a registration wall, so it no
        # longer gates PLAN_VIABLE. `max_major_courses_per_semester` stays the higher of the
        # two thresholds this counts against; `preferred_major_courses_per_semester` no longer
        # needs its own separate min() cap now that neither one is a hard violation.
        if major_courses > soft_major_cap:
            score.soft_major_overloads += 1
        completed |= {canon(c) for c in codes if c in by_code}

    score.planned_credits = sum(score.semester_credits)
    # Distribution diagnostics. NOT scored — a plan that finishes early is not illegal, and a
    # student who wants to graduate early should front-load. These make the shape visible so
    # the target-vs-cap change can be read off the report instead of inferred from
    # credit_spread alone.
    score.semesters_available = profile.semesters_to_plan
    score.semesters_used = sum(1 for c in score.semester_credits if c > 0)
    nonempty = [c for c in score.semester_credits if c > 0]
    score.credit_spread = (max(nonempty) - min(nonempty)) if nonempty else 0

    satisfied = {canon(normalize_code(c)) for c in profile.completed_courses} | seen
    score.requirement_coverage, score.missing_requirements = requirement_coverage(
        fixture, satisfied
    )
    score.violation_counts = counts
    score.viable = not counts and score.requirement_coverage >= 1.0

    # Idle capacity: credits the student could legally have taken but was not given, counted
    # only while requirements are still unmet. Without this, "schedule almost nothing" scores
    # as the cleanest strategy in the whole harness — a 1-course plan has zero violations and
    # the lowest violations-per-plan of any model. It is the failure mode the prompt's
    # "a short legal plan is better than a long illegal one" actively invites, and no existing
    # column made it visible. Zero when coverage is complete: an under-full plan that satisfied
    # the degree is not wasting anything.
    if score.requirement_coverage < 1.0:
        cap = profile.max_credits_per_semester
        horizon = max(profile.semesters_to_plan, len(score.semester_credits))
        padded = score.semester_credits + [0] * (horizon - len(score.semester_credits))
        score.idle_credits = sum(max(cap - c, 0) for c in padded[:horizon])

    if assertions:
        score.assertions_passed = check_assertions(
            assertions, semesters, score, canonical=canonical)
    return score


# --- scenario assertions -------------------------------------------------------------------
# Mode A's problem: the deterministic planner guarantees legality, so plan viability alone
# cannot separate a model that understood the student from one that returned an empty
# proposal. These assertions ask the only question left — was the student's stated ask
# actually acted on?


def check_assertions(
    assertions: list[dict[str, Any]],
    semesters: list[list[str]],
    score: PlanScore,
    proposal: dict[str, Any] | None = None,
    canonical: dict[str, str] | None = None,
) -> dict[str, bool]:
    """``canonical`` collapses approved substitutes onto their primary, so an assertion naming
    MA 16100 is honoured by a plan that scheduled MA 16500. Optional so a caller without a
    fixture still works; every caller in the harness passes it."""
    canon = (lambda c: (canonical or {}).get(c, c))
    placement: dict[str, int] = {}
    for index, codes in enumerate(semesters):
        for code in codes:
            placement.setdefault(canon(normalize_code(code)), index)

    results: dict[str, bool] = {}
    for spec in assertions:
        kind = spec.get("kind")
        if kind == "scheduled_before":
            code = canon(normalize_code(spec["course"]))
            index = placement.get(code)
            key = f"{kind}:{code}<{spec['semester_index']}"
            results[key] = index is not None and index < int(spec["semester_index"])
        elif kind == "any_scheduled_before":
            codes = [canon(normalize_code(c)) for c in spec["courses"]]
            key = f"{kind}:{'|'.join(codes)}<{spec['semester_index']}"
            results[key] = any(
                placement.get(c) is not None and placement[c] < int(spec["semester_index"])
                for c in codes
            )
        elif kind == "none_scheduled":
            # The student asked for something to be LEFT OUT. Every other assertion kind asks
            # "did the model do the thing"; this one asks "did it refrain", which is the only
            # way to score a negative preference. Note it is trivially true whenever the
            # courses could not have been scheduled anyway — see the mi-no-filler scenario's
            # notes for why that makes it a Mode B/C assertion in practice, not a Mode A one.
            codes = [canon(normalize_code(c)) for c in spec["courses"]]
            key = f"{kind}:{'|'.join(codes)}"
            results[key] = not any(placement.get(c) is not None for c in codes)
        elif kind == "max_credits_at_most":
            key = f"{kind}:{spec['value']}"
            results[key] = all(c <= int(spec["value"]) for c in score.semester_credits)
        elif kind == "max_major_courses_at_most":
            # For a student who asks for a lighter major load than the default cap allows
            # ("no more than two CS classes at a time"). The cap itself is scored as a
            # violation; this scores whether a specific ASK was honoured.
            key = f"{kind}:{spec['value']}"
            results[key] = all(
                c <= int(spec["value"]) for c in score.major_courses_per_semester
            )
        elif kind == "proposal_sets_credit_cap":
            # Mode B has no proposal; the assertion is not applicable and is skipped rather
            # than scored as a failure (counting N/A as a miss would penalise Mode B for a
            # question it was never asked).
            if proposal is not None:
                key = f"{kind}:{spec['value']}"
                results[key] = proposal.get("max_credits_per_semester") == int(spec["value"])
    return results


# --- Mode A: proposal quality ---------------------------------------------------------------


def score_proposal(
    proposal: dict[str, Any], profile: Profile, fixture: Fixture
) -> dict[str, Any]:
    """Is the model's PlanEditProposal GROUNDED — does it name things that exist?

    The app degrades a hallucinated proposal to a no-op by design, so a bad proposal never
    produces an illegal plan; it just produces a plan that ignored the student. This is where
    that silent failure becomes visible.
    """
    remaining = {normalize_code(c) for c in profile.remaining_courses}
    known_tags = {tag.lower() for course in fixture.catalog for tag in course.requirement_tags}

    reorder = [normalize_code(c) for c in proposal.get("reorder") or []]
    defer = [normalize_code(c) for c in proposal.get("defer") or []]
    tags = [str(t).strip().lower() for t in proposal.get("avoid_tags") or []]

    # Truncated: the most common failure here is a model writing a whole semester layout
    # ("fall 2026: CS 25000, CS 25100, ... [14 cr]") into a field that expects one course
    # code. Those strings are enormous and would otherwise dominate the JSONL, so keep enough
    # to recognise the shape and no more.
    ungrounded_codes = [c[:60] for c in reorder + defer if c not in remaining]
    ungrounded_tags = [t[:60] for t in tags if t and t not in known_tags]
    conflicts = sorted(set(reorder) & set(defer))
    cap = proposal.get("max_credits_per_semester")

    touched = bool(reorder or defer or tags or cap is not None)
    return {
        "proposal_touched_anything": touched,
        "proposal_grounded": not ungrounded_codes and not ungrounded_tags and not conflicts,
        "ungrounded_codes": ungrounded_codes,
        "ungrounded_tags": ungrounded_tags,
        "reorder_defer_conflict": conflicts,
        "proposed_credit_cap": cap,
        "rationale_len": len(str(proposal.get("rationale") or "")),
    }


def rationale_flags(
    rationale: str, planned_codes: set[str], known_codes: "set[str] | dict[str, Any]"
) -> list[str]:
    """Course codes the rationale claims about that aren't in the plan or the catalog.

    Same triage-only status as ``scorers.faithfulness_flags``: it catches entity invention,
    not "CS 38100 will be easier in the spring"-style relational fiction.

    ``known_codes`` takes a plain membership container rather than a ``Fixture`` — Mode A
    passes ``ctx.fixture.by_code`` (its course universe is genuinely the fixture's), Mode B
    passes ``ctx.database.course_codes`` (the real database it was actually shown). Only ``in``
    is ever called on it, so either shape works.
    """
    flags: list[str] = []
    for subject, number in _COURSE_CODE_RE.findall(rationale.upper()):
        code = normalize_code(f"{subject} {number}")
        if code not in known_codes:
            flags.append(f"rationale cites unknown course: {code}")
        elif code not in planned_codes:
            flags.append(f"rationale cites course not in the plan: {code}")
    return flags


def plan_to_semesters(plan: Plan) -> list[list[str]]:
    return [semester.courses for semester in plan.semesters]
