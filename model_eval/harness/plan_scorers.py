"""Plan-of-study scoring. The headline instrument of this harness.

Unlike faithfulness (a heuristic that needs your eyes), everything here is FULLY AUTOMATIC
and decidable: a plan either schedules CS 38100 before its prerequisites or it doesn't. The
only judgment call baked in is the fixture itself — see the provenance header in
``plan_fixtures/cs_machine_intelligence.yaml``. Wrong fixture rows bias every model the same
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
    term_offering_violation  scheduled in a term the course is not offered
    credit_cap_violation     semester credits exceed the profile's cap
    duplicate_course         the same course twice, or a course already completed
    major_overload_violation more than ``max_major_courses_per_semester`` courses from the
                             major's own subject in one term (default: more than 3 CS courses)

  Soft (recorded, NOT part of PLAN_VIABLE):

    soft_major_overloads     semesters holding more major-subject courses than
                             ``preferred_major_courses_per_semester`` but within the hard cap
                             (default: exactly 3 CS courses)

  Completeness:

    requirement_coverage     fraction of the fixture's requirement groups satisfied by
                             completed + planned courses
    missing_requirements     which groups are short, and by what

  PLAN_VIABLE = zero hard violations AND requirement_coverage == 1.0 within the horizon.
  That conjunction is the number to compare models on. ``violations`` is reported alongside
  it so a near-miss (one bad term offering) is visibly different from a plan that invented
  six courses.

DELIBERATELY NOT SCORED HERE: whether the plan is *pleasant* — workload balance, spreading
theory courses, summer usage. Those are preferences, not correctness, and rolling them into
the headline number would let a model trade a real prerequisite violation against a nicer
credit distribution. They are recorded as diagnostics (``credit_spread``) and left out of
PLAN_VIABLE on purpose.

THE MAJOR-COURSE CAP IS THE ONE EXCEPTION, and it is deliberate. Four or five CS courses in a
term is not a registration wall — the registrar will happily let a student sign up for it — so
by the standard above it belongs with the preferences. It is scored anyway because a plan a
student abandons in week six is not a plan, and because the models kept producing exactly that:
the transcripts are full of 4- and 5-CS semesters, and so, until this landed, was the
deterministic planner's own output. The soft/hard split is what keeps the trade honest: three
CS courses is a bad term and is recorded as one, four is treated as the plan being wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .fixtures import Fixture, SamplePlan
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
            alias = f" (counts as {key})" if key != code else ""
            if key in seen:
                flag("duplicate_course", f"{code}{alias} appears more than once")
            elif key in completed:
                flag("duplicate_course",
                     f"{code}{alias} was already completed before the plan starts")
            seen.add(key)

            credits += course.credits
            if is_major_course(code, profile.major_subject):
                major_courses += 1
            if course.offered_terms and term not in course.offered_terms:
                flag(
                    "term_offering_violation",
                    f"{code} in {term} {year}; offered {'/'.join(course.offered_terms)}",
                )
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
        if credits > profile.max_credits_per_semester:
            flag(
                "credit_cap_violation",
                f"semester {index + 1} has {credits} credits, cap is "
                f"{profile.max_credits_per_semester}",
            )
        subject = profile.major_subject
        if major_courses > profile.max_major_courses_per_semester:
            flag(
                "major_overload_violation",
                f"semester {index + 1} has {major_courses} {subject} courses, the limit is "
                f"{profile.max_major_courses_per_semester}",
            )
        elif major_courses > soft_major_cap:
            # SOFT: recorded, never flagged. It stays out of `counts` so it cannot change
            # PLAN_VIABLE — a term the student would find heavy is not a term that stops them
            # registering, and conflating the two would let a model trade a prereq violation
            # against a nicer workload.
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


def rationale_flags(rationale: str, planned_codes: set[str], fixture: Fixture) -> list[str]:
    """Course codes the rationale claims about that aren't in the plan or the catalog.

    Same triage-only status as ``scorers.faithfulness_flags``: it catches entity invention,
    not "CS 38100 will be easier in the spring"-style relational fiction.
    """
    flags: list[str] = []
    for subject, number in _COURSE_CODE_RE.findall(rationale.upper()):
        code = normalize_code(f"{subject} {number}")
        if code not in fixture.by_code:
            flags.append(f"rationale cites unknown course: {code}")
        elif code not in planned_codes:
            flags.append(f"rationale cites course not in the plan: {code}")
    return flags


def requirement_report_check(
    claimed: Any, fixture: Fixture, profile: Profile, semesters: list[list[str]],
    score: PlanScore,
) -> dict[str, Any]:
    """What the model SAID it covered, against what it actually did. Diagnostic only.

    `requirements_covered` is a required key per requirement group in the Mode B grammar (see
    prompts.plan_schema) — the model cannot close the object without accounting for the last
    group as well as the first, which is aimed squarely at the positional drop-off. This
    function is what tells you whether forcing the accounting made the model *cover* more or
    merely *claim* more, and those are completely different outcomes:

      claimed ⊇ covered, groups still missing -> it accounted, saw the gap, and shipped anyway
      claimed names courses not in the plan   -> the accounting itself is fiction
      claimed ≈ covered, fewer groups missing -> the forcing function worked

    NOTHING HERE FEEDS `requirement_coverage`. Coverage is recomputed from the semesters against
    the fixture, exactly as before, so a model that asserts CS 18000 satisfies `science` gains
    nothing — it just lands in `requirement_claims_unplanned` instead.
    """
    if not isinstance(claimed, dict):
        return {"requirement_report_present": False}

    # COMPLETED COURSES COUNT. A requirement is satisfied by what the student has already
    # taken as much as by what the plan schedules, so "cs-core is covered by CS 18000
    # (completed) + CS 25200 (planned)" is a correct claim, not a fabricated one. Checking only
    # against the semesters flagged all 5 of mi-spring-start's completed courses as phantom on
    # the first run of this scorer — the metric was wrong, not the model. Canonicalised on both
    # sides so an approved substitute (MA 16500 for MA 16100) is not a phantom either.
    canonical = fixture.canonical
    def resolve(code: str) -> str:
        normalized = normalize_code(code)
        return canonical.get(normalized, normalized)

    satisfied = {resolve(c) for semester in semesters for c in semester}
    satisfied |= {resolve(c) for c in profile.completed_courses}
    known = {g["id"] for g in fixture.requirement_groups}
    # A claim naming a course the plan does not contain: the accounting is decorative.
    phantom: list[str] = []
    # A group the model itself left empty — it noticed the gap rather than missing it.
    declared_empty: list[str] = []
    for gid, codes in claimed.items():
        if gid not in known or not isinstance(codes, list):
            continue
        if not codes:
            declared_empty.append(gid)
        phantom += [str(c) for c in codes if resolve(str(c)) not in satisfied]

    missing_groups = {m.split(":")[0] for m in score.missing_requirements}
    return {
        "requirement_report_present": True,
        # Groups the model admitted it could not cover. Read next to `missing_requirements`:
        # agreement means it knew; disagreement means the accounting did not reflect the plan.
        "requirement_claims_empty": sorted(declared_empty),
        "requirement_claims_unplanned": sorted(set(phantom)),
        # Did the model's own accounting agree with the scorer about WHICH groups fell short?
        "requirement_report_agrees": sorted(declared_empty) == sorted(missing_groups),
    }


def self_report_check(
    parsed_semesters: list[Any], score: PlanScore
) -> dict[str, Any]:
    """Did the model's own stated totals match what it actually scheduled?

    Mode B/C ask the model to fill `total_credits` and `major_course_count` per semester. Those
    values are never trusted — every score is recomputed from `courses` — but comparing them
    separates two failures that look identical in the violation counts:

      stated == actual, and over the cap  -> it knew and did it anyway (attention/priority)
      stated != actual                    -> it lost track of its own semester (arithmetic)

    A prompt fix helps the first; neither a prompt nor a grammar helps the second, which is an
    argument for the Mode C loop instead. Absent fields (an older record, or a model that
    skipped them despite the schema) count as neither right nor wrong.
    """
    credit_ok = credit_seen = major_ok = major_seen = 0
    for index, entry in enumerate(parsed_semesters):
        if not isinstance(entry, dict):
            continue
        actual_credits = (score.semester_credits[index]
                          if index < len(score.semester_credits) else None)
        actual_major = (score.major_courses_per_semester[index]
                        if index < len(score.major_courses_per_semester) else None)
        stated_credits, stated_major = entry.get("total_credits"), entry.get("major_course_count")
        if isinstance(stated_credits, int) and actual_credits is not None:
            credit_seen += 1
            credit_ok += stated_credits == actual_credits
        if isinstance(stated_major, int) and actual_major is not None:
            major_seen += 1
            major_ok += stated_major == actual_major
    return {
        "credits_self_reported_ok": (credit_ok == credit_seen) if credit_seen else None,
        "credits_self_report_errors": credit_seen - credit_ok,
        "major_self_reported_ok": (major_ok == major_seen) if major_seen else None,
        "major_self_report_errors": major_seen - major_ok,
    }


def plan_to_semesters(plan: Plan) -> list[list[str]]:
    return [semester.courses for semester in plan.semesters]


# --- Mode C: did the published sample plan get used, or copied? ------------------------------


def template_anchoring(
    sample: SamplePlan, semesters: list[list[str]], hallucinated: list[str],
    canonical: dict[str, str] | None = None,
) -> dict[str, Any]:
    """How closely a plan tracks the published sample plan. DIAGNOSTIC — never scored.

    Mode C's risk is not that a model ignores the reference; it is that the model reproduces it
    for a student it does not describe. Seven of the eight scenarios differ from the sample
    plan's assumed student (spring start, 12-credit cap, courses already completed, four
    semesters left), so on those the *right* answer looks LESS like the template, not more.
    A rise in ``slot_match`` next to a rise in term-offering or credit-cap violations is the
    signature of anchoring, and no other column in the harness shows it.

    Computed for Mode B as well as Mode C. Mode B never sees the sample plan, so its slot_match
    is the base rate — how much of the template a model reproduces from parametric memory alone
    — and the Mode C minus Mode B delta is what the reference material actually bought.

    ``off_catalog_from_reference`` splits hallucinations by origin: a code the sample plan
    dangled but the catalog does not have, versus one the model invented freehand. Only the
    first is a cost the reference material created. NOTE: with the current fixture this is
    zero by construction — every code the plan cites was adopted into the catalog on
    2026-07-27 — so it reads as a guard against a future major's plan, not a live signal.

    Substitutes are collapsed onto their primary first, so a model that schedules MA 16500 in
    the slot the plan gives MA 16100 is correctly counted as having followed the template.
    """
    canon = (lambda c: (canonical or {}).get(c, c))
    slot_of = {canon(k): v for k, v in sample.slot_of.items()}
    matched = placed = 0
    for index, codes in enumerate(semesters):
        for raw in codes:
            code = canon(normalize_code(raw))
            if code not in slot_of:
                continue
            placed += 1
            matched += slot_of[code] == index

    off_catalog = {normalize_code(c) for c in sample.off_catalog_codes}
    from_reference = sorted({normalize_code(c) for c in hallucinated} & off_catalog)
    return {
        "template_slot_match": round(matched / placed, 4) if placed else None,
        "template_courses_placed": placed,
        "template_courses_available": len(slot_of),
        "off_catalog_from_reference": from_reference,
        "sample_plan_hash": sample.plan_hash,
    }
