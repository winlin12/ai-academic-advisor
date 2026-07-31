"""MODE C — retry-to-convergence.

Modes A and B ask "how good is one attempt?". This asks a different question: **given repeated
attempts, how many does a model need — and how long does it take — to land a plan with no
errors in it?** That is closer to how the product is actually used. A student does not accept
or discard a single plan; they iterate with the assistant until the thing is clean.

THE NAME "MODE C" IS REUSED, DELIBERATELY, AND THERE IS ONE THING TO WATCH.
`results_old8/runs_*.jsonl` on this box already contains records with `"stage": "plan_mode_c"`
carrying a completely different meaning — the removed reference arm (Mode B plus the
department's published sample plan, later Mode B with scorer feedback). Those records are dead
data from a deleted arm, and this mode now owns the name.

What keeps the two apart is not the label, it is everything else: this mode writes
`runs_convergence.jsonl` (the old arm wrote `runs_baseline.jsonl`), its report reads only that
file, and its records carry a `variant` field and a static-prompt hash the old ones do not have.
An analysis that globs `results*/runs_baseline.jsonl` across directories is the one place the
collision could bite — check `variant` is present before pooling anything called `plan_mode_c`.

THREE VARIANTS, TRACKED SEPARATELY, NEVER AVERAGED.

  blind     — same prompt every attempt, new sample (seed moves, temperature optionally
              climbs). No information about what went wrong. Measures the variance of the
              model's own distribution: how often does a good plan fall out by luck?
  feedback  — the validator's specific violations are appended to the next attempt's user
              message. Measures self-correction: can the model read a structured error and
              fix it?
  repair    — the HARNESS deletes the violating placements (deterministic, no model needed)
              and FREEZES everything that validates; the model is only ever asked to fill what
              is still missing. Measures the thing the product would actually ship: how many
              rounds of "fill the gaps" to finish a degree-complete plan.

These are different capabilities and a single "attempts to converge" column that mixed them
would describe neither. They are separate variants in the record and separate columns in the
report.

`repair` IS SCORED ON A DIFFERENT CRITERION, and it has to be — see the block above
`auto_repair`. Its repair step guarantees a clean plan by construction, so "no violations" stops
being evidence about the model; it is scored STRICTLY instead (clean AND every requirement group
covered). Comparing its `attempts p50` against the other two is comparing two different bars,
and the report says so next to the table.

CENSORING IS THE POINT OF THE STATISTICS. A case that runs out of clock at attempt 7 has NOT
told you the model needs 7 attempts — it has told you the model needs *more than* 7. Recording
that as a failure (0%) understates the model and recording it as 7 overstates it. Both are
wrong in ways that move rankings. Every case therefore carries `censored` and `censor_reason`,
and `convergence_report.py` uses a Kaplan-Meier estimator so the censored cases contribute what
they actually know rather than being dropped or flattened.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not touch `plan_scorers.score_plan`. The
convergence criterion is computed FROM that scorer's output (`violation_counts`), so Mode C and
Mode B are validating against one implementation. A second copy of the prereq/credit/term logic
that drifted from the first would be undetectable from the outside and would land squarely in
the convergence numbers.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import plan_scorers, scorers
from .fixtures import Fixture, Scenario
from .llamacpp_client import LlamaCppClient, LlamaCppError
from .planner import Profile, generate_plan
from .prompts import json_response_format, plan_schema

# The spec's definition of a valid plan of study, verbatim: prerequisites, credit-hour caps,
# term offerings. See `CONVERGENCE_CRITERION_CAVEAT` for what this leaves out and why the
# report prints a second column next to it.
DEFAULT_VIOLATION_CLASSES = (
    "prereq_violation",
    "credit_cap_violation",
    "term_offering_violation",
)

CONVERGENCE_CRITERION_CAVEAT = """\
`converged` uses the three violation classes Mode C was specified with, over FILLED slots only,
with no requirement-coverage condition. An EMPTY PLAN THEREFORE CONVERGES ON ATTEMPT 1: zero
filled slots is zero violations. That is not a hypothetical — the feedback variant actively
pushes toward it, because deleting the course named in a violation always removes that
violation. `converged_strict` (full PLAN_VIABLE: all seven violation classes plus 100%
requirement coverage) and `degenerate` are recorded on every attempt so this is visible rather
than silently inflating the headline. Read them together."""


# --- locked slots -------------------------------------------------------------------------


@dataclass(frozen=True)
class LockedSlot:
    """A (semester, course) pair the student fixed. Never revisable."""

    semester_index: int
    course: str


@dataclass
class LockedSlots:
    path: Path
    slots_hash: str
    by_scenario: dict[str, list[LockedSlot]]

    def for_scenario(self, scenario_id: str) -> list[LockedSlot]:
        return self.by_scenario.get(scenario_id, [])


def load_locked_slots(path: Path) -> LockedSlots:
    import hashlib

    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw.decode("utf-8")) or {}
    by_scenario: dict[str, list[LockedSlot]] = {}
    for scenario_id, rows in (data.get("scenarios") or {}).items():
        by_scenario[scenario_id] = [
            LockedSlot(
                semester_index=int(row["semester_index"]),
                course=plan_scorers.normalize_code(str(row["course"])),
            )
            for row in rows or []
        ]
    return LockedSlots(
        path=Path(path),
        slots_hash=hashlib.sha256(raw).hexdigest()[:16],
        by_scenario=by_scenario,
    )


def apply_locked_slots(
    semesters: list[list[str]], locked: list[LockedSlot], horizon: int
) -> tuple[list[list[str]], list[str]]:
    """Strip every locked course out of the model's plan, then re-insert it where it was pinned.

    This is the "strip out of the per-attempt pass and re-insert as fixed constraints" rule,
    and doing it in that order matters. If a locked course were merely *checked* in place, a
    model that moved CS 25100 from semester 3 to semester 1 would be scored on the plan it
    wrote rather than the plan the student would receive — and the student receives the locked
    position, because that is what "locked" means in the product. Re-inserting makes the scored
    artifact the real one.

    The model's ATTEMPT to move a locked course is not discarded, though: it is returned as an
    alteration and recorded. A model that keeps fighting the locks is a different failure from
    one that respects them and still cannot sequence, and only this list separates them.
    """
    pinned = {slot.course for slot in locked}
    alterations: list[str] = []

    width = max(horizon, len(semesters), *( [s.semester_index + 1 for s in locked] or [0] ))
    out: list[list[str]] = [[] for _ in range(width)]

    for index, courses in enumerate(semesters):
        for raw in courses:
            code = plan_scorers.normalize_code(raw)
            if code in pinned:
                target = next(s.semester_index for s in locked if s.course == code)
                if index != target:
                    alterations.append(
                        f"{code} moved to semester {index + 1}; locked to semester {target + 1}"
                    )
                continue  # dropped here, re-inserted below at the pinned index
            out[index].append(raw)

    for slot in locked:
        if slot.course not in {plan_scorers.normalize_code(c)
                               for courses in semesters for c in courses}:
            alterations.append(f"{slot.course} was dropped; locked to "
                               f"semester {slot.semester_index + 1}")
        out[slot.semester_index].append(slot.course)

    # Trailing empties are meaningful in this mode ("an empty slot is a legitimate terminal
    # state"), but empties BEYOND the horizon are an artifact of a model writing extra
    # semesters, and they would inflate `semesters_available`. Trim only past the horizon.
    while len(out) > horizon and not out[-1]:
        out.pop()
    return out, alterations


def locked_slot_problems(
    fixture: Fixture, scenario: Scenario, locked: list[LockedSlot]
) -> list[str]:
    """Is this locked set even satisfiable? Run before spending a single GPU-minute.

    An unreachable lock ("CS 38100 in semester 1, with nothing behind it") makes convergence
    impossible for reasons that have nothing to do with the model — every model would time out
    and the benchmark would report a capability difference that is really a fixture bug. This
    is the same guard `run.py check` applies to the scenarios themselves.
    """
    problems: list[str] = []
    known = set(fixture.by_code)
    horizon = scenario.profile.semesters_to_plan
    terms = plan_scorers._terms(scenario.profile, horizon)
    canonical = fixture.canonical
    completed = {canonical.get(plan_scorers.normalize_code(c), c)
                 for c in scenario.profile.completed_courses}

    seen: dict[str, int] = {}
    for slot in locked:
        if slot.course not in known:
            problems.append(f"{scenario.id}: locked course {slot.course} is not in the catalog")
            continue
        if not 0 <= slot.semester_index < horizon:
            problems.append(
                f"{scenario.id}: {slot.course} is locked to semester "
                f"{slot.semester_index + 1}, outside the {horizon}-semester horizon"
            )
            continue
        if slot.course in seen:
            problems.append(f"{scenario.id}: {slot.course} is locked twice")
        seen[slot.course] = slot.semester_index

        if canonical.get(slot.course, slot.course) in completed:
            problems.append(
                f"{scenario.id}: {slot.course} is locked but the student already completed it"
            )
        course = fixture.by_code[slot.course]
        term = terms[slot.semester_index][0]
        if course.offered_terms and term not in course.offered_terms:
            problems.append(
                f"{scenario.id}: {slot.course} is locked to {term} but is offered only in "
                f"{'/'.join(course.offered_terms)}"
            )

    # A locked course whose prerequisite is ALSO locked, later or in the same term, can never
    # be satisfied no matter what the model does with the free slots.
    for slot in locked:
        course = fixture.by_code.get(slot.course)
        if course is None:
            continue
        for prereq in course.prereqs:
            code = plan_scorers.normalize_code(prereq)
            if canonical.get(code, code) in completed:
                continue
            if code in seen and seen[code] >= slot.semester_index:
                problems.append(
                    f"{scenario.id}: {slot.course} (semester {slot.semester_index + 1}) needs "
                    f"{code}, which is locked to semester {seen[code] + 1} — unsatisfiable"
                )
            elif slot.semester_index == 0 and code not in completed:
                problems.append(
                    f"{scenario.id}: {slot.course} is locked to the FIRST semester but needs "
                    f"{code}, which the student has not completed — unsatisfiable"
                )
    return problems


# --- the convergence criterion ---------------------------------------------------------------


_PREREQ_RE = re.compile(r"^(?P<code>[A-Z]{2,5} \d{3,5}) in semester \d+ needs (?P<needs>.+) first$")


def filter_excluded_edges(
    score: plan_scorers.PlanScore, excluded: set[str]
) -> tuple[list[str], dict[str, int]]:
    """Drop prereq violations for edges the operator has excluded, e.g. ``CS 38100->MA 26100``.

    Mode C compounds: every attempt re-encounters the same edge, so ONE wrong prerequisite in
    the fixture does not cost a model one violation, it costs it the entire convergence run.
    That is why this exists and why Mode B has no equivalent — there the same bad edge costs a
    single scored plan and shows up as a constant offset across models.

    Excluding an edge is a loud, recorded act: the excluded set travels in the meta file and is
    printed in the report header. It is NOT a way to make a model look better; it is a way to
    stop a known-bad fixture row from eating the measurement while it is being fixed.
    """
    if not excluded:
        return list(score.violations), dict(score.violation_counts)

    kept: list[str] = []
    counts: dict[str, int] = {}
    for violation in score.violations:
        kind, _, detail = violation.partition(": ")
        if kind == "prereq_violation":
            match = _PREREQ_RE.match(detail)
            if match:
                code = match.group("code")
                needs = [n.strip() for n in match.group("needs").split(",")]
                if all(f"{code}->{n}" in excluded for n in needs):
                    continue
        kept.append(violation)
        counts[kind] = counts.get(kind, 0) + 1
    return kept, counts


# --- deterministic repair + the validated-placement ratchet -----------------------------------
#
# WHY THIS VARIANT EXISTS. The `feedback` arm asks the model to delete a course the validator
# has ALREADY identified by name, semester and reason. That is not a reasoning task — it is a
# deletion the harness can do itself, correctly, every time, in microseconds. Worse, it costs a
# whole round trip and the model routinely rewrites the entire plan to do it: gemma4-e4b's first
# case went 17 -> 4 -> 10 -> 14 violations, discarding good placements on every pass, because
# nothing anchored the work it had already got right.
#
# So `repair` splits the job along the line where the split actually is:
#
#   the HARNESS removes violations   — deterministic, decidable, no model needed
#   the MODEL fills what is missing  — the part that needs judgement
#
# and every placement that survives validation is FROZEN into the locked set for the next
# attempt. The frozen set only grows, so good work is never lost and progress is monotonic —
# a ratchet, not a re-roll.
#
# THE CONVERGENCE CRITERION MUST CHANGE WITH IT, and this is not optional. Under the specified
# criterion (zero violations over filled slots, no coverage condition) this variant would
# converge on attempt 1 for EVERY model, every time — the repair step guarantees a clean plan by
# construction, so "clean" stops being evidence about the model at all. `repair` is therefore
# scored on the STRICT criterion: clean AND every requirement group covered. What it measures is
# how many rounds of "fill the gaps" it takes to finish a degree-complete plan.

_V_SEMESTER = re.compile(r"semester (\d+)")
_V_CODE = re.compile(r"^([A-Z]{2,5} \d{3,5})")


def parse_violation(violation: str) -> tuple[str, str | None, int | None]:
    """(kind, offending course, 0-based semester) from a scorer violation string.

    The scorer stays the single source of truth for DETECTING a violation; this only decides
    which placement to drop. Reimplementing the detection here is the one thing that would make
    Mode C and Mode B disagree silently.
    """
    kind, _, detail = violation.partition(": ")
    code_match = _V_CODE.match(detail)
    code = code_match.group(1) if code_match else None
    sem_match = _V_SEMESTER.search(detail)
    index = int(sem_match.group(1)) - 1 if sem_match else None
    return kind, code, index


def _dependency_closure(fixture: Fixture, codes: Iterable[str]) -> set[str]:
    """Everything ``codes`` transitively depend on, prerequisites and corequisites alike."""
    out: set[str] = set()
    stack = list(codes)
    while stack:
        course = fixture.by_code.get(stack.pop())
        if course is None:
            continue
        for dep in tuple(course.prereqs) + tuple(course.coreqs):
            dep = plan_scorers.normalize_code(dep)
            if dep not in out:
                out.add(dep)
                stack.append(dep)
    return out


def _removal_priority(
    fixture: Fixture, code: str, depended_on: set[str]
) -> tuple[int, int, str]:
    """Cheapest thing to drop first — and "cheapest" is mostly about what else breaks.

    THE DEPENDENCY TIER DOMINATES, and it is the fix for a real bug: the first version ranked
    only on requirement membership and credits, so a credit-cap violation in semester 1 was
    cleared by deleting MA 16100 — a five-credit course that half the plan sits behind. That
    stranded CS 18200, whose removal stranded CS 25100, which the student had LOCKED, leaving a
    violation the harness had made and could not clear. One repair pass removed 19 courses and
    still ended dirty.

    So: drop leaves first. A course nothing else in the plan depends on can go without
    consequence; one that others sit behind cascades, and cascading is how a minimal repair
    turns into demolition. Only within a tier does filler-before-required and
    bigger-before-smaller apply (bigger clears a credit cap in fewer removals).
    """
    in_group = any(code in g.get("courses", []) for g in fixture.requirement_groups)
    course = fixture.by_code.get(code)
    return (
        2 if code in depended_on else 0,
        1 if in_group else 0,
        # Tie-break on the code so a repair is deterministic; the credit term is folded in
        # ahead of it via the tuple below.
        f"{999 - (course.credits if course else 0):03d}{code}",
    )


def auto_repair(
    fixture: Fixture,
    profile: Profile,
    semesters: list[list[str]],
    locked: list[LockedSlot],
    *,
    max_passes: int = 12,
) -> tuple[list[list[str]], list[str], list[str]]:
    """Delete violating placements until the plan is clean. Never touches a locked slot.

    Returns (repaired, removed, unrepairable). ``unrepairable`` is non-empty when the only way
    to clear a violation would be to move a course the STUDENT pinned — the harness must not,
    so it reports the conflict instead of silently overriding the person it is planning for.

    Removal cascades on purpose: dropping CS 25100 can invalidate a CS 37300 that depended on
    it, so the loop re-scores after each pass rather than computing one removal set up front.
    """
    pinned = {slot.course for slot in locked}
    # PROTECTED = the pins themselves plus everything they transitively depend on. Removing a
    # course a locked one sits behind creates a violation on a placement the student fixed —
    # which the harness then cannot clear, because it may not move the pin either. It is the
    # harness manufacturing an unrepairable plan out of a repairable one.
    protected = pinned | _dependency_closure(fixture, pinned)
    working = [list(courses) for courses in semesters]
    removed: list[str] = []
    unrepairable: list[str] = []

    for _ in range(max_passes):
        score = plan_scorers.score_plan(fixture, profile, working)
        if not score.violation_counts:
            break

        # Recomputed every pass: removing a course can turn another one into a leaf, and a
        # stale set would keep protecting something nothing depends on any more.
        scheduled = [plan_scorers.normalize_code(c) for s in working for c in s]
        depended_on = _dependency_closure(fixture, scheduled) & set(scheduled)

        drop: list[tuple[int, str]] = []
        blocked: list[str] = []
        for violation in score.violations:
            kind, code, index = parse_violation(violation)

            if kind in ("credit_cap_violation", "major_overload_violation"):
                # No single course is "the" offender; the semester is over a limit. Shed the
                # cheapest non-locked courses until it is not.
                if index is None or not 0 <= index < len(working):
                    continue
                is_major = kind == "major_overload_violation"
                candidates = [
                    c for c in (plan_scorers.normalize_code(x) for x in working[index])
                    if c not in protected
                    and (not is_major or plan_scorers.is_major_course(c, profile.major_subject))
                ]
                if not candidates:
                    blocked.append(
                        f"{violation} (every course in it is locked or holds up a locked one)")
                    continue
                candidates.sort(key=lambda c: _removal_priority(fixture, c, depended_on))
                drop.append((index, candidates[0]))
                continue

            if code is None:
                continue
            if code in pinned:
                blocked.append(f"{violation} (the offending course is locked)")
                continue

            if kind == "duplicate_course" and "appears more than once" in violation:
                # Keep the FIRST placement, drop the later one — the earlier slot is the one
                # the rest of the plan may already be sequenced behind.
                #
                # ALIAS DUPLICATES BREAK THE NAIVE VERSION, and did: the scorer detects
                # duplicates on the CANONICAL code, so a plan holding both MA 27101 and
                # MA 26100 (approved equivalents) is flagged as "MA 27101 (counts as MA 26100)
                # appears more than once". Searching for the literal "MA 27101" finds exactly
                # one occurrence, the keep-first rule drops nothing, and a duplicate survives a
                # repair pass that is supposed to guarantee a clean plan. When the literal code
                # appears once, IT is the duplicate — the earlier occurrence is spelled
                # differently — so it is the one to remove.
                positions = [
                    i for i, courses in enumerate(working)
                    if code in (plan_scorers.normalize_code(x) for x in courses)
                ]
                if len(positions) > 1:
                    drop.append((positions[1], code))
                elif positions:
                    drop.append((positions[0], code))
                continue

            if index is None:
                for i, courses in enumerate(working):
                    if code in (plan_scorers.normalize_code(x) for x in courses):
                        drop.append((i, code))
                        break
            else:
                drop.append((index, code))

        if not drop:
            unrepairable = blocked
            break

        # ONE REMOVAL PER COURSE PER PASS. Two rules can name the same course in two different
        # semesters — a junk CS 38100 in semester 1 trips `prereq_violation` while the LEGITIMATE
        # CS 38100 in semester 4 trips `duplicate_course` — and applying both deletes the course
        # outright, taking a requirement group with it. Silent, and it cost 11pp of coverage in
        # testing before it was caught.
        #
        # Keeping the FIRST is what makes it correct rather than merely safe: violations are
        # appended in semester order, so the earliest bad placement is dropped and the later
        # legitimate one survives to be re-scored on the next pass.
        seen_codes: set[str] = set()
        deduped: list[tuple[int, str]] = []
        for index, code in drop:
            if code in seen_codes:
                continue
            seen_codes.add(code)
            deduped.append((index, code))
        drop = deduped

        for index, code in drop:
            if not 0 <= index < len(working):
                continue
            for position, raw in enumerate(working[index]):
                if plan_scorers.normalize_code(raw) == code:
                    working[index].pop(position)
                    removed.append(f"{code} (semester {index + 1})")
                    break
    else:  # pragma: no cover — 12 passes without converging means the cascade is cyclic
        unrepairable.append("repair did not stabilise within max_passes")

    return working, removed, unrepairable


def index_by_label(
    raw: list[Any], profile: Profile, *, delta: bool
) -> list[list[str]]:
    """Map the model's semester objects onto horizon indices, by term/year when it matters.

    POSITIONAL INDEXING IS WRONG FOR A DELTA, and this was a real bug with a real cost. Once the
    repair prompt started asking for only the semesters being CHANGED, a reply of one semester
    was a one-element array — and reading it positionally put it in semester 1. Live case,
    gemma4-e4b / mi-sophomore-catchup: the model correctly answered "COM 11400, spring 2027",
    the harness filed it under semester 1 where COM 11400 is illegal, auto-repair deleted it,
    and the case looped at 88.9% coverage for seven attempts. The model was right every time and
    the labels it sent were being discarded.

    Full-plan replies keep POSITIONAL indexing, deliberately. Mode B scores that way on purpose
    (`_term_labels_ok` is a separate diagnostic, so a model that sequences correctly but labels
    the calendar wrong is not punished twice), and Mode C's first attempt must stay comparable
    to it.

    In delta mode an unmatched label is DROPPED rather than guessed into a position, because a
    position carries no meaning in a partial reply — guessing would reintroduce exactly the bug
    this function exists to fix.
    """
    from .planner import first_planning_term, next_term

    calendar: dict[tuple[str, Any], int] = {}
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for index in range(profile.semesters_to_plan):
        calendar[(term, year)] = index
        term, year = next_term(term, year)

    filled = [e for e in raw if isinstance(e, dict) and (e.get("courses") or [])]
    matched = sum(
        1 for e in filled
        if (str(e.get("term", "")).lower(), e.get("year")) in calendar
    )
    # A REPLY THAT IGNORED THE DELTA INSTRUCTION. If nothing matched and the shape is a whole
    # plan, the model sent everything with bad labels rather than a delta — falling back to
    # position recovers it. Dropping the lot would score a mislabelled calendar as a total
    # planning failure, which is the double-punishment `_term_labels_ok` exists to avoid.
    positional = not delta or (matched == 0 and len(raw) >= profile.semesters_to_plan)

    out: list[list[str]] = [[] for _ in range(profile.semesters_to_plan)]
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        courses = [str(c) for c in (entry.get("courses") or [])]
        if not courses:
            continue
        index = calendar.get((str(entry.get("term", "")).lower(), entry.get("year")))
        if index is None:
            if not positional:
                # A labelled delta the calendar does not recognise. Position means nothing in a
                # partial reply, so guessing one would recreate the bug this function fixes.
                # The loss is visible: `delta_unmatched` counts it on the record.
                continue
            index = position
        if 0 <= index < len(out):
            out[index].extend(courses)
    return out


def unmet_candidates(fixture: Fixture, profile: Profile,
                     semesters: list[list[str]]) -> set[str]:
    """Courses that would advance an unmet requirement group, and are not already placed."""
    canonical = fixture.canonical
    canon = lambda code: canonical.get(code, code)
    have = {canon(plan_scorers.normalize_code(c)) for s in semesters for c in s}
    have |= {canon(plan_scorers.normalize_code(c)) for c in profile.completed_courses}

    wanted: set[str] = set()
    for group in fixture.requirement_groups:
        courses = group.get("courses", [])
        if group.get("kind") == "all":
            wanted.update(c for c in courses if canon(c) not in have)
        else:
            credits = sum(fixture.credits(c) for c in courses if canon(c) in have)
            if credits < float(group.get("choose_credits") or 0):
                wanted.update(c for c in courses if canon(c) not in have)
    return wanted


def legal_slots(
    fixture: Fixture, profile: Profile, semesters: list[list[str]], code: str
) -> list[int]:
    """Every semester ``code`` could legally be added to. 1-based, for printing.

    THE HARNESS ALREADY KNOWS THIS, so it should not be the model's job to search for it —
    the same argument that moved violation deletion out of the model. And the search is
    genuinely hard here: with CS 25200 parked at semester 7, CS 37300, CS 35400, CS 49000 and
    CS 42200 were each legal in EXACTLY ONE semester out of eight. Asking a 4B model to find a
    single legal slot by trial and error, while also re-transcribing twenty-five confirmed
    placements, is a search problem wearing a planning problem's clothes.
    """
    base = len(plan_scorers.score_plan(fixture, profile, semesters).violations)
    out: list[int] = []
    for index in range(max(len(semesters), profile.semesters_to_plan)):
        trial = [list(courses) for courses in semesters]
        while len(trial) <= index:
            trial.append([])
        trial[index].append(code)
        if len(plan_scorers.score_plan(fixture, profile, trial).violations) <= base:
            out.append(index + 1)
    return out


def placement_hints(
    fixture: Fixture, profile: Profile, semesters: list[list[str]], limit: int = 12
) -> list[str]:
    """"CS 37300 — can go in: semester 8 (spring 2030)". Computed, not guessed.

    THE TERM LABEL IS PART OF THE ANSWER, not decoration. The reply schema is keyed on
    term/year, so a hint that says only "semester 3" leaves the model one lookup away from
    using it — and that lookup is where it failed: told "semester 3, semester 4", gemma4-e4b
    answered "fall 2026", which is semester 1, where the course busts the credit cap. Printing
    the label the model must actually emit removes the translation step entirely. Same argument
    as computing the slots in the first place.
    """
    from .planner import first_planning_term, next_term

    calendar: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(profile.semesters_to_plan):
        calendar.append((term, year))
        term, year = next_term(term, year)

    def label(slot: int) -> str:
        if 1 <= slot <= len(calendar):
            return f"semester {slot} ({calendar[slot - 1][0]} {calendar[slot - 1][1]})"
        return f"semester {slot}"

    hints: list[str] = []
    for code in sorted(unmet_candidates(fixture, profile, semesters))[:limit]:
        slots = legal_slots(fixture, profile, semesters, code)
        course = fixture.by_code.get(code)
        credits = f" ({course.credits} cr)" if course else ""
        where = (", ".join(label(s) for s in slots) if slots
                 else "NOWHERE right now — something already placed is blocking it")
        hints.append(f"{code}{credits} — can go in: {where}")
    return hints


def is_deadlocked(fixture: Fixture, profile: Profile,
                  semesters: list[list[str]]) -> bool:
    """Can ANY still-needed course be added anywhere without creating a violation?

    A DIRECT TEST, replacing the stall counter that missed this. Coverage creeping up by one
    group every few attempts keeps resetting a stall heuristic while the plan is already
    unfinishable — which is exactly what happened live: gemma4-e4b filled all 8 semesters with
    validated optional selectives, and CS 37300 (a REQUIRED course) had nowhere left to go. The
    prompt still said "add courses to the empty space". There was no empty space.

    So the question is asked literally: try every needed course in every semester and see if a
    single placement is legal. If none is, no amount of prompting will help and the frozen set
    has to give something back.
    """
    wanted = unmet_candidates(fixture, profile, semesters)
    if not wanted:
        return False
    base = len(plan_scorers.score_plan(fixture, profile, semesters).violations)
    for code in wanted:
        for index in range(len(semesters)):
            trial = [list(courses) for courses in semesters]
            trial[index].append(code)
            if len(plan_scorers.score_plan(fixture, profile, trial).violations) <= base:
                return False
    return True


def surplus_placements(
    fixture: Fixture, profile: Profile, semesters: list[list[str]], pinned: set[str]
) -> list[tuple[str, int]]:
    """Placements whose removal costs no requirement coverage — pure capacity, no progress.

    THE OTHER HALF OF THE DEADLOCK, and the one the prerequisite logic cannot see. A model that
    fills every term with legal optional selectives has built a plan that validates perfectly
    and can never be completed: CS 33400, CS 44800, CS 45800, CS 47500 and STAT 41600 were all
    frozen in the live run while CS 37300 — required — had no seat. None of those five was
    blocking anything by ORDER; they were blocking by occupancy.

    "Costs no coverage" is decided by removing the course and re-scoring, not by guessing from
    tags: a course can be in a `choose` group and still be surplus once that group's credit
    target is already met by others.
    """
    base = plan_scorers.score_plan(fixture, profile, semesters).requirement_coverage
    out: list[tuple[str, int]] = []
    for index, courses in enumerate(semesters):
        for raw in courses:
            code = plan_scorers.normalize_code(raw)
            if code in pinned:
                continue
            trial = [list(c) for c in semesters]
            trial[index] = [c for c in trial[index]
                            if plan_scorers.normalize_code(c) != code]
            if plan_scorers.score_plan(
                    fixture, profile, trial).requirement_coverage >= base:
                out.append((code, index))
    return out


def release_blockers(
    fixture: Fixture,
    profile: Profile,
    semesters: list[list[str]],
    locked: list[LockedSlot],
    *,
    limit: int = 3,
) -> tuple[list[list[str]], list[str]]:
    """Un-freeze placements that are LEGAL BUT TOO LATE, so the plan can finish.

    THE BUG THIS FIXES WAS THE HARNESS'S, NOT A MODEL'S, and it invalidated the repair arm's
    first live run. Freezing on "no violation" is too weak a test: a placement can be perfectly
    legal where it sits and still make the rest of the degree unreachable. Observed, gemma4-e4b /
    mi-fresh-start: CS 25200 validated at semester 7 and was frozen there forever. It gates the
    whole cs-elective group, so every course behind it — CS 30700, CS 35200, CS 42200, CS 42600,
    CS 45600, CS 49000 — had nowhere legal to go and was deleted on every subsequent pass.
    STAT 35000 froze at semester 8 and did the same to CS 37300. Coverage sat at 77.8% for eight
    attempts and the case was recorded as the model failing to converge. It was not: the ratchet
    had built a trap and then scored the model for being in it.

    So the ratchet is no longer unconditional. When the repaired plan is clean but coverage has
    STOPPED IMPROVING, the frozen courses standing in front of the unmet requirements are
    released — latest position first, since a late prerequisite blocks the most — and the model
    gets to place them again. This is the chain-prerequisite deletion the pure ratchet lacked.

    Student-locked slots are never released. If the deadlock traces to a pin, it is the
    student's constraint and the honest outcome is a plan that cannot cover everything.
    """
    canonical = fixture.canonical
    canon = lambda code: canonical.get(code, code)
    completed = {canon(plan_scorers.normalize_code(c)) for c in profile.completed_courses}
    pinned = {slot.course for slot in locked}
    position: dict[str, int] = {
        canon(plan_scorers.normalize_code(code)): index
        for index, courses in enumerate(semesters)
        for code in courses
    }
    scheduled = set(position)

    # What is still missing, and therefore what has to become placeable.
    score = plan_scorers.score_plan(fixture, profile, semesters)
    if score.requirement_coverage >= 1.0:
        return semesters, []

    wanted: set[str] = set()
    for group in fixture.requirement_groups:
        courses = group.get("courses", [])
        if group.get("kind") == "all":
            short = [c for c in courses if canon(c) not in scheduled | completed]
        else:
            have = sum(fixture.credits(c) for c in courses
                       if canon(c) in scheduled | completed)
            short = ([] if have >= float(group.get("choose_credits") or 0)
                     else [c for c in courses if canon(c) not in scheduled | completed])
        wanted.update(short)

    # Everything those courses sit behind that is ALREADY PLACED and not pinned. Those are the
    # only placements whose position can be the obstacle.
    blockers: dict[str, int] = {}
    for code in wanted:
        course = fixture.by_code.get(code)
        if course is None:
            continue
        for dep in _dependency_closure(fixture, [code]) | {
            plan_scorers.normalize_code(d)
            for d in tuple(course.prereqs) + tuple(course.coreqs)
        }:
            dep = canon(dep)
            if dep in position and dep not in pinned and dep not in completed:
                blockers[dep] = position[dep]

    # Latest first: a prerequisite parked at semester 7 blocks strictly more than one at 3.
    ranked = sorted(blockers, key=lambda c: (-blockers[c], c))[:limit]

    # CAPACITY, not just order. If the plan is deadlocked, freeing prerequisites is useless
    # while every term is still full — so surplus placements (removal costs no coverage) are
    # released too, latest first. Without this the live run sat at 8 full semesters with a
    # required course homeless and nothing the release could do about it.
    if is_deadlocked(fixture, profile, semesters):
        surplus = surplus_placements(fixture, profile, semesters, pinned)
        for code, index in sorted(surplus, key=lambda pair: (-pair[1], pair[0])):
            if len(ranked) >= limit:
                break
            if code not in ranked:
                ranked.append(code)

    released: list[str] = []
    out = [list(courses) for courses in semesters]
    for code in ranked:
        for index, courses in enumerate(out):
            for spot, raw in enumerate(courses):
                if canon(plan_scorers.normalize_code(raw)) == code:
                    courses.pop(spot)
                    released.append(f"{code} (was semester {index + 1})")
                    break
    return out, released


def freeze_placements(semesters: list[list[str]]) -> list[LockedSlot]:
    """Every surviving placement becomes a lock for the next attempt. This is the ratchet."""
    return [
        LockedSlot(semester_index=index, course=plan_scorers.normalize_code(code))
        for index, courses in enumerate(semesters)
        for code in courses
    ]


@dataclass
class AttemptResult:
    attempt: int
    converged: bool
    converged_strict: bool
    degenerate: bool
    violations: list[str]
    violation_counts: dict[str, int]
    requirement_coverage: float
    filled_slots: int
    planned_credits: int
    semesters: list[list[str]]
    locked_alterations: list[str]
    structure_ok: bool
    parse_failure_reason: str | None
    elapsed_s: float
    ttft_s: float | None
    eval_count: int | None
    prompt_eval_count: int | None
    predicted_ms: float | None
    output: str
    seed: int
    temperature: float
    feedback_given: list[str] = field(default_factory=list)

    def as_record(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "converged": self.converged,
            "converged_strict": self.converged_strict,
            "degenerate": self.degenerate,
            "violations": self.violations,
            "violation_counts": self.violation_counts,
            "requirement_coverage": round(self.requirement_coverage, 4),
            "filled_slots": self.filled_slots,
            "planned_credits": self.planned_credits,
            "semesters": self.semesters,
            "locked_alterations": self.locked_alterations,
            "structure_ok": self.structure_ok,
            "parse_failure_reason": self.parse_failure_reason,
            "elapsed_s": round(self.elapsed_s, 3),
            "ttft_s": self.ttft_s,
            "eval_count": self.eval_count,
            "prompt_eval_count": self.prompt_eval_count,
            "predicted_ms": self.predicted_ms,
            "tokens_per_s": _tokens_per_s(self.eval_count, self.predicted_ms),
            "seed": self.seed,
            "temperature": self.temperature,
            "feedback_given": self.feedback_given,
        }


def _tokens_per_s(eval_count: int | None, predicted_ms: float | None) -> float | None:
    """Generation throughput from llama.cpp's own timings.

    REPORTED, NEVER FOLDED INTO A QUALITY NUMBER. A model needing 3 attempts at 23 tok/s beats
    a model needing 1 attempt at 6 tok/s on wall-clock and loses on reasoning, and only showing
    both numbers keeps those two facts apart. See the report's throughput column.
    """
    if not eval_count or not predicted_ms:
        return None
    return round(eval_count / (predicted_ms / 1000.0), 2)


def score_attempt(
    fixture: Fixture,
    profile: Profile,
    semesters: list[list[str]],
    *,
    violation_classes: tuple[str, ...],
    excluded_edges: set[str],
    locked_alterations: list[str],
) -> tuple[plan_scorers.PlanScore, list[str], dict[str, int], bool, bool, bool]:
    """Score one attempt and decide convergence.

    Returns (score, kept_violations, kept_counts, converged, converged_strict, degenerate).
    """
    score = plan_scorers.score_plan(fixture, profile, semesters)
    violations, counts = filter_excluded_edges(score, excluded_edges)

    converged = all(counts.get(cls, 0) == 0 for cls in violation_classes)
    converged_strict = not counts and score.requirement_coverage >= 1.0
    # DEGENERATE = converged by having nothing to be wrong about. The empty-plan escape hatch
    # the criterion leaves open, made countable. A plan covering under half its requirements is
    # not a plan a student can use, however clean it is.
    degenerate = converged and (score.planned_credits == 0
                                or score.requirement_coverage < 0.5)
    return score, violations, counts, converged, converged_strict, degenerate


def format_feedback(
    violations: list[str], locked_alterations: list[str], top_n: int | None
) -> list[str]:
    """The validator's findings, as the next attempt sees them.

    TRUNCATION IS A DESIGN CHOICE, not a token saving. Handing back the complete violation list
    turns the task into transcription — the model is told exactly which courses are wrong and
    where, and convergence measures reading comprehension rather than planning. A real advising
    UI surfaces the first few problems, so `feedback_top_n` defaults to a small number and the
    count of what was withheld is stated, so the model knows the list is partial rather than
    believing it has fixed everything.
    """
    items = list(locked_alterations) + list(violations)
    if top_n is None or len(items) <= top_n:
        return items
    withheld = len(items) - top_n
    return items[:top_n] + [f"(and {withheld} more problem(s) not shown)"]


# --- the retry loop ---------------------------------------------------------------------------


def run_case(
    ctx,
    client: LlamaCppClient,
    model_cfg: dict,
    scenario: Scenario,
    locked: list[LockedSlot],
    *,
    variant: str,
    conv_cfg: dict[str, Any],
    out,
) -> dict[str, Any]:
    """One (model, scenario, variant) test case: retry until converged, out of clock, or out of
    attempts.

    EVERY intermediate attempt is written, not just the last one. Whether a model is *making
    progress* (7 violations, then 4, then 1) or just resampling noise (7, 6, 8, 7) is the most
    diagnostic thing this mode produces, and it is invisible in a record that keeps only the
    terminal state.
    """
    fixture: Fixture = ctx.fixture
    profile = scenario.profile
    timeout_s = float(conv_cfg["timeout_s"])
    max_attempts = int(conv_cfg["max_attempts"])
    top_n = conv_cfg.get("feedback_top_n")
    classes = tuple(conv_cfg.get("violation_classes") or DEFAULT_VIOLATION_CLASSES)
    excluded = set(conv_cfg.get("excluded_prereq_edges") or [])
    base_seed = int(ctx.cfg["run"]["seed"])
    stride = int(conv_cfg.get("seed_stride", 1000))
    temp_step = float(conv_cfg.get("temperature_step", 0.0))
    temp_max = float(conv_cfg.get("temperature_max", 1.0))
    base_temp = float(ctx.cfg["run"]["temperature"])

    attempts: list[AttemptResult] = []
    feedback: list[str] = []
    started = time.monotonic()
    censored, censor_reason = True, "max_attempts"
    static_hash = ""
    # REPAIR's state. `working_locked` is the student's pins plus every placement that has
    # survived validation so far — it only grows. `missing` is what the model is asked to add
    # next. For the other two variants these stay at their initial values and nothing ratchets.
    working_locked = list(locked)
    missing: list[str] = []
    hints: list[str] = []
    repair_removed: list[str] = []
    repair_released: list[str] = []
    repair_unrepairable: list[str] = []
    frozen_by_attempt: list[int] = []
    released_by_attempt: list[int] = []
    stall, last_coverage = 0, -1.0
    stall_patience = int(conv_cfg.get("repair_stall_patience", 2))
    release_max = int(conv_cfg.get("repair_release_max", 3))
    # PLATEAU STOP. A case whose progress metric has not moved for N attempts is not going to
    # move; the remaining attempts are GPU time spent re-confirming that. `plateau` is its own
    # censor reason and — importantly — it is still CENSORED, not a failure: the model might
    # have converged at attempt 9, we simply stopped asking. Kaplan-Meier treats it as censored
    # at the attempt we stopped, which is exactly what we know and no more.
    plateau_patience = int(conv_cfg.get("plateau_patience", 3))
    plateau_variants = set(conv_cfg.get("plateau_variants") or ["repair", "feedback"])
    plateau = 0
    last_progress: tuple[int, float] | None = None
    # How many releases have fired since the metric last actually moved. Only the first one
    # buys the model extra patience; after that the releases are demonstrably not working.
    releases_since_progress = 0

    for attempt in range(1, max_attempts + 1):
        elapsed = time.monotonic() - started
        if elapsed >= timeout_s:
            censored, censor_reason = True, "timeout"
            break

        # BLIND moves the sample and nothing else; FEEDBACK moves the context and nothing else.
        # Letting feedback also reseed would confound self-correction with resampling luck —
        # the single comparison this mode exists to make. REPAIR moves neither: what changes
        # between its attempts is the frozen set, which is the harness's doing, not the
        # sampler's.
        if variant == "blind":
            seed = base_seed + attempt * stride
            temperature = min(base_temp + temp_step * (attempt - 1), temp_max)
            given: list[str] = []
        else:
            seed = base_seed
            temperature = base_temp
            given = list(feedback) if variant == "feedback" else []

        system, user, static_hash = ctx.prompts.plan_convergence(
            scenario, working_locked, feedback=given,
            confirmed=(variant == "repair" and attempt > 1),
            missing_requirements=missing if variant == "repair" else None,
            placement_hints=hints if variant == "repair" else None,
        )
        options = {
            "temperature": temperature,
            "top_p": ctx.cfg["run"].get("top_p", 0.9),
            "seed": seed,
            "max_tokens": ctx.cfg["run"]["max_plan_tokens"],
        }
        # The per-request timeout never outlives the case's remaining clock: a 600 s request
        # inside a 600 s case would let one attempt consume the whole window and report
        # "1 attempt, censored", which describes the timeout rather than the model.
        remaining = max(1.0, timeout_s - (time.monotonic() - started))
        client.timeout_s = min(ctx.cfg["run"]["request_timeout_s"], remaining)

        t0 = time.monotonic()
        try:
            res = client.chat(
                system, user, options=options, think=model_cfg.get("think"),
                response_format=json_response_format(
                    "PlanOfStudy",
                    plan_schema(
                        ctx.cfg["run"]["planning_terms"],
                        [c.code for c in fixture.catalog],
                        [g["id"] for g in fixture.requirement_groups],
                    ),
                ),
            )
        except LlamaCppError as exc:
            # A transport failure is not evidence about the model's planning, but it IS part of
            # the wall clock the student would experience, so the attempt is recorded and the
            # loop continues rather than abandoning the case.
            attempts.append(AttemptResult(
                attempt=attempt, converged=False, converged_strict=False, degenerate=False,
                violations=[f"request_error: {exc}"], violation_counts={"request_error": 1},
                requirement_coverage=0.0, filled_slots=0, planned_credits=0, semesters=[],
                locked_alterations=[], structure_ok=False, parse_failure_reason=str(exc),
                elapsed_s=time.monotonic() - t0, ttft_s=None, eval_count=None,
                prompt_eval_count=None, predicted_ms=None, output="", seed=seed,
                temperature=temperature, feedback_given=given,
            ))
            continue

        parsed = scorers.extract_json(res.text)
        if not parsed or not isinstance(parsed.get("semesters"), list):
            reason = "no JSON object" if not parsed else "no semesters array"
            attempts.append(AttemptResult(
                attempt=attempt, converged=False, converged_strict=False, degenerate=False,
                violations=[f"unparseable: {reason}"], violation_counts={"unparseable": 1},
                requirement_coverage=0.0, filled_slots=0, planned_credits=0, semesters=[],
                locked_alterations=[], structure_ok=False, parse_failure_reason=reason,
                elapsed_s=res.total_s, ttft_s=res.ttft_s, eval_count=res.eval_count,
                prompt_eval_count=res.prompt_eval_count, predicted_ms=res.predicted_ms,
                output=res.text, seed=seed, temperature=temperature, feedback_given=given,
            ))
            feedback = [f"Your previous reply was not usable JSON ({reason}). "
                        f"Return the plan as a single JSON object."]
            continue

        # Delta replies (repair, attempt 2+) are indexed by their term/year labels; full plans
        # stay positional so attempt 1 remains comparable to Mode B. See `index_by_label`.
        released_now_flag = False
        delta = variant == "repair" and attempt > 1
        raw_semesters = index_by_label(parsed["semesters"], profile, delta=delta)
        semesters, alterations = apply_locked_slots(
            raw_semesters, working_locked, profile.semesters_to_plan
        )

        if variant == "repair":
            # THE HARNESS CLEARS THE VIOLATIONS. Anything the validator can name, it can
            # delete — asking a model to do it costs a round trip and, in practice, costs the
            # rest of the plan too.
            semesters, removed_now, unrepairable = auto_repair(
                fixture, profile, semesters, locked
            )
            repair_removed += removed_now
            repair_unrepairable = unrepairable

        score, violations, counts, converged, strict, degenerate = score_attempt(
            fixture, profile, semesters,
            violation_classes=classes, excluded_edges=excluded,
            locked_alterations=alterations,
        )
        missing = list(score.missing_requirements)
        if variant == "repair":
            hints = placement_hints(fixture, profile, semesters)

        if variant == "repair":
            # STRICT, and it has to be. The repair step guarantees a clean plan by
            # construction, so "no violations" says nothing about the model here — only
            # "clean AND the degree is actually covered" does. Using the loose criterion would
            # score every model as converging on attempt 1.
            converged = strict
            degenerate = False

            # THE RATCHET IS CONDITIONAL. While coverage is still climbing, freezing is pure
            # gain. Once it stops climbing on a clean plan, the frozen set is no longer
            # protecting progress — it is the thing preventing it, and holding on would score
            # the model for a deadlock the harness built. See `release_blockers`.
            released_now: list[str] = []
            if not converged:
                if score.requirement_coverage <= last_coverage + 1e-9:
                    stall += 1
                else:
                    stall = 0
                last_coverage = max(last_coverage, score.requirement_coverage)
                # DEADLOCK OVERRIDES PATIENCE. The stall counter is a heuristic and it missed
                # the real case: coverage ticking up one group every few attempts kept
                # resetting it while the plan was already unfinishable. `is_deadlocked` asks the
                # question directly — can any needed course go anywhere at all — so a plan with
                # no legal move is broken open immediately instead of burning the attempt
                # budget being asked for the impossible.
                if is_deadlocked(fixture, profile, semesters):
                    stall = stall_patience
                if stall >= stall_patience:
                    semesters, released_now = release_blockers(
                        fixture, profile, semesters, locked, limit=release_max)
                    repair_released += released_now
                    stall = 0
                    if released_now:
                        # RECOMPUTE WHAT IS MISSING. `missing` was derived from the plan BEFORE
                        # the release, so it does not mention the courses the release just took
                        # out — and that list is the only thing telling the model what to add.
                        # Observed live: the breaker pulled CS 18200 and CS 25000, the prompt
                        # still said only "cs-core: missing CS 25200", and the model was asked
                        # to restore one course while two others had silently vanished from the
                        # confirmed block. It cannot fix what it is not told is gone.
                        missing = list(
                            plan_scorers.score_plan(
                                fixture, profile, semesters
                            ).missing_requirements
                        )
                        hints = placement_hints(fixture, profile, semesters)
            released_by_attempt.append(len(released_now))
            released_now_flag = bool(released_now)
            if released_now_flag:
                releases_since_progress += 1
            working_locked = freeze_placements(semesters)
            frozen_by_attempt.append(len(working_locked))
        attempts.append(AttemptResult(
            attempt=attempt, converged=converged, converged_strict=strict, degenerate=degenerate,
            violations=violations, violation_counts=counts,
            requirement_coverage=score.requirement_coverage,
            filled_slots=sum(1 for s in semesters for _ in s),
            planned_credits=score.planned_credits,
            semesters=semesters, locked_alterations=alterations,
            structure_ok=True, parse_failure_reason=None,
            elapsed_s=res.total_s, ttft_s=res.ttft_s, eval_count=res.eval_count,
            prompt_eval_count=res.prompt_eval_count, predicted_ms=res.predicted_ms,
            output=res.text, seed=seed, temperature=temperature, feedback_given=given,
        ))
        ctx.write_transcript(
            model_cfg["name"], f"mode_c_{variant}", f"{scenario.id}__a{attempt}", 0,
            system, user, res.text,
            verdict=f"converged={converged} strict={strict} degenerate={degenerate} "
                    f"violations={counts or 'none'} coverage={score.requirement_coverage:.0%}",
        )
        if converged:
            censored, censor_reason = False, ""
            break

        # NO PROGRESS FOR `plateau_patience` ATTEMPTS -> stop asking.
        #
        # The metric is the PAIR (violations, coverage), not coverage alone: for `repair`
        # coverage is the live number (its plan is always clean, so violations are pinned at 0),
        # while for `feedback` the violation count is what has to fall and coverage can sit
        # still legitimately. One rule reads both.
        #
        # `blind` is excluded by default and that is not an oversight — repetition IS its
        # measurement. It draws independent samples to estimate how often a good plan falls out
        # by luck, so identical results across draws are the expected case, and early-stopping
        # on them would truncate the very distribution the variant exists to measure.
        if variant in plateau_variants:
            progress = (len(violations), round(score.requirement_coverage, 4))
            if progress == last_progress:
                plateau += 1
                # A RELEASE EARNS ONE MORE ATTEMPT, NOT AMNESTY — and only the FIRST release
                # since the last real progress earns it. Decrementing on every release was
                # still not enough: mi-ai-early released three courses on all eight stalled
                # attempts, so +1/-1 cancelled out and the stop stayed unreachable while the
                # same three courses were released and re-added forever. A livelock. Once a
                # release has failed to move the metric, further releases stop buying credit
                # and the plateau accumulates.
                if released_now_flag and releases_since_progress <= 1:
                    plateau = max(0, plateau - 1)
            else:
                plateau = 0
                releases_since_progress = 0
            last_progress = progress
            if plateau >= plateau_patience:
                censored, censor_reason = True, "plateau"
                break

        feedback = format_feedback(violations, alterations, top_n)
    else:
        censored, censor_reason = True, "max_attempts"

    wall = time.monotonic() - started
    converged_at = next((a.attempt for a in attempts if a.converged), None)
    record = {
        "model": model_cfg["name"],
        "stage": "plan_mode_c",
        "variant": variant,
        "scenario": scenario.id,
        "expect_unsatisfiable": scenario.expect_unsatisfiable,
        "static_hash": static_hash,
        "fixture_hash": fixture.fixture_hash,
        "locked_slots": [{"semester_index": s.semester_index, "course": s.course}
                         for s in locked],
        "converged": converged_at is not None,
        "attempts_to_converge": converged_at,
        "attempts_used": len(attempts),
        "wall_clock_s": round(wall, 3),
        # RIGHT-CENSORED. `attempts_to_converge` is null and `attempts_used` is a LOWER BOUND on
        # what this model would have needed. Never read the second as the first.
        "censored": censored,
        "censor_reason": censor_reason if censored else None,
        "timeout_s": timeout_s,
        "max_attempts": max_attempts,
        "converged_strict": any(a.converged_strict for a in attempts),
        "degenerate_convergence": bool(converged_at is not None
                                       and attempts[converged_at - 1].degenerate),
        "violations_by_attempt": [len(a.violations) for a in attempts],
        "coverage_by_attempt": [round(a.requirement_coverage, 4) for a in attempts],
        # REPAIR only. `frozen_by_attempt` must be non-decreasing — it is the ratchet, and if it
        # ever falls, the freeze has a bug and the variant is not doing what it claims.
        "frozen_by_attempt": frozen_by_attempt,
        # A DROP IN `frozen_by_attempt` IS ONLY LEGITIMATE WHERE THIS IS NON-ZERO. The ratchet
        # guard reads the two together: a fall with a release is the deadlock-breaker doing its
        # job, a fall without one is the freeze losing work, which is a bug.
        "released_by_attempt": released_by_attempt,
        "repair_released": repair_released,
        "repair_released_total": len(repair_released),
        "repair_removed": repair_removed,
        "repair_removed_total": len(repair_removed),
        # Non-empty means the ONLY way to clear a violation was to move a course the student
        # pinned. The harness refuses; the case is reported rather than quietly overridden.
        "repair_unrepairable": repair_unrepairable,
        "tokens_per_s": _median([t for t in (_tokens_per_s(a.eval_count, a.predicted_ms)
                                             for a in attempts) if t]),
        "locked_alterations_total": sum(len(a.locked_alterations) for a in attempts),
        "attempts": [a.as_record() for a in attempts],
    }
    out.write(json.dumps(record, default=str) + "\n")
    out.flush()
    return record


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 3)


# --- preflight ---------------------------------------------------------------------------------


def preflight(ctx, locked_slots: LockedSlots, conv_cfg: dict[str, Any]) -> list[str]:
    """Everything checkable before a GPU-minute is spent. Returns the problems found.

    The prerequisite-risk report is the important half. Mode C COMPOUNDS a bad prereq edge:
    the same wrong edge is re-hit on every attempt, so it does not cost a model one violation,
    it costs it the whole run and turns into a fabricated convergence difference.
    """
    problems: list[str] = []
    for scenario in ctx.fixture.scenarios:
        problems += locked_slot_problems(
            ctx.fixture, scenario, locked_slots.for_scenario(scenario.id)
        )
        # Can the deterministic planner still solve it with the locks in place? It cannot be
        # asked to honour them directly (it has no notion of a pin), so this is the weaker but
        # still useful check: the scenario itself must be reachable, exactly as `run.py check`
        # requires, or convergence is impossible for reasons unrelated to the model.
        if scenario.expect_unsatisfiable:
            continue
        plan = generate_plan(scenario.profile, ctx.fixture.catalog)
        score = plan_scorers.score_plan(
            ctx.fixture, scenario.profile, plan_scorers.plan_to_semesters(plan)
        )
        if not score.viable:
            problems.append(
                f"{scenario.id}: the deterministic planner cannot produce a viable plan — "
                f"convergence would be measuring a fixture bug, not a model."
            )
    return problems


def run_convergence(
    root: Path, *, models: list[str] | None = None, brackets: list[str] | None = None,
    variants: list[str] | None = None, scenarios: list[str] | None = None,
) -> Path:
    """Top level for Mode C. Own results file, own lock, own schema.

    The server lifecycle is deliberately identical to `runner.run_model`'s — same launch flags,
    same warmup-and-discard, same stop between models — because a convergence number taken
    against a differently-configured server is not comparable to anything, including itself.
    """
    from .runner import _env_record, _run_lock, _warmup, load_context
    from .llamacpp_client import LlamaCppClient
    from .server import gpu_memory_used_mb, start_server

    ctx = load_context(root)
    conv_cfg = dict(ctx.cfg.get("convergence") or {})
    if not conv_cfg:
        raise SystemExit("config.yaml has no `convergence:` block.")

    slots_path = root / conv_cfg["locked_slots"]
    locked_slots = load_locked_slots(slots_path)

    selected = [
        m for m in ctx.cfg["models"]
        if (not models or m["name"] in models)
        and (not brackets or m.get("bracket") in brackets)
    ]
    if not selected:
        raise SystemExit("No models matched the filter.")

    variant_list = list(
        variants or conv_cfg.get("enabled_variants") or ["repair", "feedback", "blind"])
    unknown = [v for v in variant_list if v not in ("blind", "feedback", "repair")]
    if unknown:
        raise SystemExit(f"Unknown variant(s): {', '.join(unknown)}")

    cases = [s for s in ctx.fixture.scenarios if not scenarios or s.id in scenarios]
    if not cases:
        raise SystemExit("No scenarios matched the filter.")

    print(prereq_risk_note(ctx.fixture, conv_cfg))
    problems = preflight(ctx, locked_slots, conv_cfg)
    if problems:
        print(f"\n❌ preflight found {len(problems)} problem(s) — no GPU time spent:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("\n✅ preflight passed: every locked slot is legal where it is pinned.\n")

    # CROSS-TAG GUARD. `_run_lock` only refuses a second run under the SAME tag, which is
    # enough when the only tags are baseline/mitigated (they are never wanted at once anyway).
    # Mode C breaks that assumption: `.convergence.lock` and `.baseline.lock` are different
    # files, so nothing structural stops `converge` starting on top of a running sweep — and
    # they would fight over the GPU and over port 8099, producing two sets of plausible-looking
    # latency numbers that describe neither run. Convergence is the more fragile of the two,
    # since its whole output is wall-clock.
    for other in sorted(ctx.results_dir.glob(".*.lock")):
        if other.name == ".convergence.lock":
            continue
        try:
            pid = int(other.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid > 0 and Path(f"/proc/{pid}").exists():
            raise SystemExit(
                f"refusing to start: {other.name} is held by a LIVE process (pid {pid}). "
                f"Mode C measures wall-clock time; sharing the GPU with another run would "
                f"make every number in it meaningless. Wait for that run, or stop it."
            )
        print(f"[converge] ignoring stale lock {other.name} (pid {pid or 'unknown'} is gone)")

    out_path = ctx.results_dir / "runs_convergence.jsonl"
    with _run_lock(ctx.results_dir, "convergence"):
        _write_meta(ctx, conv_cfg, locked_slots, selected, variant_list, cases)
        mode = "a" if out_path.exists() else "w"
        total = len(selected) * len(variant_list) * len(cases)
        print(f"[converge] {len(selected)} model(s) x {len(variant_list)} variant(s) x "
              f"{len(cases)} scenario(s) = {total} case(s) -> {out_path} (mode={mode})")
        print(f"[converge] worst case {total * conv_cfg['timeout_s'] / 3600:.1f} h if every "
              f"case runs to its {conv_cfg['timeout_s']} s timeout")

        with out_path.open(mode) as out:
            for model_cfg in selected:
                name = model_cfg["name"]
                print(f"\n=== {name} ({model_cfg.get('bracket')}) ===")
                vram_baseline = gpu_memory_used_mb()
                try:
                    handle = start_server(ctx.cfg, model_cfg, ctx.log_dir, host=ctx.host)
                except (RuntimeError, TimeoutError) as exc:
                    print(f"[SKIP] {name}: {exc}")
                    out.write(json.dumps(
                        {"model": name, "stage": "error", "error": str(exc)}) + "\n")
                    out.flush()
                    continue
                client = LlamaCppClient(handle.base_url,
                                        timeout_s=ctx.cfg["run"]["request_timeout_s"])
                try:
                    _warmup(ctx, client, model_cfg)
                    env = _env_record(ctx, handle, client, model_cfg, vram_baseline)
                    env["stage"] = "env"
                    out.write(json.dumps(env, default=str) + "\n")
                    out.flush()
                    print(f"[env] {name}: vram_delta={env['vram_delta_mb']} "
                          f"offload={env['offload']} n_ctx={env['server_n_ctx']}")
                    for variant in variant_list:
                        for scenario in cases:
                            rec = run_case(
                                ctx, client, model_cfg, scenario,
                                locked_slots.for_scenario(scenario.id),
                                variant=variant, conv_cfg=conv_cfg, out=out,
                            )
                            status = (f"converged @ attempt {rec['attempts_to_converge']}"
                                      if rec["converged"]
                                      else f"CENSORED ({rec['censor_reason']})")
                            print(f"  [{name}/{variant}] {scenario.id}: {status} "
                                  f"in {rec['wall_clock_s']:.0f}s "
                                  f"({rec['attempts_used']} attempt(s))")
                finally:
                    handle.stop(
                        trim_log_lines=ctx.cfg["llamacpp"].get("keep_log_lines", 400))
                    time.sleep(ctx.cfg["llamacpp"].get("cooldown_s", 5))
    return out_path


def _write_meta(ctx, conv_cfg, locked_slots: LockedSlots, selected, variants, cases) -> None:
    scenario = ctx.fixture.scenarios[0]
    meta = {
        "tag": "convergence",
        "mode": "D",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": ctx.host,
        "models": [m["name"] for m in selected],
        "variants": list(variants),
        "scenarios": [s.id for s in cases],
        "run": ctx.cfg["run"],
        "llamacpp": ctx.cfg["llamacpp"],
        "convergence": conv_cfg,
        "fixture": {
            "path": ctx.fixture.path.name,
            "hash": ctx.fixture.fixture_hash,
            "verified": ctx.fixture.verified,
        },
        "mock_db": {"path": ctx.database.path.name, "hash": ctx.database.db_hash},
        # Locked slots are prompt input, not scoring authority — recorded here (and per record)
        # rather than folded into fixture_hash, so adding them cannot invalidate a Mode A/B row.
        "locked_slots": {
            "path": locked_slots.path.name,
            "hash": locked_slots.slots_hash,
            "scenarios": {k: [{"semester_index": s.semester_index, "course": s.course}
                              for s in v]
                          for k, v in locked_slots.by_scenario.items()},
        },
        "static_hashes": {
            "plan_mode_c": ctx.prompts.plan_convergence(
                scenario, locked_slots.for_scenario(scenario.id))[2],
        },
    }
    (ctx.results_dir / "meta_convergence.json").write_text(json.dumps(meta, indent=2))


def prereq_risk_note(fixture: Fixture, conv_cfg: dict[str, Any]) -> str:
    """The honest state of the prerequisite edges, printed before every Mode C run."""
    edges = sum(len(c.prereqs) for c in fixture.catalog)
    excluded = list(conv_cfg.get("excluded_prereq_edges") or [])
    lines = [
        f"prerequisite edges in the fixture: {edges} across {len(fixture.catalog)} courses",
        f"fixture verified: {fixture.verified}",
    ]
    if not fixture.verified:
        lines.append(
            "  ⚠️  ALL prereq edges are hand-written — Purdue publishes them only in Banner, "
            "whose robots.txt disallows crawling. Mode C compounds a wrong edge across every "
            "attempt of every case, so this is a larger threat here than in Mode A/B."
        )
    lines.append(
        f"excluded edges: {', '.join(excluded) if excluded else 'none'}"
        + ("" if excluded else "  (set convergence.excluded_prereq_edges to exclude known-bad "
                              "edges, e.g. ['CS 38100->MA 26100'])")
    )
    return "\n".join(lines)
