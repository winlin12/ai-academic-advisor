"""MODE C — what the student does after reading the plan.

Modes A and B answer "how good is one attempt?". This module answers the question a person
actually has in front of a plan they have just read: *this is close, finish it* / *I don't like
this, try again* / *scrap it*. `model_eval/harness/convergence.py` measured exactly those three
policies, tracked them separately, and never averaged them — because they are different
capabilities. They are shipped here under the names a student would use:

    FILL       (harness `repair`)   Everything that validates is FROZEN. The checker deletes
                                    what is illegal, and the model is asked only to fill what
                                    is still missing — with the legal slots for each gap
                                    computed and handed to it. Good work is never re-rolled.

    REGENERATE (harness `feedback`) The model is told, specifically, what is wrong with the
                                    plan it produced and must fix it itself. Seed and
                                    temperature are held FIXED, so what changes between the two
                                    attempts is the context alone.

    START OVER (harness `blind`)    A fresh sample. The model is told nothing about the last
                                    plan; the seed moves and the temperature creeps up. This is
                                    the one to reach for when the plan is not wrong so much as
                                    not what the student wanted.

WHY THE SEED POLICY DIFFERS PER MODE, and it is not an implementation detail. If REGENERATE
also re-seeded, a better second plan would be indistinguishable between "the model read the
error and fixed it" and "a different sample happened to be better" — the exact confound the
harness holds seed fixed to avoid. FILL moves neither seed nor temperature: what changes
between its attempts is the frozen set, which is the app's doing rather than the sampler's.

FILL IS SCORED — and presented — DIFFERENTLY. Its repair step guarantees a clean plan by
construction, so "no violations" says nothing; the honest question is whether the degree is
actually covered. That is what the UI reports back for it.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.models.schemas import StudentProfile
from app.services.ai_planner import (
    PLAN_SYSTEM,
    AiPlanDraft,
    AiPlanResult,
    _salvage_semesters,
    _to_plan_response,
    _warnings,
    plan_schema,
    profile_block,
)
from app.services.llamacpp_client import LlamaCppClient, ModelResponseError
from app.services.plan_context import approx_tokens, render_program_context
from app.services.plan_validation import (
    legal_slots,
    normalize_code,
    repair_plan,
    term_sequence,
    unmet_candidates,
    validate_semesters,
)
from app.services.planner import planning_terms
from app.services.planner_db import ProgramCatalog, requirement_coverage

logger = logging.getLogger(__name__)

RefineMode = Literal["fill", "regenerate", "start-over"]

# How many violations to show the model. A full list makes REGENERATE close to transcription —
# the model is told exactly which courses are wrong and where, and stops planning. 5 is roughly
# what an advising UI would surface, and the prompt states how many were withheld so the model
# knows the list is partial.
FEEDBACK_TOP_N = 5

# Cap on the computed placement hints, for the same reason the export is budgeted: these are
# the longest variable-length block in the prompt.
HINT_LIMIT = 12

# START OVER creeps the temperature up per attempt, exactly as the blind variant does. A
# student pressing it repeatedly is telling us the current distribution is not producing what
# they want, and re-sampling at the same temperature mostly reproduces it.
TEMPERATURE_STEP = 0.05
TEMPERATURE_MAX = 0.80


# Composed ON TOP of PLAN_SYSTEM, never by editing it. Mode C's whole claim is that it is the
# same planning task under repetition; if the two prompts drifted, a difference in outcome could
# be a prompt difference and there would be no way to tell. Verbatim from
# model_eval/harness/prompts.py::PLAN_LOCKED_RULES.
PLAN_LOCKED_RULES = """\
Some slots in this student's plan are LOCKED. A locked slot is a course the student has
already decided to take in a specific semester.

- Locked courses are FIXED. Keep every one exactly where it is listed. Do not move it to
  another semester, do not remove it, do not schedule it a second time somewhere else.
- Plan everything else AROUND them. A locked course still consumes credits and still counts
  toward the major-course limit in the semester it sits in, and its own prerequisites must
  still be scheduled in EARLIER semesters than the one it is locked to.
- Locked courses still satisfy requirement groups and still satisfy prerequisites for later
  courses, exactly as any other scheduled course does.
- Every other slot is yours to fill or leave empty."""


class RefineOutcome(BaseModel):
    """What a refine call did, over and above the plan itself."""

    mode: str
    # False when the model was never called — nothing to fill, or it was unreachable.
    used_model: bool = True
    # FILL only: the placements that SURVIVED and were frozen against further change. Note
    # "survived", not "never moved" — the checker runs before the freeze, so a course that was
    # sitting before its own prerequisite is slid to a legal term and then frozen there.
    # Claiming these were untouched would be false for exactly the courses the student most
    # wants to know about.
    kept: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    note: str = ""


def _label_index(profile: StudentProfile, count: int) -> dict[tuple[str, int], int]:
    """(term, year) -> semester index. FILL's replies are keyed on the label, not on position.

    A delta reply only names the semesters it is adding to, so its first entry is not
    semester 1. Reading it positionally silently relocates every course in it — which is how a
    "fill semester 7" answer lands in semester 1 and busts the credit cap.
    """
    return {(term, year): index
            for index, (term, year) in enumerate(term_sequence(profile, count))}


def _locked_block(
    semesters: list[list[str]], profile: StudentProfile, catalog: ProgramCatalog
) -> list[str]:
    """The frozen plan, grouped by semester and labelled with the term the model must emit."""
    terms = term_sequence(profile, len(semesters))
    lines: list[str] = []
    for index, codes in enumerate(semesters):
        if not codes:
            continue
        term, year = terms[index]
        described = ", ".join(
            f"{code} ({catalog.credits(code)} cr)" if catalog.credits(code) else code
            for code in codes
        )
        lines.append(f"- Semester {index + 1} ({term} {year}): {described}")
    return lines


def _hints(
    semesters: list[list[str]], profile: StudentProfile, catalog: ProgramCatalog
) -> list[str]:
    """"CS 37300 (3 cr) — can go in: semester 8 (spring 2030)". Computed, never guessed."""
    terms = term_sequence(profile, profile.semesters_to_plan)

    def label(index: int) -> str:
        if 0 <= index < len(terms):
            return f"semester {index + 1} ({terms[index][0]} {terms[index][1]})"
        return f"semester {index + 1}"

    out: list[str] = []
    for code in unmet_candidates(semesters, profile, canonical=catalog.canonical)[:HINT_LIMIT]:
        slots = legal_slots(semesters, profile, catalog.courses, code,
                            canonical=catalog.canonical)
        credits = catalog.credits(code)
        where = (", ".join(label(s) for s in slots) if slots
                 else "NOWHERE right now — something already placed is blocking it")
        out.append(f"{code}{f' ({credits} cr)' if credits else ''} — can go in: {where}")
    return out


# Saying these were CHECKED is the point: the model is not being asked to re-plan them, and a
# plan it is told is already verified is one it has no reason to rewrite.
_CONFIRMED_HEADER = (
    "ALREADY CONFIRMED — these placements have been checked and are correct. Keep every one "
    "exactly as it is and do not repeat them elsewhere:"
)

_HINTS_HEADER = (
    "WHERE THEY CAN GO — already checked against prerequisites, term offerings, the credit cap "
    "and the per-semester limit on your major's courses. Use one of the terms listed; any other "
    "term will be rejected:"
)

_DELTA_INSTRUCTION = (
    "Return ONLY the semesters you are adding courses to, with just the courses you are ADDING. "
    "The confirmed placements above are kept automatically — do not repeat them."
)


# Room reserved for what the char-count estimate cannot see: the chat template's own wrapping
# and the grammar's overhead. Mirrors ai_planner._WINDOW_SAFETY_TOKENS.
_WINDOW_SAFETY_TOKENS = 600

# A FILL reply is a delta — usually two or three semesters — so it needs a fraction of a full
# plan's budget. Naming it separately is what buys the room the locked block and the placement
# hints cost, instead of squeezing the catalog export they are meant to be read against.
_DELTA_MAX_TOKENS = 700

# Below this there is nothing worth asking for — a plan's `semesters` payload ran ~470-550
# tokens at the median across model_eval's corpus, and even a delta needs a few hundred.
_MIN_USEFUL_TOKENS = 300


def _system_for(catalog: ProgramCatalog, user: str, *, extra_rules: str = "") -> str:
    """The static half, with the catalog export sized against what the variable half took.

    MODE C's prompts are BIGGER than Mode B's — the locked block, the missing-requirement list
    and the computed placement hints all sit on top of the same rules and the same export — and
    the context window did not grow to match. Rendering the export at Mode B's fixed budget
    overflowed it: measured on a real Machine Intelligence plan, FILL was left 354 tokens to
    reply in, fell under the useful-plan floor, and skipped the model entirely. The student
    pressed Fill and nothing happened.

    So the export is budgeted LAST, against what is actually left. The locked block and hints
    are what the model is being asked to act on, so they are never the thing that gets trimmed;
    the export's own trimming (selective menus, announced in the document) absorbs the
    difference.
    """
    rules = f"{PLAN_SYSTEM}\n\n{extra_rules}" if extra_rules else PLAN_SYSTEM
    spent = approx_tokens(rules) + approx_tokens(user)
    available = (settings.llamacpp_context_tokens - spent - _DELTA_MAX_TOKENS
                 - _WINDOW_SAFETY_TOKENS)
    budget = min(settings.plan_context_token_budget, max(available, 0))
    return f"{rules}\n\n{render_program_context(catalog, budget_tokens=budget)}"


def _fill_prompts(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    frozen: list[list[str]],
    missing: list[str],
) -> tuple[str, str]:
    parts = [profile_block(profile, catalog), ""]

    locked = _locked_block(frozen, profile, catalog)
    if locked:
        # Saying these were CHECKED is the point: the model is not being asked to re-plan them,
        # and a plan it is told is already verified is one it has no reason to rewrite.
        parts += [_CONFIRMED_HEADER, *locked, ""]
    if missing:
        parts += ["STILL MISSING — the plan does not yet satisfy these requirement groups:",
                  *(f"- {item}" for item in missing), ""]

    hints = _hints(frozen, profile, catalog)
    if hints:
        parts += [_HINTS_HEADER, *hints, ""]
    # ONLY THE ADDITIONS. The confirmed placements are re-inserted regardless of what comes
    # back, so demanding the full plan buys nothing and costs a great deal: it turns a
    # two-course edit into a twenty-five-course transcription, and a small model spends its
    # attention copying rather than planning.
    parts.append(_DELTA_INSTRUCTION)
    user = "\n".join(parts)
    return _system_for(catalog, user, extra_rules=PLAN_LOCKED_RULES), user


_FEEDBACK_HEADER = (
    "Your previous plan was checked and these problems were found. Fix them and return a "
    "corrected plan:"
)

_NOTHING_WRONG = (
    "Nothing was found wrong with this plan; the student simply wants a different arrangement "
    "of the same requirements."
)


def _feedback_prompts(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    problems: list[str],
    missing: list[str],
) -> tuple[str, str]:
    parts = [profile_block(profile, catalog), ""]
    shown = problems[:FEEDBACK_TOP_N]
    if shown:
        withheld = len(problems) - len(shown)
        parts += [_FEEDBACK_HEADER, *(f"- {item}" for item in shown)]
        if withheld:
            # Stated so the model knows the list is partial rather than exhaustive — otherwise
            # it "fixes" five things and reports the plan clean.
            parts.append(f"- ...and {withheld} more of the same kinds.")
        parts.append("")
    if missing:
        parts += ["The plan also does not yet satisfy these requirement groups:",
                  *(f"- {item}" for item in missing), ""]
    parts.append("Produce the corrected plan of study.")
    user = "\n".join(parts)
    return _system_for(catalog, user), user


def _merge_delta(
    frozen: list[list[str]],
    draft: AiPlanDraft,
    profile: StudentProfile,
) -> tuple[list[list[str]], list[str]]:
    """Fold a delta reply into the frozen plan, keyed on the term label. Returns (plan, added).

    The frozen placements are re-inserted regardless of what came back — that is what makes
    the freeze a ratchet rather than a suggestion. A model that ignores the instruction and
    repeats a confirmed course simply has the duplicate dropped here.
    """
    index_of = _label_index(profile, profile.semesters_to_plan)
    merged = [list(term) for term in frozen]
    already = {normalize_code(code) for term in frozen for code in term}
    added: list[str] = []

    for semester in draft.semesters:
        index = index_of.get((semester.term.lower(), int(semester.year)))
        if index is None or index >= len(merged):
            continue
        for raw in semester.courses:
            code = normalize_code(raw)
            if code in already:
                continue
            merged[index].append(code)
            already.add(code)
            added.append(code)
    return merged, added


async def _call(
    client: LlamaCppClient,
    system: str,
    user: str,
    catalog: ProgramCatalog,
    *,
    seed: int | None,
    temperature: float | None = None,
) -> AiPlanDraft | None:
    """One grammar-constrained plan call. Returns None when nothing usable came back."""
    schema = plan_schema(planning_terms(), [c.code for c in catalog.courses])
    room = (settings.llamacpp_context_tokens - approx_tokens(system) - approx_tokens(user)
            - _WINDOW_SAFETY_TOKENS)
    max_tokens = min(settings.llamacpp_plan_max_tokens, max(room, 0))
    if max_tokens < _MIN_USEFUL_TOKENS:
        logger.warning("refine: prompt leaves only %d tokens for the plan; skipping the model",
                       max_tokens)
        return None
    if temperature is not None:
        client = LlamaCppClient(temperature=temperature)
    try:
        return await client.propose(
            system, user, AiPlanDraft, schema=schema, seed=seed, max_tokens=max_tokens
        )
    except (ModelResponseError, ValidationError) as exc:
        salvaged = _salvage_semesters(getattr(exc, "raw", "") or "")
        if salvaged is None:
            logger.warning("refine: unusable reply (%s)", exc)
        return salvaged


def _finish(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    semesters: list[list[str]],
    *,
    model: str,
    mode: str,
    rationale: str,
    kept: list[str],
    added: list[str],
    removed: list[str],
    seed: int | None,
    used_model: bool = True,
    note: str = "",
) -> tuple[AiPlanResult, RefineOutcome]:
    """Repair, validate, score coverage, and package — the same tail every mode ends in.

    Nothing skips the checker. A refined plan that bypassed it would be the one way an illegal
    schedule reaches the screen, and FILL in particular hands the model a delta that it merges
    into a plan the model never saw in full.
    """
    repaired, dropped, inserted, _ = repair_plan(
        semesters, profile, catalog.courses, canonical=catalog.canonical
    )

    # NOTHING IS AUTO-FILLED, in any mode. A deterministic backfill used to run here for
    # REGENERATE, because a thin model draft could otherwise cost a student most of their
    # degree. It bought a coverage number by choosing the student's electives for them — and a
    # selective group is a MENU, so "first two in catalog order" is a coin toss, not advice.
    # The guard that actually protects a bad Regenerate is the no-worse-than-baseline check
    # below, which needs no invented courses to work.
    #
    # What the model did not schedule is reported as unfilled and shown, per requirement group,
    # by `requirement_progress` — for the student to fill themselves.
    scheduled = {code for term in repaired for code in term}
    unplanned = [code for code in profile.remaining_courses if code not in scheduled]
    backfilled: list[str] = []
    final = validate_semesters(repaired, profile, catalog.courses, canonical=catalog.canonical)

    canonical = catalog.canonical
    satisfied = {canonical.get(c, c) for c in profile.completed_courses}
    satisfied |= {canonical.get(code, code) for term in repaired for code in term}
    coverage, missing = requirement_coverage(catalog, satisfied)

    removed = sorted({*removed, *dropped})
    added = [*added, *inserted, *backfilled]

    plan = _to_plan_response(
        profile, catalog, repaired, unplanned,
        _warnings(final, removed, added, missing),
    )
    result = AiPlanResult(
        plan=plan, rationale=rationale, model=model, used_model=used_model,
        model_placed=[c for term in repaired for c in term],
        removed=removed, backfilled=added,
        violations=[str(v) for v in final.violations],
        requirement_coverage=coverage, missing_requirements=missing, seed=seed,
    )
    outcome = RefineOutcome(
        mode=mode, used_model=used_model, kept=kept, added=added, removed=removed, note=note,
    )
    return result, outcome


async def refine_plan(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    semesters: list[list[str]],
    mode: RefineMode,
    *,
    client: LlamaCppClient,
    seed: int | None = None,
    attempt: int = 1,
) -> tuple[AiPlanResult, RefineOutcome]:
    """Run one refinement round. ``semesters`` is the plan currently on the student's screen."""
    current = [[normalize_code(code) for code in term] for term in semesters]

    if mode == "start-over":
        return await _start_over(profile, catalog, client=client, seed=seed, attempt=attempt)
    if mode == "regenerate":
        return await _regenerate(profile, catalog, current, client=client, seed=seed)
    return await _fill(profile, catalog, current, client=client, seed=seed)


async def _fill(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    current: list[list[str]],
    *,
    client: LlamaCppClient,
    seed: int | None,
) -> tuple[AiPlanResult, RefineOutcome]:
    """Freeze what validates; ask the model only for what is missing."""
    # The checker clears the violations first. Anything the validator can name, it can fix —
    # asking a model to do it costs a round trip and, in practice, costs the rest of the plan.
    frozen, removed, inserted, _check = repair_plan(
        current, profile, catalog.courses, canonical=catalog.canonical
    )
    # `kept` is what SURVIVED untouched, so the courses repair had to insert are excluded —
    # they are things this round added, and reporting them as both kept and added tells the
    # student their plan already contained a course it did not.
    kept = [code for term in frozen for code in term if code not in set(inserted)]

    canonical = catalog.canonical
    satisfied = {canonical.get(c, c) for c in profile.completed_courses}
    satisfied |= {canonical.get(code, code) for term in frozen for code in term}
    _coverage, missing = requirement_coverage(catalog, satisfied)
    outstanding = unmet_candidates(frozen, profile, canonical=canonical)

    if not missing and not outstanding:
        # Nothing to fill. Say so instead of spending thirty seconds to be told the same by a
        # model — and without touching a plan the student is happy with.
        return _finish(
            profile, catalog, frozen, model=client.model, mode="fill",
            rationale="This plan already covers every requirement group in your program and "
                      "schedules everything you have left, so there was nothing to fill in.",
            kept=kept, added=inserted, removed=removed, seed=seed, used_model=False,
            note="nothing-to-fill",
        )

    system, user = _fill_prompts(profile, catalog, frozen, missing)
    draft = await _call(client, system, user, catalog, seed=seed)
    if draft is None:
        return _finish(
            profile, catalog, frozen, model=client.model, mode="fill",
            rationale="The model could not be reached, so your plan is unchanged apart from "
                      "the checks that run without it.",
            kept=kept, added=inserted, removed=removed, seed=seed, used_model=False,
            note="model-unavailable",
        )

    merged, added = _merge_delta(frozen, draft, profile)
    return _finish(
        profile, catalog, merged, model=client.model, mode="fill",
        rationale=draft.rationale or (
            f"Filled {len(added)} more course(s) into the terms that had room, keeping "
            f"everything already scheduled exactly where it was."),
        kept=kept, added=[*inserted, *added], removed=removed, seed=seed,
    )


def _reseed(seed: int | None) -> int:
    """A different sample from the same request. Derived from the old seed rather than random
    so a caller that passed one still gets a reproducible result."""
    import random

    return random.randrange(2**31) if seed is None else (seed * 6364136223846793005 + 1) % 2**31


async def _regenerate(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    current: list[list[str]],
    *,
    client: LlamaCppClient,
    seed: int | None,
) -> tuple[AiPlanResult, RefineOutcome]:
    """Tell the model what is wrong and make it fix the plan itself. Seed held FIXED."""
    check = validate_semesters(current, profile, catalog.courses, canonical=catalog.canonical)
    canonical = catalog.canonical
    satisfied = {canonical.get(c, c) for c in profile.completed_courses}
    satisfied |= {canonical.get(code, code) for term in current for code in term}
    _coverage, missing = requirement_coverage(catalog, satisfied)

    problems = [str(v) for v in check.violations]
    nothing_wrong = not problems and not missing
    if nothing_wrong:
        # The plan is clean AND complete, so there is no error to feed back.
        problems = [_NOTHING_WRONG]

    system, user = _feedback_prompts(profile, catalog, problems, missing)
    # SEED HELD FIXED — but only when there is actually feedback to give. What differs from the
    # previous attempt is then the context alone, so a better plan is attributable to the model
    # reading the error rather than to a luckier sample; that is the one comparison this mode
    # exists to make, and re-seeding would confound it.
    #
    # WITH NOTHING WRONG, holding it fixed is not rigour, it is a bug: same prompt plus same
    # seed reproduces the plan byte for byte, so a student pressing Regenerate on a clean plan
    # watches nothing happen. Measured exactly that way in the browser. There is no comparison
    # left to protect once the violation list is empty, so the sample moves instead.
    draft = await _call(
        client, system, user, catalog,
        seed=(_reseed(seed) if nothing_wrong else seed),
    )
    if draft is None:
        return _finish(
            profile, catalog, current, model=client.model, mode="regenerate",
            rationale="The model could not be reached, so your plan is unchanged.",
            kept=[], added=[], removed=[], seed=seed, used_model=False,
            note="model-unavailable",
        )

    rebuilt = [[normalize_code(c) for c in s.courses]
               for s in draft.semesters][:profile.semesters_to_plan]

    # NEVER HAND BACK A WORSE PLAN THAN THE ONE BEING CORRECTED. "Fix the problems in this
    # plan" that answers with a plan covering less of the degree has not fixed anything, and
    # the student did not ask to gamble — they asked for a correction. Same "no worse than
    # baseline" rule `advisor_agent.revise_plan` applies to Mode A proposals, and the reason it
    # is needed here is that this mode holds the seed fixed on purpose: it cannot resample its
    # way out of a bad draft, so the guard is the only thing standing between a bad draft and
    # the screen. A student who wants a different plan rather than a corrected one has
    # START OVER, which is allowed to differ freely.
    corrected, corrected_outcome = _finish(
        profile, catalog, rebuilt, model=client.model, mode="regenerate",
        rationale=draft.rationale or "Reworked the plan to address the problems found in it.",
        kept=[], added=[], removed=[], seed=seed,
    )
    # The BASELINE gets the identical treatment — the same repair — so the comparison is like
    # for like rather than the model's cleaned-up plan against a raw previous one.
    baseline, baseline_outcome = _finish(
        profile, catalog, current, model=client.model, mode="regenerate",
        rationale="", kept=[], added=[], removed=[], seed=seed, note="kept-previous",
    )

    if corrected.requirement_coverage < baseline.requirement_coverage:
        logger.info("regenerate produced %.0f%% coverage against %.0f%% for the plan it was "
                    "correcting; keeping the better one",
                    corrected.requirement_coverage * 100,
                    baseline.requirement_coverage * 100)
        winner, outcome = baseline, baseline_outcome
    else:
        winner, outcome = corrected, corrected_outcome

    # Reported against the plan AS SHIPPED — after repair and backfill — not against the raw
    # draft, so "added"/"removed" describe what the student will actually see change.
    before = {code for term in current for code in term}
    after = {c.code for s in winner.plan.semesters for c in s.courses}
    outcome.kept = sorted(before & after)
    outcome.added = sorted(after - before)
    outcome.removed = sorted(before - after)

    # THE MESSAGE HAS TO MATCH WHAT HAPPENED. The baseline branch is not "your plan, untouched"
    # — it has been through repair and backfill like every other path, so on a thin plan it can
    # add twenty courses while still being the branch that "kept" the previous one. Saying "I
    # couldn't improve on your plan" over a plan that visibly just grew is simply false, so the
    # wording is chosen from the actual diff rather than from which branch won.
    if winner is baseline:
        winner.rationale = (
            "I couldn't produce a better plan than the one you had, so I kept it and filled in "
            f"{len(outcome.added)} course(s) that were missing."
            if outcome.added else
            "I couldn't improve on the plan you already have, so I've left it as it was. Try "
            "\u201cStart over\u201d for a different plan, or tell me in the chat what to change."
        )
    return winner, outcome


async def _start_over(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    *,
    client: LlamaCppClient,
    seed: int | None,
    attempt: int,
) -> tuple[AiPlanResult, RefineOutcome]:
    """A fresh sample, told nothing about the plan it replaces.

    Delegates to the ordinary Mode B generator: "told nothing" is exactly the first-attempt
    prompt, and re-implementing it here would be a second copy free to drift. The seed moves and
    the temperature creeps, which is what makes repeated presses explore rather than repeat.
    """
    from app.services.ai_planner import generate_ai_plan

    temperature = min(settings.llamacpp_temperature + TEMPERATURE_STEP * max(attempt - 1, 0),
                      TEMPERATURE_MAX)
    sampler = LlamaCppClient(model=client.model, temperature=temperature)
    result = await generate_ai_plan(profile, catalog, client=sampler, seed=seed)
    outcome = RefineOutcome(
        mode="start-over",
        used_model=result.used_model,
        added=result.backfilled,
        removed=result.removed,
        note=f"temperature={temperature:.2f}",
    )
    return result, outcome


__all__ = ["PLAN_LOCKED_RULES", "RefineMode", "RefineOutcome", "refine_plan"]
