"""Revise-plan agent: the model proposes, the deterministic planner disposes.

The flow is deliberately asymmetric so the model can never emit an illegal schedule:

    profile ──> generate_plan() ──> baseline legal plan
              │
              └─> model sees (profile + baseline + requirement context + feedback)
                  and returns a PlanEditProposal (reorder / defer / avoid_tags / credit cap)
              │
              └─> apply proposal to the profile ──> generate_plan() re-validates it
                    ├─ no worse than baseline ─> return revised plan + rationale
                    └─ worse (more unplanned) ─> feed the planner's warnings back, retry (≤ N)

The planner remains the single source of truth for legality (prereqs, term offerings, credit
caps). The model only reorders codes the planner already knows how to schedule; a hallucinated
proposal degrades to a no-op because unknown codes and out-of-range caps are dropped on apply.

The proposal is a structured output (``VllmClient.propose`` with the PlanEditProposal
schema). llama.cpp's grammar-constrained ``response_format`` is stricter than Ollama's old
syntax-only ``format=<schema>``, but still doesn't guarantee every schema keyword, so
``propose()`` itself retries once on a malformed response; the retry loop below is for a
*different* concern — *semantic* quality (a proposal that legally parsed but made the plan
worse).
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import ValidationError

from app.models.schemas import (
    Course,
    PlanEditProposal,
    PlanResponse,
    RevisePlanResponse,
    StudentProfile,
)
from app.services.vllm_client import VllmClient, ModelResponseError
from app.services.planner import generate_plan

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3


# Static by design (prompt-cache prefix). Field-level semantics live on PlanEditProposal's
# Field descriptions, which travel with the structured-output schema — no JSON scaffolding
# in the prompt.
_SYSTEM_PROMPT = (
    "You are an assistant that tunes a college course plan from a student's feedback. You are "
    "NOT an official advisor and you must not invent courses, prerequisites, or requirements. "
    "A deterministic planner ENFORCES legality (prerequisites, term offerings, credit caps, "
    "and a limit on how many of the major's own courses share a semester); "
    "you only express preferences over the courses already listed. "
    # Without this, `reorder` reads as "put these in the first semester" and a model asked to
    # prioritise the CS track lists five CS courses expecting all five to land at once. The
    # planner spreads them instead, so the rationale it writes has to describe what will
    # actually happen. Kept UP HERE, next to the other "what the planner does for you" text,
    # so it is nowhere near the credit-cap paragraph below — see that comment.
    "It will not put more than the stated number of major-subject courses in one semester, so "
    "reordering several of them to the front moves them earlier in the ORDER — it does not "
    "stack them into the same term. "
    "Only use course codes and tags that appear in the context. Leave a list empty if it does "
    "not apply. Never put a course in both reorder and defer. "
    # LAST, AND SELF-CONTAINED. It used to open "The credit cap is the exception:" — fine when
    # the preceding sentence was about the planner owning legality, ambiguous once a
    # major-course sentence sat between them, because "the exception" then had two possible
    # referents. model_eval 2026-07-29: gemma4-e4b went 2/3 -> 0/3 on the explicit-cap scenario
    # when that text landed, while every other model held 3/3. Small models are evidently
    # sensitive to anything nearby that reads as "caps are not your job", which is the exact
    # failure this paragraph was written to fix. Name the field as the model's own job, and
    # leave nothing after it.
    "One thing IS yours to set: when the student asks for a specific per-semester credit load, "
    "you MUST set max_credits_per_semester to that number. Writing it only in the rationale "
    "has no effect."
)


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
               r"[^.\n]{0,40}?(\b\d{1,2}\b)\s*(?:cr\b|credits?\b|credit hours?\b)", re.IGNORECASE),
    # "12 credits a semester" / "12 credit semesters" with no leading verb
    re.compile(r"\b(\d{1,2})\s*(?:cr\b|credits?\b|credit hours?\b)"
               r"[^.\n]{0,20}?(?:a|per|each|every)\s+(?:semester|term)", re.IGNORECASE),
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


def _proposal_schema(profile: StudentProfile, catalog: list[Course]) -> dict:
    """PlanEditProposal's schema, narrowed to the codes and tags this student actually has.

    Grammar-level grounding, not prompt-level. Free-form string fields let a model write a whole
    semester layout ("FALL 2026:CS25000,CS25200,MA26100[14CR]") into a field that expects one
    course code, or invent a tag like 'seminar' that no course carries — 16% of parsed proposals
    in model_eval did exactly that, and ``_apply_proposal`` silently drops every one of them, so
    the student's request disappears with no error anywhere. An enum makes those strings
    undecodable instead, and costs nothing: constrained decoding over a short enum is cheaper
    than free-form text.

    Empty enums are skipped — an enum with no members is not a legal grammar.
    """
    schema = json.loads(PlanEditProposal.model_json_schema_json()) \
        if hasattr(PlanEditProposal, "model_json_schema_json") \
        else json.loads(json.dumps(PlanEditProposal.model_json_schema()))
    known = {course.code for course in catalog}
    codes = [code for code in profile.remaining_courses if code in known]
    tags = sorted({tag for course in catalog for tag in course.requirement_tags})
    if codes:
        for field in ("reorder", "defer"):
            schema["properties"][field]["items"] = {"type": "string", "enum": list(codes)}
    if tags:
        schema["properties"]["avoid_tags"]["items"] = {"type": "string", "enum": tags}
    return schema


def _course_context(profile: StudentProfile, catalog: list[Course]) -> str:
    """Compact list of the courses the model is allowed to move, with their tags.

    Restricting the model's palette to the student's remaining courses (and surfacing each
    course's tags) is what keeps ``avoid_tags``/``reorder`` grounded instead of invented.
    """
    by_code = {course.code: course for course in catalog}
    lines: list[str] = []
    for code in profile.remaining_courses:
        course = by_code.get(code)
        if course is None:
            continue
        tags = ", ".join(course.requirement_tags) or "none"
        lines.append(f"- {course.code} \"{course.title}\" ({course.credits} cr; tags: {tags})")
    return "\n".join(lines) or "(no known remaining courses)"


def _plan_summary(plan: PlanResponse) -> str:
    lines: list[str] = []
    for semester in plan.semesters:
        codes = ", ".join(course.code for course in semester.courses) or "(empty)"
        lines.append(f"- {semester.term} {semester.year}: {codes} [{semester.total_credits} cr]")
    if plan.unplanned_courses:
        lines.append(f"- Unplanned: {', '.join(plan.unplanned_courses)}")
    return "\n".join(lines) or "(no semesters planned)"


def _collect_warnings(plan: PlanResponse) -> list[str]:
    warnings = list(plan.warnings)
    for semester in plan.semesters:
        warnings.extend(semester.warnings)
    return warnings


def _user_prompt(
    profile: StudentProfile,
    plan: PlanResponse,
    catalog: list[Course],
    feedback: str,
    prior_warnings: list[str],
) -> str:
    parts = [
        f"DEGREE PROGRAM: {profile.degree_program}",
        # Major load first, credit cap second: the credit cap is the only line here the model
        # has to act on, so nothing sits between it and the feedback that could read as
        # "somebody else handles the caps". See _SYSTEM_PROMPT's closing paragraph.
        f"{profile.major_subject} courses per semester: at most "
        f"{profile.max_major_courses_per_semester} (hard limit), "
        f"{profile.preferred_major_courses_per_semester} or fewer preferred.",
        f"Current credit cap: {profile.max_credits_per_semester} per semester "
        f"(change it via max_credits_per_semester if the student asks for a different load).",
        "",
        "COURSES THAT CAN BE MOVED:",
        _course_context(profile, catalog),
        "",
        "CURRENT PLAN:",
        _plan_summary(plan),
        "",
        f"STUDENT FEEDBACK:\n{feedback}",
    ]
    if prior_warnings:
        parts += [
            "",
            "Your previous proposal left these planner warnings — fix them this time:",
            "\n".join(f"- {warning}" for warning in prior_warnings),
        ]
    return "\n".join(parts)


def _apply_proposal(
    profile: StudentProfile,
    proposal: PlanEditProposal,
    catalog: list[Course],
) -> StudentProfile:
    """Fold a proposal into a new profile the planner can re-plan from.

    Everything here is a preference, not a command: unknown codes are dropped, a course can't
    be both reordered and deferred, and the credit cap falls back to the profile's when absent.
    The planner then guarantees the result is legal regardless of what the model asked for.
    """
    by_code = {course.code: course for course in catalog}
    remaining = list(profile.remaining_courses)
    present = set(remaining)
    avoid = {tag.strip().lower() for tag in proposal.avoid_tags if tag.strip()}

    def is_avoided(code: str) -> bool:
        course = by_code.get(code)
        if course is None or not avoid:
            return False
        return any(tag.lower() in avoid for tag in course.requirement_tags)

    # Soft base order: courses carrying an avoided tag sink to the back (stable within groups).
    base = [c for c in remaining if not is_avoided(c)] + [c for c in remaining if is_avoided(c)]

    # Explicit reorder/defer override the soft base; reorder wins if a code appears in both.
    reorder = [c for c in proposal.reorder if c in present]
    reorder_set = set(reorder)
    defer = [c for c in proposal.defer if c in present and c not in reorder_set]
    pinned = reorder_set | set(defer)
    middle = [c for c in base if c not in pinned]
    new_remaining = reorder + middle + defer

    return profile.model_copy(
        update={
            "remaining_courses": new_remaining,
            "max_credits_per_semester": proposal.max_credits_per_semester
            or profile.max_credits_per_semester,
        }
    )


def _severity(plan: PlanResponse) -> int:
    """How incomplete a plan is. A revision is accepted only if it is no worse than baseline."""
    return len(plan.unplanned_courses)


async def propose_edit(
    profile: StudentProfile,
    catalog: list[Course],
    plan: PlanResponse,
    feedback: str,
    *,
    client: VllmClient,
    seed: int | None = None,
) -> PlanEditProposal:
    """MODE A, on its own: turn free-text feedback into a structured, grounded proposal.

    Split out of ``revise_plan`` so the AI-planner path can reuse it. This is the part of Mode A
    that has to understand the student; who then BUILDS the schedule from the proposal — the
    greedy planner (``revise_plan`` below) or Mode B (``ai_revise_plan``) — is a separate
    decision, and the eval measured this step in isolation for exactly that reason.
    """
    prompt = _user_prompt(profile, plan, catalog, feedback, [])
    proposal = await client.propose(
        _SYSTEM_PROMPT, prompt, PlanEditProposal,
        schema=_proposal_schema(profile, catalog), seed=seed,
    )
    return _with_credit_cap_backstop(proposal, feedback)


def _with_credit_cap_backstop(proposal: PlanEditProposal, feedback: str) -> PlanEditProposal:
    """DETERMINISTIC BACKSTOP for the one field the model must set itself.

    If the student named a per-semester load and the model left the field null, take the number
    from the text — a regex is right every time here, and the alternative is silently ignoring
    an explicit request. The model's own value always wins when it set one.
    """
    if proposal.max_credits_per_semester is not None:
        return proposal
    asked = extract_credit_cap(feedback)
    if asked is None:
        return proposal
    logger.info("revise-plan: model left the credit cap null; using %d from the student's own "
                "wording", asked)
    return proposal.model_copy(update={"max_credits_per_semester": asked})


def apply_proposal(
    profile: StudentProfile, proposal: PlanEditProposal, catalog: list[Course]
) -> StudentProfile:
    """Public name for ``_apply_proposal`` — the AI-planner path folds proposals in too."""
    return _apply_proposal(profile, proposal, catalog)


async def revise_plan(
    profile: StudentProfile,
    catalog: list[Course],
    feedback: str,
    *,
    client: VllmClient,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    seed: int | None = None,
) -> RevisePlanResponse:
    """Run the propose → apply → re-plan → re-validate loop and return the best plan found."""
    baseline = generate_plan(profile, catalog)
    baseline_severity = _severity(baseline)

    best_plan = baseline
    best_proposal = PlanEditProposal(rationale="No changes applied; showing the baseline plan.")
    accepted_iteration = 0
    prior_warnings: list[str] = []

    for iteration in range(1, max_iterations + 1):
        prompt = _user_prompt(profile, best_plan, catalog, feedback, prior_warnings)
        try:
            proposal = await client.propose(
                _SYSTEM_PROMPT, prompt, PlanEditProposal,
                schema=_proposal_schema(profile, catalog),
                seed=None if seed is None else seed + iteration,
            )
        except (ModelResponseError, ValidationError) as exc:
            # Refusal, truncation, or a value outside the client-side constraints (e.g. a
            # credit cap past le=24) — keep the best legal plan so far instead of failing.
            logger.warning("revise-plan: unusable proposal on iteration %d: %s", iteration, exc)
            break

        proposal = _with_credit_cap_backstop(proposal, feedback)

        revised_plan = generate_plan(_apply_proposal(profile, proposal, catalog), catalog)
        best_plan, best_proposal, accepted_iteration = revised_plan, proposal, iteration

        if _severity(revised_plan) <= baseline_severity:
            break  # legal and no worse than where we started — accept it.

        prior_warnings = _collect_warnings(revised_plan)

    return RevisePlanResponse(
        plan=best_plan,
        rationale=best_proposal.rationale,
        proposal=best_proposal,
        iterations=accepted_iteration,
        planner="deterministic",
    )


async def ai_revise_plan(
    profile: StudentProfile,
    program_catalog,
    feedback: str,
    current_plan: PlanResponse | None,
    *,
    client: VllmClient,
    seed: int | None = None,
) -> RevisePlanResponse:
    """MODE A proposes, MODE B rebuilds. The chat's revise path when the plan came from the AI.

    WHY NOT JUST HAND THE FEEDBACK TO MODE B. Because Mode A is the part that was measured on
    understanding the student, and it is grammar-constrained to say something actionable: an
    enum of the student's own remaining courses for reorder/defer, an enum of real tags for
    avoid_tags, and an integer field for the credit cap that a regex backstops when the model
    leaves it null. Free text into a planner has none of those guarantees — the eval found 16%
    of unconstrained proposals naming things that do not exist, every one of which the app
    silently degrades to a no-op, so the student's request disappears with no error anywhere.

    WHY NOT LET THE DETERMINISTIC PLANNER REBUILD, as ``revise_plan`` does. Because the student
    is looking at a plan Mode B wrote, and handing back a greedy-planner layout in response to
    "move CS 38100 earlier" changes every semester on the screen to answer a question about one
    course. Rebuilding with the same planner that drew the original keeps the diff the size of
    the request.

    The proposal is applied twice over, deliberately: to the PROFILE (so the reordered
    ``remaining_courses`` and any new credit cap reach both the prompt and the repair step) and
    to the PROSE the model reads, so a preference the grammar cannot express — "less
    theory-heavy" — still lands.
    """
    from app.services.ai_planner import generate_ai_plan  # circular at module scope

    catalog = program_catalog.courses
    baseline = current_plan or generate_plan(profile, catalog)

    try:
        proposal = await propose_edit(
            profile, catalog, baseline, feedback, client=client, seed=seed
        )
    except (ModelResponseError, ValidationError) as exc:
        # The student still gets an answer: Mode B alone can act on the raw feedback, it just
        # cannot be held to the structured guarantees above.
        logger.warning("revise-plan: unusable proposal (%s); replanning from the raw request",
                       exc)
        proposal = PlanEditProposal(rationale="")

    revised_profile = _apply_proposal(profile, proposal, catalog)
    result = await generate_ai_plan(
        revised_profile, program_catalog, client=client,
        request=_revision_request(feedback, proposal), seed=seed,
    )
    return RevisePlanResponse(
        plan=result.plan,
        rationale=proposal.rationale or result.rationale,
        proposal=proposal,
        iterations=1,
        planner="ai" if result.used_model else "deterministic",
        layout=result.layout,
        removed=result.removed,
        backfilled=result.backfilled,
        requirement_coverage=result.requirement_coverage,
        missing_requirements=result.missing_requirements,
    )


def _revision_request(feedback: str, proposal: PlanEditProposal) -> str:
    """The student's words, plus the structured reading of them, for Mode B's variable tail.

    Both halves matter. The raw feedback carries intent the proposal fields cannot hold; the
    structured reading is what the grammar already validated, and repeating it as an explicit
    instruction is what stops a model from reading a long message and acting on the first
    sentence only.
    """
    parts = [feedback.strip()]
    if proposal.reorder:
        parts.append(f"Schedule these EARLIER than before: {', '.join(proposal.reorder)}.")
    if proposal.defer:
        parts.append(f"Push these to LATER semesters: {', '.join(proposal.defer)}.")
    if proposal.avoid_tags:
        parts.append(f"Prefer courses that are not tagged: {', '.join(proposal.avoid_tags)}.")
    if proposal.max_credits_per_semester:
        parts.append(f"Hard limit of {proposal.max_credits_per_semester} credits per semester.")
    return "\n".join(parts)
