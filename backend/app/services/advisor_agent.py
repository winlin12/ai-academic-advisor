"""LFM2 revise-plan agent: the model proposes, the deterministic planner disposes.

The flow is deliberately asymmetric so a local ~8B model can never emit an illegal schedule:

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
"""

from __future__ import annotations

import logging

from pydantic import ValidationError

from app.models.schemas import (
    Course,
    PlanEditProposal,
    PlanResponse,
    RevisePlanResponse,
    StudentProfile,
)
from app.services.ollama_client import ModelJSONError, OllamaClient
from app.services.planner import generate_plan

logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3


_SYSTEM_PROMPT = (
    "You are an assistant that tunes a college course plan from a student's feedback. You are "
    "NOT an official advisor and you must not invent courses, prerequisites, or requirements. "
    "A deterministic planner owns legality (prerequisites, term offerings, credit caps); you "
    "only express preferences over the courses already listed. Respond with a SINGLE JSON "
    "object and nothing else, using exactly these keys:\n"
    '  "rationale": string — one or two sentences explaining the change, for the student.\n'
    '  "reorder": string[] — course codes to take earlier, highest priority first.\n'
    '  "defer": string[] — course codes to push to later semesters.\n'
    '  "avoid_tags": string[] — requirement tags to deprioritise (e.g. "theory-heavy").\n'
    '  "max_credits_per_semester": integer or null — new per-semester credit cap if asked.\n'
    "Only use course codes and tags that appear in the context. Leave a list empty if it does "
    "not apply. Never put a course in both reorder and defer."
)


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
        f"STUDENT: {profile.name} — {profile.degree_program}",
        f"Current credit cap: {profile.max_credits_per_semester} per semester.",
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


async def revise_plan(
    profile: StudentProfile,
    catalog: list[Course],
    feedback: str,
    *,
    client: OllamaClient,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
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
            raw = await client.generate_json(_SYSTEM_PROMPT, prompt)
            proposal = PlanEditProposal.model_validate(raw)
        except (ModelJSONError, ValidationError) as exc:
            # The model produced junk — keep the best legal plan so far instead of failing.
            logger.warning("revise-plan: unusable proposal on iteration %d: %s", iteration, exc)
            break

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
    )
