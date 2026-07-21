"""Tests for the revise-plan agent.

The model transport is stubbed (canned PlanEditProposal objects, mirroring what
LlamaCppClient.propose returns after schema validation), so these run with no network — they
exercise the proposal-application logic and the propose → re-plan loop against the seed
catalog.
"""

import asyncio

from app.models.schemas import PlanEditProposal, StudentProfile
from app.services.advisor_agent import _apply_proposal, revise_plan
from app.services.catalog import load_catalog
from app.services.llamacpp_client import ModelResponseError


class _StubClient:
    """Stand-in for LlamaCppClient that returns canned proposals in order."""

    def __init__(self, responses: list[PlanEditProposal]):
        self.model = "stub-model"
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def propose(
        self, system_prompt: str, user_prompt: str, output_type: type[PlanEditProposal]
    ) -> PlanEditProposal:
        self.prompts.append(user_prompt)
        return self._responses.pop(0)


def _profile(**overrides) -> StudentProfile:
    base = dict(
        completed_courses=["CS251"],
        remaining_courses=["CS381", "CS502", "CS536"],
        start_term="fall",
        start_year=2026,
        semesters_to_plan=4,
        max_credits_per_semester=6,
    )
    base.update(overrides)
    return StudentProfile(**base)


def test_apply_proposal_reorders_defers_and_drops_unknown_codes():
    revised = _apply_proposal(
        _profile(),
        PlanEditProposal(reorder=["CS536", "FAKE999"], defer=["CS381"]),
        load_catalog(),
    )
    # CS536 pulled to the front, CS381 pushed to the back, unknown code ignored.
    assert revised.remaining_courses == ["CS536", "CS502", "CS381"]


def test_apply_proposal_avoid_tags_sinks_matching_courses():
    revised = _apply_proposal(
        _profile(remaining_courses=["CS381", "CS536", "CS541"]),
        PlanEditProposal(avoid_tags=["theory-heavy"]),  # only CS381 is theory-heavy
        load_catalog(),
    )
    assert revised.remaining_courses[-1] == "CS381"


def test_apply_proposal_keeps_profile_cap_when_proposal_omits_one():
    revised = _apply_proposal(_profile(max_credits_per_semester=9), PlanEditProposal(), load_catalog())
    assert revised.max_credits_per_semester == 9


def test_revise_plan_applies_credit_cap_and_returns_rationale():
    client = _StubClient(
        [PlanEditProposal(rationale="Capping the load.", max_credits_per_semester=3)]
    )
    result = asyncio.run(
        revise_plan(_profile(max_credits_per_semester=9), load_catalog(), "lighter please", client=client)
    )

    assert result.rationale == "Capping the load."
    assert result.iterations == 1
    assert result.proposal.max_credits_per_semester == 3
    assert all(semester.total_credits <= 3 for semester in result.plan.semesters)


def test_revise_plan_falls_back_to_baseline_on_refusal():
    class _RefusingClient:
        model = "stub"

        async def propose(self, system_prompt: str, user_prompt: str, output_type) -> PlanEditProposal:
            raise ModelResponseError("stop_reason=refusal")

    result = asyncio.run(revise_plan(_profile(), load_catalog(), "whatever", client=_RefusingClient()))

    # No usable proposal -> the baseline plan is returned unchanged, flagged as 0 iterations.
    assert result.iterations == 0
    assert result.plan.semesters
