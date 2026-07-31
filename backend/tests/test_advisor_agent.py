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
        self.schemas: list[dict | None] = []

    async def propose(
        self, system_prompt: str, user_prompt: str, output_type: type[PlanEditProposal],
        schema: dict | None = None,
    ) -> PlanEditProposal:
        self.prompts.append(user_prompt)
        # Mirrors the real client: callers may narrow the schema per request (course-code and
        # tag enums built from this student's own courses). Captured so a test can assert on it.
        self.schemas.append(schema)
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

        async def propose(self, system_prompt: str, user_prompt: str, output_type,
                          schema: dict | None = None) -> PlanEditProposal:
            raise ModelResponseError("stop_reason=refusal")

    result = asyncio.run(revise_plan(_profile(), load_catalog(), "whatever", client=_RefusingClient()))

    # No usable proposal -> the baseline plan is returned unchanged, flagged as 0 iterations.
    assert result.iterations == 0
    assert result.plan.semesters


def test_proposal_schema_is_narrowed_to_this_student_and_catalog():
    """Grammar-level grounding: the model must not be *able* to name a course or tag that
    doesn't exist. 16% of parsed proposals in model_eval were ungrounded — whole semester
    layouts written into `reorder`, tags like 'seminar' invented wholesale — and every one was
    silently dropped by _apply_proposal, costing the student their request with no error."""
    from app.services.advisor_agent import _proposal_schema

    catalog = load_catalog()
    profile = _profile()
    schema = _proposal_schema(profile, catalog)

    for field in ("reorder", "defer"):
        assert schema["properties"][field]["items"]["enum"] == profile.remaining_courses, field
    tags = schema["properties"]["avoid_tags"]["items"]["enum"]
    assert tags == sorted(set(tags)) and "required" in tags
    # ...and the narrowed schema is what actually reaches the client.
    client = _StubClient([PlanEditProposal(rationale="ok")])
    asyncio.run(revise_plan(profile, catalog, "make it lighter", client=client))
    assert client.schemas[0] is not None
    assert client.schemas[0]["properties"]["reorder"]["items"]["enum"]


def test_credit_cap_is_recovered_from_the_students_own_wording():
    """The one field the model must set itself, backstopped deterministically.

    A student naming a number is a parsing problem, and a regex is right every time where the
    best model measured in model_eval was right 0/3 — it left the field null and wrote the
    number into the rationale, where nothing consumes it.
    """
    client = _StubClient([PlanEditProposal(rationale="sure", max_credits_per_semester=None)])
    result = asyncio.run(revise_plan(
        _profile(), load_catalog(), "please cap me at 12 credits a semester", client=client))

    assert result.proposal.max_credits_per_semester == 12
    assert all(s.total_credits <= 12 for s in result.plan.semesters)


def test_credit_cap_backstop_never_overrides_the_model():
    client = _StubClient([PlanEditProposal(rationale="sure", max_credits_per_semester=9)])
    result = asyncio.run(revise_plan(
        _profile(), load_catalog(), "please cap me at 12 credits a semester", client=client))

    assert result.proposal.max_credits_per_semester == 9
