"""Tests for MODE B — the path where the model writes the schedule.

The model is stubbed throughout. What is being tested is everything AROUND the generation: that
the prompt tells the student's truth, that a truncated draft is salvaged rather than thrown
away, that NOTHING is added the model did not ask for, and that a model failure still produces
a plan the student can use.
"""

import asyncio

import pytest

from app.models.schemas import Course, StudentProfile
from app.services.ai_planner import (
    AiPlanDraft,
    AiSemester,
    _salvage_semesters,
    build_prompts,
    generate_ai_plan,
    plan_schema,
)
from app.services.llamacpp_client import ModelResponseError
from app.services.planner_db import ProgramCatalog, RequirementGroup


def _course(code, credits=3, groups=(), terms=("fall", "spring")):
    return Course(
        code=code, title=f"{code} title", credits=credits,
        prereq_groups=[list(g) for g in groups],
        prereqs=sorted({c for g in groups for c in g}),
        offered_terms=list(terms),
    )


def _catalog() -> ProgramCatalog:
    courses = [
        _course("CS 18000", credits=4),
        _course("CS 18200", groups=[["CS 18000"]]),
        _course("CS 25100", groups=[["CS 18200"]]),
        _course("MA 16100", credits=5),
        _course("ENGL 10600"),
    ]
    return ProgramCatalog(
        program_id="11111111-1111-1111-1111-111111111111",
        name="Computer Science, BS",
        catalog_year="2026-2027",
        groups=[
            RequirementGroup(id="core", name="CS Core", selective=False, credits_min=10.0,
                             options=[["CS 18000"], ["CS 18200"], ["CS 25100"]]),
            RequirementGroup(id="gen", name="General education", selective=False,
                             credits_min=8.0, options=[["MA 16100"], ["ENGL 10600"]]),
        ],
        courses=courses,
    )


def _profile(**overrides) -> StudentProfile:
    base = {
        "profile_label": "Test", "degree_program": "Computer Science",
        "program_id": "11111111-1111-1111-1111-111111111111",
        "completed_courses": [], "start_term": "fall", "start_year": 2026,
        "semesters_to_plan": 4, "max_credits_per_semester": 12, "major_subject": "CS",
        "max_major_courses_per_semester": 2, "preferred_major_courses_per_semester": 2,
        "remaining_courses": ["CS 18000", "CS 18200", "CS 25100", "MA 16100", "ENGL 10600"],
    }
    return StudentProfile(**{**base, **overrides})


class _StubClient:
    """Returns a canned draft, or raises to exercise the failure paths."""

    def __init__(self, draft=None, error=None):
        self.model = "stub-model"
        self._draft = draft
        self._error = error
        self.seeds: list[int | None] = []
        self.max_tokens: list[int | None] = []

    async def propose(self, system_prompt, user_prompt, output_type, schema=None, *,
                      seed=None, max_tokens=None):
        self.seeds.append(seed)
        self.max_tokens.append(max_tokens)
        if self._error is not None:
            raise self._error
        return self._draft


# --- the prompt ------------------------------------------------------------------------------


def test_the_prompt_names_the_target_load_not_only_the_ceiling():
    """A ceiling alone produced a monotonic slide to half the opening load across 162 eval
    plans, because nothing ever told the model the room was supposed to be used."""
    _system, user = build_prompts(_profile(), _catalog())
    assert "the load to reach, not a ceiling" in user
    assert "credits still outstanding" in user
    assert "12 is the hard limit" in user


def test_the_completed_list_is_stated_as_closed():
    """Two of six models filled in what they thought "transferred in my math credits" meant and
    dropped a course nothing said was done."""
    _system, user = build_prompts(_profile(completed_courses=["MA 16100"]), _catalog())
    assert "Already completed (complete list): MA 16100" in user
    # With nothing completed there is no list to call closed.
    _system, empty = build_prompts(_profile(completed_courses=[]), _catalog())
    assert "Already completed: none" in empty


def test_the_calendar_is_spelled_out_term_by_term():
    _system, user = build_prompts(_profile(start_term="spring", start_year=2027), _catalog())
    assert "spring 2027 -> fall 2027 -> spring 2028 -> fall 2028" in user


def test_the_students_request_travels_in_the_variable_tail_only():
    """The ~11k-token export must stay byte-identical between students or llama-server's KV
    cache re-reads it on every request."""
    system_a, user_a = build_prompts(_profile(), _catalog(), "graduate early")
    system_b, user_b = build_prompts(_profile(profile_label="Other"), _catalog(),
                                     "keep fridays free")
    assert system_a == system_b
    assert "graduate early" in user_a and "graduate early" not in user_b
    assert "keep fridays free" in user_b


def test_the_profile_label_never_reaches_the_model():
    """profile_label is a local display label the student may not even set — it names the
    saved profile, not the person, and must never appear in what an external llama-server
    process sees or logs."""
    _system, user = build_prompts(_profile(profile_label="my backup CS plan"), _catalog())
    assert "my backup CS plan" not in user
    assert "DEGREE PROGRAM: Computer Science" in user


def test_the_schema_constrains_terms_and_courses_to_what_exists():
    schema = plan_schema(["fall", "spring"], ["CS 18000", "CS 18200"])
    semester = schema["properties"]["semesters"]["items"]["properties"]
    assert semester["term"]["enum"] == ["fall", "spring"]
    assert semester["courses"]["items"]["enum"] == ["CS 18000", "CS 18200"]
    # An enum with no members is not a legal grammar.
    assert "enum" not in plan_schema(["fall"], [])["properties"]["semesters"]["items"][
        "properties"]["courses"]["items"]


# --- salvage ---------------------------------------------------------------------------------


def test_a_draft_truncated_mid_json_keeps_its_complete_semesters():
    truncated = (
        '{"semesters":[{"term":"fall","year":2026,"courses":["CS 18000"]},'
        '{"term":"spring","year":2027,"courses":["CS 18200"]},'
        '{"term":"fall","year":2027,"courses":["CS 251'
    )
    draft = _salvage_semesters(truncated)
    assert draft is not None
    assert [s.courses for s in draft.semesters] == [["CS 18000"], ["CS 18200"]]


def test_salvage_is_not_fooled_by_a_brace_inside_a_string():
    raw = '{"semesters":[{"term":"fall","year":2026,"courses":["CS {18000}"]}'
    draft = _salvage_semesters(raw)
    assert draft is not None
    assert draft.semesters[0].courses == ["CS {18000}"]


def test_salvage_returns_none_when_nothing_complete_was_produced():
    assert _salvage_semesters('{"semesters":[{"term":"fa') is None
    assert _salvage_semesters("") is None
    assert _salvage_semesters('{"rationale":"I refuse"}') is None


# --- nothing is auto-filled -------------------------------------------------------------------


def test_courses_the_model_left_out_are_reported_unfilled_not_quietly_chosen():
    """A selective group is a MENU. Picking the first options in catalog order is a coin toss,
    not advice, and a student who sees a course in their schedule is entitled to assume somebody
    chose it. The gap is surfaced instead."""
    draft = AiPlanDraft(
        semesters=[AiSemester(term="fall", year=2026, courses=["CS 18000"])],
        rationale="",
    )
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=_StubClient(draft)))

    scheduled = {c.code for s in result.plan.semesters for c in s.courses}
    assert scheduled == {"CS 18000"}, "nothing was added that the model did not ask for"
    assert set(result.plan.unplanned_courses) == {
        "CS 18200", "CS 25100", "MA 16100", "ENGL 10600",
    }
    assert result.requirement_coverage < 1.0
    assert result.missing_requirements, "the gaps are named, not hidden"


def test_a_forced_prerequisite_is_added_but_a_free_choice_is_not():
    """The one thing repair may add is a prerequisite of something already in the plan — that is
    forced by the plan the model wrote, not a choice made on the student's behalf."""
    draft = AiPlanDraft(
        semesters=[AiSemester(term="fall", year=2026, courses=["CS 18200"])],
        rationale="",
    )
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=_StubClient(draft)))

    scheduled = {c.code for s in result.plan.semesters for c in s.courses}
    assert scheduled == {"CS 18000", "CS 18200"}, "CS 18000 is forced; nothing else is"
    assert result.backfilled == ["CS 18000"]


# --- end to end, with the model stubbed ------------------------------------------------------


def test_a_good_draft_survives_and_is_reported_as_the_models_work():
    draft = AiPlanDraft(
        semesters=[
            AiSemester(term="fall", year=2026, courses=["CS 18000", "MA 16100"]),
            AiSemester(term="spring", year=2027, courses=["CS 18200", "ENGL 10600"]),
            AiSemester(term="fall", year=2027, courses=["CS 25100"]),
        ],
        rationale="Foundations first.",
    )
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=_StubClient(draft)))

    assert result.used_model
    assert result.rationale == "Foundations first."
    assert result.requirement_coverage == pytest.approx(1.0)
    assert result.removed == []
    assert [c.code for c in result.plan.semesters[0].courses] == ["CS 18000", "MA 16100"]


def test_a_draft_that_forgets_a_root_course_keeps_its_chain():
    """The regression that motivated move-before-delete: one omission, not a deleted degree."""
    draft = AiPlanDraft(
        semesters=[
            AiSemester(term="fall", year=2026, courses=["CS 18200"]),
            AiSemester(term="spring", year=2027, courses=["CS 25100"]),
        ],
        rationale="",
    )
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=_StubClient(draft)))

    scheduled = {c.code for s in result.plan.semesters for c in s.courses}
    assert {"CS 18000", "CS 18200", "CS 25100"} <= scheduled
    assert not [w for w in result.plan.warnings if "prereq_violation" in w]
    # The courses the model simply never mentioned stay unfilled — the rescue is for the
    # prerequisite chain it broke, not a licence to finish the plan on its behalf.
    assert set(result.plan.unplanned_courses) == {"MA 16100", "ENGL 10600"}


def test_a_downed_model_still_produces_a_plan_and_says_so():
    client = _StubClient(error=ModelResponseError("no schema-valid output", raw=""))
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=client))

    assert result.used_model is False
    assert "deterministic planner" in result.rationale
    assert result.plan.semesters, "a fallback plan is still a plan"
    assert any("deterministic planner" in w for w in result.plan.warnings)


def test_a_truncated_model_response_is_salvaged_rather_than_discarded():
    raw = ('{"semesters":[{"term":"fall","year":2026,"courses":["CS 18000","MA 16100"]},'
           '{"term":"spring","year":2027,"courses":["CS 18200","ENG')
    client = _StubClient(error=ModelResponseError("truncated", raw=raw))
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=client))

    assert result.used_model, "the draft was usable — this is not a model outage"
    assert "CS 18000" in result.model_placed


def test_the_seed_reaches_the_model_so_regenerate_means_something():
    draft = AiPlanDraft(semesters=[], rationale="")
    client = _StubClient(draft)
    asyncio.run(generate_ai_plan(_profile(), _catalog(), client=client, seed=4242))
    assert client.seeds == [4242]


def test_the_output_budget_is_what_the_context_window_leaves_over(monkeypatch):
    from app.services import ai_planner

    # A window barely larger than the prompt must not ask for a 2048-token plan; a request that
    # overflows is truncated mid-JSON, not rejected, and reads as "the planner was unavailable".
    monkeypatch.setattr(ai_planner.settings, "llamacpp_context_tokens", 100_000)
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    asyncio.run(generate_ai_plan(_profile(), _catalog(), client=client))
    assert client.max_tokens == [ai_planner.settings.llamacpp_plan_max_tokens]

    monkeypatch.setattr(ai_planner.settings, "llamacpp_context_tokens", 1_000)
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    result = asyncio.run(generate_ai_plan(_profile(), _catalog(), client=client))
    assert client.max_tokens == [], "the model should not have been called at all"
    assert result.used_model is False
    assert "context window" in result.rationale
