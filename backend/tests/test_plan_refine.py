"""Tests for MODE C — the three things a student does after reading a plan.

The model is stubbed. What is under test is the policy around it, and the policies differ in
ways that are invisible from the output alone: FILL must FREEZE what already validates, and
must not re-seed; REGENERATE must hold the seed fixed so a better second plan is attributable
to the feedback rather than to a luckier sample; START OVER must move the seed so pressing it
twice explores instead of redrawing.
"""

import asyncio

import pytest

from app.models.schemas import Course, StudentProfile
from app.services.ai_planner import AiPlanDraft, AiSemester
from app.services.vllm_client import ModelResponseError
from app.services.plan_refine import refine_plan
from app.services.planner_db import ProgramCatalog, RequirementGroup


def _course(code, credits=3, groups=(), terms=("fall", "spring")):
    return Course(
        code=code, title=f"{code} title", credits=credits,
        prereq_groups=[list(g) for g in groups],
        prereqs=sorted({c for g in groups for c in g}),
        offered_terms=list(terms),
    )


def _catalog() -> ProgramCatalog:
    return ProgramCatalog(
        program_id="11111111-1111-1111-1111-111111111111",
        name="Computer Science, BS",
        catalog_year="2026-2027",
        groups=[
            RequirementGroup(id="core", name="CS Core", selective=False, credits_min=10.0,
                             options=[["CS 18000"], ["CS 18200"], ["CS 25100"]]),
            RequirementGroup(id="gen", name="General education", selective=False,
                             credits_min=8.0, options=[["MA 26100"], ["ENGL 10600"]]),
        ],
        courses=[
            _course("CS 18000", credits=4),
            _course("CS 18200", groups=[["CS 18000"]]),
            _course("CS 25100", groups=[["CS 18200"]]),
            _course("MA 26100", credits=4),
            _course("ENGL 10600"),
        ],
    )


def _profile(**overrides) -> StudentProfile:
    base = {
        "name": "Test", "degree_program": "Computer Science",
        "program_id": "11111111-1111-1111-1111-111111111111",
        "completed_courses": [], "start_term": "fall", "start_year": 2026,
        "semesters_to_plan": 4, "max_credits_per_semester": 12, "major_subject": "CS",
        "max_major_courses_per_semester": 2, "preferred_major_courses_per_semester": 2,
        "remaining_courses": ["CS 18000", "CS 18200", "CS 25100", "MA 26100", "ENGL 10600"],
    }
    return StudentProfile(**{**base, **overrides})


class _StubClient:
    def __init__(self, draft=None, error=None):
        self.model = "stub-model"
        self._draft = draft
        self._error = error
        self.seeds: list[int | None] = []
        self.systems: list[str] = []
        self.users: list[str] = []

    async def propose(self, system_prompt, user_prompt, output_type, schema=None, *,
                      seed=None, max_tokens=None):
        self.seeds.append(seed)
        self.systems.append(system_prompt)
        self.users.append(user_prompt)
        if self._error is not None:
            raise self._error
        return self._draft


def _run(profile, catalog, semesters, mode, client, **kw):
    return asyncio.run(refine_plan(profile, catalog, semesters, mode, client=client, **kw))


# --- the duplicate the user asked about ------------------------------------------------------


def test_fill_drops_the_LATER_duplicate_and_freezes_the_earlier_one():
    """MA 26100 listed twice comes back once, in the slot it first had, and frozen."""
    plan = [["MA 26100", "CS 18000"], ["CS 18200"], ["MA 26100"], []]
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))

    result, outcome = _run(_profile(), _catalog(), plan, "fill", client)

    scheduled = [c.code for s in result.plan.semesters for c in s.courses]
    assert scheduled.count("MA 26100") == 1
    assert "MA 26100" in [c.code for c in result.plan.semesters[0].courses]
    assert result.plan.semesters[2].courses == []
    assert "MA 26100" in outcome.removed
    assert "MA 26100" in outcome.kept, "the surviving copy is frozen, not re-rolled"


# --- FILL: the freeze is the whole point ------------------------------------------------------


def test_fill_freezes_every_validating_placement_and_keeps_it_where_it_was():
    plan = [["CS 18000"], ["CS 18200"], [], []]
    draft = AiPlanDraft(
        semesters=[AiSemester(term="fall", year=2027, courses=["CS 25100", "MA 26100"])],
        rationale="Filled the gap.",
    )
    client = _StubClient(draft)

    result, outcome = _run(_profile(), _catalog(), plan, "fill", client)

    assert [c.code for c in result.plan.semesters[0].courses] == ["CS 18000"]
    assert [c.code for c in result.plan.semesters[1].courses] == ["CS 18200"]
    assert set(outcome.kept) == {"CS 18000", "CS 18200"}
    assert "CS 25100" in outcome.added


def test_fill_reads_the_delta_by_TERM_LABEL_not_by_position():
    """A delta reply names only the semesters it touches, so its first entry is not semester 1.

    Reading it positionally silently relocates every course in it — which is how an answer
    meant for semester 3 lands in semester 1 and busts the credit cap.
    """
    plan = [["CS 18000"], ["CS 18200"], [], []]
    draft = AiPlanDraft(
        semesters=[AiSemester(term="fall", year=2027, courses=["CS 25100"])],  # semester 3
        rationale="",
    )
    result, _outcome = _run(_profile(), _catalog(), plan, "fill", _StubClient(draft))

    assert [c.code for c in result.plan.semesters[2].courses] == ["CS 25100"]
    assert all(c.code != "CS 25100" for c in result.plan.semesters[0].courses)


def test_fill_tells_the_model_what_is_confirmed_what_is_missing_and_where_it_can_go():
    plan = [["CS 18000"], ["CS 18200"], [], []]
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    _run(_profile(), _catalog(), plan, "fill", client)

    user = client.users[0]
    assert "ALREADY CONFIRMED" in user
    assert "Semester 1 (fall 2026): CS 18000" in user
    assert "STILL MISSING" in user
    assert "WHERE THEY CAN GO" in user
    # The legal slots are computed, and the label is the one the reply schema is keyed on.
    assert "CS 25100" in user and "semester 3 (fall 2027)" in user
    assert "Return ONLY the semesters you are adding courses to" in user
    # The locked-slot rules ride on top of the shared planning rules, never instead of them.
    assert "Some slots in this student's plan are LOCKED" in client.systems[0]
    assert "PREREQUISITES COME FIRST" in client.systems[0]


def test_fill_does_not_call_the_model_when_there_is_nothing_to_fill():
    complete = [["CS 18000", "MA 16100"], ["CS 18200", "ENGL 10600"], ["CS 25100"], []]
    profile = _profile(remaining_courses=["CS 18000", "CS 18200", "CS 25100", "ENGL 10600"])
    catalog = _catalog()
    catalog.courses.append(_course("MA 16100", credits=4))
    catalog.groups[1].options = [["ENGL 10600"]]

    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    result, outcome = _run(profile, catalog, complete, "fill", client)

    assert client.seeds == [], "no reason to spend 30s being told there is nothing to do"
    assert outcome.used_model is False
    assert outcome.note == "nothing-to-fill"
    assert "nothing to fill" in result.rationale


def test_fill_leaves_the_plan_intact_when_the_model_is_unreachable():
    plan = [["CS 18000"], ["CS 18200"], [], []]
    client = _StubClient(error=ModelResponseError("down", raw=""))
    result, outcome = _run(_profile(), _catalog(), plan, "fill", client)

    assert outcome.used_model is False
    assert outcome.note == "model-unavailable"
    assert [c.code for c in result.plan.semesters[0].courses] == ["CS 18000"]


# --- REGENERATE: the seed is held fixed on purpose --------------------------------------------


def test_regenerate_tells_the_model_what_was_wrong():
    # CS 25100 before its prerequisite is a violation the checker can name precisely.
    plan = [["CS 25100"], ["CS 18000"], ["CS 18200"], []]
    client = _StubClient(AiPlanDraft(
        semesters=[
            AiSemester(term="fall", year=2026, courses=["CS 18000"]),
            AiSemester(term="spring", year=2027, courses=["CS 18200"]),
            AiSemester(term="fall", year=2027, courses=["CS 25100"]),
        ],
        rationale="Fixed the ordering.",
    ))
    result, _outcome = _run(_profile(), _catalog(), plan, "regenerate", client, seed=7)

    user = client.users[0]
    assert "Your previous plan was checked and these problems were found" in user
    assert "CS 25100" in user
    assert client.seeds == [7], "the seed is held FIXED — only the context changes"
    assert result.rationale == "Fixed the ordering."


def test_regenerate_says_so_when_nothing_is_actually_wrong_AND_moves_the_seed():
    """A clean plan has no feedback to give, so holding the seed would reproduce it exactly.

    Fixing the seed is what makes "did the model fix the error?" answerable — but with no error
    there is no such question left to protect, and the student pressing Regenerate would watch
    nothing happen. Measured that way in the browser before this branch existed.
    """
    plan = [["CS 18000", "MA 26100"], ["CS 18200", "ENGL 10600"], ["CS 25100"], []]
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    _run(_profile(), _catalog(), plan, "regenerate", client, seed=3)

    assert "Nothing was found wrong with this plan" in client.users[0]
    assert client.seeds != [3], "with nothing to fix, the sample has to move"


def test_regenerate_holds_the_seed_when_there_IS_something_to_fix():
    plan = [["CS 25100"], ["CS 18000"], ["CS 18200"], []]
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    _run(_profile(), _catalog(), plan, "regenerate", client, seed=3)
    assert client.seeds == [3]


def test_regenerate_caps_how_many_problems_it_shows_and_says_it_capped_them():
    from app.services.plan_refine import FEEDBACK_TOP_N

    # Every course before its prerequisites, plus duplicates: many violations at once.
    plan = [["CS 25100", "CS 18200", "CS 25100"], ["CS 25100"], ["CS 18200"], []]
    client = _StubClient(AiPlanDraft(semesters=[], rationale=""))
    _run(_profile(), _catalog(), plan, "regenerate", client, seed=1)

    # Only the problems block — the unmet-requirement list that follows it is bulleted too.
    after = client.users[0].split("Fix them and return a corrected plan:")[1].splitlines()
    listed: list[str] = []
    for line in after:
        if not line.strip():
            if listed:          # blank line AFTER the block ends it; the leading one does not
                break
            continue
        listed.append(line)

    assert len(listed) <= FEEDBACK_TOP_N + 1, "the list is a sample, not a transcription"
    # The count of what was withheld is stated, or the model "fixes" five and calls it clean.
    assert any("more of the same kinds" in line for line in listed)


# --- START OVER: a fresh sample, told nothing -------------------------------------------------


def test_start_over_tells_the_model_nothing_about_the_plan_it_replaces():
    plan = [["CS 25100"], ["CS 18000"], [], []]
    client = _StubClient(AiPlanDraft(
        semesters=[AiSemester(term="fall", year=2026, courses=["CS 18000"])], rationale="Fresh."
    ))
    _result, outcome = _run(_profile(), _catalog(), plan, "start-over", client, seed=11)

    assert outcome.mode == "start-over"
    # It delegates to the ordinary first-attempt generator, so nothing about the old plan can
    # leak in — no locked block, no violation list.
    assert outcome.kept == []


def test_start_over_creeps_the_temperature_so_repeated_presses_explore():
    from app.services import plan_refine

    plan = [["CS 18000"], [], [], []]
    seen: list[float] = []

    class _Recorder(_StubClient):
        pass

    original = plan_refine.VllmClient

    def _capture(*args, **kwargs):
        seen.append(kwargs.get("temperature"))
        return _Recorder(AiPlanDraft(semesters=[], rationale=""))

    plan_refine.VllmClient = _capture
    try:
        for attempt in (1, 2, 5):
            _run(_profile(), _catalog(), plan, "start-over", _StubClient(), attempt=attempt)
    finally:
        plan_refine.VllmClient = original

    assert seen == sorted(seen), "temperature never goes down"
    assert seen[0] < seen[-1], "pressing it again samples more widely"
    assert seen[-1] <= plan_refine.TEMPERATURE_MAX


# --- every mode goes through the checker ------------------------------------------------------


@pytest.mark.parametrize("mode", ["fill", "regenerate", "start-over"])
def test_no_mode_can_return_an_illegal_plan(mode):
    """Including FILL, whose merge hands the model a delta it never saw in full context."""
    plan = [["CS 25100", "MA 26100"], ["CS 18000"], ["MA 26100"], []]
    # A draft that would be illegal if taken at face value: a duplicate and a bad ordering.
    draft = AiPlanDraft(
        semesters=[
            AiSemester(term="fall", year=2026, courses=["CS 25100", "CS 18000"]),
            AiSemester(term="spring", year=2027, courses=["MA 26100", "MA 26100"]),
        ],
        rationale="",
    )
    result, _outcome = _run(_profile(), _catalog(), plan, mode, _StubClient(draft))

    assert result.violations == [] or all(
        "heavy_major_load" in v or "over_requested_credits" in v for v in result.violations
    ), result.violations
    codes = [c.code for s in result.plan.semesters for c in s.courses]
    assert len(codes) == len(set(codes)), "no course appears twice in any mode's output"
