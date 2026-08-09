"""Tests for the degree checklist — the view that answers "am I going to graduate".

No database, no model. The interesting cases are all about what counts as ONE requirement:
alternatives are one slot, a selective group is a credit target rather than a course list, and a
substitute the student already took satisfies the primary it stands in for.
"""

import pytest

from app.models.schemas import Course, StudentProfile
from app.services import plan_requirements
from app.services.plan_requirements import requirement_progress, semester_labels
from app.services.planner_db import ProgramCatalog, RequirementGroup


@pytest.fixture(autouse=True)
def _no_real_suggestions(monkeypatch):
    """Keeps this file's "no database, no model" promise: `requirement_progress` normally
    calls `suggest_courses`, which hits a live embedding model + Postgres. Stubbed here to
    empty by default; the real path is covered in test_plan_suggestions.py, and the wiring
    (that a real return value reaches `UnresolvedRequirementView`) is covered below."""
    monkeypatch.setattr(plan_requirements, "suggest_courses", lambda *a, **k: [])


def _course(code, credits=3, groups=(), terms=("fall", "spring")):
    return Course(
        code=code, title=f"{code} title", credits=credits,
        prereq_groups=[list(g) for g in groups],
        prereqs=sorted({c for g in groups for c in g}),
        offered_terms=list(terms),
    )


def _catalog(groups, courses, aliases=None) -> ProgramCatalog:
    return ProgramCatalog(
        program_id="p1", name="Computer Science, BS", catalog_year="2026-2027",
        groups=groups, courses=courses, aliases=aliases or {},
    )


def _profile(**overrides) -> StudentProfile:
    base = {
        "profile_label": "Test", "program_id": "p1", "completed_courses": [], "remaining_courses": [],
        "start_term": "fall", "start_year": 2026, "semesters_to_plan": 4,
        "max_credits_per_semester": 15, "major_subject": "CS",
        "max_major_courses_per_semester": 3, "preferred_major_courses_per_semester": 3,
    }
    return StudentProfile(**{**base, **overrides})


def _status_map(group):
    return [(slot.status, slot.filled_by or slot.label) for slot in group.slots]


# --- the three states -------------------------------------------------------------------------


def test_completed_planned_and_unfilled_are_told_apart():
    groups = [RequirementGroup(id="core", name="Core", selective=False, credits_min=9.0,
                               options=[["CS 18000"], ["CS 18200"], ["CS 25100"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4), _course("CS 18200"),
                                _course("CS 25100")])
    profile = _profile(completed_courses=["CS 18000"], remaining_courses=["CS 18200",
                                                                          "CS 25100"])

    result = requirement_progress(profile, catalog, [["CS 18200"], [], [], []])
    core = result.groups[0]

    assert _status_map(core) == [
        ("completed", "CS 18000"),
        ("planned", "CS 18200"),
        ("unfilled", "CS 25100"),
    ]
    assert core.satisfied is False
    assert result.groups_satisfied == 0
    assert result.credits_completed == 4
    assert result.credits_planned == 3


def test_a_planned_slot_carries_the_semester_it_sits_in():
    groups = [RequirementGroup(id="core", name="Core", selective=False, credits_min=3.0,
                               options=[["CS 18000"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4)])
    result = requirement_progress(_profile(remaining_courses=["CS 18000"]), catalog,
                                  [[], [], ["CS 18000"], []])
    assert result.groups[0].slots[0].semester_index == 2


def test_a_group_is_satisfied_only_when_nothing_in_it_is_unfilled():
    groups = [RequirementGroup(id="core", name="Core", selective=False, credits_min=6.0,
                               options=[["CS 18000"], ["CS 18200"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4), _course("CS 18200")])
    profile = _profile(completed_courses=["CS 18000"], remaining_courses=["CS 18200"])

    assert requirement_progress(profile, catalog, [[], [], [], []]).groups_satisfied == 0
    full = requirement_progress(profile, catalog, [["CS 18200"], [], [], []])
    assert full.groups_satisfied == 1
    assert full.groups[0].satisfied is True


# --- what counts as ONE requirement -----------------------------------------------------------


def test_alternatives_are_one_slot_not_two():
    """"MA 26100 or MA 27101" is one requirement; showing two would tell a student to take the
    regular AND the honours version of the same class."""
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=4.0,
                               options=[["MA 26100", "MA 27101"]])]
    catalog = _catalog(groups, [_course("MA 26100", 4), _course("MA 27101", 5)])

    result = requirement_progress(_profile(remaining_courses=["MA 26100"]), catalog,
                                  [[], [], [], []])
    slots = result.groups[0].slots
    assert len(slots) == 1
    assert slots[0].status == "unfilled"
    assert slots[0].label == "MA 26100 or MA 27101"
    assert {o.code for o in slots[0].options} == {"MA 26100", "MA 27101"}


def test_taking_either_alternative_fills_the_slot():
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=4.0,
                               options=[["MA 26100", "MA 27101"]])]
    catalog = _catalog(groups, [_course("MA 26100", 4), _course("MA 27101", 5)])
    result = requirement_progress(_profile(completed_courses=["MA 27101"]), catalog,
                                  [[], [], [], []])
    assert _status_map(result.groups[0]) == [("completed", "MA 27101")]
    assert result.groups[0].satisfied is True


def test_an_approved_substitute_fills_the_primary_it_stands_in_for():
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=5.0,
                               options=[["MA 16100"]])]
    catalog = _catalog(groups, [_course("MA 16100", 5), _course("MA 16500", 5)],
                       aliases={"MA 16500": "MA 16100"})
    result = requirement_progress(_profile(completed_courses=["MA 16500"]), catalog,
                                  [[], [], [], []])
    assert result.groups[0].satisfied is True


def test_a_selective_group_is_a_credit_target_with_ONE_hole_for_the_shortfall():
    """Not one hole per missing course: a 6-credit gap might be one course or two, and that is
    the student's choice to make."""
    groups = [RequirementGroup(id="sel", name="Selectives", selective=True, credits_min=6.0,
                               options=[["CS 40000"], ["CS 41000"], ["CS 42000"]])]
    catalog = _catalog(groups, [_course(c) for c in ("CS 40000", "CS 41000", "CS 42000")])

    result = requirement_progress(_profile(), catalog, [["CS 40000"], [], [], []])
    slots = result.groups[0].slots

    assert _status_map(slots and result.groups[0]) == [
        ("planned", "CS 40000"),
        ("unfilled", "3 more credits from this list"),
    ]
    assert result.groups[0].credits_filled == 3.0


def test_a_selective_group_is_satisfied_once_the_credits_are_there():
    groups = [RequirementGroup(id="sel", name="Selectives", selective=True, credits_min=6.0,
                               options=[["CS 40000"], ["CS 41000"], ["CS 42000"]])]
    catalog = _catalog(groups, [_course(c) for c in ("CS 40000", "CS 41000", "CS 42000")])
    result = requirement_progress(_profile(), catalog, [["CS 40000"], ["CS 41000"], [], []])
    assert result.groups[0].satisfied is True
    assert all(slot.status != "unfilled" for slot in result.groups[0].slots)


# --- the options offered for a hole -----------------------------------------------------------


def test_options_carry_only_the_semesters_the_course_would_legally_land_in():
    """An option offered here must not be rejectable on click, so the legality check is the same
    one the edit route runs."""
    groups = [
        RequirementGroup(id="core", name="Core", selective=False, credits_min=7.0,
                         options=[["CS 18000"], ["CS 18200"]]),
    ]
    catalog = _catalog(groups, [_course("CS 18000", 4),
                                _course("CS 18200", groups=[["CS 18000"]])])
    profile = _profile(remaining_courses=["CS 18000", "CS 18200"])

    # CS 18000 is in semester 1, so CS 18200 is legal from semester 2 onward and never before.
    result = requirement_progress(profile, catalog, [["CS 18000"], [], [], []])
    unfilled = [s for s in result.groups[0].slots if s.status == "unfilled"]
    assert len(unfilled) == 1
    option = unfilled[0].options[0]
    assert option.code == "CS 18200"
    assert option.legal_semesters == [1, 2, 3]


def test_an_option_with_nowhere_to_go_is_reported_with_an_empty_term_list():
    """Its prerequisite is not scheduled anywhere, so there is no legal slot. Saying so beats
    offering a term that would be rejected."""
    groups = [RequirementGroup(id="core", name="Core", selective=False, credits_min=3.0,
                               options=[["CS 25100"]])]
    catalog = _catalog(groups, [_course("CS 25100", groups=[["CS 18200"]]),
                                _course("CS 18200")])
    result = requirement_progress(_profile(remaining_courses=["CS 25100"]), catalog,
                                  [[], [], [], []])
    assert result.groups[0].slots[0].options[0].legal_semesters == []


def test_a_course_already_in_the_plan_is_never_offered_again():
    groups = [RequirementGroup(id="sel", name="Selectives", selective=True, credits_min=9.0,
                               options=[["CS 40000"], ["CS 41000"]])]
    catalog = _catalog(groups, [_course("CS 40000"), _course("CS 41000")])
    result = requirement_progress(_profile(), catalog, [["CS 40000"], [], [], []])
    unfilled = next(s for s in result.groups[0].slots if s.status == "unfilled")
    assert {o.code for o in unfilled.options} == {"CS 41000"}


def test_semester_labels_match_the_planning_calendar():
    labels = semester_labels(_profile(start_term="spring", start_year=2027))
    assert labels == ["Spring 2027", "Fall 2027", "Spring 2028", "Fall 2028"]


@pytest.mark.parametrize("plan", [[[], [], [], []], [["CS 18000"], [], [], []]])
def test_filled_slots_never_carry_options(plan):
    """A filled slot's alternatives are noise, and on a seventeen-option menu a lot of it."""
    groups = [RequirementGroup(id="core", name="Core", selective=False, credits_min=4.0,
                               options=[["CS 18000"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4)])
    result = requirement_progress(_profile(remaining_courses=["CS 18000"]), catalog, plan)
    for slot in result.groups[0].slots:
        if slot.status != "unfilled":
            assert slot.options == []


# --- requirements the catalog states but never enumerates --------------------------------------


def test_unresolved_requirements_are_reported_and_never_counted_as_satisfied():
    """The bug this fixes: Women's, Gender & Sexuality Studies showed TWO requirement groups —
    both from the major — and 100% coverage, while the Liberal Arts core, the university core,
    the world-language requirement and 20 credits of electives were absent from the screen.
    NOT DECIDABLE is a different fact from NOT MET, and both differ from "not shown at all".
    """
    from app.services.planner_db import UnresolvedRequirement

    groups = [RequirementGroup(id="core", name="Major Core", selective=False, credits_min=3.0,
                               options=[["CS 18000"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4)])
    catalog.unresolved = [
        UnresolvedRequirement(id="ucc", name="University Core Requirements",
                              requirement_type="core", credits_min=None, scope="university",
                              raw_text="For a complete listing, visit the Senate Website."),
        UnresolvedRequirement(id="lang", name="Core III: Linguistic Diversity",
                              requirement_type="major", credits_min=8.0, scope="program",
                              raw_text="Completion of 10100, 10200 in one world language."),
    ]

    result = requirement_progress(_profile(completed_courses=["CS 18000"]), catalog,
                                  [[], [], [], []])

    # The checkable group really is satisfied — and the fraction says CHECKABLE, not "all".
    assert result.groups_satisfied == 1
    assert result.groups_total == 1
    # The other two are shown, with the catalog's own words, and counted separately.
    assert [u.name for u in result.unresolved] == [
        "University Core Requirements", "Core III: Linguistic Diversity",
    ]
    assert result.unresolved[1].credits_min == 8.0
    assert "world language" in (result.unresolved[1].raw_text or "")
    # They are NOT groups: nothing can plan, check or satisfy them.
    assert all(g.id not in {"ucc", "lang"} for g in result.groups)
    # Scope carries through so the checklist can tell "every Purdue student has this" apart
    # from "specific to this major" instead of filing both under the same department.
    assert [u.scope for u in result.unresolved] == ["university", "program"]


def test_unresolved_requirements_carry_suggested_courses(monkeypatch):
    """`suggest_courses` is a semantic guess (see plan_suggestions.py) — this only checks that
    whatever it returns reaches the view and is passed the right exclusion set, not that the
    guess itself is any good."""
    from app.services.planner_db import UnresolvedRequirement
    from app.services.plan_suggestions import SuggestedCourse

    calls = []

    def fake_suggest(name, raw_text, *, exclude=frozenset(), limit=5):
        calls.append((name, exclude))
        return [SuggestedCourse(code="COM 11400", title="Fundamentals of Speech", credits=3)]

    monkeypatch.setattr(plan_requirements, "suggest_courses", fake_suggest)

    groups = [RequirementGroup(id="core", name="Major Core", selective=False, credits_min=3.0,
                               options=[["CS 18000"]])]
    # The suggested code must be a real catalog course, same as any other addable option —
    # `_options_for` (the same function real slots use) looks its title/credits up here.
    catalog = _catalog(groups, [_course("CS 18000", 4), _course("COM 11400", 3)])
    catalog.unresolved = [
        UnresolvedRequirement(id="ucc", name="University Core Requirements",
                              requirement_type="core", credits_min=None, scope="university",
                              raw_text="Visit the Senate Website."),
    ]

    result = requirement_progress(_profile(completed_courses=["CS 18000"]), catalog,
                                  [["CS 25100"], [], [], []])

    # Same shape (and the same "choose a term and add it" data) as a real slot's options —
    # a guess at WHICH requirement a course satisfies is not a guess at whether it can be
    # legally scheduled.
    suggested = result.unresolved[0].suggested_courses
    assert [o.code for o in suggested] == ["COM 11400"]
    assert suggested[0].title == "COM 11400 title"
    assert suggested[0].credits == 3
    # Completed AND planned courses both went into the exclusion set the suggester saw.
    assert calls == [("University Core Requirements", frozenset({"CS 18000", "CS 25100"}))]


def test_the_catalog_prose_does_not_repeat_the_requirement_title():
    from app.services.plan_requirements import _without_title

    assert _without_title("Core II: Social Diversity\nCulture and religion play a role.",
                          "Core II: Social Diversity") == "Culture and religion play a role."
    # Only an exact leading match is stripped; anything else is the catalog's words.
    assert _without_title("Something else entirely.", "Core II") == "Something else entirely."
    assert _without_title(None, "Core II") is None


# --- computed requirements (credit-threshold rules, not course lists) -------------------------
#
# "Upper Level Requirement" is real, unmodified Purdue catalog text — verified against the
# live crawled row for Computer Science: Machine Intelligence.
UPPER_LEVEL_TEXT = (
    "Resident study at Purdue University for at least two semesters and the enrollment in "
    "and completion of at least 32 semester hours of coursework required and approved for "
    "the completion of the degree. These courses are expected to be at least junior-level "
    "(30000+) courses.\nStudents should be able to fulfill most, if not all, of these "
    "credits within their major requirements; there should be a clear pathway for students "
    "to complete any credits not completed within their major."
)


def test_parses_the_real_upper_level_requirement_text():
    from app.services.plan_requirements import _parse_credit_threshold_rule

    assert _parse_credit_threshold_rule(UPPER_LEVEL_TEXT) == (32.0, 30000)


@pytest.mark.parametrize("text", [
    None, "", "Some unrelated requirement text with no numbers in this particular shape.",
    "At least 32 semester hours but no course-number floor stated anywhere here.",
    "Courses must be junior-level (30000+) but no credit-hour minimum stated.",
])
def test_unrelated_or_partial_text_is_not_mistaken_for_the_rule(text):
    from app.services.plan_requirements import _parse_credit_threshold_rule

    assert _parse_credit_threshold_rule(text) is None


@pytest.mark.parametrize(("code", "expected"), [
    ("STAT 35000", 35000), ("CS 18000", 18000), ("HEBR 12100", 12100), ("NOAG 101", 101),
])
def test_course_number_extraction(code, expected):
    from app.services.plan_requirements import _course_number

    assert _course_number(code) == expected


def test_a_credit_threshold_requirement_becomes_a_computed_check_not_an_unresolved_one():
    from app.services.planner_db import UnresolvedRequirement

    groups = [RequirementGroup(id="major", name="Machine Intelligence Core", selective=False,
                               credits_min=3.0, options=[["STAT 35000"]])]
    catalog = _catalog(groups, [_course("STAT 35000", 3), _course("CS 18000", 4)])
    catalog.unresolved = [
        UnresolvedRequirement(id="upper", name="Upper Level Requirement",
                              requirement_type="university", credits_min=None,
                              scope="university", raw_text=UPPER_LEVEL_TEXT),
    ]

    result = requirement_progress(
        _profile(completed_courses=["STAT 35000"]), catalog, [["CS 18000"], [], [], []],
    )

    # Moved out of `unresolved` entirely — this one CAN be checked.
    assert result.unresolved == []
    assert len(result.computed) == 1
    computed = result.computed[0]
    assert computed.name == "Upper Level Requirement"
    assert computed.credits_required == 32.0
    assert computed.credits_filled == 3.0  # only STAT 35000 clears the 30000 floor
    assert computed.satisfied is False
    assert [c.code for c in computed.contributing_courses] == ["STAT 35000"]
    assert computed.contributing_courses[0].status == "completed"


def test_a_course_already_counted_toward_a_major_group_also_counts_toward_the_computed_check():
    """The double-counting the catalog's own text asks for: "Students should be able to
    fulfill most, if not all, of these credits within their major requirements." A course
    satisfying `groups` is not excluded from `computed` — they are independent views over the
    same completed/planned set, not a shared, subtractive pool."""
    from app.services.planner_db import UnresolvedRequirement

    groups = [RequirementGroup(id="major", name="Machine Intelligence Core", selective=False,
                               credits_min=3.0, options=[["STAT 35000"]])]
    catalog = _catalog(groups, [_course("STAT 35000", 3)])
    catalog.unresolved = [
        UnresolvedRequirement(id="upper", name="Upper Level Requirement",
                              requirement_type="university", credits_min=None,
                              scope="university", raw_text=UPPER_LEVEL_TEXT),
    ]

    result = requirement_progress(
        _profile(completed_courses=["STAT 35000"]), catalog, [[], [], [], []],
    )

    # Satisfies the major group...
    assert result.groups[0].satisfied is True
    assert result.groups[0].credits_filled == 3.0
    # ...AND independently counts toward the computed check, not zero because "already used".
    assert result.computed[0].credits_filled == 3.0


def test_below_the_course_number_floor_does_not_count():
    from app.services.planner_db import UnresolvedRequirement

    groups = [RequirementGroup(id="major", name="Core", selective=False, credits_min=4.0,
                               options=[["CS 18000"]])]
    catalog = _catalog(groups, [_course("CS 18000", 4)])
    catalog.unresolved = [
        UnresolvedRequirement(id="upper", name="Upper Level Requirement",
                              requirement_type="university", credits_min=None,
                              scope="university", raw_text=UPPER_LEVEL_TEXT),
    ]

    result = requirement_progress(
        _profile(completed_courses=["CS 18000"]), catalog, [[], [], [], []],
    )

    assert result.computed[0].credits_filled == 0.0
    assert result.computed[0].contributing_courses == []


def test_an_unrelated_unresolved_requirement_is_unaffected_by_computed_detection():
    from app.services.planner_db import UnresolvedRequirement

    catalog = _catalog([], [])
    catalog.unresolved = [
        UnresolvedRequirement(id="lang", name="World Language Courses",
                              requirement_type="world_language", credits_min=None,
                              scope="university", raw_text="Choose one world language."),
    ]

    result = requirement_progress(_profile(), catalog, [[], [], [], []])

    assert result.computed == []
    assert [u.name for u in result.unresolved] == ["World Language Courses"]
