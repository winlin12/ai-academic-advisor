"""Tests for the program-catalog layer: how crawled requirement rows become a plannable degree.

The SQL is not exercised here (that needs a database); what is, is the interpretation — which
is where the crawled catalog is genuinely ambiguous and where getting it wrong puts a student
in both the regular and the honours version of the same class.
"""

import pytest

from app.models.schemas import Course, StudentProfile
from app.services.planner_db import (
    DEFAULT_OPTION_CREDITS,
    ProgramCatalog,
    RequirementGroup,
    _is_selective,
    detect_language_sequences,
    narrow_language_menu,
    normalize_course_code,
    requirement_coverage,
    select_remaining_courses,
)


def _catalog(groups, courses) -> ProgramCatalog:
    return ProgramCatalog(
        program_id="p1", name="Test Program", catalog_year="2026-2027",
        groups=groups, courses=courses,
    )


def _course(code, credits=3):
    return Course(code=code, title=f"{code} title", credits=credits)


def _credits_of(catalog):
    return catalog.credits


# --- selective detection ---------------------------------------------------------------------


def test_a_group_whose_options_exceed_its_credit_target_is_a_menu():
    """Purdue files "Required CS Major Math Courses (7-8 credits)" under `major` with four
    options worth 15 credits. It is a menu wearing the word "Required" in its title, and the
    arithmetic is the only signal that says so."""
    assert _is_selective("major", 7.0, [4.0, 5.0, 3.0, 3.0]) is True


def test_a_group_whose_options_exactly_meet_its_target_is_a_take_all():
    """"Required CS Major Core Courses (21 credits)" lists six courses totalling exactly 21."""
    assert _is_selective("major", 21.0, [4.0, 3.0, 3.0, 4.0, 3.0, 4.0]) is False


def test_an_elective_group_is_a_menu_whatever_its_arithmetic_says():
    assert _is_selective("elective", None, [3.0, 3.0, 3.0]) is True
    assert _is_selective("choose_credits", 3.0, [3.0]) is True


def test_a_group_with_no_credit_target_is_not_guessed_into_a_menu():
    assert _is_selective("foundation", None, [3.0, 3.0]) is False


# --- alternatives ("MA 26100 or MA 27101") ----------------------------------------------------


def test_alternatives_count_as_one_requirement_not_two():
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=7.0,
                               options=[["MA 26100", "MA 27101"], ["MA 26500", "MA 35100"]])]
    catalog = _catalog(groups, [_course("MA 26100", 4), _course("MA 27101", 5),
                                _course("MA 26500"), _course("MA 35100")])

    remaining = select_remaining_courses(groups, set(), _credits_of(catalog))
    assert remaining == ["MA 26100", "MA 26500"], "one from each run, never the honours twin too"


def test_having_taken_the_alternative_satisfies_the_requirement():
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=7.0,
                               options=[["MA 26100", "MA 27101"]])]
    catalog = _catalog(groups, [_course("MA 26100", 4), _course("MA 27101", 5)])

    assert select_remaining_courses(groups, {"MA 27101"}, _credits_of(catalog)) == []
    coverage, missing = requirement_coverage(catalog, {"MA 27101"})
    assert coverage == pytest.approx(1.0)
    assert missing == []


def test_a_take_all_group_reports_only_the_runs_that_are_short():
    groups = [RequirementGroup(id="math", name="Math", selective=False, credits_min=7.0,
                               options=[["MA 26100", "MA 27101"], ["MA 26500"]])]
    catalog = _catalog(groups, [_course("MA 26100", 4), _course("MA 27101", 5),
                                _course("MA 26500")])
    coverage, missing = requirement_coverage(catalog, {"MA 27101"})
    assert coverage == pytest.approx(0.0)
    assert missing == ["Math: missing MA 26500"]


# --- selection order and credit targets -------------------------------------------------------


def test_required_groups_come_before_every_selective_one():
    groups = [
        RequirementGroup(id="pick", name="Electives", selective=True, credits_min=3.0,
                         options=[["CS 40000"], ["CS 41000"]]),
        RequirementGroup(id="core", name="Core", selective=False, credits_min=6.0,
                         options=[["CS 18000"], ["CS 18200"]]),
    ]
    catalog = _catalog(groups, [_course(c) for c in
                                ("CS 40000", "CS 41000", "CS 18000", "CS 18200")])
    assert select_remaining_courses(groups, set(), _credits_of(catalog)) == [
        "CS 18000", "CS 18200", "CS 40000",
    ]


def test_a_selective_group_takes_just_enough_to_reach_its_credit_target():
    groups = [RequirementGroup(id="pick", name="Electives", selective=True, credits_min=6.0,
                               options=[["CS 40000"], ["CS 41000"], ["CS 42000"]])]
    catalog = _catalog(groups, [_course(c) for c in ("CS 40000", "CS 41000", "CS 42000")])
    assert select_remaining_courses(groups, set(), _credits_of(catalog)) == [
        "CS 40000", "CS 41000",
    ]


def test_completed_courses_count_toward_a_selective_groups_target_first():
    groups = [RequirementGroup(id="pick", name="Electives", selective=True, credits_min=6.0,
                               options=[["CS 40000"], ["CS 41000"], ["CS 42000"]])]
    catalog = _catalog(groups, [_course(c) for c in ("CS 40000", "CS 41000", "CS 42000")])
    assert select_remaining_courses(groups, {"CS 40000"}, _credits_of(catalog)) == ["CS 41000"]


def test_a_selective_group_with_no_target_degrades_to_choose_one():
    groups = [RequirementGroup(id="pick", name="Electives", selective=True, credits_min=None,
                               options=[["CS 40000"], ["CS 41000"]])]
    catalog = _catalog(groups, [_course("CS 40000"), _course("CS 41000")])
    assert select_remaining_courses(groups, set(), _credits_of(catalog)) == ["CS 40000"]


def test_an_option_with_no_credits_on_file_is_assumed_standard_size():
    groups = [RequirementGroup(id="pick", name="Electives", selective=True, credits_min=3.0,
                               options=[["UNKNOWN 10000"]])]
    catalog = _catalog(groups, [])          # the code resolves to no course row at all
    assert DEFAULT_OPTION_CREDITS == 3.0
    assert select_remaining_courses(groups, set(), _credits_of(catalog)) == ["UNKNOWN 10000"]


# --- derived facts ----------------------------------------------------------------------------


def test_the_major_subject_is_derived_from_the_programs_own_required_courses():
    """It is a fact about the program, not about the student — and a profile naming the wrong
    one silently disables the per-semester major-course cap."""
    groups = [
        RequirementGroup(id="core", name="Core", selective=False, credits_min=9.0,
                         options=[["ME 20000"], ["ME 27000"], ["MA 16100"]]),
        # A selective group full of CS must not outvote the required ME core.
        RequirementGroup(id="pick", name="Electives", selective=True, credits_min=3.0,
                         options=[["CS 40000"], ["CS 41000"], ["CS 42000"]]),
    ]
    catalog = _catalog(groups, [])
    assert catalog.major_subject() == "ME"


def test_alias_chains_collapse_onto_one_primary():
    catalog = ProgramCatalog(
        program_id="p1", name="P", catalog_year="2026-2027",
        courses=[_course("MA 16100"), _course("MA 16500"), _course("MA 16900")],
        aliases={"MA 16500": "MA 16100", "MA 16900": "MA 16500"},
    )
    assert catalog.canonical["MA 16900"] == "MA 16100"
    assert catalog.canonical["MA 16100"] == "MA 16100"


def test_a_cyclic_alias_pair_cannot_hang_the_resolver():
    catalog = ProgramCatalog(
        program_id="p1", name="P", catalog_year="2026-2027",
        courses=[_course("A 10000"), _course("B 10000")],
        aliases={"A 10000": "B 10000", "B 10000": "A 10000"},
    )
    assert set(catalog.canonical) == {"A 10000", "B 10000"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("cs18000", "CS 18000"), ("CS  18000", "CS 18000"), ("CS-18000", "CS 18000"),
     ("ENGL", "ENGL"), ("18000", "18000")],
)
def test_normalize_course_code(raw, expected):
    assert normalize_course_code(raw) == expected


# --- language-sequence detection and narrowing -------------------------------------------------
#
# Titles below are the REAL titles Purdue's own crawled catalog uses (verified against the
# live "Language and Culture Electives/Foreign Language Requirement" group for Computer
# Science: Machine Intelligence) — not invented for the test. ASL's titles are deliberately
# included as-is ("American Sign Language I", no "Level" word) because that's the one real
# gap in the current detector: see its assertion below.


def _lang_course(code, title):
    return Course(code=code, title=title, credits=3)


def _real_language_group() -> tuple[RequirementGroup, dict[str, Course]]:
    rows = [
        ("SPAN 10100", "Spanish Level I"), ("SPAN 10200", "Spanish Level II"),
        ("SPAN 20100", "Spanish Level III"),
        ("FR 10100", "French Level I"), ("FR 10200", "French Level II"),
        ("FR 20100", "French Level III"),
        ("ARAB 10100", "Standard Arabic Level I"), ("ARAB 10200", "Standard Arabic Level II"),
        ("ARAB 20100", "Standard Arabic Level III"),
        ("HEBR 10100", "Modern Hebrew Level I"), ("HEBR 12100", "Biblical Hebrew Level I"),
        ("HEBR 12200", "Biblical Hebrew Level II"), ("HEBR 20100", "Modern Hebrew Level III"),
        ("ASL 10100", "American Sign Language I"), ("ASL 10200", "American Sign Language II"),
        ("ASL 20100", "American Sign Language III"),
    ]
    by_code = {code: _lang_course(code, title) for code, title in rows}
    group = RequirementGroup(
        id="lang", name="Language and Culture Electives/Foreign Language Requirement",
        selective=True, credits_min=3.0, options=[[code] for code, _ in rows],
    )
    return group, by_code


def test_detects_one_sequence_per_subject_code_sorted_by_level():
    group, by_code = _real_language_group()
    sequences = detect_language_sequences(group, by_code)

    assert set(sequences) == {"SPAN", "FR", "ARAB", "HEBR"}, (
        "ASL is a known gap: its real titles never say the word 'Level', only a bare Roman "
        "numeral ('American Sign Language I'), which the detector deliberately does not match "
        "on its own — a bare trailing Roman numeral is too easy to false-positive on an "
        "unrelated title ('World War II')."
    )
    assert [o.level for o in sequences["SPAN"]] == [1, 2, 3]
    assert [o.code for o in sequences["SPAN"]] == ["SPAN 10100", "SPAN 10200", "SPAN 20100"]


def test_hebrews_two_tracks_share_one_subject_code_by_design():
    """Modern and Biblical Hebrew are not interchangeable, but they share the HEBR subject
    code, and `language_of_interest` speaks in subject codes — so they merge into one
    sequence here. A known, accepted limitation (see `detect_language_sequences`'s
    docstring), not a bug this test should hide."""
    group, by_code = _real_language_group()
    sequences = detect_language_sequences(group, by_code)
    assert len(sequences["HEBR"]) == 4


def test_a_non_language_group_is_never_detected():
    group = RequirementGroup(
        id="core", name="Required CS Major Core Courses (21 credits)", selective=False,
        credits_min=21.0, options=[["CS 18000"], ["CS 18200"]],
    )
    by_code = {"CS 18000": _course("CS 18000", 4), "CS 18200": _course("CS 18200", 3)}
    assert detect_language_sequences(group, by_code) == {}


def test_no_language_chosen_narrows_to_one_entry_level_course_per_language():
    group, by_code = _real_language_group()
    narrowed = narrow_language_menu(group, by_code, StudentProfile())
    assert sorted(narrowed.courses) == ["ARAB 10100", "FR 10100", "HEBR 10100", "SPAN 10100"]
    assert narrowed.recommended_code is None


def test_choosing_a_language_shows_its_full_sequence_and_recommends_the_students_level():
    group, by_code = _real_language_group()
    profile = StudentProfile(language_of_interest="span", language_proficiency=2)
    narrowed = narrow_language_menu(group, by_code, profile)
    assert narrowed.courses == ["SPAN 10100", "SPAN 10200", "SPAN 20100"]
    assert narrowed.recommended_code == "SPAN 10200"


def test_a_level_the_sequence_does_not_reach_recommends_the_highest_available():
    group, by_code = _real_language_group()
    profile = StudentProfile(language_of_interest="SPAN", language_proficiency=3)
    narrowed = narrow_language_menu(group, by_code, profile)
    assert narrowed.recommended_code == "SPAN 20100"


def test_narrow_language_menu_is_a_no_op_on_a_non_language_group():
    group = RequirementGroup(
        id="core", name="Required CS Major Core Courses (21 credits)", selective=False,
        credits_min=21.0, options=[["CS 18000"]],
    )
    by_code = {"CS 18000": _course("CS 18000", 4)}
    profile = StudentProfile(language_of_interest="SPAN")
    assert narrow_language_menu(group, by_code, profile) is group
