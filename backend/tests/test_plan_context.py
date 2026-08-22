"""Tests for the catalog export Mode B reads — the model's entire view of the world.

Everything the model gets right or wrong about term offerings, prerequisites and alternatives
comes from this document, so what is asserted here is that each of those facts is (a) present
and (b) findable, which are different properties. The eval's offering note exists precisely
because a fact stated once on one row of fifty-six is present but not findable.
"""

from app.models.schemas import Course
from app.services.plan_context import (
    approx_tokens,
    offering_note,
    render_program_context,
)
from app.services.planner_db import ProgramCatalog, ProseRule, RequirementGroup


def _course(code, credits=3, terms=("fall", "spring", "summer"), groups=()):
    return Course(
        code=code, title=f"{code} title", credits=credits, offered_terms=list(terms),
        prereq_groups=[list(g) for g in groups],
        prereqs=sorted({c for g in groups for c in g}),
    )


def _catalog(**overrides) -> ProgramCatalog:
    base = {
        "program_id": "p1",
        "name": "Computer Science, BS",
        "catalog_year": "2026-2027",
        "groups": [
            RequirementGroup(id="core", name="CS Core", selective=False, credits_min=7.0,
                             options=[["CS 18000"], ["CS 18200"]]),
            RequirementGroup(id="math", name="Math", selective=False, credits_min=4.0,
                             options=[["MA 26100", "MA 27101"]]),
        ],
        "courses": [
            _course("CS 18000", credits=4),
            _course("CS 18200", groups=[["CS 18000"]]),
            _course("CS 43900", terms=("fall",)),
            _course("MA 26100", credits=4),
            _course("MA 27101", credits=5),
        ],
        "prereq_rows": {
            "CS 18200": {"raw_text": "Prerequisite: CS 18000 with a minimum grade of C.",
                         "prereq_groups": [["CS 18000"]], "coreq_codes": [],
                         "confidence": "high"},
        },
        "offering_rows": {"CS 43900": {"terms_observed": 3}},
    }
    return ProgramCatalog(**{**base, **overrides})


def test_the_term_restricted_courses_are_named_above_the_rows():
    """A constraint stated once as one field among fifty-six near-identical rows is present but
    not findable — which is how term-offering violations kept firing on the same two courses."""
    export = render_program_context(_catalog())
    assert "NOT EVERY COURSE RUNS EVERY TERM" in export
    assert "FALL ONLY — never spring/summer: CS 43900" in export
    assert "Every course not named above runs in all three terms." in export


def test_the_offering_note_is_derived_from_the_rows_it_sits_above():
    """Hand-written offering notes are how the eval's fixture came to record a fall-only course
    as spring-only. This one cannot disagree with its data because it is read off it."""
    note = "\n".join(offering_note([
        {"course_code": "CS 10000", "offered_terms": ["fall", "spring", "summer"]},
        {"course_code": "CS 43900", "offered_terms": ["fall"]},
        {"course_code": "CS 47300", "offered_terms": ["fall", "spring"]},
        {"course_code": "CS 99999", "offered_terms": []},
    ]))
    assert "NEVER OFFERED, do not schedule at all: CS 99999" in note
    assert "FALL ONLY — never spring/summer: CS 43900" in note
    assert "fall/spring only, never summer: CS 47300" in note
    assert "CS 10000" not in note, "unrestricted courses are the default, not a listed case"


def test_no_note_at_all_when_nothing_is_restricted():
    assert offering_note([
        {"course_code": "CS 10000", "offered_terms": ["fall", "spring", "summer"]},
    ]) == []


def test_alternatives_are_one_option_row_carrying_its_substitutes():
    """"MA 26100 or MA 27101" printed as two rows reads as two things to schedule."""
    export = render_program_context(_catalog())
    assert '"alternatives":["MA 27101"]' in export
    assert '"course_code":"MA 26100"' in export


def test_requirement_types_use_the_vocabulary_the_rules_prose_defines():
    """The model is told what `all_of` and `choose_credits` mean and nothing about the
    crawler's own `foundation`/`college`/`major` spellings."""
    catalog = _catalog(groups=[
        RequirementGroup(id="pick", name="Selectives", selective=True, credits_min=6.0,
                         options=[["CS 43900"]]),
    ])
    export = render_program_context(catalog)
    assert '"requirement_type":"choose_credits"' in export
    assert "foundation" not in export


def test_prerequisites_carry_their_raw_text_confidence_and_chain_depth():
    export = render_program_context(_catalog())
    assert '"prereq_groups":[["CS 18000"]]' in export
    assert '"confidence":"high"' in export
    # chain_depth is computed, and the prose tells the model it is not a year.
    assert '"chain_depth":1' in export
    assert "It is NOT a year, a class standing, or a semester number" in export


def test_courses_are_listed_in_prerequisite_order():
    export = render_program_context(_catalog())
    courses_block = export.split("## courses")[1]
    assert courses_block.index('"CS 18000"') < courses_block.index('"CS 18200"')


def test_the_export_says_when_it_trimmed_a_menu():
    """A model that believes a trimmed menu is the whole menu confidently reports a requirement
    as uncoverable."""
    big = _catalog(groups=[
        RequirementGroup(id="pick", name="Selectives", selective=True, credits_min=6.0,
                         options=[[f"CS {40000 + i}"] for i in range(200)]),
    ])
    export = render_program_context(big, budget_tokens=1_500)
    assert "were omitted to fit the context window" in export
    # The hard constraints are never what gets trimmed.
    assert "## course_prerequisites" in export
    assert "## course_planner_terms" in export


def test_prose_only_requirements_never_reach_the_export():
    """The catalog states university core, world language and "choose 1 course in 6
    disciplines" in prose, with no course list. They used to be exported to the model as
    `unresolved_requirement_groups`, described as real requirements it could do nothing about;
    removed 2026-08-12 (see `planner_db.ProseRule`) so the prompt is only things the model can
    actually schedule. `ProgramCatalog.prose_rules` has ONE reader and it is not this one."""
    catalog = _catalog()
    catalog.prose_rules = [
        ProseRule(id="u1", name="University Core Requirements", credits_min=None,
                  raw_text="Choose 1 course in 6 different disciplines."),
    ]
    export = render_program_context(catalog)
    assert "unresolved" not in export
    assert "University Core Requirements" not in export
    assert "6 different disciplines" not in export


def test_the_export_is_deterministic_for_a_given_database_state():
    assert render_program_context(_catalog()) == render_program_context(_catalog())


def test_token_estimate_is_conservative_for_dense_json():
    """4 chars/token is a prose rule of thumb. This document is JSON, measured at ~2.2, and the
    optimistic estimate silently overflowed the context window on every plan."""
    row = '{"course_code":"CS 25100","credit_hours_min":3,"title":"Data Structures"}'
    assert approx_tokens(row) > len(row) // 4
