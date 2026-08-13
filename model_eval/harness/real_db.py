"""Loads a `CatalogDatabase` (see `catalog_export.py`) straight from the real
`catalog_ingestion`/`advisor` Postgres DB — no JSON files, no fixture, no mock.

KEPT STANDALONE ON PURPOSE (`run.py`'s own docstring: "nothing from the app"). This does not
import `backend/app/services/planner_db.py` or `plan_context.py`, even though both do almost
exactly this job for the live app — see `plan_context.py`'s own docstring, "keeping the two
byte-comparable is the whole point". Reusing the app's code here would collapse that
independence: the eval is supposed to be a check ON the app, not a wrapper around it. So the
handful of queries below are a separate, hand-written read of the same tables. Accept the
duplication; it is what keeps this a real second opinion.

THE COURSE UNIVERSE IS BROADER THAN "WHAT A REQUIREMENT NAMES". A first version only pulled
courses that appear as a `requirement_options` row for THIS program — for the real,
crawled "Computer Science: Machine Intelligence" program, that was 37 courses for an entire
CS degree, because most of its own department's electives are simply real courses that no
specific requirement row happens to name (Purdue's crawled catalog states "Selectives (6
credits)" with a sample list, not an exhaustive one — see planner_db.py's own
`_is_selective`/menu-vs-list note). A student's actual elective menu is every course in the
department, and a model choosing from 37 options is being tested on a narrower, easier problem
than the one it will face in production. `broaden_subjects` pulls in every undergraduate course
under the given subject prefixes regardless of whether any `requirement_options` row names it,
tagged in the rendered export as electives an unlisted option can still satisfy — see
`catalog_export.render_context`'s "not named in any requirement_options row" column note.

`broaden_subjects=None` (the default) AUTO-DETECTS the one subject prefix required most often
by the program's OWN `requirement_options` rows, rather than a fixed subject applied to every
program. This is not cosmetic: a hard-coded `("CS",)` default, run against a program with
nothing to do with computer science (Women's, Gender & Sexuality Studies, say), pulled in ~140
irrelevant CS electives and — because some of those CS courses have prerequisites in MA/STAT
that the WGSS program never otherwise needed — produced `course_prerequisites` rows pointing at
courses that were never in the export at all. `--major` in `run.py` makes it trivial to point
this loader at an arbitrary program, so a subject guess that is only right for one hard-coded
program stops being safe the moment it is used for its actual purpose. See the closure step
below for the other half of that same fix.

BUDGETED, because broadening can genuinely overflow the context: Purdue's catalog has ~140 CS
course codes across every level ever offered. `budget_tokens` caps the broadened ADDITIONS
only — every course a requirement actually names is never trimmed, since dropping one would
silently make a real requirement unsatisfiable. Undergraduate-numbered (< 50000), lowest
course number first, is the priority order for what survives the cut: intro-adjacent electives
are more likely to matter to an undergraduate's plan than a 400-level elective four years out.

EVERY PREREQUISITE/COREQUISITE REFERENCE RESOLVES WITHIN THE EXPORT. A course pulled in by
broadening can have a prerequisite that nothing else in the program happens to need — the WGSS
incident above, or more mundanely, a CS elective whose prereq is a specific math course this
program's own math requirement doesn't name. Left alone, that produces a `course_prerequisites`
row naming a course with no `courses` row, which `catalog_export.integrity_problems` correctly
flags and which a real deployment would never produce (a live app resolves prerequisites
against the whole catalog, not a per-request export). The fix is a small bounded closure after
prerequisites are fetched: anything referenced as a prereq or coreq that is not already in the
export gets pulled in too, permanently (not subject to the elective trim — dropping a course's
own prerequisite while keeping the course would be worse than not broadening at all).

WHAT ELSE DOES NOT SURVIVE THE TRIP:

  * `requirement_options` here is FLAT, exactly like `catalog_export`'s shape always was —
    "MA 26100 or MA 27101" ends up as two separate rows, not one option with alternatives.
    `plan_context.py` added an `alternatives` field for the live app; this loader does not.

  * Requirement groups that carry no course options AND no children — a credit target or a
    type like "university"/"core"/"world_language" stated in prose instead of as a course list
    — have no rows to put in `requirement_groups`/`requirement_options`, and this loader now
    DROPS them entirely. In the real catalog they are the MAJORITY (see
    `app/services/planner_db.py`'s `UnresolvedRequirement` docstring: 749 of 928 crawled
    programs have at least one). They used to be read into a model-visible
    `unresolved_requirement_groups` table; that table came out 2026-08-12 (see
    `catalog_export.TABLES` for why) because showing a model a requirement and then telling it
    there is nothing to do about it was costing attention and buying nothing scorable. The one
    prose requirement that DOES get shown is shown as coursework instead: University Core is
    resolved into the synthesized per-competency `ucc-*` groups below, off each course's own
    UCC attributes. Everything else waits for a representation that can be planned against.

  * `courses.attributes` is empty unless `attributes_raw` happens to hold a JSON tag list —
    true only for the handful of courses a prior eval fixture seeded; real crawled courses
    carry none, because Acalog does not publish course-level attribute tags at all.

TWO-PHASE SINCE 2026-08-06: a real program's own open elective menus (gen-ed "Culture Course"
lists, department breadth requirements) can be far bigger than any CS-broadening ever was — the
Machine Intelligence program's own requirement_options name 994 distinct courses before a
single broadened CS elective is added, which renders to ~175k tokens against a 32k context
window on its own. `fetch_real_db_base` does every Postgres round trip ONCE, for the maximal
universe any scenario could need (every course any menu names, prerequisite-closed). Everything
that used to be a single `load_real_db` call is now that fetch, plus a second, pure-Python,
per-scenario pass (`build_scenario_database`) that ranks and budgets each open elective menu by
that scenario's own `gen_ed_preference`/`world_language` — see both functions' docstrings.
`load_real_db` remains as a one-shot wrapper for callers that don't need per-scenario behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .catalog_export import TABLES, CatalogDatabase, chain_depths, render_context
from .fixtures import Fixture, synthesize_scenarios
# `normalize_code` only — so a hand-written `manual_course_aliases` key is matched against the
# crawl the same way a model's own course code is ("ma16500", "MA-16500" -> "MA 16500"), rather
# than silently never matching over a spacing difference. plan_scorers imports fixtures/planner
# and never real_db, so this direction adds no cycle.
from .plan_scorers import normalize_code
from .planner import ALL_TERMS, Course

logger = logging.getLogger(__name__)

# Same two-signal rule `app/services/planner_db._is_selective` uses, reproduced rather than
# imported (see the module docstring on why). A group whose options total MORE credits than it
# requires is a menu even when the crawler filed it under a "required"-sounding type.
_SELECTIVE_TYPES = {"choose_credits", "choose", "selective", "elective", "free_elective"}
_DEFAULT_OPTION_CREDITS = 3.0

_UNIVERSITY_REQUIREMENT_TYPES = {"university", "core", "world_language"}
_COLLEGE_REQUIREMENT_TYPES = {"college"}

# Purdue's "First-Year Engineering" program is the College of Engineering-wide gate every
# engineering major clears before/while declaring — see `is_college_of_engineering` and
# `_fye_course_codes`, added 2026-08-07. There is no FK from a major's own program row to this
# one (verified: `programs.college_id`/`department_id` are NULL for every row in this crawl, and
# `colleges`/`departments` are both empty tables), so it is looked up by this fixed id.
FYE_PROGRAM_ID = "bebe0ff3-d003-4d43-8674-3b39cc11407d"

# NO STRUCTURED "COLLEGE" DATA EXISTS IN THIS CRAWL to check against — see `FYE_PROGRAM_ID`'s
# note. This is a NAME heuristic, not a database fact: verified by hand against Purdue's actual
# College of Engineering school list, biased conservative (only matches a known CoE school
# name), and explicitly excludes two patterns that contain "Engineering" but are NOT this
# college: "Engineering Technology" (Purdue Polytechnic Institute) and anything named "Computer
# Science" (College of Science — including its "Computer Science: Software Engineering"
# concentration). Re-verify by hand before trusting this for a program not already checked.
_COLLEGE_OF_ENGINEERING_SCHOOLS = (
    "First-Year Engineering",
    "Aeronautical and Astronautical Engineering",
    "Aeronautics and Astronautics",
    "Agricultural and Biological Engineering",
    "Biomedical Engineering",
    "Chemical Engineering",
    "Civil Engineering",
    "Computer Engineering",
    "Construction Engineering",
    "Electrical Engineering",
    "Electrical and Computer Engineering",
    "Engineering Education",
    "Environmental and Ecological Engineering",
    "Environmental and Natural Resources Engineering",
    "Industrial Engineering",
    "Materials Engineering",
    "Materials Science and Engineering",
    "Mechanical Engineering",
    "Multidisciplinary Engineering",
    "Nuclear Engineering",
    "Interdisciplinary Engineering",
)
_NOT_COLLEGE_OF_ENGINEERING_MARKERS = ("Engineering Technology", "Computer Science")


def is_college_of_engineering(program_name: str) -> bool:
    """Best-effort, name-only classification of whether `program_name` is a College of
    Engineering program — see the constants above for why nothing more authoritative exists in
    this crawl. Purdue's "First-Year Engineering must finish in year one" rule applies to every
    College of Engineering major and NO ONE ELSE; getting this wrong in either direction either
    drops a real constraint or invents one for a student it does not apply to, so callers must
    gate the FYE deadline note/check on this and nothing looser.
    """
    name = program_name or ""
    if any(marker in name for marker in _NOT_COLLEGE_OF_ENGINEERING_MARKERS):
        return False
    return any(school in name for school in _COLLEGE_OF_ENGINEERING_SCHOOLS)


def _fye_course_codes(cur: psycopg.Cursor) -> frozenset[str]:
    """Every course that satisfies one of Purdue's 8 numbered First-Year Engineering
    requirements, read from the FYE program's own DB rows (see `FYE_PROGRAM_ID`) — the exact
    same 8 requirements, verified by hand against `catalog.purdue.edu`, that gate T2M into every
    College of Engineering major. `ENGL 10600`/`COM 11400` are added by hand alongside the DB's
    own "Requirement #8" options (`SCLA 11000`/`SCLA 11100`): the crawl only resolves the
    SCLA-sequence alternative to course codes, not the generic Written/Oral Communication UCC
    categories that are this requirement's more commonly taken path — the same category gap
    `plan_fixtures/*.yaml`'s own gen-ed groups already work around for the same reason.
    """
    cur.execute(
        """
        SELECT ro.course_code_raw
        FROM requirement_groups rg
        JOIN requirement_options ro ON ro.requirement_group_id = rg.id
        WHERE rg.program_id = %s AND rg.name LIKE 'Requirement #%%'
        """,
        (FYE_PROGRAM_ID,),
    )
    codes = {row["course_code_raw"] for row in cur.fetchall()}
    codes |= {"ENGL 10600", "COM 11400"}
    return frozenset(codes)


def _scope_for(requirement_type: str) -> str:
    """Mirrors `app/services/planner_db._scope_for` (kept standalone — see module docstring)."""
    if requirement_type in _UNIVERSITY_REQUIREMENT_TYPES:
        return "university"
    if requirement_type in _COLLEGE_REQUIREMENT_TYPES:
        return "college"
    return "program"


# SCOPE TIERS, for `fetch_real_db_base`'s `scope_filter`. A degree is nested — a major student
# owes the major AND the college core AND the university core — so "program" alone is never the
# right answer for a real major, and `scope_filter=None` (every tier, the default everywhere in
# the harness today) is. Kept as a named capability for a caller that deliberately wants one
# tier: `run.py`'s old per-fixture `requirement_scope` setting is gone with the fixture files,
# and every real-major fixture set it to "major", i.e. exactly this default.
#
# It used to map each fixture scope to exactly ONE `_scope_for` tier and drop every group not in
# it, so `requirement_scope: major` discarded all 12 of CS's college/university groups. The
# effect was not that those requirements scored badly — they were not shown to the model or
# scored at all, while the course MENUS that satisfy them survived, because those menus carry
# `requirement_type = NULL` and `_scope_for` calls anything untyped "program". The model was
# handed 619 culture courses and 103 history courses and never told it owed a world language or
# a University Core.
SCOPE_TIERS = {
    "major": frozenset({"program", "college", "university"}),
    "college": frozenset({"college", "university"}),
    "university": frozenset({"university"}),
}

# Purdue's University Core Curriculum competencies, as they appear in the catalog's own option
# text: "MA 16100 - Plane Analytic Geometry And Calculus I (UCC: QR) Credit Hours: 5.00".
#
# THE CORE IS ENTIRELY RECOVERABLE FROM THIS, and from nothing else in the crawl. Every
# `requirement_groups` row that *states* the core ("University Core Requirements", type `core`)
# is prose-only — 0 options, 0 children, 463 characters of raw_text — so the requirement is
# unscorable from the group tree. But 40,007 `requirement_options` rows carry a UCC tag inline
# (HUM 24289, BSS 6376, SCI 3001, STS 1900, QR 1767, IL 1243, AI 615, OC 563, WC 253 — measured
# 2026-08-11), which is exactly the nine competencies the prose names. Parsing them onto the
# course gives the model, and the scorer, a checkable core requirement without any crawl change.
_UCC_RE = re.compile(r"UCC:\s*([A-Z]{2,3})\b")

# Competency code -> the name the catalog gives it, for the rendered requirement.
UCC_COMPETENCIES = {
    "WC": "Written Communication",
    "OC": "Oral Communication",
    "IL": "Information Literacy",
    "QR": "Quantitative Reasoning",
    "SCI": "Science",
    "STS": "Science, Technology & Society",
    "HUM": "Human Cultures: Humanities",
    "BSS": "Human Cultures: Behavioral/Social Science",
    "AI": "AI Working Competency",
}

# MEASURED against this export's own JSON (dense, punctuation- and code-heavy) rather than
# assumed from the 4:1 prose rule of thumb — see `app/services/plan_context.py`'s identical
# constant and the incident that made it explicit: an under-count here does not fail loudly,
# it silently truncates every generated plan mid-JSON.
_CHARS_PER_TOKEN = 2.2

# Broadened electives only ever pull undergraduate-level coursework. A bachelor's degree plan
# has no business drawing on a 500-level graduate seminar as a "here's another option" filler.
_MAX_UNDERGRAD_NUMBER = 50000

_COURSE_NUMBER_RE = re.compile(r"(\d+)$")

# Subject prefixes this catalog's language departments crawl under — used to recognize a
# requirement group as language-relevant FROM ITS OPTION CODES, not its crawled name text, so
# detection survives catalog wording changes and works for any program, not only the one this
# was built against.
_LANGUAGE_SUBJECTS = {
    "SPAN", "FR", "GER", "ITAL", "CHNS", "RUSS", "JPNS", "KOR", "ARAB", "PTGS",
    "LATN", "HEBR", "ASL",
}

# THE ONE PLACE THIS HARNESS EVER STOPPED A COURSE COUNTING TWICE. `Curriculum Note`'s crawled
# raw_text states a genuine exclusivity rule — "Courses taken to meet the Foreign Language and
# Culture requirement may not also be used to meet the General Education or Great Issues
# requirements" — and the Gen-Ed merge below enforced it by withholding every language-subject
# course from the pooled Gen-Ed group (see that call site for the incident that prompted it).
#
# OFF SINCE 2026-08-12, by decision: the harness now counts EVERY instance a course is used,
# with no exception, so a French course fills the language requirement and the Gen-Ed pool at
# the same time. Real exclusivity rules like this one exist and this is knowingly not modelling
# them; a plan is scored generously here rather than failed for an overlap the scorer cannot
# see the whole of. Flip to True to restore the crawled rule — nothing else has to change.
_ENFORCE_LANGUAGE_GENED_EXCLUSION = False

# The crawled group whose raw prose (`College of Science Core Requirements`) states the real
# world-language rule: "Language and Culture (1-9 credits) Complete ONE of the Options from
# this list." Named exactly here rather than re-detected, same as `force_selective_groups` in
# config.yaml — this one specific group name is tied directly to the two-path modeling in
# `build_scenario_database._language_path_options`, not a generic heuristic.
_WORLD_LANGUAGE_GROUP_NAME = "Language and Culture Electives/Foreign Language Requirement"

# Hand-picked, well-known Purdue gen-eds — the "whatever works" / "most popular" / no-preference
# default. There is no enrollment data in either crawled database (Acalog does not publish seat
# counts, and PurdueIO's advisor schema was never asked to track them), so unlike every other
# ranking below, this list cannot be measured off the data; it is a deliberate editorial choice.
# A program whose menus don't happen to name any of these just falls through to the
# (subject, course number) filler ranking — never an error, never an empty menu.
_DEFAULT_GEN_ED_CODES = (
    "ENGL 10600", "COM 11400", "PHIL 11000", "POL 10100", "PSY 12000", "SOC 10000",
    "HIST 10300", "ECON 21000", "ANTH 10000", "COM 22400",
)

# Free-text theme -> substrings matched against a course's title (lowercased). Deliberately a
# small, literal table rather than an embedding/LLM match: this decides what goes INTO the
# static prompt, so it has to be exactly reproducible the same way everything else that lands in
# the static hash is. A preference that matches nothing here falls back to `_rank_by_default` —
# never an error, never an empty menu.
_GEN_ED_THEMES: dict[str, tuple[str, ...]] = {
    "leadership": ("leadership", "team", "collaborat", "management", "project", "organiz"),
    "writing": ("writ", "composition", "rhetoric", "literary", "literature"),
    "art": ("art", "design", "music", "theatre", "dance", "film"),
    "history": ("history", "civilization"),
    "science": ("science", "biology", "chemistry", "physics", "geology", "astro"),
    "business": ("business", "account", "market", "entrepreneur", "econom", "finance"),
    "psychology": ("psycholog", "behavior", "mind", "cognit"),
    "culture": ("culture", "cultural", "society", "global", "world"),
    "language": ("language", "linguistic"),
}

# Preference strings that mean "no real preference" — resolved to the same hardcoded default
# list an absent preference gets. "Most popular"/"popular" live here too: there is no enrollment
# data to rank by (see `_DEFAULT_GEN_ED_CODES`), so it is a synonym for the default, not a
# distinct ranking.
_NO_PREFERENCE_PHRASES = {
    "", "whatever works", "whatever", "i don't care", "i dont care", "no preference",
    "none", "any", "anything", "doesn't matter", "doesnt matter", "idc", "most popular",
    "popular", "no preference given",
}


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _is_selective(requirement_type: str, credits_min: float | None,
                  option_credits: list[float]) -> bool:
    if (requirement_type or "") in _SELECTIVE_TYPES:
        return True
    return bool(credits_min) and sum(option_credits) > float(credits_min) + 0.01


# --- prerequisites ------------------------------------------------------------------------------
#
# TWO SOURCES, TRIED IN THAT ORDER, because the same fact lives in two shapes depending on how a
# box was populated:
#
#   advisor.course_prerequisites   the pre-parsed table (migration 003), already AND-of-ORs.
#                                  Absent on a box that never ran those migrations — this file
#                                  swallows `UndefinedTable` rather than failing, which is how
#                                  it can be missing and stay quiet.
#   prerequisite_rules             `catalog_ingestion`'s own table, written by
#                                  `ingest.courses.upsert_prerequisite_rule` from whatever text
#                                  reached it. Holds an AND/OR TREE in `parsed_json`, which
#                                  `_groups_from_tree` flattens to the same AND-of-ORs shape.
#
# WHY THIS MATTERS MORE THAN IT LOOKS. When neither has rows, every course exports with no
# `prereq_groups` — and "no prerequisites" is not an error anywhere downstream, it is a legal,
# quiet state. The whole harness then runs with prerequisite ordering unconstrained: the models
# are never told the order, `prereq_violation` can never fire, and Mode C's repair loop has
# nothing to repair. Measured 2026-08-12 on this box: 0 edges across every program. The prompt
# still says PREREQUISITES COME FIRST, so a run in that state measures models against a rule
# whose data is missing. `run.py check` prints the edge count for exactly this reason.
_PREREQ_CODE_RE = re.compile(r"\b([A-Z]{2,6})\s+(\d{3,5}[A-Z]?)\b")


# =================================================================================================
# HARDCODED PREREQUISITES — TEMPORARY. PLANNED FOR REPLACEMENT.
# =================================================================================================
#
# THIS TABLE IS HAND-TYPED, NOT CRAWLED, and it exists because there is currently no automated
# way to get prerequisites into this database at all:
#
#   * Acalog (catalog.purdue.edu) publishes title, credits and description per course. No
#     prerequisites. Verified 2026-08-12 against CS 18200's own course page.
#   * PurdueIO (api.purdue.io) has no prerequisite field on the Course entity at all.
#   * Banner self-service publishes them, per course, at
#     `bwckctlg.p_disp_course_detail?cat_term_in=...&subj_code_in=CS&crse_numb_in=18200` — and
#     that host's robots.txt is `User-agent: * / Disallow: /`.
#
# So this fills the gap for the ONE THING that cannot be measured without it: prerequisite
# ordering is the largest Mode B failure class in every sweep recorded here, PREREQUISITES COME
# FIRST is the top rule in the prompt, and with an empty prerequisite table `prereq_violation`
# can never fire — the harness would be asking models to obey a rule it cannot score.
#
# WHAT REPLACES IT. Any of: a data extract from the Office of the Registrar (Banner stores these
# as structured rules, not prose — the ask is a report, not a scrape); permission to fetch the
# Banner pages directly, which `catalog-ingest import-prerequisites` already parses; or a
# published Purdue API that carries them. Whichever lands, it writes `prerequisite_rules` and
# `_prereq_rows` picks it up automatically — this table is a FALLBACK, consulted only for a
# course the database has nothing for, so real data always wins and nothing here needs deleting
# on the day it arrives.
#
# SCOPE: the required-course cores of the 16 curated sweep programs (config.yaml `sweep.curated`)
# — 257 distinct required courses, of which the ones below are the ones that actually chain.
#
# RULES FOR EDITING THIS TABLE, and they matter more than the table:
#
#   1. NEVER GUESS. An invented edge is worse than a missing one: a missing edge under-charges
#      every model equally, while a wrong edge charges a model for a schedule the registrar
#      would have accepted, and Mode C re-hits it on every attempt (see `prereq_risk_note`).
#      Everything outside the STEM cores below — NUR, CM, EDCI, COM, POL, AGEC, SOC, EDPS — is
#      deliberately ABSENT rather than approximated. Those programs' plans are currently scored
#      without prerequisite constraints, which is the honest state, not a silent one.
#   2. Verify against the course's own Banner page before adding a row, and put the date in the
#      comment for that subject block.
#   3. AND-of-ORs, same shape as `courses.prereq_groups`: [["CS 25000"], ["CS 25100", "CS
#      25300"]] is CS 25000 AND (CS 25100 OR CS 25300).
#
# Sourced from Purdue's published course descriptions and the department-published prerequisite
# charts (e.g. cs.purdue.edu's own prereq chart) as of 2026-08-12.
_HARDCODED_PREREQS: dict[str, list[list[str]]] = {
    # --- Computer Science. The 18000->18200/24000->25000/25100->25200 spine is the whole
    # reason Mode B's prerequisite rule exists; every CS scenario runs through it.
    "CS 18200": [["CS 18000"], ["MA 16100", "MA 16500", "MA 16200", "MA 16600"]],
    "CS 24000": [["CS 18000"]],
    "CS 25000": [["CS 18200"], ["CS 24000"]],
    "CS 25100": [["CS 18200"], ["CS 24000"]],
    "CS 25200": [["CS 25000"], ["CS 25100"]],
    "CS 37300": [["CS 25100"]],
    "CS 38100": [["CS 25100"]],
    "CS 47100": [["CS 25100"]],
    # --- Mathematics. The calculus sequence, both the MA 161/162 and the MA 165/166 tracks,
    # which are alternates of each other rather than a sequence.
    "MA 16200": [["MA 16100"]],
    "MA 16600": [["MA 16500"]],
    "MA 16020": [["MA 16010"]],
    "MA 26100": [["MA 16200", "MA 16600"]],
    "MA 26500": [["MA 16200", "MA 16600"]],
    "MA 26600": [["MA 26100"]],
    "MA 41600": [["MA 26100"]],
    # --- Statistics.
    "STAT 35000": [["MA 16200", "MA 16600"]],
    # --- Physics. PHYS 172/272 is the engineering/science sequence.
    "PHYS 27200": [["PHYS 17200"], ["MA 16200", "MA 16600"]],
    "PHYS 22100": [["PHYS 22000"]],
    "PHYS 24100": [["PHYS 17200"]],
    # --- Chemistry. The 255/256 organic sequence and its labs; the labs are corequisites of
    # their lectures, which this table cannot express (see `_HARDCODED_COREQS`).
    "CHM 11200": [["CHM 11100"]],
    "CHM 11620": [["CHM 11610"]],
    "CHM 25600": [["CHM 25500"]],
    "CHM 33900": [["CHM 25600"]],
    # --- Biology.
    "BIOL 11100": [["BIOL 11000"]],
    "BIOL 24200": [["BIOL 24100"]],
    # --- Electrical & Computer Engineering. ECE 20001/20002 with their paired labs.
    "ECE 20002": [["ECE 20001"]],
    "ECE 20008": [["ECE 20007"]],
    # --- Mechanical Engineering. Statics before mechanics of materials and dynamics.
    "ME 27400": [["ME 27000"]],
    "ME 32300": [["ME 27000"]],
}

# COREQUISITES — satisfied by the same semester OR earlier, unlike a prerequisite. Same
# provenance, same rules, same fate as `_HARDCODED_PREREQS` above. The lab-with-lecture pairs
# are the whole content: scoring a lab as needing its lecture STRICTLY earlier would charge
# every model for the schedule the catalog itself prescribes.
_HARDCODED_COREQS: dict[str, list[str]] = {
    "CHM 25501": ["CHM 25500"],
    "CHM 25601": ["CHM 25600"],
    "CHM 33901": ["CHM 33900"],
    "ECE 20007": ["ECE 20001"],
    "ME 30801": ["ME 30800"],
    "ME 32301": ["ME 32300"],
}


def _groups_from_tree(node: Any) -> list[list[str]]:
    """An AND/OR prerequisite tree (`prerequisite_rules.parsed_json`) as AND-of-ORs.

    `[["CS 25000"], ["CS 25100", "CS 25300"]]` means CS 25000 AND (CS 25100 OR CS 25300) — the
    shape every consumer in this harness already speaks. A nested AND inside an OR cannot be
    expressed in it and is flattened to the OR of every course underneath, which is the SAFE
    direction: it can only ever accept a schedule the strict reading would reject, never charge
    a model for a violation the catalog does not state.
    """
    if not isinstance(node, dict):
        return []
    kind = node.get("type")
    if kind == "COURSE":
        code = str(node.get("course") or "").upper().strip()
        return [[code]] if code else []
    if kind == "AND":
        out: list[list[str]] = []
        for child in node.get("children") or []:
            out += _groups_from_tree(child)
        return out
    if kind == "OR":
        options: list[str] = []
        for child in node.get("children") or []:
            for group in _groups_from_tree(child):
                options += group
        # Deduplicated, order preserved — the same course can appear twice under a hand-written
        # "A or (A with a lab)" and an option list that repeats itself reads as a parser bug.
        seen: set[str] = set()
        deduped = [c for c in options if not (c in seen or seen.add(c))]
        return [deduped] if deduped else []
    return []


def _codes_in(text: str | None) -> list[str]:
    """Course codes named in a free-text field (`courses.corequisites_raw`), deduplicated."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for subject, number in _PREREQ_CODE_RE.findall(text.upper()):
        code = f"{subject} {number}"
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


def find_program_id(cur: psycopg.Cursor, name_ilike: str) -> list[dict[str, Any]]:
    """Programs whose name matches `name_ilike` (a SQL ILIKE pattern) — for picking a
    `program_id` by hand. Returns id/name/catalog_year/requirement group count so a caller can
    tell a genuinely-crawled program (many groups) from a synthetic fixture-mirror (few)."""
    cur.execute(
        """
        SELECT p.id::text AS id, p.name, cy.label AS catalog_year,
               (SELECT count(*) FROM requirement_groups rg WHERE rg.program_id = p.id) AS n_groups
        FROM programs p
        JOIN catalog_years cy ON cy.id = p.catalog_year_id
        WHERE p.name ILIKE %s
        ORDER BY n_groups DESC
        """,
        (name_ilike,),
    )
    return list(cur.fetchall())


def resolve_major_interactive(pg_url: str, search_text: str, *, limit: int = 10) -> str:
    """Search `programs.name` for `search_text` (case-insensitive substring — the caller does
    not need Postgres wildcard syntax or an exact title) and return one `program_id`.

    One match resolves silently. Several print up to `limit` of them, numbered 0-`limit-1`
    (widest, most-requirement-groups first — see `find_program_id` — so the genuinely-crawled
    program tends to sort near the top rather than a same-named synthetic fixture-mirror), and
    prompt on stdin for a number. This is what lets `run.py --major "computer science"` work
    without knowing the catalog spells a given track `"Computer Science: Machine Intelligence"`
    while another calls its parallel `"Computer Science, BS — Machine Intelligence
    concentration"`.
    """
    with psycopg.connect(pg_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        matches = find_program_id(cur, f"%{search_text}%")

    if not matches:
        raise LookupError(f"no program matches {search_text!r}")
    if len(matches) == 1:
        chosen = matches[0]
        print(f"[major] one match: {chosen['name']} ({chosen['catalog_year']}, "
              f"{chosen['n_groups']} requirement groups) -> {chosen['id']}")
        return chosen["id"]

    shown = matches[:limit]
    print(f"[major] {len(matches)} programs match {search_text!r}"
          + (f", showing the first {limit}" if len(matches) > limit else "") + ":")
    for index, row in enumerate(shown):
        print(f"  {index}  {row['name']} ({row['catalog_year']}, "
              f"{row['n_groups']} requirement groups)")
    choice = input(f"Pick 0-{len(shown) - 1}: ").strip()
    try:
        picked = int(choice)
        if not (0 <= picked < len(shown)):
            raise ValueError
    except ValueError:
        raise LookupError(f"{choice!r} is not a choice between 0 and {len(shown) - 1}") from None

    chosen = shown[picked]
    print(f"[major] picked: {chosen['name']} -> {chosen['id']}")
    return chosen["id"]


def _course_number(code: str) -> int:
    match = _COURSE_NUMBER_RE.search(code)
    return int(match.group(1)) if match else 0


def _attributes(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(tag) for tag in parsed] if isinstance(parsed, list) else []


@dataclass
class GroupMeta:
    """One `requirement_groups` row, with its full (untrimmed) option list and the data needed
    to rank/trim it per scenario. Built once in `fetch_real_db_base`; never mutated after."""
    group_id: str
    name: str
    requirement_type: str          # "all_of" (kept whole, always) or "choose_credits" (ranked)
    credits_min: float | None
    display_order: int
    option_codes: list[str]        # catalog order, already stale-filtered
    option_credits: dict[str, float]
    # Set only when >=50% of this group's OWN options share one `_LANGUAGE_SUBJECTS` prefix —
    # a single-department menu like "Spanish (SPAN)". A menu that MIXES several languages (the
    # crawled "Language and Culture Electives/Foreign Language Requirement" umbrella) is left
    # unset here on purpose; `build_scenario_database` matches it by CODE prefix instead, since
    # no single subject dominates it.
    language_subject: str | None = None


@dataclass
class RealDatabaseBase:
    """Everything `fetch_real_db_base` reads from Postgres for one program, ahead of any
    scenario's own gen-ed/world-language preference. `build_scenario_database` turns one of
    these plus a scenario's preferences into the `CatalogDatabase` that scenario is actually
    shown — pure Python, no further database round trips, because every course any scenario
    could possibly select was already fetched and prerequisite-closed here.
    """
    program_id: str
    program_row: dict[str, Any]
    always_groups: list[GroupMeta]      # requirement_type == all_of — never trimmed
    elective_groups: list[GroupMeta]    # requirement_type == choose_credits — scenario-ranked
    broadened_codes: list[str]          # existing major-subject broadening candidates, ranked
    broaden_subjects: tuple[str, ...]
    course_aliases_rows: list[dict[str, Any]]
    program_notes_rows: list[dict[str, Any]]
    courses_by_code: dict[str, dict[str, Any]]
    prereq_by_code: dict[str, dict[str, Any]]
    # See `is_college_of_engineering`/`_fye_course_codes` — False/empty for every other program,
    # computed once here rather than per scenario since neither depends on gen_ed_preference or
    # world_language.
    college_of_engineering: bool = False
    fye_course_codes: frozenset[str] = frozenset()

    @property
    def always_codes(self) -> set[str]:
        return {c for g in self.always_groups for c in g.option_codes} | {
            row["course_code"] for row in self.course_aliases_rows
        }


def fetch_real_db_base(
    pg_url: str,
    program_id: str,
    *,
    broaden_subjects: tuple[str, ...] | None = None,
    force_selective_groups: tuple[str, ...] = (),
    manual_course_aliases: Mapping[str, str] | None = None,
    extra_course_codes: tuple[str, ...] = (),
    scope_filter: str | None = None,
) -> RealDatabaseBase:
    """Read every table `catalog_export.TABLES` names, for one program, from the real Postgres
    DB. One-time and comparatively expensive (several round trips over what can be thousands of
    courses); returns the MAXIMAL universe any scenario could need, prerequisite-closed, with
    nothing yet trimmed for gen-ed preference or world language — see `build_scenario_database`
    for the cheap, per-scenario, database-free second half.

    `force_selective_groups` — exact `requirement_groups.name` matches to treat as
    `choose_credits` regardless of what the structural heuristics below decide. This crawl
    leaves most groups' `requirement_type`/`credits_min` unset, and a single-subject group with
    a descriptive (non-generic) name is exactly the case none of those heuristics can safely
    call — see `config.yaml`'s `real_db.force_selective_groups` for the ones verified by hand
    for this program and why.

    `manual_course_aliases` — `{alias_code: primary_code}` equivalences the crawl did not
    record, applied exactly like the synthetic "or"-chain aliases below and for the same reason.
    Needed when a group lists two ALTERNATIVE SEQUENCES as four independently required rows
    (`is_required` true, `is_selective_option` false on every one), which is what the chain
    detector keys off and therefore cannot see — see `config.yaml`'s
    `real_db.manual_course_aliases` for the pairs verified by hand for this program.

    `scope_filter` — one of `_scope_for`'s own return values (`"program"`, `"college"`,
    `"university"`), or `None` (default: no filtering, every existing caller's behaviour is
    unchanged). When set, only `requirement_groups` rows whose `_scope_for(requirement_type)`
    matches are kept — everything else (its options, and anything only reachable by merging it
    into a synthetic gen-ed/TWTP/lab-science group below) is dropped before classification, so
    Mode B/C can be pointed at "just this program's own requirements" without a separate
    program_id. Added so a fixture can ask to see only its `"program"`-scoped groups —
    university/college-wide requirements (gen-ed, First-Year Engineering, ...) are exactly the
    content the new standalone school/gen-ed pseudo-major fixtures test in isolation instead.

    Raises `LookupError` for an unknown `program_id`, or one with nothing to plan from, rather
    than silently building an empty database that would render as a program of nothing.
    """
    with psycopg.connect(pg_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id::text AS id, p.name, p.degree_type, p.program_type, p.campus,
                   p.total_credits_min, cy.label AS catalog_year,
                   cy.start_year, cy.end_year, cy.is_archived
            FROM programs p
            JOIN catalog_years cy ON cy.id = p.catalog_year_id
            WHERE p.id = %s
            """,
            (program_id,),
        )
        program = cur.fetchone()
        if program is None:
            raise LookupError(f"no program with id {program_id!r} in {pg_url}")

        # ONE extra query, ONLY for College of Engineering programs — every other program pays
        # nothing for this. Computed here, not near the `return` below, because `cur`'s
        # connection is already closed by then — the `with` block ends partway through this
        # function, well before its final `return`. See `is_college_of_engineering`'s docstring
        # for why this classification must be exact.
        college_of_engineering = is_college_of_engineering(program["name"])
        fye_codes = _fye_course_codes(cur) if college_of_engineering else frozenset()

        cur.execute(
            """
            SELECT rg.id::text AS group_id, rg.name, rg.requirement_type, rg.credits_min,
                   rg.display_order, rg.parent_group_id::text AS parent_group_id,
                   COALESCE(c.course_code, ro.course_code_raw) AS course_code,
                   ro.credits AS option_credits, ro.is_selective_option,
                   -- Carries the "(UCC: QR)" competency tag; see `_UCC_RE`. The only place in
                   -- the crawl the University Core is recoverable from.
                   ro.option_text
            FROM requirement_groups rg
            JOIN requirement_options ro ON ro.requirement_group_id = rg.id
            LEFT JOIN courses c ON c.id = ro.course_id
            WHERE rg.program_id = %s
              AND COALESCE(rg.requirement_type, '') <> 'plan_of_study'
            ORDER BY rg.display_order ASC, ro.display_order ASC
            """,
            (program_id,),
        )
        raw_options = [row for row in cur.fetchall() if (row["course_code"] or "").strip()]

        groups_meta: dict[str, dict[str, Any]] = {}
        option_codes: dict[str, list[str]] = {}
        option_credits: dict[str, dict[str, float]] = {}
        # In-line "X or Y" continuations (`is_selective_option` on the row BEFORE them, same
        # signal `app/services/planner_db._fetch_groups` chains on) — real options, but not an
        # extra requirement, so their credits must not double-count toward the group's own
        # total. The alias-based dedupe below already catches the subset of these pairs that
        # also happen to be registered `course_aliases` (MA 26100/MA 27101); it does NOT catch
        # a plain in-catalog "or" between two courses with no alias relationship (Mechanical
        # Engineering's "PHYS 24100 or PHYS 27200" inside its Major Requirements group) —
        # uncounted here, that pair inflated the group's apparent total to 65 against a
        # credits_min of 61 and made a fully-required 24-course core misread as a menu.
        chain_alternates: dict[str, set[str]] = {}
        # alternate code -> the HEAD code of its "or" run (the first, non-continuation entry),
        # plus which group it was found in. NOT the same claim as "these two courses are
        # interchangeable" in general — `is_selective_option` chains an entire elective MENU
        # exactly the same way it chains a genuine two-way substitute (Purdue's "Language and
        # Culture Electives" is one long "or" run of 40+ unrelated language courses; so is
        # "CS 47100 or CS 47300" inside an AI-course CHOOSE-one group). Only inside a group this
        # module goes on to classify `all_of` (required) does a same-group "or" have exactly one
        # reading — a required group has no real choice by definition, so an internal "or" can
        # only mean "one specific line, two ways to satisfy it," never "pick your favorite."
        # That's why this is filtered to `always_gids` below, well after classification, rather
        # than treated as a general alias signal here.
        chain_primary: dict[str, str] = {}
        chain_primary_gid: dict[str, str] = {}
        chaining: dict[str, bool] = {}
        chain_head: dict[str, str] = {}
        # course code -> the University Core competencies it carries. Collected from EVERY
        # option row across the program before any scope/elective filtering runs, because a
        # course's UCC tag is a property of the course, not of whichever menu happened to list
        # it — CS 18000 is `UCC: IL` wherever it appears.
        ucc_by_code: dict[str, set[str]] = {}
        for row in raw_options:
            gid = row["group_id"]
            code = row["course_code"].strip()
            for tag in _UCC_RE.findall(row.get("option_text") or ""):
                if tag in UCC_COMPETENCIES:
                    ucc_by_code.setdefault(code, set()).add(tag)
            meta = groups_meta.setdefault(gid, {
                "name": row["name"] or "Requirement",
                "requirement_type": row["requirement_type"] or "",
                "credits_min": row["credits_min"],
                "display_order": row["display_order"] or 0,
                "parent_group_id": row["parent_group_id"],
            })
            credits = (float(row["option_credits"]) if row["option_credits"] is not None
                      else _DEFAULT_OPTION_CREDITS)
            if chaining.get(gid) and code not in option_codes.get(gid, []):
                chain_alternates.setdefault(gid, set()).add(code)
                head = chain_head.get(gid)
                if head and code not in chain_primary:
                    chain_primary[code] = head
                    chain_primary_gid[code] = gid
            else:
                chain_head[gid] = code
            chaining[gid] = bool(row["is_selective_option"])
            option_codes.setdefault(gid, [])
            if code not in option_codes[gid]:
                option_codes[gid].append(code)
            option_credits.setdefault(gid, {})[code] = credits

        # SCOPE FILTER, applied BEFORE container detection/classification/merges below, so a
        # dropped group simply never exists for the rest of this function rather than needing
        # every later step (`_collapse_subtree`, `_apply_merge`, `_is_elective`) to know about
        # it separately. Safe because every one of those already conditions on
        # `option_codes.get(gid)` being present as its "does this group carry real options"
        # check — a scope-dropped gid fails that check the same way a genuinely optionless
        # group already did, so no downstream code needed to change. `scope_filter=None` (the
        # default) is a no-op: every existing caller's behaviour is untouched byte-for-byte.
        if scope_filter is not None:
            # A SET OF TIERS, not one tier — see `SCOPE_TIERS`. A bare string is
            # still accepted (and still means exactly that one tier) so any caller passing
            # `_scope_for`'s own vocabulary directly keeps working unchanged.
            allowed = ({scope_filter} if isinstance(scope_filter, str) else frozenset(scope_filter))
            dropped_gids = {gid for gid, meta in groups_meta.items()
                            if _scope_for(meta["requirement_type"]) not in allowed}
            for gid in dropped_gids:
                groups_meta.pop(gid, None)
                option_codes.pop(gid, None)
                option_credits.pop(gid, None)

        # THE WHOLE REQUIREMENT-GROUP TREE for this program (not just groups with their own
        # options) — needed to recognize a "browse by subject/department" CONTAINER: a group
        # with no options of its own but several child groups that each carry a menu (this
        # crawl's "Approved Courses by Subject", holding History/English/Spanish/... as
        # separate children). Every child of a container is a menu-and-alternative BY
        # STRUCTURE, regardless of how the crawler tagged its own `requirement_type` — most of
        # them carry none at all, which is exactly why `_is_selective` alone (below) is not
        # enough to classify them.
        cur.execute(
            "SELECT id::text AS id, name, parent_group_id::text AS parent_group_id, credits_min "
            "FROM requirement_groups WHERE program_id = %s",
            (program_id,),
        )
        tree_rows = cur.fetchall()
        tree_children: dict[str | None, list[dict[str, Any]]] = {}
        for row in tree_rows:
            tree_children.setdefault(row["parent_group_id"], []).append(row)
        child_count: dict[str, int] = Counter(
            r["parent_group_id"] for r in tree_rows if r["parent_group_id"]
        )
        # THRESHOLD, NOT "ANY MULTI-CHILD NODE": ordinary organizational grouping (a
        # concentration's "required courses" + "electives" pair, a lab-science parent with two
        # sequence options) also shows up as a small multi-child parent, and its children are
        # NOT interchangeable alternatives to each other the way subject-browsing menus are.
        # This program's own "Approved Courses by Subject" has 53 children; nothing else in it
        # tops 18 — a wide enough gap that 20 catches the genuine browse-by-department
        # containers without also catching ordinary 3-4-child organizational nodes.
        container_ids = {
            r["id"] for r in tree_rows
            if not option_codes.get(r["id"]) and child_count.get(r["id"], 0) >= 20
        }

        all_named_codes = {c for codes in option_codes.values() for c in codes}
        if not all_named_codes:
            raise LookupError(
                f"program {program_id!r} ({program['name']!r}) has zero requirement_options "
                f"rows — every requirement group is either childless prose or a container. "
                f"There is nothing for Mode B to plan from."
            )

        # STALE REQUIREMENT OPTIONS ARE DROPPED, NOT RAISED — see the module docstring's
        # incident note. Checked across EVERY named code, elective menus included (not just
        # what used to be `required_codes`), since a stale code inside a gen-ed menu would
        # dangle a `courses` reference exactly the same way.
        cur.execute(
            "SELECT DISTINCT course_code FROM courses WHERE course_code = ANY(%s)",
            (sorted(all_named_codes),),
        )
        existing_codes = {row["course_code"] for row in cur.fetchall()}
        stale_codes = all_named_codes - existing_codes
        if stale_codes:
            logger.warning(
                "real_db: dropping %d requirement_options course(s) not in the current "
                "PurdueIO catalog (stale department listing, not a crawl bug): %s",
                len(stale_codes), ", ".join(sorted(stale_codes)),
            )
            for gid in list(option_codes):
                option_codes[gid] = [c for c in option_codes[gid] if c not in stale_codes]
                for code in stale_codes:
                    option_credits.get(gid, {}).pop(code, None)

        # Alias substitutes (MA 16500 for MA 16100, etc.) are separate real courses — a
        # student who took the substitute needs its title/credits looked up too, so its code
        # has to be in `courses` even though it never appears as a requirement_options row
        # itself. Fetched before `courses` so the expansion below can widen the code set first.
        cur.execute(
            """
            SELECT ca.alias_code, c.course_code AS primary_code, ca.reason
            FROM course_aliases ca
            JOIN courses c ON c.id = ca.course_id
            WHERE c.course_code = ANY(%s)
            """,
            (sorted(all_named_codes - stale_codes),),
        )
        course_aliases_rows = [
            {"course_code": row["alias_code"], "alias_of": row["primary_code"],
             "reason": row["reason"] or "approved substitute"}
            for row in cur.fetchall()
        ]

        # DEDUPE ALIAS PAIRS WITHIN THE SAME GROUP. Acalog sometimes lists an honors/alternate
        # course as its own "or" option in requirement_options (MA 27101 next to MA 26100 —
        # "MA 26100 ... 4.00 or MA 27101 ... 5.00") on the SAME page that ALSO backs a
        # `course_aliases` "approved substitute" relationship for the identical pair. Left in
        # as two independent options, the alias inflates the group's apparent credit sum:
        # `_is_selective` below reads MA 26100 (4) + MA 27101 (5) + MA 26500 (3) + MA 35100
        # (3) = 15 credits against a `credits_min` of 7 and calls a two-slot REQUIRED group
        # ("one Calc-III variant AND one Linear-Algebra variant") a free-choice menu — the
        # student never actually gets to pick any 7 credits' worth from that list, only one
        # course per slot. It also hands the model a straight contradiction: `all_of` says
        # "take every option" on the very courses `course_aliases` says "take one, never
        # both" about (confirmed on this program's own "Mathematics" group: MA 16100/16500
        # and MA 16200/16600 are two alternative Calc I/II sequences, not four courses).
        # Keep only the canonical code per pair — the alias is still fully creditable
        # wherever scoring reads `course_aliases`, it just stops being listed as a SECOND,
        # independent option alongside the course it stands in for.
        alias_of_by_code = {row["course_code"]: row["alias_of"] for row in course_aliases_rows}
        for gid, codes in option_codes.items():
            keep = [c for c in codes if alias_of_by_code.get(c) not in codes]
            if len(keep) != len(codes):
                for code in set(codes) - set(keep):
                    option_credits.get(gid, {}).pop(code, None)
                option_codes[gid] = keep

        # LARGE, UNTYPED MENUS (this crawl leaves `requirement_type`/`credits_min` unset on
        # most of them — "Culture Course List" itself included) still have to be recognized as
        # electives, not just the container's own children. No real "take every one of these"
        # requirement in this program's own core/major/math groups tops 21 credits across a
        # handful of courses; a group naming more than this many options is, in practice,
        # always a menu. Kept generous on purpose — a false "elective" here only costs ranking
        # (the group is never left EMPTY, see `build_scenario_database`'s guaranteed top-1),
        # while a false "all_of" would render a real menu whole and reproduce the exact
        # context-overflow bug this module was rewritten to fix.
        _LARGE_MENU_THRESHOLD = 12

        # GENERIC "Option N" / "Sequence N" NAMES are this crawl's OWN tell for "one of several
        # parallel alternatives" — Acalog names a real, single required list after what it IS
        # ("Required CS Major Core Courses"), and only reaches for a placeholder like "Sequence
        # 1" / "Sequence 2" / "Option 1" when the page is presenting parallel tracks (the
        # chemistry sequence vs. the biology sequence vs. the earth-science sequence, each an
        # alternative path to the same lab-science requirement). None of those groups ever get a
        # `requirement_type`/`credits_min` in this crawl, so this is a real, additional signal,
        # not a duplicate of the checks above.
        # Prefix match, not exact — "Sequence 1 (for life scientists who are not Biology
        # majors)" is the same placeholder pattern as a bare "Sequence 1", just with the
        # audience it applies to appended.
        _GENERIC_TRACK_NAME_RE = re.compile(r"^(option|sequence|choice|track)\b", re.I)

        def _is_elective(gid: str, meta: dict[str, Any]) -> bool:
            if meta["name"] in force_selective_groups:
                return True
            counted_credits = [
                v for c, v in option_credits.get(gid, {}).items()
                if c not in chain_alternates.get(gid, set())
            ]
            if _is_selective(meta["requirement_type"], meta["credits_min"], counted_credits):
                return True
            if meta.get("parent_group_id") in container_ids:
                return True
            if _GENERIC_TRACK_NAME_RE.match(meta["name"].strip()):
                return True
            # EVERYTHING BELOW IS AN UNTYPED-MENU BACKSTOP, NOT A SEPARATE VOTE. Both signals
            # exist for the crawl's MAJORITY case — a group with no `credits_min` at all (see
            # each one's own comment: "this crawl leaves requirement_type/credits_min unset on
            # most of them", "None of those groups ever get a requirement_type/credits_min").
            # A group that DOES carry a `credits_min` already got a confident answer from the
            # sum check above; letting size or subject spread override that confident "no" is
            # what misread Mechanical Engineering's own 24-course, 5-subject (ME/ECE/MA/MSE/
            # PHYS — the major's required math, physics, materials and circuits courses, not
            # alternatives to each other) Major Requirements group as a menu.
            if meta["credits_min"]:
                return False
            if len(option_codes.get(gid, [])) > _LARGE_MENU_THRESHOLD:
                return True
            # HETEROGENEOUS SUBJECTS: a group whose options span more than one department
            # (Technical Writing & Presenting: CHM/COM/EDCI) is, in every case this program
            # names, several departments' equivalent versions of the SAME requirement — nobody
            # takes a chemistry course, a communications course, AND an education course to
            # satisfy one "Technical Writing & Presenting" line. A single-subject group with
            # multiple options (Statistics: five STAT courses; PHYSICS I: three PHYS courses)
            # is NOT caught by this — those may be genuine alternative tracks too, but nothing
            # in this crawl distinguishes that case from a real multi-course requirement, and
            # guessing wrong in that direction is the worse failure (see the size-threshold
            # comment above for why "false elective" is the safe side to err on when unsure —
            # this specific signal just doesn't reach that far).
            subjects = {code.split(" ", 1)[0] for code in option_codes.get(gid, [])}
            return len(subjects) > 1

        elective_gids = [gid for gid, meta in groups_meta.items() if _is_elective(gid, meta)]
        always_gids = [gid for gid in groups_meta if gid not in elective_gids]

        # SYNTHETIC ALIASES FOR UNLINKED IN-LINE "OR" ALTERNATES, NOW THAT `always_gids` IS
        # KNOWN. `chain_primary`/`chain_primary_gid` were built earlier from every
        # `is_selective_option` chain in the crawl; filtered here to chains inside a group THIS
        # module classified `all_of` — see that dict's own comment for why an elective group's
        # chain must NOT get this treatment (Mechanical Engineering's Major Requirements ->
        # PHYS 24100/PHYS 27200 is the case this fixes: a required group whose only "or" is a
        # genuine equivalent, unregistered in `course_aliases`, which otherwise made the group
        # need both at once — impossible — and every plan score as missing one of them no
        # matter what it scheduled).
        #
        # ANY RUN LENGTH, NOT JUST PAIRS — every alt_code in a chain aliases to that chain's own
        # HEAD, independently; `requirement_coverage`'s "kind: all" check only needs each alias
        # to canonicalise to the same primary, so a 3-, 4-, ... way run is exactly as sound as a
        # 2-way one, not a different, riskier case. This used to be capped at run length 1 (i.e.
        # exactly one alternate) out of caution that a longer run might really be an elective
        # menu misfiled into an `all_of` group — but that risk is already handled: `_is_elective`
        # decides `always_gids` vs `elective_gids` BEFORE this loop ever runs, so a chain reaching
        # here has already been judged "this group is genuinely required," length included in
        # what it saw. WAS the exact case for the cap's removal: CS MI's own "Required Courses
        # for Machine Intelligence" has "MA 41600 or STAT 41600 or STAT 51200" as its last
        # 3-credit slot — MA 41600/STAT 41600 are a genuine cross-listing (same Probability
        # course, two departments) and STAT 51200 ("Applied Regression Analysis") is a real,
        # different course that satisfies the same slot, not a third listing of Probability. The
        # old length-1 cap left all three listed as independently required (18 credits of "must
        # take every one" against a `credits_min` of 12), which both mis-scored any plan that
        # picked one of the three and let a model burn a course slot scheduling MA 41600 AND
        # STAT 41600 back to back, not realizing they were the same course. Aliasing every
        # alternate to its run's head fixes both: the group's own `courses` list drops to just
        # the head, matching `credits_min` exactly, and `course_aliases` (shown to the model,
        # "take one, never both") now names the whole run.
        existing_alias_codes = {row["course_code"] for row in course_aliases_rows}
        for alt_code, primary_code in chain_primary.items():
            gid = chain_primary_gid.get(alt_code)
            if gid not in always_gids or alt_code in existing_alias_codes:
                continue
            course_aliases_rows.append({
                "course_code": alt_code, "alias_of": primary_code,
                "reason": "in-line requirement alternate (crawled 'or')",
            })
            if alt_code in option_codes.get(gid, []):
                option_codes[gid] = [c for c in option_codes[gid] if c != alt_code]
                option_credits.get(gid, {}).pop(alt_code, None)

        # HAND-STATED EQUIVALENCES THE CRAWL NEVER RECORDED AS ONE. The loop above can only see
        # alternation the catalog expressed as an in-line "or" (`is_selective_option` chains).
        # An Acalog page that lists two ALTERNATIVE SEQUENCES as plain consecutive rows under one
        # heading produces four rows with `is_required` true and `is_selective_option` false on
        # every one, and nothing in the crawl distinguishes that from a genuine "take all four".
        # CS MI's "Mathematics" group is exactly this: MA 16100/16200 (the 5-credit sequence) and
        # MA 16500/16600 (the 4-credit one) are two tracks through the same requirement, and a
        # student takes ONE — but the group scored as needing all four, so every plan in every
        # mode was reported `short: missing MA 16500, MA 16600` no matter what it scheduled, and
        # the coverage number it fed was wrong for a reason the model had no way to fix.
        #
        # Applied here, AFTER the chain aliases, so a crawled `course_aliases` row or a detected
        # chain always wins over a hand-written line that has gone stale against a re-crawl.
        # Same "crawl doesn't structure it, so state it once by hand" pattern — and the same
        # obligation to re-verify after a re-crawl — as `force_selective_groups` above.
        # RECOMPUTED, not reused: the chain loop above appends to `course_aliases_rows` without
        # updating `existing_alias_codes`, and the whole question here is "did anything already
        # alias this code?". Note that being in `chain_primary` is NOT enough to skip on — the
        # chain loop only acts on chains found inside an `always_gid`, so a pair the crawl
        # records as an "or" in some OTHER group (a sample-schedule or credit-bounded sibling)
        # reaches here still unaliased. That is exactly the CS MI case below.
        existing_alias_codes = {row["course_code"] for row in course_aliases_rows}
        for alias_code, primary_code in (manual_course_aliases or {}).items():
            alias_code, primary_code = normalize_code(alias_code), normalize_code(primary_code)
            if alias_code in existing_alias_codes:
                continue
            gid = next((g for g in always_gids if alias_code in option_codes.get(g, [])), None)
            if gid is None:
                # Not in any required group of THIS program — the pair is for another program,
                # or a re-crawl moved it. Silent by design: `manual_course_aliases` is one list
                # shared by every school the sweep runs, so "not applicable here" is the normal
                # case, not a misconfiguration.
                continue
            if primary_code not in option_codes.get(gid, []):
                # BOTH HALVES OF THE PAIR MUST BE IN THE SAME REQUIRED GROUP for the alias to
                # mean anything, and the common reason they aren't is not a stale config line —
                # it is a program that genuinely requires only ONE of the two sequences. Most
                # engineering programs really do require MA 16500 + MA 16600 specifically; there
                # is no alternation to collapse there, and aliasing MA 16500 onto an MA 16100
                # the requirement never named would invent a substitution the catalog doesn't
                # allow. Debug, not warning: across a multi-school sweep this is the majority
                # case for these pairs, so warning-level would be pure noise.
                logger.debug(
                    "manual alias %s -> %s not applied in group %r: the primary is not one of "
                    "its options, so this program requires the alias in its own right.",
                    alias_code, primary_code, groups_meta[gid]["name"],
                )
                continue
            course_aliases_rows.append({
                "course_code": alias_code, "alias_of": primary_code,
                "reason": "hand-verified equivalent sequence (not structured by the crawl)",
            })
            existing_alias_codes.add(alias_code)
            option_codes[gid] = [c for c in option_codes[gid] if c != alias_code]
            option_credits.get(gid, {}).pop(alias_code, None)

        always_codes = {c for gid in always_gids for c in option_codes.get(gid, [])} | {
            row["course_code"] for row in course_aliases_rows
        }
        elective_codes_all = {c for gid in elective_gids for c in option_codes.get(gid, [])}

        # AUTO-DETECT THE SUBJECT TO BROADEN when the caller didn't name one: the single
        # subject prefix required most often, counted off `always_codes` ONLY (the program's
        # genuinely-required courses — major core, math, sequences), not the elective/gen-ed
        # menus. Counting elective codes too would let a program with an enormous open-elective
        # menu in some other subject outvote its own major; see the module docstring for the
        # WGSS incident this auto-detect already guards against on the other axis.
        if broaden_subjects is None:
            subject_counts = Counter(code.split(" ", 1)[0] for code in always_codes)
            broaden_subjects = (subject_counts.most_common(1)[0][0],) if subject_counts else ()

        # THE BROADENING. Every undergraduate course under `broaden_subjects` that ISN'T
        # already named by SOME requirement — real electives, not tied to a specific
        # requirement row, exactly the ones the crawled catalog's "Selectives (N credits)"
        # groups gesture at with a sample list rather than an exhaustive one.
        broadened_codes: list[str] = []
        if broaden_subjects:
            cur.execute(
                "SELECT DISTINCT course_code FROM courses WHERE course_code SIMILAR TO %s",
                ("(" + "|".join(re.escape(s) for s in broaden_subjects) + ") %",),
            )
            broadened_codes = sorted(
                (row["course_code"] for row in cur.fetchall()
                 if row["course_code"] not in always_codes | elective_codes_all
                 and _course_number(row["course_code"]) < _MAX_UNDERGRAD_NUMBER),
                key=_course_number,
            )

        # THE MAXIMAL UNIVERSE: every code any scenario could ever end up rendering.
        codes = always_codes | elective_codes_all | set(broadened_codes)

        cur.execute(
            """
            SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min, attributes_raw
            FROM courses WHERE course_code = ANY(%s) ORDER BY course_code
            """,
            (sorted(codes),),
        )
        courses_by_code = {row["course_code"]: row for row in cur.fetchall()}
        # COURSES THE STUDENT HOLDS BUT THIS PROGRAM NEVER LISTS. A CS major's completed
        # transcript routinely contains ENGL 10600 and COM 11400 — real Purdue courses with real
        # credit hours (4 and 3) that appear nowhere in the CS requirement universe, so the
        # query above never sees them. Without this they were priced at zero and the student was
        # reported short of a graduation minimum they had actually met.
        #
        # Fetched into `courses_by_code` only, which feeds `all_credits`; they are NOT added to
        # the rendered `courses` table, because the model has no reason to be shown courses that
        # are neither a requirement of this program nor available to schedule — the student has
        # already taken them.
        wanted_extra = sorted(set(extra_course_codes) - set(courses_by_code))
        if wanted_extra:
            cur.execute(
                """
                SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min,
                       attributes_raw
                FROM courses WHERE course_code = ANY(%s) ORDER BY course_code
                """,
                (wanted_extra,),
            )
            for row in cur.fetchall():
                courses_by_code.setdefault(row["course_code"], row)
        # UCC COMPETENCIES ONTO THE COURSE, so a course's line in the rendered database says
        # which University Core requirements taking it would satisfy. `attributes_raw` stays the
        # crawl's own field; these are merged with it in `_course_row` rather than overwriting.
        for code, row in courses_by_code.items():
            row["ucc_tags"] = sorted(ucc_by_code.get(code, ()))
        missing_required = always_codes - set(courses_by_code)
        if missing_required:
            raise LookupError(
                f"requirement_options names {len(missing_required)} required course(s) not in "
                f"`courses`: {', '.join(sorted(missing_required))}"
            )
        # A broadened or elective-menu candidate that doesn't resolve to a real `courses` row
        # (crawl gap) is dropped rather than raised — it was never load-bearing.
        broadened_codes = [c for c in broadened_codes if c in courses_by_code]
        for gid in list(option_codes):
            option_codes[gid] = [c for c in option_codes[gid] if c in courses_by_code]
        codes = always_codes | {c for gid in elective_gids for c in option_codes.get(gid, [])} \
            | set(broadened_codes)

        def _advisor_rows(fetch_codes: set[str], query: str) -> dict[str, dict[str, Any]]:
            if not fetch_codes:
                return {}
            try:
                cur.execute(query, (sorted(fetch_codes),))
            except psycopg.errors.UndefinedTable:
                conn.rollback()
                return {}
            return {row["course_code"]: row for row in cur.fetchall()}

        def _prereq_rows(fetch_codes: set[str]) -> dict[str, dict[str, Any]]:
            """Prerequisites for `fetch_codes`, from whichever source this database has. See
            the `_groups_from_tree` section above for the two shapes and why it matters."""
            if not fetch_codes:
                return {}
            rows = _advisor_rows(
                fetch_codes,
                "SELECT course_code, raw_text, prereq_groups, coreq_codes, confidence "
                "FROM advisor.course_prerequisites WHERE course_code = ANY(%s)",
            )
            if rows:
                return rows
            cur.execute(
                """
                SELECT DISTINCT ON (c.course_code)
                       c.course_code, c.corequisites_raw,
                       pr.raw_text, pr.parsed_json, pr.parse_confidence
                FROM courses c
                JOIN prerequisite_rules pr ON pr.course_id = c.id
                WHERE c.course_code = ANY(%s) AND pr.parsed_json IS NOT NULL
                ORDER BY c.course_code, pr.parse_confidence
                """,
                (sorted(fetch_codes),),
            )
            out: dict[str, dict[str, Any]] = {}
            for row in cur.fetchall():
                groups = _groups_from_tree(row["parsed_json"])
                if not groups:
                    continue
                out[row["course_code"]] = {
                    "course_code": row["course_code"],
                    "raw_text": row["raw_text"],
                    "prereq_groups": groups,
                    "coreq_codes": _codes_in(row["corequisites_raw"]),
                    "confidence": row["parse_confidence"] or "medium",
                }

            # LAST, AND ONLY FOR A COURSE NOTHING ABOVE ANSWERED — see `_HARDCODED_PREREQS`.
            # Real data always wins: the day an extract or an authorised fetch lands in
            # `prerequisite_rules`, that course stops reading this table with no code change
            # and no row to delete here.
            for code in sorted(fetch_codes):
                if code in out:
                    continue
                groups = _HARDCODED_PREREQS.get(code)
                coreqs = _HARDCODED_COREQS.get(code)
                if not groups and not coreqs:
                    continue
                out[code] = {
                    "course_code": code,
                    "raw_text": "hand-entered (see real_db._HARDCODED_PREREQS)",
                    "prereq_groups": groups or [],
                    "coreq_codes": list(coreqs or []),
                    # NOT "high", whatever the source. `confidence` is printed next to every
                    # prerequisite in the export the model reads, and a hand-typed edge has not
                    # been read off the registrar's own page the way a crawled one has.
                    "confidence": "medium",
                }
            return out

        prereq_by_code = _prereq_rows(codes)

        # CLOSE OVER PREREQUISITE/COREQUISITE REFERENCES across the WHOLE maximal universe —
        # see the module docstring. Bounded to 4 rounds — real prerequisite chains are shallow,
        # this just caps the pathological case. Closure additions are folded into `codes` (so
        # every scenario's own, later, in-memory closure in `build_scenario_database` can find
        # them) but deliberately NOT into `always_codes`: a course pulled in only to satisfy an
        # ELECTIVE's prerequisite should still be absent from a scenario that never selects that
        # elective.
        for _ in range(4):
            referenced = {
                code for row in prereq_by_code.values()
                for group in (row.get("prereq_groups") or []) for code in group
            } | {
                code for row in prereq_by_code.values() for code in (row.get("coreq_codes") or [])
            }
            missing = referenced - codes
            if not missing:
                break
            cur.execute(
                """
                SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min, attributes_raw
                FROM courses WHERE course_code = ANY(%s) ORDER BY course_code
                """,
                (sorted(missing),),
            )
            found = {row["course_code"]: row for row in cur.fetchall()}
            if not found:
                # Referenced codes that aren't real courses in `courses` either (a crawl gap in
                # the prerequisite text itself) — nothing more to resolve them with.
                break
            courses_by_code.update(found)
            codes |= set(found)
            prereq_by_code.update(_prereq_rows(set(found)))

        # Offering-term data (`advisor.course_planner_terms`) is deliberately NOT fetched here
        # — see `catalog_export.py`'s module docstring: Purdue's own published pattern is not
        # always a reliable predictor of a future term, so the harness stopped scoring plans
        # against it. Every consumer reads `offered_terms` defensively, so simply never
        # populating it is what turns the constraint off everywhere at once. `advisor
        # .course_workload` is likewise no longer fetched — see the same module's TABLES
        # comment on why.

        cur.execute(
            "SELECT note_type, note_text FROM program_notes WHERE program_id = %s",
            (program_id,),
        )
        program_notes_rows = [
            {"program_id": program_id, "note_type": row["note_type"], "note_text": row["note_text"]}
            for row in cur.fetchall()
        ]

    # SUBTREE MERGES. `_is_elective`'s heuristics correctly flag every individual leaf below
    # (containers, generic "Sequence N"/"Option N" names, `force_selective_groups`) as
    # SELECTIVE, but each leaf still rendered as its OWN top-level requirement — a CS major was
    # being told it owed separate credits in 53 departments for one Gen-Ed requirement, and a
    # separate full lab sequence in chemistry AND biology AND earth science AND physics for
    # one Lab Science requirement. Fixing that needs the ACTUAL crawled tree shape, not more
    # heuristics — printed by hand 2026-08-06 (`requirement_groups.parent_group_id`, walked to
    # every option-holding leaf) and hard-coded below by name, the same "crawl doesn't
    # structure it, so state it once by hand" pattern as `force_selective_groups`.
    def _collapse_subtree(root_name: str) -> tuple[str, list[str]] | None:
        """Every option-holding leaf under the group named `root_name`, found by walking the
        real tree (`tree_children`) rather than guessed — the actual shape mixes containers
        nested several levels deep with generically- and descriptively-named leaves in ways no
        single heuristic catches. `None` if no group has that name (a re-crawl renamed it —
        loud by omission, since the merge below then silently does nothing rather than erroring
        on a missing group).
        """
        root = next((r for r in tree_rows if r["name"] == root_name), None)
        if root is None:
            return None
        leaves: list[str] = []
        stack = [root["id"]]
        while stack:
            gid = stack.pop()
            if option_codes.get(gid):
                leaves.append(gid)
            stack.extend(c["id"] for c in tree_children.get(gid, []))
        return root["id"], leaves

    def _apply_merge(gids: list[str], *, group_id: str, name: str, credits_min: float,
                     exclude: frozenset[str] = frozenset()) -> None:
        """Pool `gids`' combined options into one new `choose_credits` GroupMeta and swap it
        into `elective_gids` in place of every gid it absorbed. `exclude` drops specific codes
        from the pool even though their source group named them — for a course a DIFFERENT
        requirement has already claimed exclusively (see the Gen-Ed call site below)."""
        nonlocal elective_gids
        if not gids:
            return
        codes: list[str] = []
        credits: dict[str, float] = {}
        seen: set[str] = set()
        for gid in sorted(gids, key=lambda g: groups_meta[g]["display_order"]):
            for code in option_codes.get(gid, []):
                if code in exclude:
                    continue
                if code not in seen:
                    seen.add(code)
                    codes.append(code)
                credits.setdefault(
                    code, option_credits.get(gid, {}).get(code, _DEFAULT_OPTION_CREDITS))
        groups_meta[group_id] = {
            "name": name, "requirement_type": "choose_credits", "credits_min": credits_min,
            "display_order": min(groups_meta[g]["display_order"] for g in gids),
            "parent_group_id": None,
        }
        option_codes[group_id] = codes
        option_credits[group_id] = credits
        elective_gids = [g for g in elective_gids if g not in gids] + [group_id]

    # GENERAL EDUCATION / CULTURE — `College of Science Core Requirements`'s own raw_text
    # states ONE requirement ("General Education (9 credits) - Choose courses from this list
    # to fulfill each General Education Option below to total 9 credits"), drawn from one pool.
    # Acalog's page shows that pool two ways — once flat ("Culture Course List", a SIBLING of
    # "Approved Courses by Subject", not one of its children — the tree walk alone would miss
    # it), once browsable by department ("Approved Courses by Subject" and its ~53 children) —
    # and the crawler captured both as separate groups. Real incident 2026-08-06: a CS major
    # was told it owed 3 separate credits in EVERY department (Women's, Gender & Sexuality
    # Studies, Theatre, Naval Science, every language) instead of ~9 total from one pool.
    #
    # The prose actually splits the 9 credits into three named sub-options (I/II/III) that
    # must each get their own share; this crawl does not tag which UCC category (HUM/BSS/
    # STS/...) a course satisfies (see `catalog_export.py`'s OMITTED_COLUMNS note), so one
    # pooled 9-credit target is the closest honest approximation available, not a
    # re-derivation of the three-way split.
    #
    # `Language and Culture Electives/Foreign Language Requirement` is deliberately NOT folded
    # in: `Curriculum Note`'s own raw_text states it explicitly — "Courses taken to meet the
    # Foreign Language and Culture requirement may not also be used to meet the General
    # Education or Great Issues requirements" — a genuinely separate requirement, which is why
    # `world_language` targets it on its own (see `build_scenario_database`).
    #
    # THAT RULE WAS ENFORCED BY CODE UNTIL 2026-08-12, and is now deliberately NOT enforced —
    # see `_ENFORCE_LANGUAGE_GENED_EXCLUSION` above. What it did: a department's "Approved
    # Courses by Subject" listing is not limited to upper-division culture courses, it names
    # the FULL department catalog, proficiency sequence included, for every language department
    # (French, Spanish, ...). Incident 2026-08-06: French courses rendered in BOTH the Gen-Ed
    # pool's `requirement_options` AND the dedicated Language and Culture group's, so the SAME
    # course counted toward both at once. Every course under a `_LANGUAGE_SUBJECTS` prefix was
    # therefore excluded from the Gen-Ed pool outright — not just the codes the dedicated group
    # happens to name, since a narrower exclusion still leaves a department's OTHER language
    # courses (upper-division literature, culture-in-the-language) sitting in both buckets.
    _subject_browse = _collapse_subtree("Approved Courses by Subject")
    _culture_gid = next((r["id"] for r in tree_rows if r["name"] == "Culture Course List"), None)
    _gened_leaves = (_subject_browse[1] if _subject_browse else []) + (
        [_culture_gid] if _culture_gid and option_codes.get(_culture_gid) else [])
    _gened_candidate_codes = {c for g in _gened_leaves for c in option_codes.get(g, [])}
    _lang_reserved = frozenset(
        c for c in _gened_candidate_codes if c.split(" ", 1)[0] in _LANGUAGE_SUBJECTS
    ) if _ENFORCE_LANGUAGE_GENED_EXCLUSION else frozenset()
    _apply_merge(_gened_leaves, group_id="merged-culture-gened",
                credits_min=9.0, name="General Education / Culture Course List",
                exclude=_lang_reserved)

    # TECHNICAL WRITING & PRESENTING — same shape, smaller tree: "Technical Writing &
    # Presenting (TWTP)" > "Technical Writing" > "Option 1" (7 ENGL courses) is one alternative
    # path; a sibling leaf directly under the TWTP root, also named "Technical Writing &
    # Presenting" (3 COM/CHM/EDCI courses), is the other. Both satisfy the SAME prose line
    # ("Technical Writing and Presentation (0-6 credits)"). Left unmerged, a CS major was told
    # it needed credits from both independently.
    _twtp = _collapse_subtree("Technical Writing & Presenting (TWTP)")
    _apply_merge(_twtp[1] if _twtp else [], group_id="merged-technical-writing",
                credits_min=3.0, name="Technical Writing & Presenting")

    # LABORATORY SCIENCE — "Laboratory Science (6-8 credits)" is the crawl's OWN explicit
    # credit target (used below instead of a guess); under it, "Biology Sequences",
    # "Chemistry Sequences", "Earth, Atmospheric, and Planetary Sciences Sequence", and
    # "Physics Sequences (8.0 credits required for a sequence)" are four ALTERNATIVE paths,
    # each itself a pair of "Sequence 1"/"Sequence 2" (or "PHYSICS I"/"PHYSICS II") groups. Real
    # incident 2026-08-06: Physics was previously left OUT of an earlier, heuristic version of
    # this merge (it doesn't match the "Sequence N" name pattern the other three paths do), so
    # a CS major was told it needed a full chemistry-or-biology-or-earth-science sequence AND
    # separately a full physics sequence, instead of picking exactly one of the four.
    #
    # POOLED, NOT BUNDLE-ENFORCED: this lets the `choose_credits` sum be reached by mixing
    # courses from different departments (one chemistry course + one biology course), which is
    # not a real Purdue sequence. True "pick one coherent bundle" enforcement needs a new
    # requirement shape in `plan_scorers.py`/`real_scoring.py` (both the fixture and real-DB
    # scoring path) — out of scope for today; this pooled merge is the agreed smaller fix that
    # at least stops the model being asked for four simultaneous full sequences.
    _lab_root_name = "Laboratory Science (6-8 credits)"
    _lab_root = next((r for r in tree_rows if r["name"] == _lab_root_name), None)
    _lab = _collapse_subtree(_lab_root_name)
    _apply_merge(
        _lab[1] if _lab else [], group_id="merged-lab-science",
        credits_min=float(_lab_root["credits_min"]) if _lab_root and _lab_root.get("credits_min")
        else 8.0,
        name="Laboratory Science",
    )

    # ONE REQUIREMENT, CRAWLED TWICE, INSIDE A SINGLE PROGRAM (2026-08-12). A Purdue program
    # page states its college core in its own department block AND again in the College of
    # Science core block, and the crawler captures both: CS carries `Mathematics` (no type, no
    # credit figure) and `Mathematics (8-10 credits)` (`requirement_type: college`, 8.0) with
    # the SAME four options MA 16100/16200/16500/16600, and `Statistics` (5 options) beside
    # `Statistics (3 credits)` (2 of those same 5). The student was shown Mathematics twice and
    # scored against two separate math requirements for one body of coursework.
    #
    # Same collapse, same justification as `merge_real_db_bases` does across programs — a course
    # satisfies every list it appears in, so if two same-named groups really were different
    # requirements, whatever closes one already closes the other. The survivor is the one naming
    # the MOST options (ties broken by display order, so the result does not depend on dict
    # ordering); it keeps its own option list rather than absorbing the duplicate's, since a
    # union would force an `all_of` survivor to require courses its own list never named. It
    # DOES adopt a credit target the duplicate states and it lacks — that figure is the one real
    # thing the college's copy carries.
    def _dedupe_same_name_groups() -> None:
        nonlocal always_gids, elective_gids
        by_key: dict[str, list[str]] = {}
        for gid in (*always_gids, *elective_gids):
            by_key.setdefault(
                _requirement_name_key(groups_meta[gid]["name"]), []).append(gid)
        dropped: set[str] = set()
        for gids in by_key.values():
            if len(gids) < 2:
                continue
            keep, *rest = sorted(
                gids,
                key=lambda g: (-len(option_codes.get(g, [])), groups_meta[g]["display_order"]),
            )
            if groups_meta[keep]["credits_min"] is None:
                stated = [groups_meta[g]["credits_min"] for g in rest
                          if groups_meta[g]["credits_min"] is not None]
                if stated:
                    groups_meta[keep]["credits_min"] = max(stated)
            dropped.update(rest)
            logger.info(
                "real_db: one requirement crawled under %d names, keeping %r: %s",
                len(gids), groups_meta[keep]["name"],
                ", ".join(f"{groups_meta[g]['name']!r} ({len(option_codes.get(g, []))} options)"
                          for g in (keep, *rest)),
            )
        if dropped:
            always_gids = [g for g in always_gids if g not in dropped]
            elective_gids = [g for g in elective_gids if g not in dropped]

    _dedupe_same_name_groups()

    def _language_subject(codes: list[str]) -> str | None:
        if not codes:
            return None
        subject, count = Counter(c.split(" ", 1)[0] for c in codes).most_common(1)[0]
        return subject if subject in _LANGUAGE_SUBJECTS and count / len(codes) >= 0.5 else None

    def _build_group(gid: str) -> GroupMeta:
        meta = groups_meta[gid]
        codes = option_codes.get(gid, [])
        is_elective = gid in elective_gids
        credits_min = meta["credits_min"]
        if meta["name"] == _WORLD_LANGUAGE_GROUP_NAME and credits_min is None:
            # Same "prose states a number, no structured column carries it" gap as the
            # General Education merge above — `College of Science Core Requirements`'s own
            # raw_text: "Language and Culture (1-9 credits) Complete one of the Options from
            # this list." 9 is the upper/full-completion end and also what makes the two real
            # paths (`build_scenario_database`'s `_language_path_options`: 2 sequence courses +
            # a 3rd credit either way) land exactly on target — see that function's docstring.
            credits_min = 9.0
        return GroupMeta(
            group_id=gid, name=meta["name"],
            requirement_type=("choose_credits" if is_elective else "all_of"),
            credits_min=credits_min, display_order=meta["display_order"],
            option_codes=codes, option_credits=option_credits.get(gid, {}),
            language_subject=_language_subject(codes) if is_elective else None,
        )

    always_groups = sorted((_build_group(gid) for gid in always_gids),
                           key=lambda g: g.display_order)
    elective_groups = sorted((_build_group(gid) for gid in elective_gids),
                             key=lambda g: g.display_order)

    # --- UNIVERSITY CORE, SYNTHESIZED FROM THE UCC TAGS -------------------------------------
    #
    # The crawl states the core only as prose (see `_UCC_RE`): the "University Core
    # Requirements" group is 0 options, 0 children, 463 characters of text naming nine
    # competencies. Every course that satisfies one is tagged inline in its option text and
    # nowhere else. So the core is rebuilt here as nine ordinary `choose_credits` groups — one
    # per competency, options being the courses in THIS program's universe carrying that tag —
    # which means the existing scorer, renderer and trimmer all handle it with no special case.
    #
    # SCOPED TO THIS PROGRAM'S OWN COURSE UNIVERSE, not to all 40,007 tagged rows in the crawl.
    # "Human Cultures: Humanities" has 24,289 options catalog-wide; rendering that is neither
    # possible within the context nor useful. What a student needs is "of the courses in front
    # of you, these carry HUM", and each course's own `attributes` says so too.
    #
    # Only emitted when the program actually crawls a University Core group — a fixture-sourced
    # pseudo-major or a program whose page omits it should not grow a requirement out of thin
    # air. 3.0 credits per competency is Purdue's standard single-course core slot.
    has_core_group = any(
        (r.get("name") or "") == "University Core Requirements" for r in tree_rows
    )
    if has_core_group and ucc_by_code:
        # These nine groups ARE the University Core in the export now — the prose row that used
        # to announce it (and tell the model there was nothing to schedule for it) is gone with
        # the rest of `unresolved_requirement_groups`, so what the model sees for gen-ed is a
        # real menu of real course codes and nothing else.
        universe = set(courses_by_code)
        for order, (tag, label) in enumerate(UCC_COMPETENCIES.items()):
            tagged = sorted(c for c in universe if tag in ucc_by_code.get(c, ()))
            if not tagged:
                continue  # this program's universe cannot satisfy it; say nothing rather than
                          # assert a requirement with an empty menu
            elective_groups.append(GroupMeta(
                group_id=f"ucc-{tag.lower()}",
                name=f"University Core: {label} (UCC: {tag})",
                requirement_type="choose_credits",
                credits_min=3.0,
                display_order=9000 + order,   # after the program's own groups, before nothing
                option_codes=tagged,
                option_credits={c: float(courses_by_code[c]["credit_hours_min"] or 3.0)
                                for c in tagged},
                language_subject=None,
            ))

    return RealDatabaseBase(
        program_id=program_id, program_row=program,
        always_groups=always_groups, elective_groups=elective_groups,
        broadened_codes=broadened_codes, broaden_subjects=broaden_subjects,
        course_aliases_rows=course_aliases_rows, program_notes_rows=program_notes_rows,
        courses_by_code=courses_by_code,
        prereq_by_code=prereq_by_code,
        college_of_engineering=college_of_engineering, fye_course_codes=fye_codes,
    )


def _rank_by_default(codes: list[str]) -> list[str]:
    """The no-preference / "most popular" ranking: `_DEFAULT_GEN_ED_CODES` first (in that
    list's own order, filtered to whatever this menu actually names), then everything else in
    (subject, course number) order — deterministic, and never empty just because a scenario
    named no preference."""
    preferred = [c for c in _DEFAULT_GEN_ED_CODES if c in codes]
    rest = sorted((c for c in codes if c not in preferred),
                 key=lambda c: (c.split(" ", 1)[0], _course_number(c)))
    return preferred + rest


def _rank_by_theme(
    codes: list[str], preference: str, courses_by_code: dict[str, dict[str, Any]],
) -> list[str] | None:
    """Deterministic keyword match for a free-text preference against `_GEN_ED_THEMES`.
    Returns `None` (never an empty list) if nothing in the theme table matched the preference
    text at all, so the caller falls back to `_rank_by_default` instead of silently emptying a
    requirement's menu because a student's wording didn't happen to hit the keyword table.
    """
    pref = preference.lower()
    substrings: set[str] = set()
    for theme, subs in _GEN_ED_THEMES.items():
        if theme in pref or any(s in pref for s in subs):
            substrings.update(subs)
    if not substrings:
        return None

    def _title(code: str) -> str:
        return str((courses_by_code.get(code) or {}).get("title") or "").lower()

    hits = [c for c in codes if any(s in _title(c) for s in substrings)]
    if not hits:
        return None
    misses = [c for c in codes if c not in hits]
    return sorted(hits, key=lambda c: (c.split(" ", 1)[0], _course_number(c))) \
        + _rank_by_default(misses)


def _rank_gen_ed(codes: list[str], preference: str | None, base: RealDatabaseBase) -> list[str]:
    """Best-first ranking of one elective group's options for one scenario's stated
    preference. Always returns every code that resolved to a real course — ranking changes
    ORDER (what survives a tight budget first), never membership; dropping an option outright
    is `build_scenario_database`'s job, driven by the token budget, not this function's.
    """
    pool = [c for c in codes if c in base.courses_by_code]
    pref = (preference or "").strip().lower()
    if pref in _NO_PREFERENCE_PHRASES:
        return _rank_by_default(pool)
    themed = _rank_by_theme(pool, pref, base.courses_by_code)
    return themed if themed is not None else _rank_by_default(pool)


def _language_window(codes: list[str], subject: str, level: int) -> list[str]:
    """The `level`-th course in `subject`'s sequence (1-indexed, catalog course-number order)
    plus the next two — e.g. level 3 in a 101/102/201/202/301/... sequence returns
    201, 202, 301: two semesters already behind the student, three courses still ahead of them.
    Clamped rather than erroring on a level past the end of what this program's crawled options
    include.
    """
    subject_codes = sorted((c for c in codes if c.split(" ", 1)[0] == subject), key=_course_number)
    start = max(0, level - 1)
    return subject_codes[start:start + 3]


# Levels the department ladder is rebuilt over in `_language_path_options`: Purdue numbers two
# courses per catalog "hundred", so 8 levels covers 10100 through 40200 — every numbered
# language course a bachelor's plan can reach.
_MAX_LANGUAGE_LEVEL = 8


def _expected_course_number(level: int) -> int:
    """Purdue's language-sequence numbering: two courses per catalog "hundred" (10100/10200,
    20100/20200, 30100/30200, ...), incrementing the hundred every two proficiency levels.
    Level 1 -> 10100, level 2 -> 10200, level 3 -> 20100, level 4 -> 20200, and so on. Used as a
    THRESHOLD against each candidate's own course number, not as an array index — see
    `_language_path_options` for why the two must not be conflated.
    """
    year = (level + 1) // 2
    semester = 1 if level % 2 else 2
    return year * 10000 + semester * 100


def _language_path_options(
    base: RealDatabaseBase, subject: str, level: int, sequence_codes: list[str],
) -> tuple[list[str], float]:
    """Both real paths for satisfying the world-language requirement in `subject`, starting
    from proficiency `level` — confirmed by hand, not guessed: (A) three courses in the
    department's own numbered sequence, or (B) the same first two plus one course from the SAME
    department that is NOT part of that sequence — Purdue's "culture course taught in the
    language" alternative (e.g. a literature course like SPAN 23100, which the "Approved Courses
    by Subject" merge folded into the general Gen-Ed pool but which is also a legitimate 3rd
    credit here).

    `level` is thresholded against each candidate's OWN course number
    (`_expected_course_number`), not counted as a position in `subject_sequence`. Indexing by
    position silently mis-leveled a student whenever the crawled option list for this
    requirement didn't start at level 1 — confirmed for German, whose "Language and Culture
    Electives" options begin at GER 10200 ("Level II"): `subject_sequence[0]` is already a
    level-2 course, so `subject_sequence[level - 1:]` handed a level-2 student only the LAST
    course in the list (`subject_sequence[1:]`) instead of the two courses actually ahead of
    them.

    THE SEQUENCE IS THE DEPARTMENT'S, NOT THE CRAWLED GROUP'S (2026-08-12). The requirement
    group names exactly three courses per language — Levels I, II and III (GER 10100/10200/
    20100) — because three is what a student STARTING AT LEVEL 1 takes. Read as the whole
    sequence, that list runs out for anybody placed higher: a level-2 German student got
    `ahead = [GER 10200, GER 20100]`, which is two courses, so `third_language` was empty and
    the pool collapsed to 2 sequence courses + 1 culture course against a 9-credit target —
    three forced courses with no path to choose between, and the only "alternative" a
    lower-level course (GER 10500, Accelerated Basic German) they had already placed out of.
    Level IV (GER 20200) exists and is in this program's course universe; it was simply never
    named by the group. So the ladder is rebuilt from every course in the subject whose number
    IS a sequence number (`_expected_course_number`), unioned with whatever the group named,
    and the culture alternative is likewise held to courses at or above the student's floor.

    Returns a small option pool (2 guaranteed sequence courses, plus up to one alternative
    each for the 3rd credit) and a credit target computed from real course credits so EITHER
    path satisfies it. The `choose_credits` scorer only sums credits from whatever the model
    actually schedules against a group's option list — it has no notion of "path" — so
    offering both 3rd-credit alternatives and crediting the group normally IS how "choose one
    path" gets modeled without a new bundle-choice requirement type (see the module docstring
    on `force_selective_groups` for the same "crawl doesn't structure it, so state it once by
    hand" pattern).
    """
    ladder = {_expected_course_number(lv) for lv in range(1, _MAX_LANGUAGE_LEVEL + 1)}
    subject_sequence = sorted(
        {c for c in sequence_codes if c.split(" ", 1)[0] == subject}
        | {c for c in base.courses_by_code
           if c.split(" ", 1)[0] == subject and _course_number(c) in ladder},
        key=_course_number,
    )
    floor = _expected_course_number(level)
    ahead = [c for c in subject_sequence if _course_number(c) >= floor]
    two = ahead[:2]
    third_language = ahead[2:3]
    culture = sorted(
        (c for c in base.courses_by_code
         if c.split(" ", 1)[0] == subject and c not in subject_sequence
         and _course_number(c) >= floor),
        key=_course_number,
    )[:1]
    pool = list(dict.fromkeys(two + third_language + culture))

    def _credits(code: str) -> float:
        return float(base.courses_by_code.get(code, {}).get("credit_hours_min") or 3.0)

    third_credit = min((_credits(c) for c in (third_language + culture)), default=3.0)
    target = sum(_credits(c) for c in two) + third_credit
    return pool, target


def _partial_database(
    base: RealDatabaseBase, codes: set[str], kept_by_group: dict[str, set[str]],
) -> CatalogDatabase:
    """Assemble a `CatalogDatabase` for exactly `codes` (assumed already prerequisite-closed),
    with `requirement_groups`/`requirement_options` rebuilt from `kept_by_group` — a
    `choose_credits` group with nothing kept is left OUT of `requirement_groups` entirely
    (rather than rendered as a menu with zero options, which would read as an unsatisfiable
    requirement) rather than reflecting every option the real crawl ever named.

    Used both for the final scenario database and, repeatedly, as `build_scenario_database`'s
    own token-budget probe — see `catalog_export.render_context`'s "AUTHORITATIVE measure"
    note, unchanged from the pre-refactor single-pass version this replaces.
    """
    depths = chain_depths(CatalogDatabase(
        path=Path("."), tables={
            "courses": [
                {"course_code": c,
                 "prereq_groups": (base.prereq_by_code.get(c) or {}).get("prereq_groups") or []}
                for c in codes
            ],
        }, db_hash="",
    ))
    ordered = sorted(codes, key=lambda c: (depths.get(c, 0), c))

    courses_by_code, prereq_by_code = base.courses_by_code, base.prereq_by_code
    program = base.program_row

    requirement_groups_rows: list[dict[str, Any]] = []
    requirement_options_rows: list[dict[str, Any]] = []
    for group in (*base.always_groups, *base.elective_groups):
        group_kept = (set(group.option_codes) if group.requirement_type == "all_of"
                     else kept_by_group.get(group.group_id) or set())
        if not group_kept:
            continue
        requirement_groups_rows.append({
            "id": group.group_id, "program_id": base.program_id, "parent_group_id": None,
            "name": group.name, "requirement_type": group.requirement_type,
            "credits_min": group.credits_min,
        })
        # Not just `group.option_codes ∩ group_kept`: the world-language block (above) can put a
        # code in `group_kept` that was NEVER in the crawled `option_codes` at all — the "culture
        # course in the language" alternative for path B, pulled from `base.courses_by_code`
        # rather than the group's own crawled listing. Filtering by `option_codes` first silently
        # dropped that pick's `requirement_options` row every time it fired: the course still
        # showed up in `courses` (a plain, ungrouped elective) but satisfied nothing, so path B
        # quietly lost its 3rd credit. Emit crawled-order picks first, then any of the rest.
        for code in group.option_codes:
            if code in group_kept:
                requirement_options_rows.append({
                    "requirement_group_id": group.group_id, "course_code": code,
                    "credits": group.option_credits.get(code, _DEFAULT_OPTION_CREDITS),
                })
        for code in sorted(group_kept - set(group.option_codes)):
            requirement_options_rows.append({
                "requirement_group_id": group.group_id, "course_code": code,
                "credits": group.option_credits.get(code, _DEFAULT_OPTION_CREDITS),
            })

    fixed_tables: dict[str, list[dict[str, Any]]] = {
        "catalog_years": [{
            "label": program["catalog_year"], "start_year": program["start_year"],
            "end_year": program["end_year"], "is_archived": program["is_archived"],
        }],
        "programs": [{
            "id": base.program_id, "catalog_year": program["catalog_year"],
            "name": program["name"], "degree_type": program["degree_type"],
            "program_type": program["program_type"], "campus": program["campus"],
            "total_credits_min": program["total_credits_min"],
        }],
        "requirement_groups": requirement_groups_rows,
        "requirement_options": requirement_options_rows,
        # BOTH sides must survive the trim — an alias whose PRIMARY course got left out (e.g.
        # it was only a candidate in an elective menu that lost the budget cut) is a dangling
        # `alias_of` reference, the same class of integrity bug a broadened course's own
        # missing prerequisite would be.
        "course_aliases": [
            row for row in base.course_aliases_rows
            if row["course_code"] in codes and row["alias_of"] in codes
        ],
        "program_notes": base.program_notes_rows,
    }

    def _course_row(code: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "course_code": code,
            "title": courses_by_code[code]["title"] or code,
            "credit_hours_min": courses_by_code[code]["credit_hours_min"],
            # Crawl attributes UNIONED with the parsed University Core tags — `UCC: HUM` and
            # friends. This is what makes "which of these courses satisfies Human Cultures?"
            # answerable from the rendered database instead of from a 24,000-row menu.
            "attributes": sorted(set(_attributes(courses_by_code[code]["attributes_raw"]))
                                 | {f"UCC: {t}" for t in courses_by_code[code].get("ucc_tags", ())}),
        }
        # PREREQ FIELDS INLINED, not a separate `course_prerequisites` table (removed
        # 2026-08-06 — see catalog_export.py's module docstring): present ONLY for a course
        # that actually has a prerequisite row, so a prereq-less course's line stays exactly as
        # short as the four keys above make it.
        prereq_row = prereq_by_code.get(code)
        if prereq_row:
            row["prereq_groups"] = prereq_row["prereq_groups"] or []
            row["coreq_codes"] = prereq_row["coreq_codes"] or []
            row["confidence"] = prereq_row["confidence"]
        return row

    tables: dict[str, list[dict[str, Any]]] = {
        **fixed_tables,
        "courses": [_course_row(code) for code in ordered],
    }
    # SYNTHETIC, NOT IN `TABLES` — never printed by `render_context` (which only ever loops
    # `TABLES.items()`), never hashed into `db_hash`/the static prompt hash, and skipped
    # entirely for a non-engineering program instead of an empty/absent row. `prompts.py` reads
    # this by name to decide whether to say anything about the FYE deadline at all — see
    # `is_college_of_engineering`'s docstring for why that gate must be exact.
    if base.college_of_engineering:
        tables["_fye_meta"] = [{
            "college_of_engineering": True,
            # Only codes this scenario's own course universe actually contains — a code trimmed
            # out by the broadening budget must not be named in an instruction the model cannot
            # legally schedule.
            "fye_course_codes": sorted(base.fye_course_codes & codes),
        }]
    assert set(TABLES) <= set(tables), "real_db table set drifted from catalog_export.TABLES"

    digest = hashlib.sha256()
    files: dict[str, str] = {}
    for table in TABLES:
        raw = json.dumps(tables[table], sort_keys=True, default=str).encode("utf-8")
        digest.update(raw)
        files[table] = hashlib.sha256(raw).hexdigest()[:16]

    return CatalogDatabase(
        path=Path(f"postgres://{base.program_id}"), tables=tables,
        db_hash=digest.hexdigest()[:16], files=files,
        # THE UNTRIMMED universe, so a completed course that did not survive the prompt trim
        # still has its credits when scoring counts what the student already holds. Deliberately
        # NOT part of `db_hash`/`files` above: it is not shown to the model and does not change
        # the experiment, so folding it into the hash would invalidate every prior run's static
        # hashes over a field none of them rendered.
        all_credits={code: float(row.get("credit_hours_min") or 0)
                     for code, row in base.courses_by_code.items()},
    )


def build_scenario_database(
    base: RealDatabaseBase,
    *,
    gen_ed_preference: str | None = None,
    world_language: tuple[str, int] | None = None,
    budget_tokens: int = 14336,
) -> CatalogDatabase:
    """The database ONE scenario is actually shown: `base`'s always-required courses, plus a
    preference-ranked, budget-fit selection from every elective/gen-ed menu and the
    auto-broadened major electives. No Postgres access — everything it needs was already
    fetched and prerequisite-closed by `fetch_real_db_base`.

    EVERY `choose_credits` GROUP GETS AT LEAST ONE OPTION, unconditionally, before the budget
    loop runs — the requirement itself is never allowed to render with zero ways to satisfy it,
    no matter how tight the budget. World language is the one exception with its own guarantee:
    given `world_language=(subject, level)`, both real paths for that subject
    (`_language_path_options`: the sequence courses, plus alternatives for the 3rd credit) are
    added outright, and every OTHER language-only group is excluded rather than ranked — a
    Spanish-track student doesn't need to see German options.
    """
    kept_by_group: dict[str, set[str]] = {}
    core_codes = set(base.always_codes)
    fill_candidates: list[tuple[str, str]] = []

    # GROUPS THE WORLD_LANGUAGE BLOCK OWNS ENTIRELY — computed up front and SKIPPED by the
    # general ranking loop below, single-subject groups (`group.language_subject` set, e.g. a
    # program whose "Approved Courses by Subject" was NOT merged away — see the gen-ed merge
    # above) and multi-subject ones (matched by code prefix, e.g. this program's own "Language
    # and Culture Electives/Foreign Language Requirement", which never clears the 50%
    # single-subject threshold `fetch_real_db_base` uses for the flag). Running the general
    # loop on these FIRST and narrowing them SECOND was a real bug: the general pass had
    # already picked a top choice and queued the other ~40 languages as budget-fill candidates
    # before the language-specific narrowing ever got a chance to exclude them, so a
    # Spanish-track scenario still ended up offered Arabic, Chinese, French, German options.
    language_handled_gids: set[str] = set()
    if world_language:
        subject, _ = world_language
        for group in base.elective_groups:
            if group.language_subject or any(
                c.split(" ", 1)[0] == subject for c in group.option_codes
            ):
                language_handled_gids.add(group.group_id)

    for group in base.elective_groups:
        if group.group_id in language_handled_gids:
            continue
        ranked = _rank_gen_ed(group.option_codes, gen_ed_preference, base)
        if not ranked:
            kept_by_group[group.group_id] = set()
            continue
        top, rest = ranked[0], ranked[1:]
        kept_by_group[group.group_id] = {top}
        core_codes.add(top)
        fill_candidates += [(code, group.group_id) for code in rest]

    if world_language:
        subject, level = world_language
        for group in base.elective_groups:
            if group.group_id not in language_handled_gids:
                continue
            source_codes = (
                group.option_codes if group.language_subject == subject
                else [c for c in group.option_codes if c.split(" ", 1)[0] == subject]
            )
            picked = (set(_language_path_options(base, subject, level, source_codes)[0])
                      if source_codes else set())
            kept_by_group[group.group_id] = picked
            core_codes |= picked

    # ROUND-ROBIN across groups: breadth (one more option per requirement) beats depth (one
    # group's whole ranked list) when the budget is tight, so a squeeze thins every menu a
    # little rather than emptying whichever groups happen to sort last.
    by_group: dict[str, list[str]] = {}
    for code, gid in fill_candidates:
        by_group.setdefault(gid, []).append(code)
    interleaved: list[tuple[str, str]] = []
    while any(by_group.values()):
        for gid, queue in by_group.items():
            if queue:
                interleaved.append((queue.pop(0), gid))

    spent_codes = set(core_codes)
    for code, gid in interleaved:
        # A CODE CAN BE A FILL CANDIDATE FOR MORE THAN ONE GROUP — the same course often
        # satisfies more than one open menu (a History course sitting in both "History (HIST)"
        # and the broader "Culture Course List"). `already_present` still gets a REAL budget
        # probe below (an extra `requirement_options` row for the second group costs a few
        # tokens even though `courses` doesn't grow), but on rejection only `kept_by_group[gid]`
        # is rolled back — `spent_codes` must never lose a code another group still depends on.
        # A rollback that discarded it unconditionally was a real bug: it silently deleted a
        # DIFFERENT group's already-guaranteed top pick, leaving `requirement_options` pointing
        # at a course missing from `courses`.
        already_present = code in spent_codes
        kept_by_group.setdefault(gid, set()).add(code)
        spent_codes.add(code)
        tokens = _approx_tokens(render_context(_partial_database(base, spent_codes, kept_by_group)))
        if tokens > budget_tokens:
            kept_by_group[gid].discard(code)
            if not already_present:
                spent_codes.discard(code)

    # Broadened major electives fill whatever budget remains, cheapest (lowest course number)
    # first — unchanged in spirit from the pre-refactor version: what survives is a PREFIX of
    # `base.broadened_codes`, so the cut stays legible ("everything past CS 4xxxx got
    # trimmed") rather than a scattered hole in the middle of the list.
    kept_broadened: list[str] = []
    for code in base.broadened_codes:
        trial = spent_codes | {code}
        tokens = _approx_tokens(render_context(_partial_database(base, trial, kept_by_group)))
        if tokens > budget_tokens:
            break
        spent_codes = trial
        kept_broadened.append(code)

    if len(kept_broadened) < len(base.broadened_codes):
        logger.info(
            "real_db: kept %d/%d broadened %s elective(s) within the %d-token budget "
            "(dropped everything past %s)",
            len(kept_broadened), len(base.broadened_codes), "/".join(base.broaden_subjects),
            budget_tokens, kept_broadened[-1] if kept_broadened else "(none)",
        )

    # IN-MEMORY PREREQUISITE CLOSURE — everything any of this could reference was already
    # fetched by `fetch_real_db_base`, so this never touches Postgres.
    final_codes = set(spent_codes)
    for _ in range(4):
        referenced = {
            code for c in final_codes
            for grp in (base.prereq_by_code.get(c, {}).get("prereq_groups") or [])
            for code in grp
        } | {
            code for c in final_codes for code in (base.prereq_by_code.get(c, {}).get("coreq_codes") or [])
        }
        missing = {c for c in referenced if c in base.courses_by_code} - final_codes
        if not missing:
            break
        final_codes |= missing

    return _partial_database(base, final_codes, kept_by_group)


def resolve_poid(pg_url: str, poid: str) -> str:
    """Purdue's own `programs.poid` -> this crawl's surrogate UUID.

    Fixtures and scenarios name extra programs by POID because it is the catalog's identifier
    and survives a re-crawl; every loader below wants the UUID. Raises rather than returning
    None: a second major that silently fails to resolve would score as "this student owes
    nothing extra", which looks like a passing plan instead of a broken configuration.
    """
    with psycopg.connect(pg_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute("SELECT id::text AS id, name FROM programs WHERE poid = %s", (str(poid),))
        rows = cur.fetchall()
    if not rows:
        raise LookupError(
            f"no program with poid {poid!r} in this crawl — it may not have been crawled yet "
            f"(the 2026-2027 sync runs ~2 min/program), or the poid is wrong."
        )
    return rows[0]["id"]


_CREDIT_ANNOTATION_RE = re.compile(
    r"\s*\(\s*[\d.]+(?:\s*[-–]\s*[\d.]+)?\s*(?:credit|credits|cr)\.?\s*\)\s*$", re.IGNORECASE
)


def _requirement_name_key(name: str) -> str:
    """Two crawled group names that denote the same requirement, as one comparable key.

    Programs stamp their own credit figure onto a shared requirement's name ("Mathematics" vs
    "Mathematics (8-10 credits)"), so the annotation is stripped and the rest is casefolded with
    runs of whitespace collapsed. Nothing else is normalised: the point is to catch one program
    annotating what another left bare, not to guess that differently-worded requirements match.
    """
    return " ".join(_CREDIT_ANNOTATION_RE.sub("", name).split()).casefold()


def merge_real_db_bases(
    primary: RealDatabaseBase, extras: list[tuple[RealDatabaseBase, str, str]],
) -> RealDatabaseBase:
    """One student pursuing several programs at once, as a single database to plan against.

    `extras` is `(base, kind, label)` per additional program — a second major or a minor. Their
    requirement groups are added to the primary's, each renamed `"<label>: <group>"` and
    re-keyed by program so two programs' identically-named groups ("Algebra" appears in both the
    Math major and the Math minor; "Statistics" in nearly everything) stay distinct instead of
    silently overwriting one another.

    NO DEDUPLICATION OF REQUIREMENTS, deliberately, and no attempt to net out overlap between
    the programs. Purdue does cap how much a minor may share with a major, but this harness
    models unlimited double counting by decision (2026-08-11): a course satisfies every list it
    appears in. What stops that from inventing a free degree is that credits are counted per
    DISTINCT COURSE toward the graduation minimum — see `real_scoring`'s `total_credits`. A plan
    can therefore close every group of all three programs and still fail on credits, which is
    the honest failure mode; the reverse (inventing an overlap cap the catalog does not state)
    would fail plans that are actually legal.

    Courses, prereqs and aliases are unioned. Where two programs describe the same course code,
    the PRIMARY program's row wins — same course, and the primary is the one whose scenarios,
    completed list and credit targets the fixture was written around.
    """
    always = list(primary.always_groups)
    elective = list(primary.elective_groups)
    courses = dict(primary.courses_by_code)
    prereqs = dict(primary.prereq_by_code)
    aliases = list(primary.course_aliases_rows)
    notes = list(primary.program_notes_rows)

    seen_aliases = {(r["course_code"], r["alias_of"]) for r in aliases}
    # THE UNIVERSITY CORE IS THE UNIVERSITY'S, NOT EACH PROGRAM'S. Every program this loader
    # touches grows its own synthesized `ucc-*` groups (see `UCC_COMPETENCIES`), so a student
    # doing two majors would otherwise be told to satisfy Quantitative Reasoning twice — nine
    # duplicate requirements per extra program, each a real scoring target the student does not
    # actually owe, and a large amount of prompt spent saying the same thing again. These are
    # the one class of group where "same name means same requirement" is not a guess: this
    # module generates them, from one university-wide competency list. Crawled groups are NOT
    # deduplicated on name — two programs' "Statistics" may be genuinely different requirements,
    # and merging them on a name match would erase real work.
    existing_ucc = {g.group_id for g in always + elective if g.group_id.startswith("ucc-")}
    # SHARED COLLEGE REQUIREMENTS, likewise counted once. Purdue replicates a college's core onto
    # every program page inside that college, so two College of Science majors each crawl their
    # own copy of "Great Issues in Science", "Laboratory Science", "Technical Writing &
    # Presenting" and seven more — MEASURED on CS + Data Science, 10 of Data Science's 15 groups
    # are byte-identical in name to one of CS's. A student in both majors completes that core
    # once.
    #
    # SAFE UNDER THIS HARNESS'S DOUBLE-COUNTING RULE, which is what makes a name match
    # defensible here even though `merge_real_db_bases` refuses name matching in general: a
    # course satisfies every list it appears in, so if two identically-named groups really were
    # different requirements, any course closing one already closes the other. Collapsing them
    # therefore cannot change which courses the student must take — only how many times the
    # prompt says so. What it does buy is real: without it the three-program prompt is 14.7k
    # tokens against a 14,336 budget and the run will not start.
    #
    # MATCHED ON `_requirement_name_key`, NOT BYTE-FOR-BYTE, since 2026-08-12. Programs annotate
    # the same core requirement with their own credit figure — CS crawls "Mathematics" and Data
    # Science crawls "Mathematics (8-10 credits)"; likewise "Statistics" and "Statistics (3
    # credits)" — and an exact match let both through, so a CS + Data Science student was shown
    # Mathematics twice and owed two separate math requirements for one body of coursework. The
    # annotation is the only difference between those names, and the credit figure it carries is
    # the SECOND program's, i.e. exactly the number the primary's own group already answers.
    existing_names = {_requirement_name_key(g.name) for g in always + elective}
    for base, kind, label in extras:
        prefix = f"{label} ({kind})"
        for bucket, source in ((always, base.always_groups), (elective, base.elective_groups)):
            for group in source:
                key = _requirement_name_key(group.name)
                if group.group_id in existing_ucc or key in existing_names:
                    continue
                existing_names.add(key)
                bucket.append(replace(
                    group,
                    group_id=f"{base.program_id}:{group.group_id}",
                    name=f"{prefix}: {group.name}",
                ))
        for code, row in base.courses_by_code.items():
            courses.setdefault(code, row)
        for code, row in base.prereq_by_code.items():
            prereqs.setdefault(code, row)
        for row in base.course_aliases_rows:
            key = (row["course_code"], row["alias_of"])
            if key not in seen_aliases:
                seen_aliases.add(key)
                aliases.append(row)

    return replace(
        primary, always_groups=always, elective_groups=elective, courses_by_code=courses,
        prereq_by_code=prereqs, course_aliases_rows=aliases, program_notes_rows=notes,
    )


def load_real_db(
    pg_url: str,
    program_id: str,
    *,
    broaden_subjects: tuple[str, ...] | None = None,
    force_selective_groups: tuple[str, ...] = (),
    manual_course_aliases: Mapping[str, str] | None = None,
    budget_tokens: int = 14336,
    gen_ed_preference: str | None = None,
    world_language: tuple[str, int] | None = None,
    scope_filter: str | None = None,
) -> CatalogDatabase:
    """One-shot convenience wrapper: `fetch_real_db_base` plus `build_scenario_database` in one
    call, for callers that don't need per-scenario preferences (`run.py check`/`fixture-check`,
    ad hoc scripts). Scenario-aware callers — the runner, for Mode B/C — should call the two
    functions separately: fetch once, build once per scenario. See both docstrings.
    """
    base = fetch_real_db_base(
        pg_url, program_id, broaden_subjects=broaden_subjects,
        force_selective_groups=force_selective_groups,
        manual_course_aliases=manual_course_aliases, scope_filter=scope_filter,
    )
    return build_scenario_database(
        base, gen_ed_preference=gen_ed_preference, world_language=world_language,
        budget_tokens=budget_tokens,
    )


def fixture_from_database(database: CatalogDatabase, *, slug: str = "") -> Fixture:
    """A `Fixture` built from a live `CatalogDatabase` — THE ONLY WAY ONE IS BUILT as of
    2026-08-12, when `plan_fixtures/*.yaml` was deleted (see `fixtures.py`'s module docstring).
    Courses and requirement groups come off the same database the model is shown; the students
    come from `fixtures.synthesize_scenarios`, derived from this program's own required courses.

    It started narrower — Mode C (`convergence.py`) scoring against the real program instead of
    a hand-authored file, the way Mode B already did via `real_scoring.score_against_real_db` —
    and the rest of the harness has since moved onto the same path.

    WHY A FIXTURE OBJECT, NOT A `RealScore`-NATIVE REWRITE. Every helper in `convergence.py`
    (`legal_slots`, `auto_repair`, `unmet_candidates`, `placement_hints`, `release_blockers`,
    `surplus_placements`) takes a `Fixture` and calls `plan_scorers.score_plan(fixture, ...)` —
    none of them care where the `Fixture` came from. Building one FROM the live database and
    swapping it in at `convergence.run_case`'s single binding point (`fixture = ctx.fixture`)
    means every one of those functions works completely unchanged, instead of rewriting a dozen
    call sites to a `RealScore`-shaped interface. See `real_scoring.py`'s own module docstring
    for the bug this closes: the STATIC fixture's `requirement_groups` includes three groups
    (`science`, `gen-ed-core`, `gen-ed-selective`) whose course codes were invented when the
    fixture was hand-authored and do not exist in the real database the model is actually
    shown — capping every well-behaved model's Mode C coverage at 6/9 = 66.7%, regardless of
    how well it plans, because the prompt (correctly) tells it never to invent a course code
    for exactly that category of requirement.

    KNOWN, ACCEPTED SIMPLIFICATION: `Course.prereqs` is a flat AND list — the same
    simplification the hand-authored fixture YAML already makes for every course in it (see
    `fixtures._course`). The real `prereq_groups` is AND-of-OR (`[["CS 25000"], ["CS 25100",
    "CS 25300"]]`); this takes the first alternative of each AND-slot (`[g[0] for g in
    prereq_groups]`). Loses nothing for a program with no real OR-alternatives in its
    prerequisites (true of Machine Intelligence's data as of this writing — every
    `prereq_groups` entry here has exactly one course per AND-slot). A program where that
    stops being true would need `plan_scorers`/`Course` to understand AND-of-OR directly; out
    of scope here.

    `offered_terms` is EVERY term, not empty. There is no term-offering data in this export at
    all as of 2026-08-06 (see `catalog_export.py`'s module docstring) and the prompt tells the
    model so in as many words — "place courses by prerequisite order only; do not assume or
    invent a term restriction for any course". The deterministic planner does not read a field
    defensively the way the scorers do: `planner.generate_plan` selects a course only when
    `term in course.offered_terms`, so an empty tuple there means EVERY course is unschedulable
    in every term and Mode A produces an empty plan for every student (measured: 14% coverage,
    zero violations, on a program whose requirements are entirely satisfiable). Stating "any
    term" is what the rest of the harness already asserts; leaving it empty states "no term",
    which is a different claim and a false one.
    """
    alias_of = {row["course_code"]: row["alias_of"] for row in database.rows("course_aliases")}

    catalog: list[Course] = []
    for row in database.rows("courses"):
        code = row["course_code"]
        catalog.append(Course(
            code=code,
            title=row.get("title", ""),
            credits=int(row.get("credit_hours_min") or 0),
            prereqs=tuple(
                group[0] for group in (row.get("prereq_groups") or []) if group
            ),
            coreqs=tuple(row.get("coreq_codes") or ()),
            offered_terms=tuple(ALL_TERMS),
            equivalent_to=alias_of.get(code, ""),
        ))

    # Same "skip groups with zero options" filter `real_scoring._groups_from_tables` uses —
    # this is what makes the science/gen-ed-core/gen-ed-selective ceiling disappear even
    # before any crawler work lands: those groups simply have no `requirement_options` rows
    # today, so they are never scored, and (since 2026-08-12) never shown either — a group with
    # no options is not in the export at all.
    # Once a category is resolved (see `parse.requirements.resolve_requirement_links`), it
    # gains real options and starts counting automatically — no further change needed here.
    options_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in database.rows("requirement_options"):
        options_by_group.setdefault(row["requirement_group_id"], []).append(row)

    requirement_groups: list[dict[str, Any]] = []
    for row in database.rows("requirement_groups"):
        options = options_by_group.get(row["id"])
        if not options:
            continue
        selective = row["requirement_type"] == "choose_credits"
        group: dict[str, Any] = {
            # `name`, NOT the row's uuid `id` — `plan_scorers.requirement_coverage`'s
            # "STILL MISSING" text uses this field as the model-facing LABEL
            # (`f"{group['id']}: missing ..."`), and nothing reads it as a database key
            # (confirmed: the only two uses in plan_scorers.py are both string
            # interpolation). A raw uuid there would make every hint unreadable.
            "id": row["name"],
            "name": row["name"],
            "kind": "choose" if selective else "all",
            "courses": [opt["course_code"] for opt in options],
        }
        if selective:
            group["choose_credits"] = row.get("credits_min") or 0
        requirement_groups.append(group)

    program_row = database.rows("programs")[0] if database.rows("programs") else {}
    name = program_row.get("name") or "(unknown program)"
    return Fixture(
        name=name,
        path=database.path,
        fixture_hash=database.db_hash,
        # TRUE, unlike the hand-authored fixtures this replaced. `verified: false` meant "the
        # prerequisite edges in this file were typed in by hand" — every edge here is crawled
        # (Banner course-detail pages, carrying a `confidence`), and every course, credit and
        # requirement group is the catalog's own.
        verified=True,
        catalog=catalog,
        requirement_groups=requirement_groups,
        scenarios=synthesize_scenarios(name, catalog, requirement_groups,
                                       canonical=_canonical_map(alias_of)),
        slug=slug or program_slug(name),
        real_db_program_id=str(program_row.get("id") or ""),
    )


_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _canonical_map(alias_of: dict[str, str]) -> dict[str, str]:
    """code -> the PRIMARY course it counts as, following alias chains, same rule as
    `Fixture.canonical`. Needed BEFORE the `Fixture` exists, because the students are
    synthesized as part of building it — see `select_remaining_courses`'s `canonical`."""
    out: dict[str, str] = {}
    for code in alias_of:
        target, seen = code, {code}
        while alias_of.get(target) and alias_of[target] not in seen:
            target = alias_of[target]
            seen.add(target)
        out[code] = target
    return out


def program_slug(name: str) -> str:
    """A short, filesystem-safe name for `results/results_<slug>/`.

    Derived from the program's own catalog name rather than authored, so a program nobody has
    ever run before still gets a stable directory: "Computer Science, BS - Machine Intelligence
    concentration" -> `computer-science-bs-machine-intelligence`. Truncated on a word boundary,
    because these become directory names a human reads in a listing.
    """
    words = [w for w in _SLUG_STRIP.sub("-", name.lower()).split("-") if w]
    out: list[str] = []
    for word in words:
        if sum(len(w) + 1 for w in out) + len(word) > 48:
            break
        out.append(word)
    return "-".join(out) or "program"


def list_programs(pg_url: str, *, plannable_only: bool = True) -> list[dict[str, Any]]:
    """Every program in the catalog, newest catalog year first, with its requirement-option
    count — the pool `run.py`'s sweep draws its random picks from.

    `plannable_only` drops the programs Mode B cannot run at all: a program whose requirement
    groups are all prose or all containers has zero `requirement_options` rows, and
    `fetch_real_db_base` raises `LookupError` on it by design ("there is nothing for Mode B to
    plan from"). Picking one at random would spend a sweep slot on a guaranteed failure.
    """
    with psycopg.connect(pg_url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id::text AS id, p.name, p.degree_type, cy.label AS catalog_year,
                   (SELECT count(*) FROM requirement_options ro
                     JOIN requirement_groups rg ON rg.id = ro.requirement_group_id
                    WHERE rg.program_id = p.id) AS n_options
            FROM programs p
            JOIN catalog_years cy ON cy.id = p.catalog_year_id
            ORDER BY cy.label DESC, p.name ASC
            """
        )
        rows = list(cur.fetchall())
    return [r for r in rows if r["n_options"] > 0] if plannable_only else rows


def resolve_program(pg_url: str, search_text: str) -> dict[str, Any]:
    """`--major <text>` -> exactly one program row, or a SystemExit-worthy LookupError listing
    the candidates. The non-interactive counterpart to `resolve_major_interactive`: a sweep runs
    unattended, so an ambiguous match must fail loudly rather than block on stdin.

    Matching, in order: exact name, exact slug (`program_slug`), then case-insensitive
    substring over both. Exactness wins outright so that a full program name is never reported
    ambiguous against the longer names that contain it ("Biology" vs "Biology: Ecology,
    Evolution and Environmental Science Concentration").

    SEVERAL ROWS CAN SHARE A NAME — the same program in two catalog years, and sometimes two
    rows in one year where the crawl caught a stub page as well as the real one ("Mechanical
    Engineering" exists with 87 requirement options and with 1). The widest one wins, then the
    newest: a stub with a single option is not the program anybody means, and picking it would
    hand every model an empty degree to plan.
    """
    search = search_text.strip()
    lowered = search.lower()
    programs = list_programs(pg_url)
    if not programs:
        raise LookupError("no plannable programs in the catalog — has the crawl run?")

    exact = [row for row in programs
             if row["name"].lower() == lowered or program_slug(row["name"]) == lowered]
    if exact:
        return max(exact, key=lambda r: (r["n_options"], r["catalog_year"]))
    matches = [r for r in programs
               if lowered in r["name"].lower() or lowered in program_slug(r["name"])]
    # Same collapse for a substring match that landed on one program duplicated across catalog
    # years: several rows, one program, no real ambiguity to report.
    if len({r["name"] for r in matches}) == 1:
        return max(matches, key=lambda r: (r["n_options"], r["catalog_year"]))
    if not matches:
        raise LookupError(f"--major {search_text!r} matched none of the {len(programs)} "
                          f"plannable programs in the catalog")
    if len(matches) > 1:
        listing = "\n".join(f"  - {r['name']}  [{program_slug(r['name'])}]"
                             for r in matches[:20])
        more = f"\n  ... and {len(matches) - 20} more" if len(matches) > 20 else ""
        raise LookupError(
            f"--major {search_text!r} matched {len(matches)} programs — be more specific "
            f"(the slug in brackets is always unique):\n{listing}{more}")
    return matches[0]
