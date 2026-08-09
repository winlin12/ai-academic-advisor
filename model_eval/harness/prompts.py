"""Prompt construction, mirroring the app's four live LLM call sites.

WHAT CHANGED FROM THE OLD HARNESS. It measured text-to-SQL, because the advisor used to hand
the model a schema and ask for a query. It doesn't any more: ``services/rag/pipeline.py`` does
retrieval itself (exact course-code SQL + pgvector cosine search) and the model only ever sees
already-retrieved chunks. Every SQL prompt in here has been replaced with the tasks the app
actually performs:

    plan_proposal   -> advisor_agent.revise_plan   (structured PlanEditProposal)   MODE A
    plan_freeform   -> no app call site (yet)                                      MODE B
    grounded_qa     -> rag.pipeline.answer_question
    explain_plan    -> routers/advisor.explain_plan

MODE B has no production counterpart on purpose. It asks the model to emit the whole schedule
itself — the thing the architecture currently refuses to let it do. It exists to answer "what
would we gain (or lose) by trusting the model with the schedule?", and it is the only task on
which models separate sharply, because Mode A's deterministic planner hides most of the
difference between a good model and a lazy one.

STRUCTURAL INVARIANT, unchanged from the old harness:

    [ SYSTEM = STATIC BLOCK — byte-identical for every model, every item, every run ]
    [ USER   = VARIABLE TAIL — the scenario/question and its data                   ]

Nothing non-deterministic may enter the static block: no timestamps, no RNG, no dict-ordering
roulette (all JSON dumped with sort_keys=True). sha256 of the static block is recorded with
every run, and the report refuses to pool records whose hashes disagree.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .fixtures import Fixture, Scenario
from .catalog_export import CatalogDatabase, render_context, render_context_lean
from .plan_scorers import normalize_code
from .planner import Course, Plan, Profile, first_planning_term, next_term


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _static_hash(text: str, schema: dict[str, Any] | None = None) -> str:
    """Hash of everything static the model is conditioned on — INCLUDING the response schema.

    The schema is not in the prompt text, so it is easy to think of it as separate. It is not:
    llama.cpp compiles it to a GBNF grammar and constrains decoding with it, so it shapes the
    output at least as hard as the system prompt does. Hashing only the prose let a schema edit
    slip past ``report``'s "records whose hashes disagree must never be pooled" rule — and the
    maxLength added to `rationale` on 2026-07-29 is exactly such an edit: it changes what every
    verbose model emits while leaving the prose byte-identical.

    Note the term enum applied by ``plan_schema()`` is NOT covered here; it comes from
    ``run.planning_terms``, which travels in meta_*.json and is checked there.
    """
    if schema is None:
        return _sha256(text)
    return _sha256(f"{text}\n{json.dumps(schema, sort_keys=True)}")


# =============================================================================================
# MODE A — revise-plan proposal. Copied in spirit and in wording from
# backend/app/services/advisor_agent.py::_SYSTEM_PROMPT so the harness measures the prompt the
# app really ships. If that prompt changes, change this one and expect the hash to move.
# =============================================================================================

PROPOSAL_SYSTEM = (
    "You are an assistant that tunes a college course plan from a student's feedback. You are "
    "NOT an official advisor and you must not invent courses, prerequisites, or requirements. "
    "A deterministic planner ENFORCES legality (prerequisites, term offerings, credit caps, "
    "and a limit on how many of the major's own courses share a semester); "
    "you only express preferences over the courses already listed. "
    "It will not put more than the stated number of major-subject courses in one semester, so "
    "reordering several of them to the front moves them earlier in the ORDER — it does not "
    "stack them into the same term. "
    "Only use course codes and tags that appear in the context. Leave a list empty if it does "
    "not apply. Never put a course in both reorder and defer. "
    # LAST, AND SELF-CONTAINED, both deliberately. This used to open "The credit cap is the
    # exception:" and sit immediately after the major-course sentence — which gave "the
    # exception" a second thing it could be an exception TO, and put the one field the model
    # must actively set right next to two sentences about what the planner handles for it.
    # gemma4-e4b went 2/3 -> 0/3 on the explicit-cap scenario when that text landed while every
    # other model held at 3/3, which is the same "not your job" reading that made this
    # paragraph necessary in the first place. Naming the field as the model's own job, with
    # nothing between it and the end of the prompt, is what this wording is for.
    "One thing IS yours to set: when the student asks for a specific per-semester credit load, "
    "you MUST set max_credits_per_semester to that number. Writing it only in the rationale "
    "has no effect."
)

# Mirrors app.models.schemas.PlanEditProposal. The app sends this schema to llama.cpp as a
# grammar-constrained response_format; so does the harness. Field descriptions are part of the
# prompt the model sees, so they are reproduced verbatim.
PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # maxLength MIRRORS app.models.schemas.PlanEditProposal's Field(max_length=400) — the
        # app derives its grammar from that model via model_json_schema(), so the two stay in
        # sync only if both carry the number. See that field's comment for what it is for; the
        # short version is that llama.cpp's grammar bounds the SHAPE of the JSON and not the
        # length of a string, and one model spent 390 s per record proving it.
        "rationale": {
            "type": "string", "maxLength": 400,
            "description": "One or two sentences explaining the change, addressed to the student.",
        },
        "reorder": {
            "type": "array", "items": {"type": "string"},
            "description": "Course codes to take earlier, highest priority first.",
        },
        "defer": {
            "type": "array", "items": {"type": "string"},
            "description": "Course codes to push to later semesters.",
        },
        "avoid_tags": {
            "type": "array", "items": {"type": "string"},
            "description": "Requirement tags to deprioritise (e.g. 'theory-heavy').",
        },
        "max_credits_per_semester": {
            "type": ["integer", "null"], "minimum": 1, "maximum": 24,
            "description": (
                "Set this to the number of credits the student asked for whenever they name a "
                "per-semester load (e.g. 'keep me at 12 credits'). Null only if they did not ask."
            ),
        },
    },
    "required": ["rationale", "reorder", "defer", "avoid_tags"],
}


# Turning the enums on or off changes what the model can physically emit, so it has to move the
# arm's identity. The RESOLVED enum values are per student (they come from that student's
# remaining courses), which belongs to the variable tail — so the policy name is hashed, not the
# values, and the static hash stays identical across scenarios the way the invariant requires.
# `;requirements_covered<-groups` was appended 2026-07-31 and removed again 2026-08-02 with the
# key itself (see plan_schema). It had to be named here while it existed because it was injected
# by `plan_schema()` at call time and so was invisible to the PLAN_SCHEMA template that
# `_static_hash` actually hashes. Nothing replaces it: `total_credits` and `major_course_count`
# went in the same commit and those ARE in the template, so Mode B's hash moves on their removal
# without help. Anything added back to the grammar at call time must be named here again.
#
# NOTE this string is shared with Mode A, whose grammar did not change — its static hash moves
# once here anyway (c45d38252a6a236b -> 223926a625f420ec) because the Mode B clause was living in
# the shared string. Mode A records do not pool across this commit; the format is the same.
SCHEMA_ENUM_POLICY = "enums:reorder,defer,avoid_tags,courses<-context"


def proposal_schema(remaining_courses: list[str], known_tags: list[str]) -> dict[str, Any]:
    """Mode A's schema with the code and tag fields restricted to what actually exists.

    THE POINT IS THAT THE MODEL CANNOT MISS. Free-form strings let a model write
    'FALL 2026:CS25000,CS25200,MA26100,MA26500[14CR]' into a field that expects one course code,
    and 16% of parsed proposals across three runs were ungrounded exactly that way — whole
    semester layouts in `reorder`, invented tags like 'seminar' and 'optional' in `avoid_tags`.
    The app degrades all of it to a no-op, so it costs the student their request silently.

    An enum makes those strings undecodable rather than merely discouraged, and it costs nothing
    at inference — constrained decoding over a 40-item enum is cheaper than free-form text. Same
    argument ``plan_schema`` already makes for the term enum.

    Empty lists are left as free strings on purpose: an enum with no members is not a legal
    grammar, and a student with nothing left to schedule has no proposal to constrain anyway.
    """
    schema = json.loads(json.dumps(PROPOSAL_SCHEMA))
    if remaining_courses:
        codes = {"type": "string", "enum": list(remaining_courses)}
        schema["properties"]["reorder"]["items"] = codes
        schema["properties"]["defer"]["items"] = json.loads(json.dumps(codes))
    if known_tags:
        schema["properties"]["avoid_tags"]["items"] = {
            "type": "string", "enum": sorted(known_tags)}
    return schema


def catalog_tags(catalog: list[Course]) -> list[str]:
    return sorted({tag for course in catalog for tag in course.requirement_tags})


# =============================================================================================
# MODE B — free-form plan of study.
# =============================================================================================

# STRIPPED BACK TO ESSENTIALS, 2026-07-31. This prompt had grown to ~60 lines of rules,
# rationale and worked reasoning, each addition justified by a failure it was meant to fix —
# and it was sitting on top of an ~11,000-token database export, where every extra line of
# prose competes with the data for the model's attention. It had also silently accumulated a
# VERBATIM DUPLICATE of the whole "two things that make the ordering easier" paragraph, which
# is the clearest possible sign that nobody could hold the thing in their head any more.
#
# What is below is the scored rules and nothing else: one line per constraint, in the order the
# scorer applies them, plus the minimum needed to navigate the export (which table holds what).
# The reasoning that used to be in the prompt has moved into these comments, where it belongs —
# the model never needed to know WHY a rule exists, only what it is.
#
# Anything removed here is recoverable from git. If a violation class regresses in the next
# sweep, add back the ONE line that addresses it and measure again, rather than restoring the
# essay.
#
# TWO LINES ADDED BACK, 2026-07-31 (this moved the static hash), under exactly that rule:
#
#   * PREREQUISITES first and expanded. Prerequisite ordering is the largest Mode B violation
#     class in every sweep so far, and it is the one constraint no grammar can express — the
#     course enum kills hallucinations and the required per-group keys force coverage, but
#     nothing structural can stop a model putting CS 25100 above CS 18200. Prose is the only
#     instrument left for it, so it gets the top of the list and the "strictly earlier, same
#     semester is not early enough" wording that the scorer actually applies.
#   * EVEN DISTRIBUTION, promoted from one sentence about the target to its own paragraph.
#     The measured shape (results_old9, 162 plans) was a monotonic slide from 13.7 credits in
#     the first semester to 7.1 in the eighth, so "aim for the target" was not landing on the
#     back half of the plan; naming the shape directly is the cheaper fix to try first.
#     The elective menu was widened in the fixture on the same day, which is what makes the
#     instruction actionable — a model told to fill a light term now has options to fill it
#     with. Read the two changes together; they are one intervention.
#   * NO REPEATS, split into two rules and promoted, same day. This was one line ("Never repeat
#     a course or schedule one already completed") carrying two different failures, and the
#     transcripts show it was not holding: gemma4-26b's mi-tight-horizon plan re-scheduled TEN
#     already-completed courses into a single 42-credit semester — a student with 18 courses
#     behind them handed a plan that repeats half of them. The scorer already distinguishes the
#     two cases ("appears more than once" vs "was already completed before the plan starts"), so
#     the prompt now does too, and each says what to CHECK rather than only what not to do.
#     The clause about completed courses still counting toward prerequisites and requirement
#     groups is there because the obvious over-correction — treating "don't schedule it" as
#     "it doesn't exist" — costs coverage, which is the other half of PLAN_VIABLE.
#
# TERM OFFERINGS PROMOTED, 2026-08-02 (moves the static hash again) — and REMOVED entirely,
# 2026-08-06. It was a single clause carrying a constraint models kept breaking:
# `term_offering_violation` charged for a fact stated on one field of one table. That table
# (`course_planner_terms`) and the rule pointing at it are BOTH gone now — Purdue's published
# offering pattern is not always a reliable predictor of a future term, and the harness was
# scoring plans against it as if it were a hard fact. See `catalog_export.py`'s module
# docstring for the removal and how it turns the constraint off everywhere (prompt AND
# scorer) at once, just by no longer populating `offered_terms`.
PLAN_SYSTEM = """\
You build a semester-by-semester plan of study from a read-only export of the advisor\'s
catalog database. Use only what is in that export.

Where things are:
- `courses` - every course that exists, with credit_hours_min, listed in prerequisite order.
  `prereq_groups` (AND-of-ORs) and `coreq_codes` appear directly on a course's own row, present
  only when that course actually has them.
- `requirement_groups` + `requirement_options` - the degree requirements, joined on
  course_code. `all_of` = take every option. `choose_credits` = take enough to reach
  `credits_min`.
- `unresolved_requirement_groups` - real requirements (e.g. university gen-ed) the catalog
  states in prose, not as a course list. Nothing to schedule here; they are not part of
  `requirement_groups` and "cover every requirement group" does not apply to them. Never invent
  a course code to satisfy one.
- `course_aliases` - `alias_of` is an approved substitute. Take one, never both.

Rules, all hard:
- PREREQUISITES COME FIRST. Every prerequisite of a course must be in a STRICTLY EARLIER
  semester than that course, or already completed. The same semester is NOT early enough.
  Before you place a course, check its own `prereq_groups`: every code in it must already be
  behind it. A course with no `prereq_groups` field has no prerequisites. Only `coreq_codes`
  may share a semester. This is the constraint most plans get wrong, and one mistake anywhere
  invalidates the whole plan.
- NEVER SCHEDULE A COURSE THE STUDENT HAS ALREADY TAKEN. The student\'s "Already completed"
  line is the complete list of what is done; those courses are finished and must not appear
  anywhere in your plan. They still count as satisfying prerequisites and requirement groups —
  a completed course is credit the student already holds, not work still to do. Scheduling one
  again wastes a seat the student is paying for.
- NEVER SCHEDULE THE SAME COURSE TWICE. Each course appears at most once in the WHOLE plan, not
  once per semester. Before you add a course to a semester, check the semesters you have
  already written and the completed list. Substitutes in `course_aliases` count as the same
  course: take one, never both.
- Only courses in `courses`. Never invent a code.
- Cover every requirement group. A group is already covered if the student completed its
  courses — check the completed list before scheduling anything for it.

SPREAD THE WORK EVENLY. Balance CREDIT HOURS, not course count, across ALL the semesters the
student has — courses are not the same size, so an equal number of courses per semester can
still leave credits badly uneven. Add up the credits in each semester as you place courses and
keep that sum near the target every semester, including the last ones. Do not fill the early
terms to the limit and leave the later ones light on credits, and do not leave a term light on
credits when something legal could go in it. Selective groups usually offer more options than
the student needs: when a term has room, pick an option whose prerequisites are already
satisfied."""

# LEAN VARIANT — for a `program.source: fixture` pseudo-major (school-core / gen-ed+world-
# language test cases), whose database is `catalog_export.render_context_lean`'s
# requirement-groups-only export, not `render_context`'s full one. PLAN_SYSTEM's "Where things
# are" section names `courses`/`prereq_groups`/`coreq_codes`/`course_aliases` — tables and
# fields that render_context_lean never emits — so using it unchanged here would describe a
# schema the model is not actually being shown. Composed the same "shared rules, one definition"
# way as PLAN_LOCKED_RULES below: every rule that still applies (never invent a code, never
# re-schedule a completed or already-placed course, cover every group, spread credits evenly)
# is unchanged; only the schema description and the prerequisite-checking rule (there are none
# to check here) differ.
PLAN_SYSTEM_LEAN = """\
You build a semester-by-semester plan of study from a read-only export of the advisor\'s
catalog database. Use only what is in that export.

Where things are:
- `requirement_groups` - the degree requirements for this test case, each with its own
  `options` (course code + credits) listed inline. `kind: "all"` = take every option.
  `kind: "choose"` = take enough options to reach `credits_min`.
- There is no separate course catalog and no prerequisite data here — these requirements have
  no meaningful prerequisite chain to sequence, on purpose (see the header of whichever fixture
  produced this export).

Rules, all hard:
- NEVER SCHEDULE A COURSE THE STUDENT HAS ALREADY TAKEN. The student\'s "Already completed"
  line is the complete list of what is done; those courses are finished and must not appear
  anywhere in your plan. They still count as satisfying requirement groups — a completed course
  is credit the student already holds, not work still to do.
- NEVER SCHEDULE THE SAME COURSE TWICE. Each course appears at most once in the WHOLE plan, not
  once per semester. Before you add a course to a semester, check the semesters you have
  already written and the completed list.
- Only courses named in `requirement_groups`. Never invent a code.
- Cover every requirement group. A group is already covered if the student completed its
  options — check the completed list before scheduling anything for it.

SPREAD THE WORK EVENLY. Balance CREDIT HOURS, not course count, across ALL the semesters the
student has. Add up the credits in each semester as you place courses and keep that sum near
the target every semester, including the last ones. Do not fill the early terms to the limit
and leave the later ones light on credits."""

# =============================================================================================
# MODE C — retry-to-convergence. See harness/convergence.py for what it measures.
# =============================================================================================
#
# COMPOSED ON TOP OF PLAN_SYSTEM, NEVER BY EDITING IT. Mode C's whole claim is that it measures
# the same planning task as Mode B under repetition; if the two prompts drifted, a convergence
# difference could be a prompt difference and there would be no way to tell from the results.
# So the shared rules have exactly one definition and Mode C appends the locked-slot rule,
# which is the only thing about the task that is genuinely different.
#
# The locked slots THEMSELVES are per-scenario and live in the variable tail, alongside the
# student. Only the rule is static.
PLAN_LOCKED_RULES = """\
Some slots in this student\'s plan are LOCKED. A locked slot is a course the student has
already decided to take in a specific semester.

- Locked courses are FIXED. Keep every one exactly where it is listed. Do not move it to
  another semester, do not remove it, do not schedule it a second time somewhere else.
- Plan everything else AROUND them. A locked course still consumes credits and still counts
  toward the major-course limit in the semester it sits in, and its own prerequisites must
  still be scheduled in EARLIER semesters than the one it is locked to.
- Locked courses still satisfy requirement groups and still satisfy prerequisites for later
  courses, exactly as any other scheduled course does.
- Every other slot is yours to fill or leave empty."""


def plan_schema(planning_terms: list[str],
                catalog_codes: list[str] | None = None) -> dict[str, Any]:
    """Mode B/C's response schema, with the term and course enums restricted to what exists.

    Encoding the restriction in the GRAMMAR rather than only in the prose matters: with summer
    excluded, a model literally cannot emit a summer semester, so "did it respect the calendar"
    stops competing for the model's attention with the actual planning problem. The prose rule
    stays too, because the enum alone doesn't explain *why*.

    The same reasoning extends to `courses`. Hallucinations are only 3% of Mode B violations, but
    every one of them is the same category error the proposal fields showed — the model writing
    'CS 37300:DATAMININGANDMACHINELEARNING(3CR)' where a bare code belongs. An enum ends that
    class outright. It cannot touch the other 97% (prereq order, duplicates, credit sums, the
    major cap) — no grammar can express those — which is what Mode C's feedback loop is for.

    REQUIREMENT ACCOUNTING (`requirements_covered`, a required key per group) lived here from
    2026-07-31 to 2026-08-02. The finding that motivated it stands and is worth keeping: coverage
    failures are POSITIONAL. Over 162 Mode B plans, the groups at the end of `requirement_groups`
    were dropped an order of magnitude more often than the ones at the top — mi-required 9 and
    mi-ai-choice 3, against gen-ed-core 84 and gen-ed-selective 78 — and the plans that dropped
    them had scheduled only 66 of 100 available credits. The models were not short of courses or
    of room; they worked down the list, did the major properly, and trailed off.

    WHY IT WENT ANYWAY: it cost roughly 440 output tokens per plan — Mode B's median output went
    600 -> 1040 and its median latency 32 s -> 65 s on gemma4-26b, which is ~40% of the sweep's
    wall clock — and it never produced a number the harness could not compute itself. Coverage is
    recomputed from the semesters against the fixture regardless, and Mode C's repair variant
    already feeds that COMPUTED gap back to the model (see convergence.py's `missing_requirements`
    -> prompts.revise_user). Making the model narrate an accounting we derive anyway is paying
    generation time for a diagnostic.

    If positional drop-off regresses in the next sweep, the fix to try first is prompt ORDER (put
    the groups the models trail off on where they will be attended to), not another required key.
    """
    schema = json.loads(json.dumps(PLAN_SCHEMA))
    semester = schema["properties"]["semesters"]["items"]["properties"]
    semester["term"]["enum"] = list(planning_terms)
    if catalog_codes:
        semester["courses"]["items"] = {"type": "string", "enum": sorted(catalog_codes)}
    return schema


PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semesters": {
            "type": "array",
            "description": "One entry per semester, in order, starting with the student's start term.",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "enum": ["fall", "spring", "summer"]},
                    "year": {"type": "integer"},
                    "courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Course codes exactly as written in the catalog, e.g. 'CS 25100'.",
                    },
                    # `total_credits` and `major_course_count` were required here as a FORCED
                    # SELF-CHECK (make the model state the arithmetic it keeps getting wrong)
                    # and REMOVED 2026-08-02 with `requirements_covered`, for the same reason:
                    # neither was ever read back into the plan — the scorer recomputes both from
                    # `courses` — so the model was spending generation time restating a number
                    # the harness already has. ~40 tokens per plan, ~8 per semester.
                    #
                    # WHAT IS LOST is the `credits_self_reported_ok` diagnostic, which separated
                    # "lost track of its own semester" from "knew the total and busted the cap
                    # anyway". Both now land in the credit-cap violation count undifferentiated.
                    # If that distinction is needed again, it is cheaper to recover from the
                    # rationale text than to reinstate a required key.
                },
                "required": ["term", "year", "courses"],
            },
        },
        "unplanned": {
            "type": "array", "items": {"type": "string"},
            "description": "Required course codes that did not fit in the available semesters.",
        },
        # `rationale` (free text, 2-3 sentences, up to 600 chars) lived here until 2026-08-06,
        # same removal reason as `total_credits`/`major_course_count`/`requirements_covered`
        # above: generation time spent on a field the harness does not need to score the plan.
        # It fed `rationale_flags` (a faithfulness diagnostic — did the explanation mention a
        # hallucinated or misordered course) and the review queue's free-text grading rows;
        # both still run, they just always see an empty string now. If that diagnostic is
        # needed again, this is the property to restore.
    },
    "required": ["semesters"],
}


# =============================================================================================
# Grounded QA + explain-plan (free text, no schema) — mirrors rag/pipeline.py and
# routers/advisor.py respectively.
# =============================================================================================

QA_SYSTEM = (
    "You are a college academic advisor. Answer using ONLY the CONTEXT below, which contains "
    "retrieved degree rules and course descriptions. If the context does not contain the "
    "answer, say you don't have that rule on file and suggest the student confirm with their "
    "department. Be concise, and quote the specific requirement text you relied on."
)

EXPLAIN_SYSTEM = """\
You are an AI academic planning assistant.
You are not an official academic advisor.
You must not invent courses, prerequisites, requirements, or policies.
Explain only from the supplied structured plan.
Always recommend verifying important decisions with an official advisor.
Be concise unless the student asks for depth."""


# =============================================================================================
# Builders
# =============================================================================================


# `prereq_depths`, `catalog_block` and `requirements_block` lived here until 2026-07-30. They
# rendered the catalog and the degree requirements as bespoke prompt prose — a format that
# exists nowhere in production, so Mode B was measuring a call site the app does not have.
# Mode B is now given `harness/catalog_export.render_context()`: the advisor's own database
# rows, in the real table shapes, read straight from Postgres (see harness/real_db.py). They
# are in git history if the old prompt ever needs reproducing.
#
# The `[level N]` prefix went with them, and not only because the format changed: "level 0"
# reads as "freshman year" and "level 3" as "junior year", which is a scheduling constraint the
# number never expressed. The same ordering information now travels as the ORDER of the
# `courses` rows, with `chain_depth` available on each course's own row for the cases where the
# number itself is wanted — and PLAN_SYSTEM says in as many words that it is not a year.


def _elective_preference_line(scenario: Scenario) -> str | None:
    """States the student's own gen-ed/world-language choice EXPLICITLY.

    THIS WAS MISSING, and it was a real bug: `harness/real_db.build_scenario_database` narrows
    WHICH COURSES RENDER for `world_language`/`gen_ed_preference` (a Spanish-track scenario is
    shown Spanish options and no other language's), but nothing ever told the MODEL that. A
    model reading the database export alone sees a shorter-than-usual course list with no
    connection to anything the student asked for — indistinguishable from an arbitrary
    editorial cut, or from every other language simply not existing. Found 2026-08-06 by a
    human reading a transcript and asking "does the model even know it's Spanish."
    """
    parts: list[str] = []
    if scenario.world_language:
        subject, level = scenario.world_language
        parts.append(
            f"World language: continuing {subject} (already through the level-{level - 1} "
            f"course) — pick from the {subject} options below to continue it, or the culture "
            f"course alongside it; ignore other languages' listings."
            if level > 1 else
            f"World language: starting {subject} from the beginning — pick from the {subject} "
            f"options below; ignore other languages' listings."
        )
    if scenario.gen_ed_preference:
        parts.append(f"Gen-ed preference: {scenario.gen_ed_preference}.")
    return " ".join(parts) if parts else None


def _database_degree_program(database: CatalogDatabase) -> str:
    """The major actually loaded via `real_db.program_id`/`--major`, read off the database
    itself rather than trusted from a hardcoded fixture field.

    `profile.degree_program` is authored per-scenario in `plan_fixtures/*.yaml` and was never
    kept in sync with whichever program `--major` points the real database at — the fixture
    file only picks which SCENARIOS (student names, completed-course lists, assertions) exist,
    not which catalog the model is shown. Mode B/C always render the real database, so this is
    the one source of truth for what major the student prompt should claim: pointing `--major`
    at a different program now changes what the model is TOLD, not just what it is SHOWN.
    """
    programs = database.rows("programs")
    if not programs:
        return "(unknown program)"
    program = programs[0]
    name = program.get("name") or "(unknown program)"
    degree_type = program.get("degree_type")
    return f"{name}, {degree_type}" if degree_type else name


def _profile_block(profile: Profile, fixture: Fixture | None = None,
                   scenario: Scenario | None = None,
                   database: CatalogDatabase | None = None) -> str:
    """The student, and — explicitly — the boundary of what they have already done.

    THE COMPLETED LIST IS STATED AS EXHAUSTIVE, which it was not before. Students describe
    their transcript in categories ("I transferred in with my math and gen-ed credits done"),
    and a model that reads the category instead of the list fills in what it thinks the
    category contains. results_old9, mi-spring-start: gemma4-26b and qwen3-8b dropped CS 18000
    from all three of their plans — never scheduled, never listed as unplanned — while the
    four other models scheduled it every time, and every model scheduled it 3/3 on the
    otherwise-identical fall-start scenario. The completed list there is MA 16100, MA 16200,
    ENGL 10600, COM 11400, PSY 12000: no CS at all. Nothing in the data said intro programming
    was done; "transferred in" did.

    Saying the list is closed costs one clause and removes the inference. It is a statement
    about OUR data, not a rewrite of what the student said — the vague sentence stays in the
    scenario, because students really do talk like that and a plan built for real students has
    to survive it.
    """
    completed = ", ".join(profile.completed_courses) or "none"
    calendar = " -> ".join(
        f"{term} {year}" for term, year in _term_sequence(profile)
    )
    # "complete list" earns its words: students describe their transcript in categories
    # ("my math and gen-ed credits are done") and two of six models filled in what they thought
    # the category contained, dropping CS 18000 from every spring-start plan.
    label = ("Already completed (complete list)" if profile.completed_courses
             else "Already completed")
    preference_line = _elective_preference_line(scenario) if scenario is not None else None
    degree_program = (_database_degree_program(database) if database is not None
                      else profile.degree_program)
    fye_line = _fye_deadline_line(profile, database)
    return (
        f"STUDENT: {profile.name} — {degree_program}\n"
        f"{label}: {completed}\n"
        f"{credit_load_line(profile, fixture)}\n"
        f"{major_load_line(profile)}\n"
        + (f"{preference_line}\n" if preference_line else "")
        + (f"{fye_line}\n" if fye_line else "")
        + f"Semesters available ({profile.semesters_to_plan}), in order:\n  {calendar}"
    )


def _fye_deadline_line(profile: Profile, database: CatalogDatabase | None) -> str | None:
    """One instruction naming Purdue's First-Year Engineering (FYE) deadline — ONLY for a
    College of Engineering program, and ONLY when FYE looks genuinely unfinished. Added
    2026-08-07 alongside `real_db.is_college_of_engineering`.

    ABSENT ENTIRELY for every other program (not an empty line, not a "does not apply" note):
    this rule does not exist outside the College of Engineering, and a line naming it anyway
    would be a fabricated constraint spent on tokens no other program needed.

    NO COURSE CODES NAMED, deliberately. `real_db._fye_meta` carries the flat union of every
    legal alternate across Purdue's 8 numbered FYE requirements (e.g. both `CS 15900` and
    `CHM 11610` for the ONE "First-Year Engineering Selective" slot) — printing that whole set
    as "still due" would instruct the model to schedule courses that satisfy the SAME
    requirement twice. Naming the requirement categories instead is true regardless of which
    legal path the student ends up on.
    """
    if database is None:
        return None
    meta = database.rows("_fye_meta")
    if not meta:
        return None
    fye_codes = set(meta[0]["fye_course_codes"])
    if not fye_codes:
        return None
    alias_of = {row["course_code"]: row["alias_of"] for row in database.rows("course_aliases")}
    canon = lambda code: alias_of.get(code, code)  # noqa: E731
    completed = {canon(normalize_code(c)) for c in profile.completed_courses}
    matched = sum(1 for code in fye_codes if canon(code) in completed)
    # THRESHOLD, NOT AN EXACT AUDIT. The selective requirement alone contributes several
    # alternates to `fye_codes` that were never all going to be completed at once, so "every
    # code done" is never the right bar. 60% covers a student who finished FYE through any one
    # legal combination of paths without needing to model which combination that was.
    if fye_codes and matched / len(fye_codes) >= 0.6:
        return None
    return (
        "FIRST-YEAR ENGINEERING (FYE): this is a College of Engineering program. Purdue "
        "requires Intro to Engineering I & II, Calculus I & II, Chemistry, Physics, the "
        "First-Year Engineering Selective (one of: General Chemistry II, C Programming, "
        "Fundamentals of Biology I, or Fundamentals of Biology II), and Written & Oral "
        "Communication to be completed within the student's first two semesters if not "
        "already done — this gates continuing in the major. Prioritize any of these still "
        "outstanding over later coursework."
    )


def credit_load_line(profile: Profile, fixture: Fixture | None = None) -> str:
    """The load to REACH, then the load never to exceed. Target first, and always a number.

    THE TARGET WAS MISSING ENTIRELY, and that was the bug. `target_credits_per_semester` is
    null in the fixture — deliberately, so the planner derives the split per term — and this
    function read that null as "there is no honest number to print", falling back to a line
    that named only the LIMIT. So every Mode B prompt told the model a ceiling and nothing to
    aim at, and phrased even the spreading advice as a warning against filling terms up.

    Models did what the prompt asked. results_old9, 162 Mode B plans, mean credits by semester
    position: 13.7, 12.3, 12.0, 11.1, 11.0, 8.5, 7.9, 7.1 — a monotonic slide to half the
    opening load, while `gen-ed-selective: 0/6 credits` was the single most common unmet
    requirement (57 plans). The model was leaving courses unscheduled in terms that had room,
    because nothing ever told it the room was supposed to be used.

    There IS an honest number: the same even split ``planner.semester_credit_target`` computes
    at semester 0 — credits still to place, over semesters available, clamped by the cap. The
    planner spends the whole plan aiming at it, so naming it is describing what a good plan
    looks like, not inventing a figure.
    """
    # NO CAP LANGUAGE AT ALL, as of 2026-08-06 — an explicit experiment, not a fixed
    # conclusion: transcripts read like the model was staying suspiciously close to the
    # soft-cap number every time, which is at least consistent with the prompt's own "the
    # student asked for N or fewer" wording anchoring it there regardless of the actual
    # `hard_credit_cap`. Removing both numbers from the text answers "does it still self-limit
    # without being told to, and to what." The scorer is UNCHANGED — `credit_cap_violation`/
    # `soft_credit_overages` still fire exactly as before, so a model that overloads a semester
    # now shows up as a real, measured violation instead of being warned off it in advance. If
    # this needs reverting, `limit`/`hard_cap` are still on `profile` — only the prose describing
    # them was removed here; see git history for the exact two lines to restore.
    target = profile.target_credits_per_semester
    remaining = 0
    if fixture is not None:
        remaining = sum(fixture.credits(code) for code in profile.remaining_courses)
    if not target and remaining > 0 and profile.semesters_to_plan > 0:
        target = -(-remaining // profile.semesters_to_plan)          # ceil, ints only
    if not target:
        # No fixture to count against (Mode A's call site) — name the shape only.
        return ("Spread the courses evenly across all "
                f"{profile.semesters_to_plan} semesters instead of filling the early ones and "
                "leaving the later ones empty")
    aim = int(target)
    # THE NUMERATOR IS STATED TOO. The target is remaining/semesters, and giving only the
    # quotient leaves the model no way to notice it has dropped something: a plan missing one
    # 4-credit course still looks locally sensible in every term. Naming the total makes the
    # arithmetic checkable — sum the plan, compare, and an omission is visible instead of
    # invisible. It also states the finishing condition, which is the thing a plan is FOR.
    total = f"{remaining} credits still outstanding. " if remaining else ""
    return (f"Credits: {total}Aim for {aim} per semester, spread evenly across every semester "
            f"available rather than front-loaded.")


def major_load_line(profile: Profile) -> str:
    """The major-course cap. One number when there is only one number.

    The soft preference (2 CS courses, relaxed to 3 only when nothing else fits) was dropped
    from the prompts on 2026-07-31: two limits on the same quantity, one of which the model was
    told to break when convenient, is a judgement call small models make badly, and it competes
    for attention with the credit target and the prerequisite chain. The scoring machinery for
    it is untouched — set `preferred_major_courses_per_semester` below the hard cap in the
    fixture and both the line and the preference paragraph come back.
    """
    hard = profile.max_major_courses_per_semester
    preferred = profile.preferred_major_courses_per_semester
    if preferred >= hard:
        return f"{profile.major_subject} courses per semester: at most {hard}"
    return (
        f"{profile.major_subject} courses per semester: at most {hard} (hard limit), "
        f"{preferred} or fewer preferred"
    )


def _term_sequence(profile: Profile) -> list[tuple[str, int]]:
    """Spell the calendar out term by term instead of making the model derive it.

    Two of the eval's scenarios start in spring or summer, and deriving "what comes after
    spring 2027 when summer is not schedulable" is a distraction from the actual task —
    a model that gets the sequencing right but the labels wrong looks identical in the
    scores to one that scheduled into a term that doesn't exist.
    """
    out: list[tuple[str, int]] = []
    term, year = first_planning_term(profile.start_term.lower(), profile.start_year)
    for _ in range(profile.semesters_to_plan):
        out.append((term, year))
        term, year = next_term(term, year)
    return out


def _plan_summary(plan: Plan) -> str:
    lines = [
        f"- {s.term} {s.year}: {', '.join(s.courses) or '(empty)'} [{s.total_credits} cr]"
        for s in plan.semesters
    ]
    if plan.unplanned_courses:
        lines.append(f"- Unplanned: {', '.join(plan.unplanned_courses)}")
    return "\n".join(lines) or "(no semesters planned)"


def _course_context(profile: Profile, catalog: list[Course]) -> str:
    """Port of ``advisor_agent._course_context`` — the courses the model may move, with tags."""
    by_code = {c.code: c for c in catalog}
    lines = []
    for code in profile.remaining_courses:
        course = by_code.get(code)
        if course is None:
            continue
        tags = ", ".join(course.requirement_tags) or "none"
        lines.append(f"- {course.code} \"{course.title}\" ({course.credits} cr; tags: {tags})")
    return "\n".join(lines) or "(no known remaining courses)"


class PromptBuilder:
    def __init__(self, fixture: Fixture, database: CatalogDatabase | None = None):
        self.fixture = fixture
        # MODE B'S CONTEXT IS THE DATABASE, not a prompt-authored catalog listing. Rendered once
        # here: it is byte-stable and lands in the static hash, so a change to the underlying
        # catalog correctly stops old Mode B records being comparable.
        self.database = database
        # `source: fixture` pseudo-majors (school-core / gen-ed+world-language test cases) get
        # the stripped requirement-groups-only render AND the matching system-prompt schema
        # description — see `catalog_export.render_context_lean`, `PLAN_SYSTEM_LEAN`, and
        # `real_db.real_db_base_from_fixture`'s docstrings for why. Every other fixture (the
        # default `source: real_db`) is unchanged. Picked once here, not per call, so Mode B
        # and Mode C (`_plan_static`/`plan_convergence`) can never disagree about which schema
        # this fixture's database block actually is.
        lean = fixture.source == "fixture"
        self._plan_system = PLAN_SYSTEM_LEAN if lean else PLAN_SYSTEM
        renderer = render_context_lean if lean else render_context
        self._database_block = renderer(database) if database is not None else None

    # -- Mode A: revise-plan proposal (the app's real path) ---------------------------------

    def plan_proposal(
        self, scenario: Scenario, plan: Plan, prior_warnings: list[str] | None = None
    ) -> tuple[str, str, str]:
        """Returns (system, user, static_hash). Port of ``advisor_agent._user_prompt``."""
        parts = [
            f"STUDENT: {scenario.profile.name} — {scenario.profile.degree_program}",
            # Major load FIRST, credit cap SECOND, and the "enforced by the planner, not by
            # you" tag dropped from the major-load line: it was sitting one line above the
            # only field the model is asked to set, and the credit-cap instruction is fragile
            # enough on small models without a neighbouring sentence telling them caps are
            # somebody else's problem.
            major_load_line(scenario.profile) + ".",
            f"Current credit cap: {scenario.profile.max_credits_per_semester} per semester "
            f"(change it via max_credits_per_semester if the student asks for a different load).",
            # Mode A only reports the target; it is not a field on PlanEditProposal, and the
            # planner derives it. Stated so the model can explain the plan's shape without
            # inferring that a ~11-credit term means the cap was misread.
            (f"The planner aims for about "
             f"{scenario.profile.target_credits_per_semester} credits a semester"
             if scenario.profile.target_credits_per_semester else
             "The planner spreads the courses evenly across the semesters available")
            + " rather than filling the early terms to the cap.",
            "",
            "COURSES THAT CAN BE MOVED:",
            _course_context(scenario.profile, self.fixture.catalog),
            "",
            "CURRENT PLAN:",
            _plan_summary(plan),
            "",
            f"STUDENT FEEDBACK:\n{scenario.feedback}",
        ]
        if prior_warnings:
            parts += [
                "",
                "Your previous proposal left these planner warnings — fix them this time:",
                "\n".join(f"- {w}" for w in prior_warnings),
            ]
        return (PROPOSAL_SYSTEM, "\n".join(parts),
                _static_hash(f"{PROPOSAL_SYSTEM}\n{SCHEMA_ENUM_POLICY}", PROPOSAL_SCHEMA))

    # -- Mode B: free-form plan of study ----------------------------------------------------

    def plan_freeform(self, scenario: Scenario) -> tuple[str, str, str]:
        """Mode B prompt. Uses ``plan_schema(TERM_ORDER)`` at call time in the runner."""
        static = self._plan_static()
        return (static, self._plan_user(scenario),
                _static_hash(f"{static}\n{SCHEMA_ENUM_POLICY}", PLAN_SCHEMA))

    # -- shared by Mode B and Mode C's free-form plan-of-study prompts -----------------------

    def _plan_static(self) -> str:
        if self._database_block is None:  # pragma: no cover — `run.py check` refuses this first
            raise RuntimeError(
                "Mode B needs a catalog database — pass one to PromptBuilder(database=...). "
                "See harness/real_db.py."
            )
        return f"{self._plan_system}\n\n{self._database_block}"

    def _plan_user(self, scenario: Scenario) -> str:
        return (
            f"{_profile_block(scenario.profile, self.fixture, scenario, self.database)}\n\n"
            f"STUDENT'S REQUEST:\n{scenario.feedback}\n\n"
            f"Produce the plan of study."
        )

    # -- Mode C: retry to convergence, with locked slots ------------------------------------

    def plan_convergence(
        self, scenario: Scenario, locked, feedback: list[str] | None = None,
        confirmed: bool = False, missing_requirements: list[str] | None = None,
        placement_hints: list[str] | None = None,
    ) -> tuple[str, str, str]:
        """Mode C's prompt. Returns (system, user, static_hash).

        ``feedback`` is the validator's findings from the previous attempt, and it goes in the
        VARIABLE TAIL — which is what keeps the static hash identical between the blind and
        feedback variants. That identity is load-bearing: the two variants differ only in what
        the model is told between attempts, so if their static hashes diverged, the report
        would be comparing two conditions instead of one condition under two retry policies.
        """
        static = f"{self._plan_system}\n\n{PLAN_LOCKED_RULES}\n\n{self._database_block}"
        if self._database_block is None:  # pragma: no cover — preflight refuses this first
            raise RuntimeError(
                "Mode C needs a catalog database — pass one to PromptBuilder(database=...). "
                "See harness/real_db.py."
            )
        parts = [_profile_block(scenario.profile, self.fixture, scenario, self.database), ""]
        if locked:
            terms = _term_sequence(scenario.profile)
            # Grouped by semester once the set gets large: after a repair pass the "locked" set
            # is most of a plan, and thirty separate one-line entries read as a list of
            # unrelated pins rather than as the schedule it actually is.
            by_semester: dict[int, list[str]] = {}
            for slot in sorted(locked, key=lambda s: (s.semester_index, s.course)):
                course = self.fixture.by_code.get(slot.course)
                credits = f" ({course.credits} cr)" if course else ""
                by_semester.setdefault(slot.semester_index, []).append(
                    f"{slot.course}{credits}")
            lines = []
            for index in sorted(by_semester):
                term, year = terms[index] if index < len(terms) else ("?", "?")
                lines.append(f"- Semester {index + 1} ({term} {year}): "
                             + ", ".join(by_semester[index]))
            header = (
                # The repair variant's second and later attempts. Saying these were CHECKED is
                # the point: the model is not being asked to re-plan them, only to COPY them —
                # into the SAME semester, verbatim. See the closing instruction below for why
                # "repeat, do not re-plan" replaced "omit entirely".
                "ALREADY CONFIRMED — these placements have been checked and are correct. "
                "Repeat every one of them in the exact semester listed here. Do not move one "
                "to a different semester, drop it, or add it again anywhere else:"
                if confirmed else
                "LOCKED SLOTS — keep these exactly where they are:"
            )
            parts += [header, *lines, ""]
        if missing_requirements:
            parts += [
                "STILL MISSING — the plan does not yet satisfy these requirement groups:",
                *(f"- {item}" for item in missing_requirements),
                "",
            ]
        if placement_hints:
            # THE LEGAL SLOTS, COMPUTED. Same principle as deleting violations in the harness:
            # if the answer is decidable without a model, do not spend the model on it. With a
            # late prerequisite these lists are often a single semester, and a small model
            # searching eight terms for it will simply fail.
            parts += [
                "WHERE THEY CAN GO — already checked against prerequisites, term offerings, "
                "the credit cap and the CS-per-semester limit. Use one of the terms listed; "
                "any other term will be rejected:",
                *(f"- {hint}" for hint in placement_hints),
                "",
            ]
        parts += [f"STUDENT'S REQUEST:\n{scenario.feedback}", ""]
        if feedback:
            parts += [
                "Your previous plan was checked and these problems were found. Fix them and "
                "return a corrected plan:",
                *(f"- {item}" for item in feedback),
                "",
            ]
        parts.append(
            # THE FULL PLAN, CONFIRMED COURSES REPEATED VERBATIM. Used to ask for only the
            # ADDITIONS — the confirmed placements are re-inserted by the harness regardless of
            # what comes back, so a delta reply was assumed to be pure savings. Live case,
            # qwen3.6-27b / mi-light-load: told to omit the confirmed courses, it tried to
            # reconstruct them from memory anyway ("do not repeat them elsewhere" read as "do
            # not forget them"), reproduced the whole schedule shifted by a semester, and spent
            # its entire reply on that — every real addition (PHYS 17200, SCLA 10100, the gen-ed
            # selective) landed in "unplanned". A model too large to need delta's token savings
            # is exactly the model with enough spare attention to hallucinate a state it was
            # told to leave alone. Asking it to copy the confirmed block verbatim into its
            # answer removes the ambiguity that "elsewhere" left open: there is no other place
            # for a confirmed course to legally appear, so the model is never guessing at it.
            "Return the FULL plan of study: every semester, in order, from semester 1 to the "
            "last one. For each semester, repeat every ALREADY CONFIRMED course from that exact "
            "semester above verbatim, then add whatever new courses belong alongside them. "
            "Every confirmed course must appear in your reply, in its confirmed semester, and "
            "nowhere else — do not move, drop, or re-add one elsewhere."
            if confirmed else "Produce the plan of study."
        )
        return (static, "\n".join(parts),
                _static_hash(f"{static}\n{SCHEMA_ENUM_POLICY}", PLAN_SCHEMA))

    # -- explain-plan -----------------------------------------------------------------------

    def explain_plan(self, question: str, plan_json: str) -> tuple[str, str, str]:
        user = (
            f"Student question:\n{question}\n\n"
            f"Structured plan:\n{plan_json}\n\n"
            f"Explain the plan clearly. Focus on prerequisites, semester sequencing, "
            f"warnings, and risk."
        )
        return EXPLAIN_SYSTEM, user, _sha256(EXPLAIN_SYSTEM)

    # -- grounded QA (RAG) ------------------------------------------------------------------

    def grounded_qa(self, question: str, chunks: list[dict[str, Any]]) -> tuple[str, str, str]:
        """Mirrors ``rag.pipeline`` — retrieval already happened, the model only summarizes.

        The chunks come from ``questions.yaml`` rather than from a live pgvector query on
        purpose: retrieval quality is a property of the embedding model and the corpus, not of
        the chat model being compared, and letting it vary would smear a retrieval difference
        across the model scores.
        """
        body = "\n\n".join(
            f"[{i + 1}] ({json.dumps(c.get('metadata', {}), sort_keys=True)})\n{c['content']}"
            for i, c in enumerate(chunks)
        ) or "(no matching rules were retrieved)"
        user = f"CONTEXT:\n{body}\n\nSTUDENT QUESTION:\n{question}"
        return QA_SYSTEM, user, _sha256(QA_SYSTEM)


def json_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """llama.cpp converts this to a GBNF grammar server-side — the same call the app makes."""
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}
