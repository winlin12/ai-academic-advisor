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
import math
from typing import Any

from .fixtures import Fixture, Scenario
from .catalog_export import CatalogDatabase, render_context
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
# UNRESOLVED REQUIREMENT GROUPS REMOVED, 2026-08-12 (moves the static hash again). The bullet
# describing them was four lines of "here is a category of requirement; there is nothing you can
# do about it" — a distinction with no action attached, held next to `requirement_groups`, which
# looks identical and is the one the model must act on. Transcripts read like the models were
# spending attention on it anyway. The table itself is gone from the export (see
# `catalog_export.TABLES`) and from `real_scoring`, so the prompt no longer names a section that
# is not there. Prose-stated university requirements come back when they can be shown as
# something schedulable, the way the synthesized `ucc-*` groups already are.
PLAN_SYSTEM = """\
You build a semester-by-semester plan of study from a read-only export of the advisor\'s
catalog database. Use only what is in that export.

Where things are:
- `courses` - every course that exists, with credit_hours_min, listed in prerequisite order.
  `prereq_groups` (AND-of-ORs) and `coreq_codes` appear directly on a course's own row, present
  only when that course actually has them.
- `requirement_groups` - the degree requirements, one row per group, each carrying its own
  `options` list of the courses that satisfy it. `all_of` = take every option.
  `choose_credits` = take enough to reach `credits_min`. Groups are named, not numbered.
- `course_aliases` - `alias_of` is an approved substitute. Take one, never both.
- `attributes` on a course lists the University Core competencies it carries, e.g. `UCC: QR`.
  A course tagged `UCC: QR` satisfies the "University Core: Quantitative Reasoning" group.

Rules, all hard:
- ONE COURSE CAN SATISFY SEVERAL REQUIREMENTS AT ONCE, and you should use that. A single course
  may count toward the major, the college requirement, and a University Core competency
  simultaneously — taking MA 16100 covers the major math requirement AND `UCC: QR`. Do not
  schedule a second course to cover a requirement one you already placed also satisfies. Look
  for options that close more than one group at a time; the shortest legal plan is the good one.
- BUT CREDITS COUNT ONCE. A course double-counting across three requirement lists still
  contributes its credit hours exactly once toward the degree total. The plan must reach the
  graduation credit minimum in REAL credits: completed credits plus the credits of the distinct
  courses you schedule. Satisfying every group with too few total credits does not graduate the
  student.
- UPPER LEVEL REQUIREMENT: at least 32 credit hours must come from courses numbered 30000 or
  above, counting completed and planned together. Introductory courses do not count toward it,
  so a plan made entirely of 10000- and 20000-level work fails this even when every requirement
  group is covered.
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
keep those sums CLOSE TO EACH OTHER, including the last ones. Do not fill the early terms to
the limit and leave the later ones light on credits, and do not leave a term light on credits
when something legal could go in it. Selective groups usually offer more options than
the student needs: when a term has room, pick an option whose prerequisites are already
satisfied."""

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


# =============================================================================================
# STAGE 1 — COURSE SELECTION (Mode B/C's first pass)
# =============================================================================================
#
# WHY THIS STAGE EXISTS AT ALL, as of 2026-08-13. Mode B used to be one call that asked a model
# to choose ~40 courses out of the ~130 the budget trim leaves in the export AND sequence them
# AND balance credits AND respect prerequisites. The measured result across 14 programs and 7
# models: the best model produced a legal plan 34% of the time, the worst 11%, with 5-30
# prerequisite violations each and up to 169 duplicate courses silently deleted before scoring.
# Both halves failed independently — coverage (selection) ran 52-86% while violations
# (ordering) were a separate failure — which is the signature of two tasks fused into one call.
#
# Split, each half gets a prompt it can actually hold: this one sees requirement groups and
# answers "which courses", knowing nothing about semesters; the ordering stage sees ~40 chosen
# codes and answers "in what order", never having to search a catalog. The ordering prompt is
# an order of magnitude smaller than the old fused one, which is where the latency goes.
SELECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "selections": {
            "type": "array",
            "description": "One entry per requirement group you are filling.",
            "items": {
                "type": "object",
                "properties": {
                    "group": {"type": "string",
                              "description": "The requirement group's name, copied exactly."},
                    "courses": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Course codes from THIS group's options that the "
                                       "student should take. Omit courses already completed.",
                    },
                },
                "required": ["group", "courses"],
            },
        },
    },
    "required": ["selections"],
}


def selection_schema(option_codes: list[str], group_names: list[str]) -> dict[str, Any]:
    """`SELECTION_SCHEMA` with both enums bound to this program's own rows.

    Same reasoning as `plan_schema`'s course enum, applied one stage earlier: a code the
    catalog does not contain cannot be selected, and a group name the program does not have
    cannot be invented. What the grammar CANNOT enforce is the part that matters — picking
    enough credits from the right group — which is what the scorer measures.
    """
    schema = json.loads(json.dumps(SELECTION_SCHEMA))
    item = schema["properties"]["selections"]["items"]["properties"]
    if group_names:
        item["group"]["enum"] = sorted(set(group_names))
    if option_codes:
        item["courses"]["items"] = {"type": "string", "enum": sorted(set(option_codes))}
    return schema


# =============================================================================================
# MODE A — AUDIT AN EXISTING SCHEDULE
# =============================================================================================
#
# The task the app actually has to do most often: a student arrives WITH a plan (their own
# draft, an advisor's sheet, last year's plan) and wants to know what is wrong with it. Mode A
# used to be the app's revise-plan path — the model expressed preferences (reorder/defer) and a
# deterministic planner rebuilt the schedule, which made every Mode A plan legal by
# construction and every model's numbers identical (79% viable, 0 violations, all seven models,
# measured 2026-08-12). Identical numbers are not a measurement.
#
# So the model now answers the question directly and its answer is what gets scored: which
# required courses are absent, and which placements have to move. Ordering stage 2 of Mode B is
# the same call with an empty starting schedule — one prompt, two entry points, so a fix to
# either lands in both.
AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "semesters": {
            "type": "array",
            "description": "The COMPLETE corrected schedule, every semester in order — not a "
                           "diff. Include courses that were already in the right place.",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "enum": ["fall", "spring", "summer"]},
                    "year": {"type": "integer"},
                    "courses": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["term", "year", "courses"],
            },
        },
        # NEITHER `missing` NOR `problems` SURVIVES (2026-08-15, 2026-08-16). Both asked the
        # model to narrate alongside the plan, and both cost more than they returned.
        #
        # `missing` — which required courses are absent — has no content once the ordering
        # stage is handed the exact set to place: nothing can be absent that was not given.
        # What it produced was enumeration, up to 36 entries, and repetition loops that ran to
        # the token ceiling (qwen3.8-27b alternating "ENGL 32800"/"ENGL 32700" until
        # truncation). 7% of plan records in that sweep were truncated.
        #
        # `problems` — a prose account of what could not be balanced — read well and was never
        # scored. Its cost is generation time on every single plan, and a model that stops to
        # explain its own compromises is a model not spending those tokens on the schedule.
        # The harness computes every problem worth knowing about (`real_scoring`'s violations,
        # coverage and spread) from the plan itself, against the real database, which is both
        # cheaper and not subject to the model's own account of itself.
        #
        # WHAT IS LOST, and it is worth naming: qwen3.8-27b used this field to explain that it
        # front-loaded "to satisfy prerequisite chains", which was a genuinely interesting
        # claim about its own reasoning. That belongs in a targeted experiment, not on the
        # critical path of every plan in a sweep.
    },
    "required": ["semesters"],
}


def audit_schema(planning_terms: list[str], course_codes: list[str]) -> dict[str, Any]:
    """`AUDIT_SCHEMA` with the term and course enums bound, same as `plan_schema`."""
    schema = json.loads(json.dumps(AUDIT_SCHEMA))
    semester = schema["properties"]["semesters"]["items"]["properties"]
    semester["term"]["enum"] = list(planning_terms)
    if course_codes:
        semester["courses"]["items"] = {"type": "string", "enum": sorted(set(course_codes))}
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

# CHANGED 2026-08-12, and it is a deliberate change of POLICY, not of wording. The previous
# prompt told the model to refuse whenever the retrieved context did not contain the answer
# ("say you don't have that rule on file"), and the models did exactly that — the transcripts
# are full of one-line refusals to questions a human advisor would have taken a swing at.
# Refusing is the safe failure for a system that might be quoted as authoritative; it is also
# useless to a student who asked a real question, and it was firing far more often than the
# thin-context items alone.
#
# WHAT THIS COSTS, stated plainly because it lands in the scoring: `questions.yaml` labels 18
# of 39 items `expected_behavior: abstain`, and those labels were written against the OLD
# policy. Under this prompt a best-effort answer to a thin-context question is the model
# obeying its instructions, and `scorers.score_qa` will nonetheless record `behavior_ok: False`
# for it. Expect the "Abstained OK" column to collapse on the next sweep. That number is now
# measuring the prompt, not the model — decide whether those 18 items should be re-labelled to
# "answer with a caveat" before reading anything into it.
QA_SYSTEM = (
    "You are a senior college academic advisor, answer using the context below. If not "
    "possible, answer to the best of your ability."
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

    `profile.degree_program` is the program's own catalog name (`fixtures.synthesize_scenarios`) and was never
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


def _full_program_line(scenario: Scenario | None, primary: str) -> str:
    """The student's WHOLE degree — primary program plus any second major or minor.

    `_database_degree_program` reads `programs`, which holds only the program the database was
    loaded for; a scenario's extra programs are merged in as requirement GROUPS (see
    `real_db.merge_real_db_bases`) and never reach that table. So without this the prompt
    announced "Computer Science: Machine Intelligence" to a student whose requirement list was
    14 groups of `Data Science (major): ...` and `Mathematics (minor): ...` — the model was
    expected to plan a dual major and a minor while being told it was planning one major.

    Rendered from `Scenario.additional_programs`, the same field the merge itself reads, so the
    sentence and the requirement list cannot drift apart.
    """
    extras = getattr(scenario, "additional_programs", ()) if scenario is not None else ()
    if not extras:
        return primary
    parts = [
        f"{label} (second major)" if kind == "major" else f"{label} ({kind})"
        for _poid, kind, label in extras
    ]
    return f"{primary} + {' + '.join(parts)}"


def _profile_block(profile: Profile, scenario: Scenario | None = None,
                   database: CatalogDatabase | None = None,
                   credits_remaining: float | None = None) -> str:
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
    degree_program = _full_program_line(
        scenario,
        _database_degree_program(database) if database is not None else profile.degree_program,
    )
    fye_line = _fye_deadline_line(profile, database)
    return (
        f"STUDENT: {profile.name} — {degree_program}\n"
        f"{label}: {completed}\n"
        f"{credit_load_line(profile, credits_remaining)}\n"
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


# How far either side of the even split still counts as evenly spread. One lab course wide.
# Kept for `real_scoring._CREDIT_BAND`'s cross-reference and for the fallback line below; the
# main credit instruction no longer states a band, because it no longer states a target.
_CREDIT_BAND = 2


def credit_load_line(profile: Profile, credits_remaining: float | None = None) -> str:
    """The per-semester load: what to aim for, and what never to exceed.

    THE TARGET IS DERIVED, NOT DECREED — `ceil(credits still to place / semesters available)`,
    the even split that finishes the degree exactly on time. That distinction is the whole
    reason it is safe to state a number again.

    HISTORY, because this line has been wrong in both directions. Naming only the LIMIT made
    models drift downward: results_old9, 162 Mode B plans, mean credits by semester position
    13.7, 12.3, 12.0, 11.1, 11.0, 8.5, 7.9, 7.1 — they treated the ceiling as a target and
    then undershot it. Naming a fixed per-semester QUOTA made them satisfy the quota and stop,
    leaving `gen-ed-selective: 0/6` as the single most common unmet requirement (57 of those
    162 plans). So on 2026-08-06 both numbers came out entirely, as an experiment: would models
    self-limit with no figure at all?

    THE EXPERIMENT ANSWERED, 2026-08-16, and the answer is no. With no number stated, every
    model exceeded the student's (unstated) ceiling — qwen3.8-27b on 2.40 semesters per plan,
    gemma4-26b on 0.32, an eightfold spread that is about the prompt rather than about
    planning. qwen3.8 filled to ~18 credits a term, exhausted the course list in six semesters
    and left three empty: `[12, 18, 17, 16, 17, 16, 8, 0, 0, 0]` for a student who asked for
    twelve or thirteen. Meanwhile `ORDER_SYSTEM` instructed it to "respect the student's stated
    credit ceiling", which the prompt had never stated.

    AND THE RULES THEMSELVES WERE TESTED, 2026-08-21 — the "these constraints are confusing
    the model" hypothesis, run as a throwaway arm that replaced ORDER_SYSTEM's whole rule list
    with "Give the student a balanced schedule." and dropped this line and `major_load_line`
    from the profile. qwen3.8-27b, construction-management, 5 scenarios, everything else
    identical: viable 5/5 -> 1/5, coverage 100% -> 91%, mean credit spread 12.6 -> 14.0. It
    got WORSE at the balance the rules were suspected of harming. Two rules turned out to be
    load-bearing — "place every course" (without it the model silently dropped 2-3 courses per
    plan; the enum grammar stops additions, nothing but the prompt compels placement) and the
    credit ceiling (spring-start came back `[24, 21, 20, 22, 8, 0, 0, 0]`, unregistrable, and
    still emptied three terms). The arm was deleted; this paragraph is what it bought.

    WHY THE OLD QUOTA FAILURE DOES NOT RETURN. That failure needed a model that could stop
    early — it was choosing courses as well as placing them, so hitting the number and halting
    silently dropped requirements. The ordering stage is handed a fixed set and must place all
    of it, and a course left out is now visible as a dropped course rather than an invisible
    coverage gap. The number can only shape the SHAPE of the plan, not its contents.

    `credits_remaining` comes from the courses actually being placed, computed against the same
    database the model is shown — not from a fixture's own credit table, which is how the
    previous version came to state a finishing condition for a different degree than the one in
    the prompt.
    """
    semesters = max(1, int(profile.semesters_to_plan or 1))
    ask = int(profile.max_credits_per_semester or 0)
    # `profile.hard_credit_cap` is deliberately NOT stated any more (2026-08-21) — see the
    # target note below. The scorer never enforced 18 in the first place: its hard line is 22,
    # `real_scoring._ABSURD_SEMESTER_CREDITS`, and everything between the student's ask and 21
    # is a soft overage. So this removes an assertion, not a constraint.
    if credits_remaining:
        target = math.ceil(float(credits_remaining) / semesters)
        # A RANGE, NOT A POINT. An exact per-semester figure is unreachable by construction —
        # courses come in 1, 3, 4 and 5-credit sizes, so no arrangement of them lands on the
        # same integer eight times, and a model told to hit one exactly is being set a target
        # it can only miss. Stating the band it should land in describes the same intent
        # without the false precision, and it leaves room for the lab-sized course that has to
        # go somewhere.
        if ask and target > ask:
            # THE HORIZON IS TOO SHORT FOR THE DEGREE at the load this student will carry.
            # Saying the plan does not fit is both true and actionable, and the honest response
            # is a full schedule plus a report of the overflow.
            parts = [
                f"Fill every semester to about {ask} credits — that is this student's limit. "
                f"The {credits_remaining:g} credits still to place over {semesters} semesters "
                f"works out at {target} a semester, which is above that limit, so not "
                f"everything fits: place what does, and say what does not"
            ]
        else:
            # AN OBJECTIVE, NOT A CONDITION TO SATISFY. This read "every semester should land
            # in {low}-{high}", and qwen3.8-27b explained in its own output exactly what that
            # cost: "the total credit count of the required courses (108) does not divide
            # evenly by 8 semesters to reach the target of 14 credits per semester without
            # exceeding the 16-credit ceiling in some terms or leaving others underfilled".
            # It tested the range as a hard constraint, correctly found it unsatisfiable —
            # courses come in 1, 3, 4 and 5-credit sizes and no arrangement lands on the same
            # number eight times — and then discarded the instruction entirely, producing
            # `[16, 18, 15, 17, 18, 17, 3, 0, 0, 0]` for a student who asked for thirteen.
            #
            # A constraint that cannot be met is worth nothing to a model checking whether it
            # can be met. An objective to get as close to as possible is worth something even
            # when it cannot be reached exactly, which is always. So the number is stated as
            # the aim, the impossibility of hitting it exactly is stated OUT LOUD so no
            # feasibility check has anything to discover, and what is actually forbidden — a
            # heavy term sitting next to an empty one — is stated as the rule instead.
            # NO TARGET NUMBER, 2026-08-21. The instruction is now purely RELATIVE: make the
            # semesters resemble each other. Every previous version named a figure — a limit, a
            # quota, a derived target — and each one gave the model something to test, satisfy
            # or discard instead of a shape to aim at. The derived target was the best of them
            # and still produced `[19, 19, 16, 17, 14, 14, 5, 0]` on qwen3.8-27b: it filled the
            # early terms toward the ceiling it could see and let the remainder fall off the
            # end. A comparison ("no semester much heavier than another") cannot be front-loaded
            # the way an absolute number can, because it is only satisfiable by looking at the
            # whole schedule.
            #
            # `credits_remaining` and `semesters` are still stated as FACTS — how much work
            # there is and how many terms it has to fit into — because that is the arithmetic
            # the model needs to spread anything at all. What is gone is the per-semester
            # figure derived from them.
            # THE TARGET IS BACK, 2026-08-21, AND THIS IS WHY. It came out for one day on the
            # theory that a purely RELATIVE instruction ("no semester much heavier than
            # another") cannot be front-loaded the way an absolute number can. Measured on
            # construction-management that looked free — 5 scenarios, mean spread 12.6 -> 12.4,
            # coverage held at 100%. It did not survive contact with a second program. Same
            # model, same scenario, animal-sciences/fresh-start, target vs no target:
            #
            #   with target     [18, 15, 15, 14, 13, 16, 12, 6]   8/8 terms, viable, 0 violations
            #   without target  [20, 22, 20, 21, 16, 10,  0, 0]   6/8 terms, NOT viable, 1
            #
            # With nothing to aim at, the model packs toward the only number left — the
            # student's own ceiling — overshoots it by its usual ~4, and exhausts the course
            # list two terms early. The relative sentences are kept BELOW as reinforcement;
            # they are not a substitute for a figure.
            #
            # ⚠ BOTH OF THOSE PLANS SCORE credit_spread = 12. The metric measures max-min across
            # NON-EMPTY semesters, so abandoning two terms costs nothing in that column and the
            # regression is invisible to it. Do not evaluate a change to this line on spread
            # alone — read `semesters_used` and the violation count beside it.
            parts = [
                f"Aim for about {target} credits a semester: the {credits_remaining:g} credits "
                f"still to place, divided over all {semesters} semesters. You will not hit that "
                f"exactly and you are not expected to — courses come in different sizes, so "
                f"some terms will be a credit or two either side of it. Get as close as you "
                f"can, term by term, and keep every semester within a few credits of {target}"
            ]
            parts.append(
                f"What is NOT acceptable is a heavy term next to a light or empty one. If a "
                f"semester ends up under {max(1, target - _CREDIT_BAND)} credits while another "
                f"is over {target + _CREDIT_BAND}, move courses between them until both are "
                f"near {target}. Use all {semesters} semesters — the last ones carry work too, "
                f"and finishing early while a later term sits empty is exactly the shape to "
                f"avoid"
            )
            # ONE CEILING, CAPPED AT 18, AND NO SECOND NUMBER. 2026-08-21.
            #
            # The anchor experiment: this model fills terms toward whatever ceiling the prompt
            # names, so a LOWER stated ceiling should drag the drift down with it. Measured on
            # qwen3.8-27b / construction-management, 5 scenarios, everything else held: stating
            # 15 gave a heaviest term of 19 and mean credit spread 11.6; stating nothing at all
            # gave a heaviest term of 22 (an unregistrable semester, and one lost plan) and
            # spread 12.4. So the ceiling matters — but the DRIFT ABOVE IT IS ROUGHLY CONSTANT
            # at about +4, whatever number is named. The model does not read this as a limit to
            # stay under; it reads it as the size a semester is meant to be, and then overruns.
            #
            # 15 was tried on that basis and rejected as too fictional to hold: told 15 it still
            # opened at 19. Capped at 18 the stated number is at least the true registrar
            # figure, so the prompt is not asserting something the catalog contradicts, and 18
            # still sits four credits under the scorer's hard line of 22
            # (`real_scoring._ABSURD_SEMESTER_CREDITS`) — which is the margin the drift eats.
            #
            # `min(ask, 18)` — NEVER a flat 18. Students ask for 12 and 13 in these scenarios,
            # and telling one of them "never go above 18" would RAISE their ceiling past what
            # they asked for, replacing the student's own stated preference with a number they
            # never gave. The cap only ever binds a student who asked for MORE than 18.
            #
            # THE REGISTRAR LINE IS STILL GONE. This is the only credit ceiling in the prompt;
            # there is no longer a second sentence naming a different maximum, because two
            # numbers is what the earlier versions of this instruction kept failing on.
            #
            # THE FEASIBILITY CHECK ABOVE STILL USES THE REAL `ask`, deliberately. If the stated
            # number fed that branch, changing it would flip scenarios into the "not everything
            # fits, say what did not" path and the model would start DROPPING courses — a
            # coverage collapse caused by the anchor rather than by the plan.
            if ask:
                stated = min(ask, 18)
                parts.append(f"Never go above {stated} credits in a semester")
        return ". ".join(parts)
    return ("Spread the courses evenly across all "
            f"{profile.semesters_to_plan} semesters instead of filling the early ones and "
            "leaving the later ones empty")


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


# =============================================================================================
# STAGE SYSTEM PROMPTS (2026-08-13 architecture)
# =============================================================================================

# SELECT ONLY. This prompt never mentions semesters, prerequisites or credit loads per term,
# because none of them are this stage's problem and every sentence about them is a sentence the
# model spends attention on instead of reading the option lists. The one arithmetic it must do
# is per-group credit targets.
SELECT_SYSTEM = """\
You choose which courses a student should take to finish their degree. You do NOT decide when
they take them — a later step does the scheduling, so ignore ordering entirely here.

You are given the degree's requirement groups. Each group carries its own `options`: the
courses that satisfy it, with credits.

Rules:
- `all_of` groups: the student must take EVERY option listed. Include all of them.
- `choose_credits` groups: pick options totalling at least `credits_min` credits. Pick the
  fewest courses that reach the target — not the whole list.
- A course already completed satisfies its group. Never select a completed course again.
- ONE COURSE CAN SATISFY SEVERAL GROUPS. If a course you already selected also appears in
  another group's options, that group is already paid for — do not select another course for it.
- Only course codes from that group's own `options`. Never invent one.
- Cover every group. A group you skip is a requirement the student does not graduate without."""

# ORDER / AUDIT. Shared by Mode A (a schedule already exists) and Mode B stage 2 (it does not).
# The prompt is deliberately blind to the catalog: it sees only the courses in play and their
# prerequisites, which is what makes it ~1-2k tokens instead of the ~12.5k the fused Mode B
# prompt needed.
ORDER_SYSTEM = """\
You lay out a student's courses across their semesters, and you check the result.

You are given the exact set of courses to place — the selection is already made, so never add a
course that is not in the list and never drop one that is.

Rules, all hard:
- PREREQUISITES COME FIRST. Every prerequisite of a course must be in a STRICTLY EARLIER
  semester than the course itself, or already completed. The same semester is NOT early enough.
  Corequisites may share a semester. Each course's prerequisites are listed with it.
- NEVER SCHEDULE A COMPLETED COURSE. The student's completed list is closed and final.
- EACH COURSE APPEARS EXACTLY ONCE in the whole schedule.
- SPREAD THE CREDITS EVENLY across every semester the student has, including the last ones.
  Balance credit hours, not course counts — courses are not the same size. Do not fill the
  early terms to the limit and leave the later ones nearly empty.
- Respect the student's stated credit ceiling every semester.

Return `semesters`: the COMPLETE schedule, not a diff. Every course in the list above appears
exactly once in it. Return nothing else — no commentary, no explanation, no lists of what did
not fit. The schedule is the whole answer."""

# REPAIR (Mode C). Everything ORDER_SYSTEM says still applies; this adds the one permission
# that separates Mode C from Mode B — the model may change WHICH courses are in the plan, not
# just where they sit, because a validator finding a coverage gap can only be fixed by adding a
# course the selection stage missed.
REPAIR_RULES = """\
This is a revision pass. A validator checked your previous schedule and its findings are below.

You MAY add courses that satisfy an unmet requirement, and you MAY remove courses that are not
required, in addition to moving what is already there. Every rule above still holds for the
schedule you return."""


class PromptBuilder:
    def __init__(self, fixture: Fixture, database: CatalogDatabase | None = None):
        self.fixture = fixture
        # MODE B'S CONTEXT IS THE DATABASE, not a prompt-authored catalog listing. Rendered once
        # here: it is byte-stable and lands in the static hash, so a change to the underlying
        # catalog correctly stops old Mode B records being comparable.
        self.database = database
        # ONE SCHEMA, ONE SYSTEM PROMPT. There used to be a second, "lean" pair here for
        # `program.source: fixture` pseudo-majors (school-core / gen-ed+world-language test
        # cases), whose export was requirement-groups-only and so needed its own schema
        # description. Those fixtures went with `plan_fixtures/` (2026-08-12); every program is
        # now a real crawled one rendered by `render_context`.
        self._plan_system = PLAN_SYSTEM
        self._database_block = render_context(database) if database is not None else None

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
            f"{_profile_block(scenario.profile, scenario, self.database)}\n\n"
            f"STUDENT'S REQUEST:\n{scenario.feedback}\n\n"
            f"Produce the plan of study."
        )

    # -- STAGE 1: course selection (Mode B/C's first pass) -----------------------------------

    def select_courses(self, scenario: Scenario) -> tuple[str, str, str]:
        """"Which courses does this student still need?" — no scheduling, no semesters.

        The static block is the REQUIREMENT GROUPS ONLY, not the full catalog export. That is
        the point of the split: this stage never needs a course's prerequisites, title or
        credit-hour breakdown, and the ~12.5k-token export existed only so a single fused call
        could both choose and sequence. What is left is the option lists themselves.
        """
        static = f"{SELECT_SYSTEM}\n\n{self._requirement_groups_block()}"
        completed = ", ".join(scenario.profile.completed_courses) or "none"
        user = (
            f"STUDENT: {scenario.profile.name} — {scenario.profile.degree_program}\n"
            f"Already completed (complete list): {completed}\n\n"
            f"STUDENT'S REQUEST:\n{scenario.feedback}\n\n"
            f"Choose the courses. Do not schedule them."
        )
        return static, user, _static_hash(static, SELECTION_SCHEMA)

    def _requirement_groups_block(self) -> str:
        """Every requirement group with its options and credit target, one JSON row per group.

        Built from `self.database`'s own rows rather than the fixture's, for the same reason
        Mode B's course enum is: the model must be shown, and grammar-bound to, the program it
        is actually being scored against.
        """
        if self.database is None:  # pragma: no cover — `run.py check` refuses this first
            raise RuntimeError("selection needs a catalog database — see harness/real_db.py.")
        options_by_group: dict[str, list[dict[str, Any]]] = {}
        for row in self.database.rows("requirement_options"):
            options_by_group.setdefault(row["requirement_group_id"], []).append(row)
        lines = [
            "REQUIREMENT GROUPS — the complete set for this degree. Nothing outside this list "
            "exists; never invent a course code.",
            "",
            'kind "all" = take every option. kind "choose" = take options totalling at least '
            "`credits_min` credits.",
            "",
        ]
        for row in self.database.rows("requirement_groups"):
            options = options_by_group.get(row["id"]) or []
            if not options:
                continue
            lines.append(json.dumps({
                "name": row.get("name"),
                "kind": "choose" if row.get("requirement_type") == "choose_credits" else "all",
                "credits_min": row.get("credits_min"),
                "options": [{"code": o["course_code"], "credits": o.get("credits")}
                            for o in options],
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return "\n".join(lines)

    # -- STAGE 2: ordering and audit (Mode A, and Mode B/C's second pass) --------------------

    def order_courses(
        self, scenario: Scenario, courses: list[str], *,
        existing: list[list[str]] | None = None,
        findings: list[str] | None = None,
        repair: bool = False,
        context_by_code: dict[str, str] | None = None,
    ) -> tuple[str, str, str]:
        """Place `courses` into semesters, and report what is wrong with `existing`.

        ONE BUILDER, THREE CALLERS, and the sharing is deliberate — Mode A hands it the
        student's own draft schedule (`existing`), Mode B stage 2 hands it a bare selection
        (`existing=None`), and Mode C hands it the previous attempt plus the validator's
        `findings`. They differ only in the variable tail, so the static hash is identical for
        Mode A and Mode B stage 2 and the two are directly comparable.
        """
        static = ORDER_SYSTEM + (f"\n\n{REPAIR_RULES}" if repair else "")
        # THE TARGET IS COMPUTED FROM THE COURSES IN FRONT OF THE MODEL, not from a separate
        # course list — see `credit_load_line`. Anything the database has no credit figure for
        # contributes nothing rather than a guessed default, so the target can only understate.
        by_code = ({row["course_code"]: row for row in self.database.rows("courses")}
                   if self.database is not None else {})
        credits_remaining = sum(
            float((by_code.get(code) or {}).get("credit_hours_min") or 0) for code in courses)
        parts = [_profile_block(scenario.profile, scenario, self.database, credits_remaining), ""]
        if existing is not None:
            # ONE BLOCK, TWO PROVENANCES, AND THE LABEL HAS TO SAY WHICH. Mode A's `existing`
            # is the student's own draft — a real schedule someone else wrote, which the model
            # is being asked to nudge. Mode C's `existing` is the model's OWN previous attempt,
            # which a validator just rejected. Both were once headed "THE SCHEDULE THE STUDENT
            # ALREADY HAS", so on every repair pass the model was handed its own failed draft
            # under a heading asserting the student was committed to it — directly below a
            # profile line reading "Already completed: none". Told it to change nothing and
            # that nothing had happened, in the same breath. Anchoring on a draft presented as
            # settled fact is the correct response to that heading; the heading was the bug.
            parts += [
                ("YOUR PREVIOUS ATTEMPT — your own draft, not anything the student is committed "
                 "to. Rewrite it as freely as you need to:") if repair else
                "THE SCHEDULE THE STUDENT ALREADY HAS:",
                self._semester_block(existing), ""]
        parts += [
            f"COURSES TO PLACE ({len(courses)}), with their prerequisites:",
            self._course_block(courses, context_by_code),
            "",
        ]
        if findings:
            parts += ["WHAT THE VALIDATOR FOUND IN YOUR LAST SCHEDULE:"]
            parts += [f"  - {f}" for f in findings]
            parts.append("")
        parts.append("Return the complete schedule.")
        return static, "\n".join(parts), _static_hash(static, AUDIT_SCHEMA)

    def _semester_block(self, semesters: list[list[str]]) -> str:
        return "\n".join(
            f"  Semester {index + 1}: " + (", ".join(codes) if codes else "(empty)")
            for index, codes in enumerate(semesters)
        ) or "  (no schedule yet)"

    def _course_block(self, courses: list[str],
                      context_by_code: dict[str, str] | None = None) -> str:
        """The courses to place, GROUPED UNDER THE REQUIREMENT EACH ONE SATISFIES.

        WHY GROUPED, 2026-08-14. This block was a flat list of `{"code","credits",
        "prerequisites"}` rows — everything needed to SEQUENCE a course and nothing about what
        it was FOR. Models dropped courses, and they dropped them silently and consistently:
        every model missed requirements on the same programs, because from inside a flat list
        there is no difference between omitting a course and omitting a degree requirement. The
        model could see that CS 18200 needs CS 18000; it could not see that leaving out
        BIOL 22100 fails "Science Courses".

        Grouping restores that. Each requirement group prints its own heading with its kind
        (`all of` / `choose N credits`) and its courses with credits, so an omission is
        visibly an omission FROM SOMETHING. The prerequisite and corequisite facts stay on each
        course line, because they are what the sequencing decision needs.

        A course satisfying several groups is printed under each of them — that is the honest
        rendering (one course really does pay into several requirements) and it also tells the
        model that placing it once settles more than one line item. Courses in the plan that
        belong to no group (prerequisite closure supplies these — Calculus I for a degree that
        only names Calculus III) get their own trailing section, labelled for what they are.
        """
        by_code = {}
        if self.database is not None:
            by_code = {row["course_code"]: row for row in self.database.rows("courses")}
        wanted = list(dict.fromkeys(courses))

        def course_line(code: str) -> str:
            row = by_code.get(code) or {}
            credits = row.get("credit_hours_min")
            bits = [f"{code} ({credits:g} cr)" if credits else code]
            prereqs = row.get("prereq_groups") or []
            if prereqs:
                bits.append("needs " + " and ".join(
                    " or ".join(group) for group in prereqs if group))
            coreqs = row.get("coreq_codes") or []
            if coreqs:
                bits.append("with " + ", ".join(coreqs) + " in the same semester or earlier")
            line = "      " + "; ".join(bits)
            # RETRIEVED DETAIL, ATTACHED TO THE COURSE IT DESCRIBES (2026-08-15). This used to
            # be a separate "RETRIEVED CATALOG CONTEXT" block after the course list, which
            # printed each course's code twice — once where it had to be placed and once in a
            # paragraph the model had to cross-reference by hand. Inline, the description sits
            # where the decision is made and the second copy of the code disappears, so the
            # prompt gets shorter AND the association gets easier. Trimmed, because a full
            # catalog paragraph per course would swamp the structure this block exists to show.
            detail = (context_by_code or {}).get(code)
            if detail:
                trimmed = " ".join(detail.split())
                # The ingest prefixes every course chunk with "COURSE <CODE> — ", which is the
                # code we just printed. Stripping it keeps the line reading as one statement
                # about one course instead of naming it twice.
                prefix = f"COURSE {code} —"
                if trimmed.startswith(prefix):
                    trimmed = trimmed[len(prefix):].strip()
                if len(trimmed) > 220:
                    trimmed = trimmed[:217].rstrip() + "..."
                line += f" — {trimmed}"
            return line

        options_by_group: dict[str, list[str]] = {}
        if self.database is not None:
            for row in self.database.rows("requirement_options"):
                options_by_group.setdefault(row["requirement_group_id"], []).append(
                    row["course_code"])

        lines: list[str] = []
        claimed: set[str] = set()
        if self.database is not None:
            for row in self.database.rows("requirement_groups"):
                members = [c for c in wanted
                           if c in (options_by_group.get(row["id"]) or [])]
                if not members:
                    continue
                claimed.update(members)
                if row.get("requirement_type") == "choose_credits":
                    target = row.get("credits_min")
                    # "AT LEAST", not a bare figure. "choose 3 credits" reads as a quota to hit
                    # exactly, and courses do not come in sizes that make that possible — the
                    # only Written Communication options in English Education are a 3-credit and
                    # a 2-credit course, so every satisfying answer overshoots. A model reading
                    # the phrase as an equality has no legal move and picks the closest thing to
                    # 3, which is the 2-credit course, which fails the requirement.
                    kind = (f"take at least {target:g} credits' worth from this group — more "
                            f"than {target:g} is fine, less is not"
                            if target else "choose from")
                else:
                    kind = "take all of these"
                lines.append(f"  {row.get('name')} — {kind}:")
                lines += [course_line(c) for c in members]

        leftover = [c for c in wanted if c not in claimed]
        if leftover:
            lines.append("  Prerequisites for the above (required to reach them, not a "
                         "requirement group of their own):")
            lines += [course_line(c) for c in leftover]
        return "\n".join(lines)

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
        parts = [_profile_block(scenario.profile, scenario, self.database), ""]
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
