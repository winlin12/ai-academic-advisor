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
from .planner import Course, Plan, Profile, first_planning_term, next_term


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# =============================================================================================
# MODE A — revise-plan proposal. Copied in spirit and in wording from
# backend/app/services/advisor_agent.py::_SYSTEM_PROMPT so the harness measures the prompt the
# app really ships. If that prompt changes, change this one and expect the hash to move.
# =============================================================================================

PROPOSAL_SYSTEM = (
    "You are an assistant that tunes a college course plan from a student's feedback. You are "
    "NOT an official advisor and you must not invent courses, prerequisites, or requirements. "
    "A deterministic planner owns legality (prerequisites, term offerings, credit caps); you "
    "only express preferences over the courses already listed. Only use course codes and tags "
    "that appear in the context. Leave a list empty if it does not apply. Never put a course "
    "in both reorder and defer."
)

# Mirrors app.models.schemas.PlanEditProposal. The app sends this schema to llama.cpp as a
# grammar-constrained response_format; so does the harness. Field descriptions are part of the
# prompt the model sees, so they are reproduced verbatim.
PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
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
            "description": "New per-semester credit cap, only if the student asked for one.",
        },
    },
    "required": ["rationale", "reorder", "defer", "avoid_tags"],
}


# =============================================================================================
# MODE B — free-form plan of study.
# =============================================================================================

PLAN_SYSTEM = """\
You are an academic planning assistant building a semester-by-semester plan of study for an
undergraduate student. You are not an official advisor.

Build the plan ONLY from the course catalog given to you. Every rule below is hard:

- Never schedule a course that is not in the catalog. Never invent a course code.
- A course may only be scheduled in a semester whose term appears in its "terms" list.
- Every prerequisite of a course must be scheduled in an EARLIER semester, or already be in
  the student's completed list. Taking a course in the same semester as its prerequisite does
  not satisfy it.
- No semester may exceed the student's credit limit.
- Never schedule a course the student has already completed, and never schedule the same
  course twice.
- Only use the terms listed in the student's calendar below. Summer is NOT available for
  planning even when a course is offered in it.
- Cover every degree requirement listed. For a "choose" requirement, schedule enough of its
  options to reach the stated credits.

If the requirements cannot all fit in the number of semesters available, still produce the
best legal plan you can and list what did not fit in "unplanned". A short legal plan is
better than a long illegal one."""

def plan_schema(planning_terms: list[str]) -> dict[str, Any]:
    """Mode B's response schema, with the term enum restricted to schedulable terms.

    Encoding the restriction in the GRAMMAR rather than only in the prose matters: with summer
    excluded, a model literally cannot emit a summer semester, so "did it respect the calendar"
    stops competing for the model's attention with the actual planning problem. The prose rule
    stays too, because the enum alone doesn't explain *why*.
    """
    schema = json.loads(json.dumps(PLAN_SCHEMA))
    schema["properties"]["semesters"]["items"]["properties"]["term"]["enum"] = list(planning_terms)
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
                },
                "required": ["term", "year", "courses"],
            },
        },
        "unplanned": {
            "type": "array", "items": {"type": "string"},
            "description": "Required course codes that did not fit in the available semesters.",
        },
        "rationale": {
            "type": "string",
            "description": "Two or three sentences explaining the sequencing, addressed to the student.",
        },
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


def catalog_block(catalog: list[Course]) -> str:
    """The model's entire allowed palette, sorted so the bytes are stable.

    Prereqs and terms are printed inline rather than as a separate table: a small model
    scheduling CS 38100 should not have to join two lists in its head to find out that it
    needs MA 26100 first.
    """
    lines = []
    for course in sorted(catalog, key=lambda c: c.code):
        prereqs = ", ".join(course.prereqs) or "none"
        terms = "/".join(course.offered_terms) or "unknown"
        tags = ", ".join(course.requirement_tags) or "none"
        lines.append(
            f"- {course.code} \"{course.title}\" | {course.credits} cr "
            f"| terms: {terms} | prereqs: {prereqs} | tags: {tags}"
        )
    return "\n".join(lines)


def requirements_block(fixture: Fixture) -> str:
    lines = []
    for group in fixture.requirement_groups:
        if group.get("kind") == "all":
            lines.append(f"- {group['name']}: take ALL of {', '.join(group['courses'])}")
        else:
            lines.append(
                f"- {group['name']}: choose {group.get('choose_credits', 0):g} credits from "
                f"{', '.join(group['courses'])}"
            )
    return "\n".join(lines)


def _profile_block(profile: Profile) -> str:
    completed = ", ".join(profile.completed_courses) or "none"
    calendar = " -> ".join(
        f"{term} {year}" for term, year in _term_sequence(profile)
    )
    return (
        f"STUDENT: {profile.name} — {profile.degree_program}\n"
        f"Already completed: {completed}\n"
        f"Credit limit per semester: {profile.max_credits_per_semester}\n"
        f"Semesters available ({profile.semesters_to_plan}), in order:\n  {calendar}"
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
    def __init__(self, fixture: Fixture):
        self.fixture = fixture
        self._catalog = catalog_block(fixture.catalog)
        self._requirements = requirements_block(fixture)

    # -- Mode A: revise-plan proposal (the app's real path) ---------------------------------

    def plan_proposal(
        self, scenario: Scenario, plan: Plan, prior_warnings: list[str] | None = None
    ) -> tuple[str, str, str]:
        """Returns (system, user, static_hash). Port of ``advisor_agent._user_prompt``."""
        parts = [
            f"STUDENT: {scenario.profile.name} — {scenario.profile.degree_program}",
            f"Current credit cap: {scenario.profile.max_credits_per_semester} per semester.",
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
        return PROPOSAL_SYSTEM, "\n".join(parts), _sha256(PROPOSAL_SYSTEM)

    # -- Mode B: free-form plan of study ----------------------------------------------------

    def plan_freeform(self, scenario: Scenario) -> tuple[str, str, str]:
        """Mode B prompt. Uses ``plan_schema(TERM_ORDER)`` at call time in the runner."""
        static = f"{PLAN_SYSTEM}\n\nCOURSE CATALOG:\n{self._catalog}\n\nDEGREE REQUIREMENTS:\n{self._requirements}"
        user = (
            f"{_profile_block(scenario.profile)}\n\n"
            f"STUDENT'S REQUEST:\n{scenario.feedback}\n\n"
            f"Produce the plan of study."
        )
        return static, user, _sha256(static)

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
