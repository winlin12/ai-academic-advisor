"""MODE B — the model writes the plan of study itself.

Until now this app had no Mode B call site: `model_eval/harness/prompts.py` says so in as many
words ("MODE B has no production counterpart on purpose... It exists to answer 'what would we
gain (or lose) by trusting the model with the schedule?'"). This module is the answer landing.
It is the eval's Mode B, shipped: same system prompt, same grammar-constrained response schema,
same database-export context — because the whole value of having measured Gemma 4 26B on that
task is lost if what we serve is a different task.

WHAT IS DIFFERENT FROM THE EVAL, and it is only ever additive:

    the model produces the schedule  ->  plan_validation.repair_plan deletes what is illegal
                                     ->  the deterministic planner backfills what is left over
                                     ->  the student sees a plan that is legal by construction

The eval scored raw model output because that is what it was measuring. A student does not want
a measurement, so nothing illegal survives to the screen. The eval's own Mode C established
that repair beats asking the model to fix itself — deletion is decidable without a model, costs
no GPU time, and cannot make the plan worse — so that is the strategy here, with one difference:
after repair, `planner.generate_plan` gets a pass at whatever is still unscheduled. Mode B's
measured weakness is coverage (it trails off on the requirement groups at the end of the list
and leaves late terms half empty), and the deterministic planner's measured strength is exactly
that. Neither alone is as good as the two in that order.

PROVENANCE IS REPORTED, never hidden. `AiPlanResult` carries which courses the model placed,
which were removed and why, and which the deterministic planner backfilled, so "the AI made
this" is a claim the UI can be specific about instead of a vibe.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.models.schemas import (
    PlannedCourse,
    PlanResponse,
    SemesterPlan,
    StudentProfile,
)
from app.services.vllm_client import VllmClient, ModelResponseError
from app.services.plan_context import approx_tokens, render_program_context
from app.services.plan_validation import (
    PlanValidation,
    normalize_code,
    repair_plan,
    term_sequence,
    validate_semesters,
)
from app.services.planner import generate_plan, planning_terms
from app.services.planner_db import ProgramCatalog, requirement_coverage

logger = logging.getLogger(__name__)


# =============================================================================================
# The prompt. Copied from model_eval/harness/prompts.py::PLAN_SYSTEM, deliberately verbatim.
# =============================================================================================
#
# EVERY LINE HERE WAS PAID FOR. The eval's copy carries ~80 lines of comments recording which
# failure each rule fixes and what happened when it was removed; that history is not duplicated
# here, but the rule text must not drift from it — if this prompt and the harness's disagree,
# the eval stops describing the system we run. Change both together, and re-measure.
#
# The short version of why the prompt looks like this: it was once ~60 lines of rules and
# worked examples sitting on top of an ~11,000-token database export, where every extra line of
# prose competes with the data for a small model's attention. It was stripped to the scored
# rules and nothing else, one line per constraint, in the order the checker applies them.
# Prerequisites lead because prerequisite ordering is the largest violation class in every
# sweep and the one constraint no grammar can express.
PLAN_SYSTEM = """\
You build a semester-by-semester plan of study from a read-only export of the advisor's
catalog database. Use only what is in that export.

Where things are:
- `courses` - every course that exists, with credit_hours_min. Listed in prerequisite order.
- `course_prerequisites` - `prereq_groups` is AND-of-ORs; `coreq_codes` may share a semester.
- `course_planner_terms` - `offered_terms`, the terms a course is taught in. The courses that
  are NOT taught in every term are listed by name above the rows.
- `requirement_groups` + `requirement_options` - the degree requirements, joined on
  course_code. `all_of` = take every option. `choose_credits` = take enough to reach
  `credits_min`.
- `course_aliases` - `alias_of` is an approved substitute. Take one, never both.

Rules, all hard:
- PREREQUISITES COME FIRST. Every prerequisite of a course must be in a STRICTLY EARLIER
  semester than that course, or already completed. The same semester is NOT early enough.
  Before you place a course, look it up in `course_prerequisites` and check that every code in
  its `prereq_groups` is already behind it; a course with no row there has no prerequisites.
  Only `coreq_codes` may share a semester. This is the constraint most plans get wrong, and one
  mistake anywhere invalidates the whole plan.
- NEVER SCHEDULE A COURSE THE STUDENT HAS ALREADY TAKEN. The student's "Already completed"
  line is the complete list of what is done; those courses are finished and must not appear
  anywhere in your plan. They still count as satisfying prerequisites and requirement groups —
  a completed course is credit the student already holds, not work still to do. Scheduling one
  again wastes a seat the student is paying for.
- NEVER SCHEDULE THE SAME COURSE TWICE. Each course appears at most once in the WHOLE plan, not
  once per semester. Before you add a course to a semester, check the semesters you have
  already written and the completed list. Substitutes in `course_aliases` count as the same
  course: take one, never both.
- Only courses in `courses`. Never invent a code.
- NOT EVERY COURSE RUNS EVERY TERM. Most do, and that is the trap: a handful are taught in one
  season only, and putting one of those in the wrong term is a seat that does not exist. The
  restricted courses are named in full at the top of `course_planner_terms` — read that list
  first, then place a course only in a term its `offered_terms` includes and the student's
  calendar contains. If the only term a course fits is already full, move something else out of
  that term; the course cannot move to a term it is not taught in.
- Never exceed the student's credit limit in a semester.
- Never exceed the student's limit on courses from their major's subject in a semester. This
  is separate from the credit limit; both apply.
- Cover every requirement group. A group is already covered if the student completed its
  courses — check the completed list before scheduling anything for it.

SPREAD THE WORK EVENLY. Divide the courses as evenly as you can across ALL the semesters the
student has, so every semester lands near the target credits — including the last ones. Do not
fill the early terms to the limit and leave the later ones nearly empty, and do not leave a
term light when something legal could go in it. Selective groups usually offer more options
than the student needs: when a term has room, pick an option whose prerequisites are already
satisfied and whose `offered_terms` include that term.

Put anything that does not fit in "unplanned". Never exceed a limit to fit it in."""


class AiSemester(BaseModel):
    term: str
    year: int
    courses: list[str] = Field(default_factory=list)


class AiPlanDraft(BaseModel):
    """Exactly the eval's PLAN_SCHEMA. `total_credits`, `major_course_count` and
    `requirements_covered` are deliberately absent: the eval required them as a forced
    self-check, measured that neither was ever read back (the checker recomputes both), and
    removed them when they cost ~440 output tokens and doubled median latency per plan."""

    semesters: list[AiSemester] = Field(default_factory=list)
    unplanned: list[str] = Field(default_factory=list)
    # Same bound and the same reason as PlanEditProposal.rationale: llama.cpp's grammar bounds
    # the SHAPE of the JSON, not the length of a string, and one model spent 390 s per record
    # proving it. Mode B is where it costs most — `semesters` is the only scored field and is
    # ~500 tokens, while the largest observed outputs ran to 6392 of which 88% was this string.
    rationale: str = Field(default="", max_length=600)


def plan_schema(terms: list[str], course_codes: list[str]) -> dict[str, Any]:
    """The response schema, with the term and course fields restricted to what exists.

    Encoding the restriction in the GRAMMAR rather than only in the prose is what makes it
    free: with summer excluded, the model literally cannot emit a summer semester, so "did it
    respect the calendar" stops competing for attention with the actual planning problem. Same
    for `courses` — hallucinated codes were only 3% of the eval's Mode B violations, but every
    one was the same category error (writing 'CS 37300:DATAMINING(3CR)' where a bare code
    belongs), and an enum ends that class outright.

    It cannot touch the other 97% — prereq order, duplicates, credit sums, the major cap. No
    grammar can express those, which is what plan_validation exists for.
    """
    codes: dict[str, Any] = {"type": "string"}
    if course_codes:
        # Empty enums are not a legal grammar, and a student with nothing left to schedule has
        # no plan to constrain anyway.
        codes = {"type": "string", "enum": sorted(course_codes)}
    return {
        "type": "object",
        "properties": {
            "semesters": {
                "type": "array",
                "description": "One entry per semester, in order, starting with the student's "
                               "start term.",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string", "enum": list(terms)},
                        "year": {"type": "integer"},
                        "courses": {
                            "type": "array",
                            "items": codes,
                            "description": "Course codes exactly as written in the catalog, "
                                           "e.g. 'CS 25100'.",
                        },
                    },
                    "required": ["term", "year", "courses"],
                },
            },
            "unplanned": {
                "type": "array", "items": codes,
                "description": "Required course codes that did not fit in the available "
                               "semesters.",
            },
            "rationale": {
                "type": "string", "maxLength": 600,
                "description": "Two or three sentences explaining the sequencing, addressed to "
                               "the student.",
            },
        },
        # REQUIRED, unlike the eval's schema. The eval left it optional because it scored
        # `semesters` and nothing else, and Gemma duly returned an empty string — fine for a
        # measurement, useless to a student looking at eight semesters with no explanation of
        # why they are in that order. It is last in the property order, so it costs ~50 tokens
        # after the part that matters and a truncated draft loses it rather than the plan.
        "required": ["semesters", "rationale"],
    }


# =============================================================================================
# The variable tail — the student.
# =============================================================================================


def credit_load_line(profile: StudentProfile, catalog: ProgramCatalog) -> str:
    """What is left to place, the shape to place it in, and the load never to exceed.

    NAMING ONLY THE CEILING WAS A MEASURED BUG. Across 162 Mode B plans the eval recorded mean
    credits by semester position of 13.7, 12.3, 12.0, 11.1, 11.0, 8.5, 7.9, 7.1 — a monotonic
    slide to half the opening load — while the single most common unmet requirement was a
    selective group left at 0 credits. The model was leaving courses unscheduled in terms that
    had room, because nothing ever told it the room was supposed to be used.

    The fix was a per-semester target ("aim for N"), and the TARGET CAME BACK OUT on 2026-08-12,
    matching the eval harness. A quota is not the objective — finishing the requirements is,
    evenly spread — and a student whose remaining credits divided by their semesters falls under
    the quota can only reach it by front-loading, which is the same failure one level up. The
    numerator stays, because it is the finishing condition and it makes an omission checkable:
    given only a per-term figure, a plan missing one 4-credit course looks locally sensible in
    every term. `target_credits_per_semester` is no longer read here; `planner` still honours it
    when a student asks for a specific load.
    """
    limit = profile.max_credits_per_semester
    remaining = sum(catalog.credits(code) for code in profile.remaining_courses)
    if not remaining:
        return (f"Credits per semester: never more than {limit}; spread the courses evenly "
                f"across all {profile.semesters_to_plan} semesters instead of filling the "
                f"early ones to the limit")
    return (f"Credits: {remaining} credits still outstanding — place all of them, spread "
            f"evenly across every semester available rather than front-loaded. {limit} is the "
            f"hard limit.")


def major_load_line(profile: StudentProfile) -> str:
    """The major-course cap. One number when there is only one number.

    Two limits on the same quantity, one of which the model is told to break when convenient,
    is a judgement call small models make badly and it competes for attention with the credit
    target and the prerequisite chain — so the soft preference is only stated when it is
    genuinely below the hard cap.
    """
    hard = profile.max_major_courses_per_semester
    preferred = profile.preferred_major_courses_per_semester
    if preferred >= hard:
        return f"{profile.major_subject} courses per semester: at most {hard}"
    return (f"{profile.major_subject} courses per semester: at most {hard} (hard limit), "
            f"{preferred} or fewer preferred")


def profile_block(profile: StudentProfile, catalog: ProgramCatalog) -> str:
    """The student, and — explicitly — the boundary of what they have already done.

    THE COMPLETED LIST IS STATED AS EXHAUSTIVE. Students describe their transcript in
    categories ("I transferred in with my math and gen-ed credits done"), and a model that
    reads the category instead of the list fills in what it thinks the category contains. In
    the eval, two of six models dropped CS 18000 from every spring-start plan — never
    scheduled, never listed as unplanned — on a scenario whose completed list contains no CS at
    all. Nothing in the data said intro programming was done; "transferred in" did.

    Saying the list is closed costs one clause and removes the inference. It is a statement
    about OUR data, not a rewrite of what the student said.
    """
    completed = ", ".join(profile.completed_courses) or "none"
    calendar = " -> ".join(
        f"{term} {year}"
        for term, year in term_sequence(profile, profile.semesters_to_plan)
    )
    label = ("Already completed (complete list)" if profile.completed_courses
             else "Already completed")
    return (
        f"DEGREE PROGRAM: {profile.degree_program}\n"
        f"{label}: {completed}\n"
        f"{credit_load_line(profile, catalog)}\n"
        f"{major_load_line(profile)}\n"
        f"Semesters available ({profile.semesters_to_plan}), in order:\n  {calendar}"
    )


def build_prompts(
    profile: StudentProfile, catalog: ProgramCatalog, request: str = ""
) -> tuple[str, str]:
    """(system, user). The structural invariant the eval enforces holds here too: everything
    static — the rules and the database export — is in the system message, and only the student
    and their request vary. That is what lets llama-server's KV cache reuse the ~11k-token
    prefix across every plan request for the same program instead of re-reading it each time.
    """
    system = f"{PLAN_SYSTEM}\n\n{render_program_context(catalog)}"
    parts = [profile_block(profile, catalog), ""]
    if request.strip():
        parts += [f"STUDENT'S REQUEST:\n{request.strip()}", ""]
    parts.append("Produce the plan of study.")
    return system, "\n".join(parts)


# =============================================================================================
# Result
# =============================================================================================


@dataclass
class AiPlanResult:
    plan: PlanResponse
    rationale: str
    # Provenance, so the UI can be specific about what the model actually did.
    model: str
    model_placed: list[str] = field(default_factory=list)
    # "model"   the model's own semester layout survived, repaired in place.
    # "planner" the deterministic planner laid the semesters out from the model's course
    #           ORDERING because that covered more of the degree. Both are AI-driven; only the
    #           first is the model's layout, and the warnings word themselves accordingly.
    layout: str = "model"
    removed: list[str] = field(default_factory=list)
    backfilled: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = field(default_factory=list)
    # False when the model could not be used at all and the deterministic planner answered.
    used_model: bool = True
    seed: int | None = None


def _to_plan_response(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    semesters: list[list[str]],
    unplanned: list[str],
    warnings: list[str],
) -> PlanResponse:
    by_code = catalog.by_code
    terms = term_sequence(profile, max(len(semesters), profile.semesters_to_plan))
    out: list[SemesterPlan] = []
    for index, (term, year) in enumerate(terms[:profile.semesters_to_plan]):
        codes = semesters[index] if index < len(semesters) else []
        courses = [by_code[code] for code in codes if code in by_code]
        out.append(SemesterPlan(
            term=term,
            year=year,
            courses=[PlannedCourse(code=c.code, title=c.title, credits=c.credits)
                     for c in courses],
            total_credits=sum(c.credits for c in courses),
            warnings=[],
        ))
    return PlanResponse(
        profile_label=profile.profile_label,
        degree_program=profile.degree_program,
        semesters=out,
        unplanned_courses=unplanned,
        warnings=warnings,
    )


def _warnings(
    check: PlanValidation,
    removed: list[str],
    backfilled: list[str],
    missing: list[str],
    layout: str = "model",
) -> list[str]:
    """The student-facing account of what happened to the draft.

    ``layout`` decides the WORDING, and getting it wrong is a false statement about the
    catalog. When the model's own layout survived, a dropped course really was unschedulable
    where the model put it. When the deterministic candidate won instead, the differences are
    just two plans choosing different options from the same selective menus — saying those
    courses "were not schedulable" would tell a student a course they can perfectly well take
    is unavailable.
    """
    warnings: list[str] = []
    if removed:
        warnings.append(
            ("Removed from the AI's draft because they were not schedulable there: "
             if layout == "model" else
             "In the AI's draft but not in this layout — a different set of options covers "
             "the same requirements: ")
            + ", ".join(sorted(set(removed)))
        )
    if backfilled:
        warnings.append(
            ("Added by the deterministic planner to fill terms the draft left short: "
             if layout == "model" else
             "Chosen by the deterministic planner, which laid these semesters out from the "
             "AI's course ordering: ")
            + ", ".join(backfilled)
        )
    warnings += [str(v) for v in check.soft]
    warnings += [f"Requirement not yet covered — {item}" for item in missing]
    warnings += [str(v) for v in check.hard]      # empty unless repair hit its pass limit
    return warnings


async def generate_ai_plan(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    *,
    client: VllmClient,
    request: str = "",
    seed: int | None = None,
) -> AiPlanResult:
    """Mode B end to end: model drafts, checker repairs, planner backfills.

    ``seed`` varies the draft. Without it, a student pressing "regenerate" on a plan they do
    not like gets the same plan back — at temperature 0.15 an identical request is very nearly
    deterministic.

    Falls back to the deterministic planner on any model failure. A student who cannot reach
    llama-server should still get a plan; they should just be told which one they got.
    """
    system, user = build_prompts(profile, catalog, request)
    codes = [course.code for course in catalog.courses]
    schema = plan_schema(planning_terms(), codes)

    # THE OUTPUT BUDGET IS WHAT IS LEFT, not what was configured. llama-server's context is
    # fixed at launch and a request that asks to generate past it does not fail loudly — the
    # plan is cut off mid-JSON and arrives unparseable, which reads to the student as "the AI
    # planner was unavailable". Ask for what actually fits.
    prompt_tokens = approx_tokens(system) + approx_tokens(user)
    room = settings.vllm_context_tokens - prompt_tokens - _WINDOW_SAFETY_TOKENS
    max_tokens = min(settings.vllm_plan_max_tokens, max(room, 0))
    if max_tokens < _MIN_USEFUL_PLAN_TOKENS:
        # Not enough window for even a short plan. The deterministic planner does not read the
        # export at all, so it is unaffected by whatever made this program's catalog so large.
        logger.warning(
            "Mode B prompt is ~%d tokens against a %d-token window, leaving %d for the plan; "
            "using the deterministic planner instead. Lower plan_context_token_budget or "
            "relaunch llama-server with a bigger --ctx-size.",
            prompt_tokens, settings.vllm_context_tokens, max_tokens,
        )
        return _deterministic_result(
            profile, catalog, client.model, seed,
            "This program's catalog is too large for the AI planner's context window, so this "
            "plan was built by the deterministic planner.",
        )

    draft: AiPlanDraft | None = None
    try:
        draft = await client.propose(
            system, user, AiPlanDraft, schema=schema, seed=seed, max_tokens=max_tokens,
        )
    except (ModelResponseError, ValidationError) as exc:
        draft = _salvage_semesters(getattr(exc, "raw", "") or "")
        if draft is None:
            logger.warning(
                "Mode B draft unusable (%s); falling back to the deterministic planner", exc)
            return _deterministic_result(
                profile, catalog, client.model, seed,
                "The AI planner was unavailable, so this plan was built by the deterministic "
                "planner: courses in catalog order, respecting prerequisites, term offerings "
                "and your credit cap.",
            )

    return _finish_draft(profile, catalog, draft, client.model, seed)


def _finish_draft(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    draft: AiPlanDraft,
    model: str,
    seed: int | None,
) -> AiPlanResult:
    """Repair the model's draft and report what changed. NOTHING is auto-filled.

    Split from ``generate_ai_plan`` so a salvaged truncated draft takes exactly the same path as
    a clean one — a partial plan that skipped the checker would be the one way an illegal
    schedule could reach the screen.

    TWO AUTO-FILL MECHANISMS USED TO LIVE HERE AND BOTH ARE GONE. A deterministic backfill
    packed leftover courses into whatever room was free, and a second candidate re-ran the
    greedy planner over the model's ordering and replaced the layout wholesale when it covered
    more groups. Both raised the coverage number; neither was the app's decision to make. A
    selective group is a MENU — "choose 6 credits from these seventeen" — and taking the first
    two in catalog order is a coin toss, not advice. A student who sees CS 44800 in their
    schedule is entitled to assume somebody chose it.

    So the plan comes back with visible holes. ``plan_requirements`` turns those into the thing
    the student acts on, and Mode C's "Fill the gaps" is there for when they would rather the
    model had a go.
    """
    # Only the terms the student actually has. A model that emits a ninth semester for an
    # eight-semester horizon has produced courses nobody can take; they become unplanned rather
    # than silently extending the plan past the date the student asked to graduate.
    drafted = [[normalize_code(code) for code in semester.courses]
               for semester in draft.semesters][:profile.semesters_to_plan]
    # PADDED TO THE FULL HORIZON before anything else touches it. A model that answers with
    # three semesters for an eight-semester student has not said the other five do not exist —
    # but `repair_plan` can only relocate a course into a term that is IN THE LIST it was given,
    # so an unpadded draft leaves it nothing to move into and it deletes instead. That was
    # invisible while a backfill ran afterwards and quietly put the courses back; with the
    # backfill gone it is the difference between a rescued chain and a deleted one.
    drafted += [[] for _ in range(profile.semesters_to_plan - len(drafted))]
    model_placed = [code for term in drafted for code in term]

    # RESCUE BEFORE DELETE — repair_plan places missing prerequisites and slides courses to
    # later terms, and only deletes what nothing can save. A draft that forgot one root course
    # is not a bad draft, and deleting everything downstream of the omission would destroy the
    # good work to punish the one mistake.
    repaired, removed, inserted, _ = repair_plan(
        drafted, profile, catalog.courses, canonical=catalog.canonical
    )

    scheduled = {code for term in repaired for code in term}
    unplanned = [code for code in profile.remaining_courses if code not in scheduled]
    # `inserted` is not a fill: repair only ever adds a PREREQUISITE of something the plan
    # already contains, which is forced rather than chosen.
    backfilled = list(inserted)

    final = validate_semesters(repaired, profile, catalog.courses, canonical=catalog.canonical)

    canonical = catalog.canonical
    satisfied = {canonical.get(c, c) for c in profile.completed_courses}
    satisfied |= {canonical.get(code, code) for term in repaired for code in term}
    coverage, missing = requirement_coverage(catalog, satisfied)

    plan = _to_plan_response(
        profile, catalog, repaired, unplanned,
        _warnings(final, removed, backfilled, missing),
    )
    return AiPlanResult(
        plan=plan,
        rationale=draft.rationale or "Here is a plan of study that fits your remaining "
                                     "requirements into the semesters you have.",
        model=model,
        model_placed=model_placed,
        removed=sorted(set(removed)),
        backfilled=backfilled,
        violations=[str(v) for v in final.violations],
        requirement_coverage=coverage,
        missing_requirements=missing,
        seed=seed,
    )


# Room reserved for what the estimate cannot see: the chat template's own wrapping tokens, the
# grammar's overhead, and the gap between _CHARS_PER_TOKEN and whatever this particular model's
# tokenizer does with a course code.
_WINDOW_SAFETY_TOKENS = 600

# Below this there is no point asking: a plan of study's `semesters` payload measured ~470-550
# tokens at the median across model_eval's corpus, so anything under 600 cannot hold one.
_MIN_USEFUL_PLAN_TOKENS = 600


def _deterministic_result(
    profile: StudentProfile,
    catalog: ProgramCatalog,
    model: str,
    seed: int | None,
    rationale: str,
) -> AiPlanResult:
    """The honest fallback: a real, legal plan, labelled as not having come from the model."""
    fallback = generate_plan(profile, catalog.courses)
    canonical = catalog.canonical
    satisfied = {canonical.get(c, c) for c in profile.completed_courses}
    satisfied |= {canonical.get(course.code, course.code)
                  for semester in fallback.semesters for course in semester.courses}
    coverage, missing = requirement_coverage(catalog, satisfied)
    fallback.warnings = [
        *fallback.warnings,
        "This plan came from the deterministic planner, not the AI planner.",
        *(f"Requirement not yet covered — {item}" for item in missing),
    ]
    return AiPlanResult(
        plan=fallback, rationale=rationale, model=model, used_model=False, seed=seed,
        requirement_coverage=coverage, missing_requirements=missing,
    )


def _salvage_semesters(raw: str) -> AiPlanDraft | None:
    """Recover the complete semesters from a plan that ran out of output tokens.

    A draft cut off mid-JSON is not a wasted generation: the grammar emits `semesters` first
    and one complete object at a time, so a plan truncated in semester 7 still holds six valid
    semesters — most of the work, and the repair-and-backfill steps downstream can finish the
    rest. Throwing it away instead means a student waits forty seconds to be told the planner
    was unavailable, which is both slower and less true.

    Scans with a depth counter rather than a regex because the payload nests (`courses` is an
    array inside each semester object) and because it must not be fooled by a brace inside a
    string. Returns None when nothing complete was produced.
    """
    start = raw.find('"semesters"')
    if start == -1:
        return None
    start = raw.find("[", start)
    if start == -1:
        return None

    complete: list[str] = []
    depth = 0
    in_string = False
    escaped = False
    object_start = -1
    for index in range(start + 1, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and object_start != -1:
                complete.append(raw[object_start:index + 1])
                object_start = -1
        elif char == "]" and depth == 0:
            break

    if not complete:
        return None
    try:
        draft = AiPlanDraft.model_validate_json(
            '{"semesters":[' + ",".join(complete) + "]}"
        )
    except ValidationError:
        return None
    logger.info("Salvaged %d complete semesters from a truncated Mode B draft", len(complete))
    return draft


def context_size(catalog: ProgramCatalog) -> dict[str, int]:
    """Rough token accounting for the Mode B prompt. Used by the health route so an operator
    can see the export fits the launched context window before a student finds out it doesn't."""
    export = render_program_context(catalog)
    return {
        "rules_chars": len(PLAN_SYSTEM),
        "export_chars": len(export),
        "approx_prompt_tokens": (len(PLAN_SYSTEM) + len(export)) // 4,
        "context_tokens": settings.vllm_context_tokens,
        "plan_max_tokens": settings.vllm_plan_max_tokens,
    }


__all__ = [
    "PLAN_SYSTEM",
    "AiPlanDraft",
    "AiPlanResult",
    "build_prompts",
    "context_size",
    "generate_ai_plan",
    "plan_schema",
    "profile_block",
]
