from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Course(BaseModel):
    code: str
    title: str
    credits: int = Field(ge=0)
    # The FLAT union of every prerequisite code, for display and for "why is this blocked"
    # warnings. It is not the satisfaction rule: a flat AND-list cannot express
    # "CS 25100 OR CS 25300" without demanding both — see ``prereq_groups``.
    prereqs: list[str] = Field(default_factory=list)
    # AND-of-ORs, as `advisor.course_prerequisites.prereq_groups` stores it:
    # [["CS 25000"], ["CS 25100", "CS 25300"]] = CS 25000 AND (CS 25100 OR CS 25300).
    # Empty means "fall back to ``prereqs``", which keeps every catalog source that only has a
    # flat list (courses.json, hand-built test fixtures) working unchanged.
    prereq_groups: list[list[str]] = Field(default_factory=list)
    # Satisfied by taking the course earlier OR in the same semester. The planner is
    # deliberately conservative and requires them earlier (always legal); only validation that
    # judges a plan the student already built applies the looser same-semester rule.
    coreqs: list[str] = Field(default_factory=list)
    offered_terms: list[str] = Field(default_factory=list)
    requirement_tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class AcademicFacetResponse(BaseModel):
    catalog_years: list[int] = Field(default_factory=list)
    schools: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)


class AcademicCourseResult(BaseModel):
    id: str
    subject: str
    number: str
    code: str
    title: str
    credit_hours: float | None = None
    description: str | None = None


class AcademicCourseSearchResponse(BaseModel):
    """Envelope for ``GET /v1/academic/courses/search`` so the result list is schema-typed."""

    courses: list[AcademicCourseResult] = Field(default_factory=list)


class AcademicProgramSummary(BaseModel):
    id: str
    catalog_year: int
    school: str | None = None
    program_title: str
    degree_code: str | None = None
    variant: str | None = None
    source_url: str
    parser_status: str
    block_count: int = 0
    course_count: int = 0
    linked_course_count: int = 0


class RequirementCourseOption(BaseModel):
    id: str
    sort_order: int
    course_code_text: str
    course_id: str | None = None
    course_title: str | None = None
    credits_text: str | None = None
    raw_text: str | None = None


class RequirementRuleOption(BaseModel):
    id: str
    option_index: int
    sort_order: int
    label: str | None = None
    courses: list[RequirementCourseOption] = Field(default_factory=list)


class RequirementRuleDetail(BaseModel):
    id: str
    sort_order: int
    rule_type: str
    choose_count: int | None = None
    # A "choose" rule stated in CREDITS rather than in a course count ("choose 6 credits from
    # the following"). Both forms occur in the catalog and they are not interchangeable — six
    # credits is two courses or one, depending on the courses — so the audit needs the number
    # to decide whether the rule is met.
    credits_min: float | None = None
    raw_text: str | None = None
    options: list[RequirementRuleOption] = Field(default_factory=list)


class RequirementBlockDetail(BaseModel):
    id: str
    sort_order: int
    title: str | None = None
    credits_text: str | None = None
    rules: list[RequirementRuleDetail] = Field(default_factory=list)


class AcademicProgramDetail(BaseModel):
    id: str
    catalog_year: int
    school: str | None = None
    program_title: str
    degree_code: str | None = None
    variant: str | None = None
    source_url: str
    parser_status: str
    blocks: list[RequirementBlockDetail] = Field(default_factory=list)


# --- Degree audit: the same requirement tree, annotated with what the student has done -------
#
# SATISFACTION IS THREE-VALUED, and that is the point of these models. `satisfied=None` means
# NOT DECIDABLE — a rule whose text the parser could not resolve into course options ("complete
# the College of Science core"), which is a different fact from "not yet met" and must not be
# rendered as a red cross. Anything the audit cannot decide, it says it cannot decide, and the
# totals below count only the rules it could.


class RequirementCourseProgress(RequirementCourseOption):
    satisfied: bool = False
    # "completed" | "planned" | None — which list matched. A course completed for real always
    # wins over merely being planned, so the two can never both be true.
    satisfied_by: str | None = None


class RequirementRuleOptionProgress(BaseModel):
    id: str
    option_index: int
    sort_order: int
    label: str | None = None
    courses: list[RequirementCourseProgress] = Field(default_factory=list)
    satisfied: bool | None = None


class RequirementRuleProgress(BaseModel):
    id: str
    sort_order: int
    rule_type: str
    choose_count: int | None = None
    credits_min: float | None = None
    raw_text: str | None = None
    options: list[RequirementRuleOptionProgress] = Field(default_factory=list)
    satisfied: bool | None = None


class RequirementBlockProgress(BaseModel):
    id: str
    sort_order: int
    title: str | None = None
    credits_text: str | None = None
    rules: list[RequirementRuleProgress] = Field(default_factory=list)
    satisfied: bool | None = None


class ProgramAuditResponse(BaseModel):
    """A program's requirement tree with every node annotated against a student's record.

    ``total_requirements`` counts only the DECIDABLE rules, so the fraction it forms with
    ``satisfied_requirements`` is honest: a program whose text the parser mostly could not
    resolve reports a small denominator rather than a bad score.
    """

    id: str
    catalog_year: int
    school: str | None = None
    program_title: str
    degree_code: str | None = None
    variant: str | None = None
    blocks: list[RequirementBlockProgress] = Field(default_factory=list)
    total_requirements: int = 0
    satisfied_requirements: int = 0


class StudentProfile(BaseModel):
    # A LABEL for telling profiles apart on this device — "backup plan", "what-if stats minor".
    # Optional, never required, and never sent to the model: it is not the student's name, and
    # nothing in this app should ask a prospective student to type one.
    profile_label: str = ""
    degree_program: str = "Computer Science"
    # Academic-DB program (programs.id). When set, ``remaining_courses`` is derived from the
    # program's requirement rows minus ``completed_courses`` — client-listed codes are ignored.
    program_id: str | None = None
    completed_courses: list[str] = Field(default_factory=list)
    remaining_courses: list[str] = Field(default_factory=list)
    start_term: str = "fall"
    start_year: int = 2026
    semesters_to_plan: int = Field(default=4, ge=1, le=12)
    max_credits_per_semester: int = Field(default=9, ge=1, le=24)
    # The load to AIM FOR, as opposed to the load never to exceed. None = derive it per term as
    # the even split of the credits left over the semesters left, which spreads a plan across
    # the student's whole horizon instead of filling early terms to the cap and leaving the
    # tail empty. Set it equal to max_credits_per_semester for the old front-loading shape.
    target_credits_per_semester: int | None = Field(default=None, ge=1, le=24)
    # A second load cap, on how many of the MAJOR's own courses land in one term. Credits alone
    # cannot express it: four 4-credit CS courses and one CS course plus three gen-eds are both
    # 16 credits and nothing like the same semester. ``max_`` is never exceeded (the course
    # moves to a later term instead); ``preferred_`` is only relaxed when nothing else fits the
    # room left in the term. Students overwhelmingly report 3 as the ceiling and 2 as the
    # comfortable load, which is what these defaults are.
    major_subject: str = "CS"
    max_major_courses_per_semester: int = Field(default=3, ge=1, le=8)
    preferred_major_courses_per_semester: int = Field(default=2, ge=1, le=8)
    # A world-language subject code the student wants to study, e.g. "SPAN" — narrows a
    # detected language-sequence requirement (see language_requirements.py) down to that
    # language's own courses instead of every language the catalog offers. None = undecided;
    # the requirement still narrows to one representative (lowest-level) course per language
    # rather than showing the full cross-product.
    language_of_interest: str | None = None
    # Roughly 1-3, matching Purdue's own "Level I/II/III" scheme (verified against real
    # crawled course titles — do not assume more levels exist for every language). None =
    # unknown; treated as level 1 (start from the beginning).
    language_proficiency: int | None = Field(default=None, ge=1, le=3)
    preferences: dict[str, str | int | bool | list[str]] = Field(default_factory=dict)


class PlannedCourse(BaseModel):
    code: str
    title: str
    credits: int


class SemesterPlan(BaseModel):
    term: str
    year: int
    courses: list[PlannedCourse]
    total_credits: int
    warnings: list[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    profile_label: str
    degree_program: str
    semesters: list[SemesterPlan]
    unplanned_courses: list[str]
    warnings: list[str]


class AdminTableInfo(BaseModel):
    """One browsable table and its current row count (admin visibility, read-only)."""

    name: str
    row_count: int


class AdminTablesResponse(BaseModel):
    tables: list[AdminTableInfo] = Field(default_factory=list)


class AdminTableRowsResponse(BaseModel):
    """One page of rows from a whitelisted table, serialised as plain JSON objects.

    ``columns`` is the union of keys across the returned page (JSONB rows don't carry
    the table's schema), so the UI can render a stable grid without a second query.
    """

    table: str
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class AiPlanRequest(BaseModel):
    """Ask the local model to write the plan of study itself (MODE B).

    ``request`` is optional free text the student typed alongside their profile ("I want to
    graduate a semester early", "keep my Fridays clear"); it travels in the variable tail of
    the prompt, never in the static block, so llama-server's KV cache still reuses the
    ~11k-token catalog export across students on the same program.

    ``seed`` is what makes "regenerate" mean something. At temperature 0.15 an identical
    request is very nearly deterministic, so a student who dislikes a plan and presses the
    button would otherwise be handed the same plan back. The client sends a fresh number.
    """

    profile: StudentProfile
    request: str = ""
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class AiPlanResponse(BaseModel):
    """The AI-drafted plan after deterministic repair, with its provenance made explicit.

    Every field below the plan exists so the UI never has to describe what happened vaguely.
    A student is entitled to know that four of their courses came from the model, one was
    deleted because it was scheduled before its prerequisite, and two were added by the
    deterministic planner to fill a term the model left empty.
    """

    plan: PlanResponse
    rationale: str
    model: str
    # False when llama-server was unreachable or returned nothing usable, and the deterministic
    # planner answered instead. The UI says so rather than passing it off as an AI plan.
    used_model: bool = True
    # "model"   the model's own semester layout survived (repaired in place).
    # "planner" the deterministic planner laid the semesters out from the model's course
    #           ORDERING, because that covered more of the degree. Both are AI-driven, but only
    #           the first is the model's layout — and the difference decides whether a dropped
    #           course was "unschedulable" or merely "not the option this layout picked".
    layout: str = "model"
    model_placed: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    backfilled: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = Field(default_factory=list)
    seed: int | None = None


class RefinePlanRequest(BaseModel):
    """MODE C — what the student does after reading the plan.

    Three policies, kept distinct because they are different capabilities and averaging them
    would describe none of them (see ``services/plan_refine``):

      "fill"        freeze everything that validates, ask the model only for what is missing.
      "regenerate"  tell the model exactly what is wrong and make it fix the plan itself.
                    Seed held fixed, so what changes between attempts is the context alone.
      "start-over"  a fresh sample, told nothing about the plan it replaces.

    ``plan`` is the layout currently on screen — the thing being refined. ``attempt`` only
    matters for "start-over", where it creeps the temperature so repeated presses explore
    instead of redrawing the same plan.
    """

    profile: StudentProfile
    plan: PlanResponse
    mode: Literal["fill", "regenerate", "start-over"]
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)
    attempt: int = Field(default=1, ge=1, le=20)


class PlanRequirementsRequest(BaseModel):
    """Cross-reference a program's requirements against a transcript and the plan on screen."""

    profile: StudentProfile
    plan: PlanResponse


class RefinePlanResponse(AiPlanResponse):
    """The refined plan, plus what this particular round did to it."""

    mode: str = "fill"
    # FILL only: the placements that survived the checker and were then FROZEN against further
    # change. This is the ratchet made visible — a student needs to see that pressing Fill did
    # not re-roll the semesters they were happy with. "Survived", not "never moved": a course
    # scheduled before its own prerequisite is slid to a legal term first, then frozen there.
    kept: list[str] = Field(default_factory=list)
    # A machine-readable reason when the model was not called at all ("nothing-to-fill",
    # "model-unavailable"), so the UI can say something specific rather than showing an
    # unchanged plan with no explanation.
    note: str = ""


class ExplainPlanRequest(BaseModel):
    """Ask the local model to explain an already-generated plan.

    ``plan`` is the full structured :class:`PlanResponse` (not a raw ``dict``) so malformed
    client payloads are rejected at the FastAPI boundary instead of being interpolated
    verbatim into the model prompt.
    """

    question: str = Field(min_length=1)
    plan: PlanResponse
    # Regenerate support: a chatbot's "try again" has to actually try again.
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class ExplainPlanResponse(BaseModel):
    answer: str


PlanEditOperation = Literal["move", "add", "remove"]


class PlanEditRequest(BaseModel):
    """One deterministic, direct manipulation of an existing plan — no LLM involved.

    ``target_semester`` is a zero-based index into ``plan.semesters`` (the layout being
    edited), required for ``move``/``add`` and ignored for ``remove``. ``profile`` is
    optional context: when present, its ``completed_courses`` seed the prerequisite check
    and ``max_credits_per_semester`` drives credit-cap warnings; without it the edit still
    applies but validation can only reason from the plan itself.
    """

    plan: PlanResponse
    operation: PlanEditOperation
    course_code: str = Field(min_length=1)
    target_semester: int | None = Field(default=None, ge=0)
    profile: StudentProfile | None = None

    @model_validator(mode="after")
    def _require_target_for_placement(self) -> "PlanEditRequest":
        if self.operation in ("move", "add") and self.target_semester is None:
            raise ValueError("target_semester is required for 'move' and 'add' operations")
        return self


class PlanEditProposal(BaseModel):
    """A structured edit the model proposes in response to free-text feedback.

    The model never emits a schedule directly — it only nudges the knobs the deterministic
    planner already understands, and the planner re-derives a legal plan from them. Every
    field is a preference, not a command: unknown course codes are ignored and the credit cap
    is clamped, so a hallucinated proposal degrades to a no-op rather than an illegal plan.

    This model doubles as the structured-output schema sent to the Anthropic API (the field
    descriptions below are what the model sees), so the API guarantees every proposal
    validates — there is no JSON-repair path.
    """

    # max_length is LOAD-BEARING, not tidiness. llama.cpp turns this schema into a GBNF
    # grammar, and a bare `"type": "string"` accepts any number of characters — so the
    # grammar constrains the SHAPE of the proposal and nothing about its size. model_eval
    # 2026-07-29 caught what that costs: with thinking disabled at the template level,
    # qwen3.6-35b-a3b relocated its reasoning INTO this field, wrote a 14,690-character
    # rationale on one scenario, and on three others fell into a verbatim repetition loop
    # ("I will output empty lists. But wait, I..." x11) that ran to the token ceiling
    # without ever closing the JSON — 9 of 28 proposals lost, ~390 s each. Nothing else in
    # the stack can stop that: temperature 0.15 makes a loop absorbing, and llama.cpp
    # disables repetition penalties by default (penalty_repeat 1.0, dry_multiplier 0.0).
    # A maxLength makes it structurally impossible instead of merely discouraged, and it
    # enforces the "one or two sentences" this description already asks for.
    rationale: str = Field(
        default="",
        max_length=400,
        description="One or two sentences explaining the change, addressed to the student.",
    )
    reorder: list[str] = Field(
        default_factory=list,
        description="Course codes to take earlier, highest priority first.",
    )
    defer: list[str] = Field(
        default_factory=list,
        description="Course codes to push to later semesters.",
    )
    avoid_tags: list[str] = Field(
        default_factory=list,
        description="Requirement tags to deprioritise (e.g. 'theory-heavy').",
    )
    max_credits_per_semester: int | None = Field(
        default=None,
        ge=1,
        le=24,
        # "only if the student asked for one" read as discouragement: models left this null and
        # put the requested number in the rationale, where the planner never sees it. Phrased
        # as an obligation instead, since this field is the only path a requested load has.
        description=(
            "Set this to the number of credits the student asked for whenever they name a "
            "per-semester load (e.g. 'keep me at 12 credits'). Null only if they did not ask."
        ),
    )


class RevisePlanRequest(BaseModel):
    """Ask the local model to revise a plan from free-text feedback.

    ``current_plan`` is optional context for the UI round-trip; the agent always re-derives a
    fresh baseline from ``profile`` so legality never depends on client-supplied state.

    ``planner`` decides who builds the revised schedule once the model has turned the feedback
    into a structured proposal (MODE A). "ai" re-runs Mode B with the proposal folded in, which
    is what keeps a revision looking like the plan it revised; "deterministic" hands the
    proposal to the greedy planner, which is faster and needs no GPU. The proposal step is the
    same either way — it is the only part that has to understand what the student asked for.

    ``seed`` regenerates: same feedback, different draft.
    """

    profile: StudentProfile
    feedback: str = Field(min_length=1)
    current_plan: PlanResponse | None = None
    planner: Literal["ai", "deterministic"] = "ai"
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class RevisePlanResponse(BaseModel):
    """The revised (still deterministically-validated) plan plus the model's reasoning."""

    plan: PlanResponse
    rationale: str
    proposal: PlanEditProposal
    iterations: int
    # Which path built the schedule, so the UI can label it and so a silent fallback from "ai"
    # to "deterministic" (llama-server down mid-session) is visible rather than mysterious.
    planner: Literal["ai", "deterministic"] = "deterministic"
    # See AiPlanResponse.layout — it decides whether a course missing from the revision was
    # unschedulable or merely not the option this layout picked.
    layout: str = "model"
    removed: list[str] = Field(default_factory=list)
    backfilled: list[str] = Field(default_factory=list)
    requirement_coverage: float = 0.0
    missing_requirements: list[str] = Field(default_factory=list)


class AdvisorAskRequest(BaseModel):
    """A free-text question for the local advisor model.

    The advisor answers via semantic retrieval: the question is embedded and the most similar
    catalog chunks are pulled from pgvector to ground the reply (see ``rag.pipeline``).
    """

    question: str = Field(min_length=1)
    # Regenerate support. Retrieval is deterministic and unaffected — only the wording of the
    # answer over the same retrieved chunks changes, which is exactly what a student pressing
    # "regenerate" on an unclear answer is asking for.
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)


class AdvisorSource(BaseModel):
    """One retrieved chunk that grounded an advisor answer, surfaced so the UI can cite sources.

    ``similarity`` is cosine similarity (≈1 = highly relevant); ``metadata`` carries whatever
    tags were stored with the chunk (type, code, program, ...).
    """

    id: int
    similarity: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    content: str


class AdvisorAskResponse(BaseModel):
    """The advisor's answer plus the retrieved chunks it was grounded on."""

    answer: str
    model: str
    context_char_count: int
    sources: list[AdvisorSource] = Field(default_factory=list)


class ModelOption(BaseModel):
    """One locally-servable model, as the picker shows it. Never the raw gguf filename — see
    ``services.model_manager.ModelSpec``."""

    name: str
    label: str
    blurb: str


class ModelStatusResponse(BaseModel):
    """Which model is running, which one (if any) is mid-switch, and what's available.

    ``current`` is ``None`` both before the first launch completes and after a failed switch —
    those are the same fact from a caller's point of view: nothing is answering right now.
    """

    current: str | None
    switching_to: str | None
    last_error: str | None
    available: list[ModelOption] = Field(default_factory=list)
