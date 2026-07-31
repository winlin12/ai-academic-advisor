from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Course(BaseModel):
    code: str
    title: str
    credits: int = Field(ge=0)
    prereqs: list[str] = Field(default_factory=list)
    # Satisfied by taking the course earlier OR in the same semester. The planner is
    # deliberately conservative and requires them earlier (always legal); only validation that
    # judges a plan the student already built applies the looser same-semester rule.
    coreqs: list[str] = Field(default_factory=list)
    offered_terms: list[str] = Field(default_factory=list)
    requirement_tags: list[str] = Field(default_factory=list)
    workload_score: int = Field(default=3, ge=1, le=5)
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


class StudentProfile(BaseModel):
    name: str = "Student"
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
    preferences: dict[str, str | int | bool | list[str]] = Field(default_factory=dict)


class PlannedCourse(BaseModel):
    code: str
    title: str
    credits: int
    workload_score: int


class SemesterPlan(BaseModel):
    term: str
    year: int
    courses: list[PlannedCourse]
    total_credits: int
    average_workload: float
    warnings: list[str] = Field(default_factory=list)


class PlanResponse(BaseModel):
    student_name: str
    degree_program: str
    semesters: list[SemesterPlan]
    unplanned_courses: list[str]
    warnings: list[str]


class StudentRecord(BaseModel):
    """A persisted student: the stored profile plus its row identity."""

    id: str
    name: str
    profile: StudentProfile
    created_at: datetime


class SavedPlan(BaseModel):
    """One plan saved against a student; ``feedback`` records what prompted it, if anything."""

    id: str
    feedback: str | None = None
    plan: PlanResponse
    created_at: datetime


class StudentDetail(StudentRecord):
    plans: list[SavedPlan] = Field(default_factory=list)


class SavePlanRequest(BaseModel):
    plan: PlanResponse
    feedback: str | None = None


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


class ExplainPlanRequest(BaseModel):
    """Ask the local model to explain an already-generated plan.

    ``plan`` is the full structured :class:`PlanResponse` (not a raw ``dict``) so malformed
    client payloads are rejected at the FastAPI boundary instead of being interpolated
    verbatim into the model prompt.
    """

    question: str = Field(min_length=1)
    plan: PlanResponse


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
    """

    profile: StudentProfile
    feedback: str = Field(min_length=1)
    current_plan: PlanResponse | None = None


class RevisePlanResponse(BaseModel):
    """The revised (still deterministically-validated) plan plus the model's reasoning."""

    plan: PlanResponse
    rationale: str
    proposal: PlanEditProposal
    iterations: int


class AdvisorAskRequest(BaseModel):
    """A free-text question for the local advisor model.

    The advisor answers via semantic retrieval: the question is embedded and the most similar
    catalog chunks are pulled from pgvector to ground the reply (see ``rag.pipeline``).
    """

    question: str = Field(min_length=1)


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
