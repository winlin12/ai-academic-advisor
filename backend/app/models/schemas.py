from typing import Any

from pydantic import BaseModel, Field


class Course(BaseModel):
    code: str
    title: str
    credits: int = Field(ge=0)
    prereqs: list[str] = Field(default_factory=list)
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
    completed_courses: list[str] = Field(default_factory=list)
    remaining_courses: list[str] = Field(default_factory=list)
    start_term: str = "fall"
    start_year: int = 2026
    semesters_to_plan: int = Field(default=4, ge=1, le=12)
    max_credits_per_semester: int = Field(default=9, ge=1, le=24)
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


class ExplainPlanRequest(BaseModel):
    question: str
    plan: dict


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
