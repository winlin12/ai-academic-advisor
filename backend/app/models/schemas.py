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
