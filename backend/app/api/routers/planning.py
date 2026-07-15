"""Deterministic planning routes: generate a plan, directly edit a plan.

Neither route ever touches the LLM — this is the "planner disposes" half of the
architecture, and it must stay fast and model-free so plan manipulation costs nothing but
a few milliseconds of CPU (and zero Anthropic API tokens).
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import resolve_for_planning
from app.models.schemas import PlanEditRequest, PlanResponse, StudentProfile
from app.services.plan_editor import (
    PlanEditError,
    UnknownCourseError,
    apply_plan_edit,
    resolve_edit_catalog,
)
from app.services.planner import generate_plan

router = APIRouter(prefix="/plan", tags=["planning"])


@router.post("/generate", response_model=PlanResponse)
def plan_generate(profile: StudentProfile):
    profile, catalog = resolve_for_planning(profile)
    return generate_plan(profile, catalog)


@router.post("/edit", response_model=PlanResponse)
def plan_edit(req: PlanEditRequest):
    """Apply one move/add/remove to a plan and return the re-validated plan.

    Direct manipulation, not regeneration: the course lands exactly where the student put
    it, then the layout is re-checked against prerequisites, term offerings, and the
    profile's credit cap, with violations surfaced as warnings. Error ladder: 404 for a
    course code the catalog has never heard of, 422 for edits that are structurally
    impossible (course not in plan, duplicate add, bad semester index).
    """
    codes = {course.code for semester in req.plan.semesters for course in semester.courses}
    codes.add(req.course_code)
    codes.update(req.plan.unplanned_courses)
    catalog = resolve_edit_catalog(codes)

    try:
        return apply_plan_edit(
            plan=req.plan,
            operation=req.operation,
            course_code=req.course_code,
            target_semester=req.target_semester,
            profile=req.profile,
            catalog=catalog,
        )
    except UnknownCourseError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PlanEditError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
