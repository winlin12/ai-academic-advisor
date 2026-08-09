"""Planning routes: generate a plan deterministically, have the model write one, edit one.

Two generators, and the difference is the whole architecture:

``POST /plan/generate``     the greedy deterministic planner. No LLM, a few milliseconds of
                            CPU, legal by construction. The fallback and the fast path.
``POST /plan/ai-generate``  MODE B — the model writes the schedule itself, then
                            ``plan_validation`` deletes anything illegal and the deterministic
                            planner backfills the gaps. Legal by *repair* rather than by
                            construction, which is why it reports exactly what it changed.
``POST /plan/refine``       MODE C — what a student does after READING a plan: "fill" the
                            gaps (freeze what validates, model fills the rest), "regenerate"
                            from the specific errors found, or "start-over" with a fresh
                            sample. See ``services/plan_refine``.
``POST /plan/requirements`` the degree as a CHECKLIST: which requirement each course fills and
                            which are still open, with the options for each hole. Deterministic.
``POST /plan/edit``         one direct move/add/remove, re-validated. Never touches the LLM.
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import resolve_for_ai_planning, resolve_for_planning
from app.models.schemas import (
    AiPlanRequest,
    AiPlanResponse,
    PlanEditRequest,
    PlanRequirementsRequest,
    PlanResponse,
    RefinePlanRequest,
    RefinePlanResponse,
    StudentProfile,
)
from app.services.ai_planner import generate_ai_plan
from app.services.llamacpp_client import (
    LlamaCppClient,
    LlamaCppConnectionError,
    ModelResponseError,
)
from app.services.plan_editor import (
    PlanEditError,
    UnknownCourseError,
    apply_plan_edit,
    resolve_edit_catalog,
)
from app.services.plan_refine import refine_plan
from app.services.plan_requirements import (
    RequirementProgressResponse,
    requirement_progress,
    semester_labels,
)
from app.services.planner import generate_plan

router = APIRouter(prefix="/plan", tags=["planning"])


@router.post("/generate", response_model=PlanResponse)
def plan_generate(profile: StudentProfile):
    profile, catalog = resolve_for_planning(profile)
    return generate_plan(profile, catalog)


@router.post("/ai-generate", response_model=AiPlanResponse)
async def plan_ai_generate(req: AiPlanRequest):
    """MODE B: the local model drafts the plan of study, then it is repaired and backfilled.

    Slow by the standards of this API — a 26B MoE reading an ~11k-token catalog export takes
    tens of seconds — which is why the client shows a progress state and why the deterministic
    ``/plan/generate`` still exists for anything that needs an answer now.

    No 503 for a downed model: ``generate_ai_plan`` falls back to the deterministic planner and
    returns ``used_model: false``, because a student with no llama-server should still get a
    plan and should be told which one they got. A 502/503 here means the DATABASE failed, which
    is not recoverable — there would be no catalog to plan from.
    """
    profile, catalog = resolve_for_ai_planning(req.profile)
    client = LlamaCppClient()
    try:
        result = await generate_ai_plan(
            profile, catalog, client=client, request=req.request, seed=req.seed
        )
    except LlamaCppConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return AiPlanResponse(
        plan=result.plan,
        rationale=result.rationale,
        model=result.model,
        used_model=result.used_model,
        layout=result.layout,
        model_placed=result.model_placed,
        removed=result.removed,
        backfilled=result.backfilled,
        violations=result.violations,
        requirement_coverage=result.requirement_coverage,
        missing_requirements=result.missing_requirements,
        seed=result.seed,
    )


@router.post("/refine", response_model=RefinePlanResponse)
async def plan_refine(req: RefinePlanRequest):
    """MODE C: fill the gaps, regenerate from the errors, or start over.

    All three take the plan currently on screen and return a new one that has been through the
    same checker as every other path — including "fill", whose merge step hands the model a
    delta it never saw in full context. Same error ladder as ``/ai-generate``: the model being
    down is reported in the body (``used_model: false``), not as a 5xx, because the student's
    plan is still there and still legal.
    """
    profile, catalog = resolve_for_ai_planning(req.profile)
    semesters = [[course.code for course in semester.courses]
                 for semester in req.plan.semesters]
    client = LlamaCppClient()
    try:
        result, outcome = await refine_plan(
            profile, catalog, semesters, req.mode,
            client=client, seed=req.seed, attempt=req.attempt,
        )
    except LlamaCppConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ModelResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RefinePlanResponse(
        plan=result.plan,
        rationale=result.rationale,
        model=result.model,
        used_model=result.used_model,
        layout=result.layout,
        model_placed=result.model_placed,
        removed=result.removed,
        backfilled=result.backfilled,
        violations=result.violations,
        requirement_coverage=result.requirement_coverage,
        missing_requirements=result.missing_requirements,
        seed=result.seed,
        mode=outcome.mode,
        kept=outcome.kept,
        note=outcome.note,
    )


@router.post("/requirements", response_model=RequirementProgressResponse)
def plan_requirements(req: PlanRequirementsRequest):
    """The degree as a checklist: which requirement each course fills, and what is still open.

    Deterministic and fast — no LLM, a few milliseconds. This is where the holes left by *not*
    auto-filling become something the student can act on: every unfilled slot carries the
    courses that could fill it and, for each, the semesters it would legally land in — checked
    with the same validator the edit route uses, so an option offered here cannot be rejected
    when the student picks it.
    """
    profile, catalog = resolve_for_ai_planning(req.profile)
    semesters = [[course.code for course in semester.courses]
                 for semester in req.plan.semesters]
    result = requirement_progress(profile, catalog, semesters)
    result.semester_labels = semester_labels(profile)
    return result


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
