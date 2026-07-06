"""Read-only views over the real ``catalog_ingestion`` database: facets, programs, and
course search. Purely deterministic — no LLM, no planner.

The old ``GET /catalog/courses`` fixture route was removed here (TODO Priority 5.2): it
served the bundled 8-course ``courses.json`` and duplicated ``/academic/courses/search``
against the real DB. The fixture itself lives on as the planner's documented offline
fallback (``services/catalog.py``); it just no longer masquerades as a catalog API.
"""

from fastapi import APIRouter, HTTPException

from app.api.deps import academic_db_unavailable
from app.models.schemas import (
    AcademicCourseSearchResponse,
    AcademicFacetResponse,
    AcademicProgramDetail,
    AcademicProgramSummary,
)
from app.services.academic_db import (
    fetch_academic_facets,
    fetch_program_detail,
    fetch_program_summaries,
    search_courses,
)

router = APIRouter(prefix="/academic", tags=["academic"])


@router.get("/facets", response_model=AcademicFacetResponse)
def academic_facets():
    try:
        return fetch_academic_facets()
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc


@router.get("/programs", response_model=list[AcademicProgramSummary])
def academic_programs(
    query: str | None = None,
    catalog_year: int | None = None,
    school: str | None = None,
    limit: int = 120,
):
    try:
        return fetch_program_summaries(
            query=query,
            catalog_year=catalog_year,
            school=school,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc


@router.get("/programs/{program_id}", response_model=AcademicProgramDetail)
def academic_program_detail(program_id: str):
    try:
        program = fetch_program_detail(program_id)
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc

    if program is None:
        raise HTTPException(status_code=404, detail="Academic program not found")
    return program


@router.get("/courses/search", response_model=AcademicCourseSearchResponse)
def academic_course_search(
    query: str | None = None,
    subject: str | None = None,
    limit: int = 80,
):
    try:
        courses = search_courses(query=query, subject=subject, limit=limit)
        return AcademicCourseSearchResponse(courses=courses)
    except Exception as exc:  # noqa: BLE001
        raise academic_db_unavailable(exc) from exc
