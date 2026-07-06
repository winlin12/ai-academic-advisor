"""Student/plan persistence (JSONB rows in the ``students``/``plans`` tables). REST-shaped:
the student is the resource, plans are a sub-collection."""

import psycopg
from fastapi import APIRouter, HTTPException

from app.api.deps import academic_db_unavailable
from app.models.schemas import (
    SavedPlan,
    SavePlanRequest,
    StudentDetail,
    StudentProfile,
    StudentRecord,
)
from app.services.students_db import create_student, fetch_student, list_students, save_plan

router = APIRouter(prefix="/students", tags=["students"])


@router.post("", response_model=StudentRecord, status_code=201)
def students_create(profile: StudentProfile):
    try:
        return create_student(profile)
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc


@router.get("", response_model=list[StudentRecord])
def students_list(limit: int = 100):
    try:
        return list_students(limit=limit)
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc


@router.get("/{student_id}", response_model=StudentDetail)
def students_get(student_id: str):
    try:
        student = fetch_student(student_id)
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
    if student is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


@router.post("/{student_id}/plans", response_model=SavedPlan, status_code=201)
def students_save_plan(student_id: str, req: SavePlanRequest):
    try:
        saved = save_plan(student_id, req.plan, feedback=req.feedback)
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
    if saved is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return saved
