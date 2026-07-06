"""Persistence for student profiles and their saved plans (TODO §4.4).

Rows live in the ``students`` / ``plans`` tables of the ``catalog_ingestion`` database
(migration ``002_students_plans``; ``make db-init`` also creates them). Profiles and plans
are stored as JSONB exactly as the API schemas serialise them, so this layer is a thin
store/fetch with no schema of its own.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import (
    PlanResponse,
    SavedPlan,
    StudentDetail,
    StudentProfile,
    StudentRecord,
)


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def create_student(profile: StudentProfile) -> StudentRecord:
    student_id = uuid.uuid4()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO students (id, name, profile, created_at, updated_at)
            VALUES (%s, %s, %s, now(), now())
            RETURNING id::text, name, profile, created_at
            """,
            (student_id, profile.name, json.dumps(profile.model_dump())),
        )
        row = cur.fetchone()
    return _student_record(row)


def list_students(*, limit: int = 100) -> list[StudentRecord]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id::text, name, profile, created_at
            FROM students
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [_student_record(row) for row in cur.fetchall()]


def fetch_student(student_id: str) -> StudentDetail | None:
    parsed = _parse_uuid(student_id)
    if parsed is None:
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id::text, name, profile, created_at FROM students WHERE id = %s",
            (parsed,),
        )
        student_row = cur.fetchone()
        if student_row is None:
            return None
        cur.execute(
            """
            SELECT id::text, feedback, plan, created_at
            FROM plans
            WHERE student_id = %s
            ORDER BY created_at DESC
            """,
            (parsed,),
        )
        plan_rows = cur.fetchall()

    record = _student_record(student_row)
    return StudentDetail(
        **record.model_dump(),
        plans=[_saved_plan(row) for row in plan_rows],
    )


def save_plan(student_id: str, plan: PlanResponse, *, feedback: str | None = None) -> SavedPlan | None:
    """Attach a plan to a student. Returns ``None`` when the student does not exist."""
    parsed = _parse_uuid(student_id)
    if parsed is None:
        return None
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM students WHERE id = %s", (parsed,))
        if cur.fetchone() is None:
            return None
        cur.execute(
            """
            INSERT INTO plans (id, student_id, feedback, plan, created_at)
            VALUES (%s, %s, %s, %s, now())
            RETURNING id::text, feedback, plan, created_at
            """,
            (uuid.uuid4(), parsed, feedback, json.dumps(plan.model_dump())),
        )
        row = cur.fetchone()
    return _saved_plan(row)


def _student_record(row: dict[str, Any]) -> StudentRecord:
    return StudentRecord(
        id=row["id"],
        name=row["name"],
        profile=StudentProfile.model_validate(row["profile"]),
        created_at=row["created_at"],
    )


def _saved_plan(row: dict[str, Any]) -> SavedPlan:
    return SavedPlan(
        id=row["id"],
        feedback=row["feedback"],
        plan=PlanResponse.model_validate(row["plan"]),
        created_at=row["created_at"],
    )
