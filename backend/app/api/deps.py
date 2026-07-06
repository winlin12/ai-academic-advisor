"""Cross-router helpers: shared error mapping and profile/catalog resolution.

Every domain router imports from here instead of from a sibling router, so routers stay
leaf modules with no dependencies on each other.
"""

import psycopg
from fastapi import HTTPException

from app.models.schemas import Course, StudentProfile
from app.services.planner_catalog import ProgramNotFoundError, resolve_profile_and_catalog


def academic_db_unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Academic database unavailable: {exc}")


def resolve_for_planning(profile: StudentProfile) -> tuple[StudentProfile, list[Course]]:
    """Shared profile/catalog resolution for the plan and revise-plan routes.

    Maps the bridge's failure modes onto the API's error ladder: unknown ``program_id`` → 404,
    academic DB down while a program-driven plan was requested → 503. Profiles without a
    ``program_id`` never 503 here — they fall back to the bundled fixture catalog.
    """
    try:
        return resolve_profile_and_catalog(profile)
    except ProgramNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Academic program not found") from exc
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
