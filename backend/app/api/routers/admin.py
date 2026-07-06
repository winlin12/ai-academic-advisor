"""Read-only database visibility for the web admin page (TODO Priority 6).

Deliberately view-only: no INSERT/UPDATE/DELETE routes exist here, so exposing this
router costs nothing beyond read access the terminal (``make psql``) already had. For
full editing power use the Adminer service in ``catalog_ingestion/docker-compose.yml``.
"""

import psycopg
from fastapi import APIRouter, HTTPException, Query

from app.api.deps import academic_db_unavailable
from app.models.schemas import AdminTableRowsResponse, AdminTablesResponse
from app.services.admin_db import UnknownTableError, fetch_rows, list_tables

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/tables", response_model=AdminTablesResponse)
def admin_tables():
    try:
        return AdminTablesResponse(tables=list_tables())
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc


@router.get("/tables/{table}", response_model=AdminTableRowsResponse)
def admin_table_rows(
    table: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    try:
        return fetch_rows(table, limit=limit, offset=offset)
    except UnknownTableError as exc:
        raise HTTPException(status_code=404, detail=f"Table '{table}' is not browsable") from exc
    except psycopg.Error as exc:
        raise academic_db_unavailable(exc) from exc
