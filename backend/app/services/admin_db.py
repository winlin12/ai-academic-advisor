"""Read-only browsing over the catalog database for the admin UI (TODO Priority 6).

Strictly a viewer: every query is a SELECT over an explicit whitelist of tables, so this
layer can never be talked into touching anything else — table names are compared against
``BROWSABLE_TABLES`` before they go anywhere near SQL, and rows are serialised with
``to_jsonb`` so Postgres (not Python) handles UUID/timestamp conversion.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import AdminTableInfo, AdminTableRowsResponse

# Every table the admin UI may read, in the order the UI should list them. Anything not
# here 404s. ``academic_rules`` is included for row inspection, but its pgvector column is
# stripped (HIDDEN_COLUMNS) — a 768-float embedding per row is noise, not information.
BROWSABLE_TABLES: tuple[str, ...] = (
    "students",
    "plans",
    "programs",
    "requirement_groups",
    "requirement_options",
    "courses",
    "subjects",
    "colleges",
    "catalog_years",
    "academic_rules",
)

HIDDEN_COLUMNS: dict[str, tuple[str, ...]] = {
    "academic_rules": ("embedding",),
}

MAX_PAGE_SIZE = 200


class UnknownTableError(ValueError):
    """Raised when a requested table is not in the browsable whitelist."""


def row_select_expression(table: str) -> str:
    """The SELECT expression for one row of ``table`` as JSONB, minus hidden columns.

    ``table`` must already be whitelisted — this only builds the projection.
    """
    expr = "to_jsonb(t)"
    for column in HIDDEN_COLUMNS.get(table, ()):
        expr += f" - '{column}'"
    return expr


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


def list_tables() -> list[AdminTableInfo]:
    """Row counts for every whitelisted table that actually exists in the database."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        existing = {row["tablename"] for row in cur.fetchall()}
        infos: list[AdminTableInfo] = []
        for table in BROWSABLE_TABLES:
            if table not in existing:
                continue
            cur.execute(f"SELECT count(*) AS n FROM {table}")  # noqa: S608 — whitelisted name
            infos.append(AdminTableInfo(name=table, row_count=cur.fetchone()["n"]))
    return infos


def fetch_rows(table: str, *, limit: int = 50, offset: int = 0) -> AdminTableRowsResponse:
    """One page of rows from a whitelisted table, serialised as plain JSON objects.

    Ordered by ``ctid`` — no assumption about which columns exist, and stable enough for
    an admin browser paging through a table that isn't being rewritten mid-scroll.
    """
    if table not in BROWSABLE_TABLES:
        raise UnknownTableError(table)
    limit = max(1, min(limit, MAX_PAGE_SIZE))
    offset = max(0, offset)

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) AS n FROM {table}")  # noqa: S608 — whitelisted name
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT {row_select_expression(table)} AS row FROM {table} t "  # noqa: S608
            "ORDER BY t.ctid LIMIT %s OFFSET %s",
            (limit, offset),
        )
        rows: list[dict[str, Any]] = [record["row"] for record in cur.fetchall()]

    columns = sorted({key for row in rows for key in row})
    return AdminTableRowsResponse(
        table=table,
        columns=columns,
        rows=rows,
        total=total,
        limit=limit,
        offset=offset,
    )
