"""SQLite build + read-only execution + execution-accuracy comparison.

Candidate SQL runs on a read-only connection (URI mode=ro AND PRAGMA query_only) so a
model emitting DROP/UPDATE fails at execution — which is exactly the grade it earns.

Correctness is Spider-style EXECUTION ACCURACY: gold and candidate result sets are
compared as multisets of per-row VALUE BAGS (each row's values stringified and sorted,
column names/order ignored). This accepts any SQL formulation returning the same data.
Known limitation, stated honestly: a query returning coincidentally identical values via
wrong logic scores a false positive; with a small seed DB this is rare but not zero —
spot-check a sample of SQL_CORRECT passes manually.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

MAX_ROWS = 500  # cap runaway cartesian joins; a correct advising query never needs this


def build_db(schema_sql: Path, db_file: Path) -> None:
    db_file.unlink(missing_ok=True)
    conn = sqlite3.connect(db_file)
    try:
        conn.executescript(Path(schema_sql).read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


def ro_connect(db_file: Path) -> sqlite3.Connection:
    if not Path(db_file).exists():
        raise FileNotFoundError(f"{db_file} missing — run `python run.py db` first")
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def execute_select(
    conn: sqlite3.Connection, sql: str
) -> tuple[list[str], list[tuple]]:
    """Run one statement, return (column_names, rows). Raises sqlite3.Error on any
    failure — including write attempts, which the read-only connection rejects."""
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise sqlite3.ProgrammingError("multiple SQL statements are not allowed")
    cur = conn.execute(sql)
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchmany(MAX_ROWS)
    return columns, rows


def rows_equal(a: list[tuple], b: list[tuple]) -> bool:
    def bag(rows: list[tuple]) -> Counter:
        return Counter(tuple(sorted(str(v) for v in row)) for row in rows)

    return bag(a) == bag(b)
