"""Sync course offerings from PurdueIO into ``advisor.course_offerings``.

    python -m app.services.offerings.sync --terms 12
    python -m app.services.offerings.sync --all --subjects CS,MA,STAT,PHYS

WHAT THIS FIXES. Purdue's Acalog catalog publishes no term offerings, so the planner used to
treat every course as available every term (``planner_catalog._course_from_row`` hard-coded
``offered_terms=TERM_ORDER``). That is not a harmless default — term availability is most of
what makes course sequencing hard, and a planner that thinks everything is always available
produces schedules a student cannot register for.

OFFERINGS ARE OBSERVED, NOT DECLARED. PurdueIO has no "offered in" field either. What it has
is ``Classes`` — a (Course, Term, Campus) triple that actually ran. So "CS 47100 is a fall
course" is an inference from sightings. The sync therefore records counts, not just booleans,
and the planner is expected to treat a one-sighting pattern differently from a nine-sighting
one. Absence is especially weak evidence: a course with two observations says nothing
reliable about the terms it was not seen in.

WHY NOT THE LOCAL MIRROR. ``purdueio-postgres`` already holds Classes/Terms and querying it
would need no network at all — but the local ``catalogsync`` container only keeps ~4 recent
terms, and four terms means each season is observed once or twice. The public API has 55
terms back to 2008. Depth is the whole point of the inference, so the API is the default
source; ``--source local`` exists for a fast offline refresh of recent terms.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings

logger = logging.getLogger(__name__)

ODATA_URL = "https://api.purdue.io/odata"
USER_AGENT = (
    "PurdueAcademicPlannerBot/1.0 (student hobby project; not affiliated with Purdue; "
    "contact: bingdaddycompany@gmail.com)"
)

# Banner term codes end in a season digit-pair: 10=fall, 20=spring, 30=summer, 13=winter.
# The leading four digits are the ACADEMIC year, which is why 202710 is "Fall 2026" — the
# calendar year is one less for fall. Getting this backwards shifts every fall course by a
# year and is invisible in aggregate, so it is derived from the code and cross-checked
# against the term's own name below.
_SEASON_BY_SUFFIX = {"10": "fall", "20": "spring", "30": "summer", "13": "winter"}
_TERM_NAME_RE = re.compile(r"(Fall|Spring|Summer|Winter)\s+(\d{4})", re.IGNORECASE)


def _get(url: str, *, retries: int = 3) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return json.load(resp)
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def _pages(entity: str, query: str) -> Iterator[dict[str, Any]]:
    """Follow @odata.nextLink. The API enforces server-driven paging and rejects $top."""
    url: str | None = f"{ODATA_URL}/{entity}?{query}" if query else f"{ODATA_URL}/{entity}"
    while url:
        data = _get(url)
        yield from data.get("value", [])
        url = data.get("@odata.nextLink")


def parse_term(code: str, name: str | None) -> tuple[str, int] | None:
    """(season, calendar_year) from a Banner term code, cross-checked against its name.

    The name is authoritative when the two disagree — the code's academic-year convention is
    a convention, and a mismatch means the convention changed, not that the name is wrong.
    """
    code = (code or "").strip()
    season_from_name = year_from_name = None
    if name:
        match = _TERM_NAME_RE.search(name)
        if match:
            season_from_name = match.group(1).lower()
            year_from_name = int(match.group(2))
    if season_from_name and year_from_name:
        return season_from_name, year_from_name
    if len(code) == 6 and code[:4].isdigit():
        season = _SEASON_BY_SUFFIX.get(code[4:])
        if season:
            academic_year = int(code[:4])
            return season, academic_year - 1 if season == "fall" else academic_year
    return None


def fetch_terms(limit: int | None) -> list[dict[str, Any]]:
    terms = [t for t in _pages("Terms", "") if t.get("StartDate")]
    terms.sort(key=lambda t: t["StartDate"], reverse=True)
    return terms[:limit] if limit else terms


def fetch_offerings_for_term(
    term: dict[str, Any], subjects: set[str] | None
) -> list[dict[str, Any]]:
    """Every (course, term) pair that ran, expanded with subject + number in one request set."""
    filt = urllib.parse.quote(f"TermId eq {term['Id']}", safe="")
    expand = urllib.parse.quote("Course($expand=Subject)", safe="")
    rows: dict[str, dict[str, Any]] = {}
    for cls in _pages("Classes", f"$filter={filt}&$expand={expand}"):
        course = cls.get("Course") or {}
        subject = (course.get("Subject") or {}).get("Abbreviation")
        number = course.get("Number")
        if not subject or not number:
            continue
        subject = subject.strip().upper()
        if subjects and subject not in subjects:
            continue
        code = f"{subject} {number.strip()}"
        row = rows.setdefault(
            code, {"course_code": code, "subject": subject, "number": number.strip(),
                   "class_count": 0}
        )
        row["class_count"] += 1
    return list(rows.values())


def fetch_offerings_local(dsn: str, subjects: set[str] | None) -> list[dict[str, Any]]:
    """Same shape, read straight out of the local PurdueIO mirror. No network."""
    clause = "AND s.\"Abbreviation\" = ANY(%s)" if subjects else ""
    params = (sorted(subjects),) if subjects else ()
    with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT t."Code"  AS term_code,
                   t."Name"  AS term_name,
                   s."Abbreviation" AS subject,
                   c."Number"       AS number,
                   count(*)         AS class_count
            FROM "Classes" cl
            JOIN "Terms"    t ON t."Id" = cl."TermId"
            JOIN "Courses"  c ON c."Id" = cl."CourseId"
            JOIN "Subjects" s ON s."Id" = c."SubjectId"
            WHERE s."Abbreviation" IS NOT NULL AND c."Number" IS NOT NULL {clause}
            GROUP BY 1, 2, 3, 4
            """,
            params,
        )
        return [dict(r) for r in cur.fetchall()]


# --- persistence -------------------------------------------------------------------------


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))


def upsert_offerings(conn: psycopg.Connection, rows: list[dict[str, Any]], run_id: uuid.UUID) -> int:
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO advisor.course_offerings
              (course_code, subject, number, term_code, term_name, season,
               calendar_year, class_count, sync_run_id)
            VALUES (%(course_code)s, %(subject)s, %(number)s, %(term_code)s, %(term_name)s,
                    %(season)s, %(calendar_year)s, %(class_count)s, %(sync_run_id)s)
            ON CONFLICT (course_code, term_code) DO UPDATE
              SET class_count = EXCLUDED.class_count,
                  term_name   = EXCLUDED.term_name,
                  season      = EXCLUDED.season,
                  calendar_year = EXCLUDED.calendar_year,
                  sync_run_id = EXCLUDED.sync_run_id
            """,
            [{**r, "sync_run_id": run_id} for r in rows],
        )
    return len(rows)


def rebuild_patterns(conn: psycopg.Connection) -> int:
    """Recompute every pattern from the raw observations.

    Wholesale rather than incremental on purpose: patterns are a pure function of
    ``course_offerings``, and an incremental update that drifted from that function would be
    both silent and wrong in the direction of "we already know this course's schedule".
    """
    with conn.cursor() as cur:
        cur.execute("TRUNCATE advisor.course_offering_patterns")
        cur.execute(
            """
            INSERT INTO advisor.course_offering_patterns (
                course_code, subject, number,
                offered_fall, offered_spring, offered_summer, offered_winter,
                fall_terms, spring_terms, summer_terms, winter_terms,
                terms_observed, first_term_code, last_term_code, updated_at)
            SELECT
                course_code, min(subject), min(number),
                bool_or(season = 'fall'),   bool_or(season = 'spring'),
                bool_or(season = 'summer'), bool_or(season = 'winter'),
                count(*) FILTER (WHERE season = 'fall'),
                count(*) FILTER (WHERE season = 'spring'),
                count(*) FILTER (WHERE season = 'summer'),
                count(*) FILTER (WHERE season = 'winter'),
                count(DISTINCT term_code), min(term_code), max(term_code), now()
            FROM advisor.course_offerings
            GROUP BY course_code
            """
        )
        return cur.rowcount


# --- entry point -------------------------------------------------------------------------


def run_sync(
    *, dsn: str, source: str, term_limit: int | None, subjects: set[str] | None
) -> dict[str, Any]:
    run_id = uuid.uuid4()
    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        conn.execute(
            "INSERT INTO advisor.offering_sync_runs (id, source, status, metadata) "
            "VALUES (%s, %s, 'running', %s)",
            (run_id, source, json.dumps(
                {"term_limit": term_limit, "subjects": sorted(subjects) if subjects else "all"})),
        )
        conn.commit()

        errors: list[str] = []
        total = terms_done = 0
        try:
            if source == "local":
                rows = []
                for row in fetch_offerings_local(dsn, subjects):
                    parsed = parse_term(row["term_code"], row.get("term_name"))
                    if not parsed:
                        errors.append(f"unparseable term {row['term_code']}")
                        continue
                    season, year = parsed
                    rows.append({
                        "course_code": f"{row['subject'].strip().upper()} {row['number'].strip()}",
                        "subject": row["subject"].strip().upper(),
                        "number": row["number"].strip(),
                        "term_code": row["term_code"], "term_name": row.get("term_name"),
                        "season": season, "calendar_year": year,
                        "class_count": row["class_count"],
                    })
                total += upsert_offerings(conn, rows, run_id)
                terms_done = len({r["term_code"] for r in rows})
            else:
                for term in fetch_terms(term_limit):
                    parsed = parse_term(term.get("Code", ""), term.get("Name"))
                    if not parsed:
                        errors.append(f"unparseable term {term.get('Code')}")
                        continue
                    season, year = parsed
                    try:
                        found = fetch_offerings_for_term(term, subjects)
                    except Exception as exc:  # noqa: BLE001 — one bad term must not kill the run
                        errors.append(f"{term.get('Code')}: {exc}")
                        logger.warning("term %s failed: %s", term.get("Code"), exc)
                        continue
                    rows = [
                        {**r, "term_code": term.get("Code"), "term_name": term.get("Name"),
                         "season": season, "calendar_year": year}
                        for r in found
                    ]
                    total += upsert_offerings(conn, rows, run_id)
                    terms_done += 1
                    conn.commit()
                    print(f"  {term.get('Code')} {term.get('Name'):<14} {len(rows):>6} courses")

            patterns = rebuild_patterns(conn)
            conn.execute(
                "UPDATE advisor.offering_sync_runs SET status='completed', completed_at=now(), "
                "terms_synced=%s, offerings_seen=%s, errors=%s WHERE id=%s",
                (terms_done, total, json.dumps(errors[:50]), run_id),
            )
            conn.commit()
            return {"terms": terms_done, "offerings": total, "patterns": patterns,
                    "errors": errors}
        except Exception as exc:
            conn.rollback()
            conn.execute(
                "UPDATE advisor.offering_sync_runs SET status='failed', completed_at=now(), "
                "errors=%s WHERE id=%s",
                (json.dumps([str(exc)]), run_id),
            )
            conn.commit()
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=settings.purdueio_database_url,
                    help="purdueio database (default: settings.purdueio_database_url)")
    ap.add_argument("--source", choices=["odata", "local"], default="odata",
                    help="odata = public API with deep history; local = the mirror's recent terms")
    ap.add_argument("--terms", type=int, default=12,
                    help="how many most-recent terms to sync (odata only; default 12 = 4 years)")
    ap.add_argument("--all", action="store_true", help="every term the API has (55, back to 2008)")
    ap.add_argument("--subjects", help="comma-separated subject filter, e.g. CS,MA,STAT,PHYS")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    subjects = (
        {s.strip().upper() for s in args.subjects.split(",") if s.strip()}
        if args.subjects else None
    )
    result = run_sync(
        dsn=args.dsn, source=args.source,
        term_limit=None if args.all else args.terms, subjects=subjects,
    )
    print(f"\nterms={result['terms']} offerings={result['offerings']} "
          f"patterns={result['patterns']}")
    if result["errors"]:
        print(f"{len(result['errors'])} error(s):")
        for err in result["errors"][:10]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
