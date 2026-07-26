"""Crawl Banner course-detail pages for prerequisites and store them in the advisor schema.

    # scoped — just the courses the eval fixture uses (fast; ~40 pages)
    python -m app.services.prerequisites.sync --codes "CS 18000,CS 25100,CS 38100"

    # a whole subject
    python -m app.services.prerequisites.sync --subjects CS,MA,STAT,PHYS

    # every course PurdueIO knows about (long — 11k pages at the crawl delay)
    python -m app.services.prerequisites.sync --all

This is the "similar to catalogsync" crawler, in the shape the rest of the stack uses: a CLI
that a compose service and a Make target drive (see catalog_ingestion/Makefile
``sync-prereqs``). It is single-threaded, rate-limited and page-cached by construction — see
``banner.py`` for why that is not negotiable against Banner.

Course LIST comes from PurdueIO (the ``Courses``/``Subjects`` tables in the purdueio database);
prerequisite TEXT comes from Banner. The two are joined here so the crawler never guesses which
courses exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.services.prerequisites.banner import BannerFetcher
from app.services.prerequisites.parser import parse_prereq_html

logger = logging.getLogger(__name__)


def _course_list(
    dsn: str, codes: list[str] | None, subjects: set[str] | None, limit: int | None
) -> list[tuple[str, str, str]]:
    """(course_code, subject, number) triples to crawl, sourced from PurdueIO's Courses table.

    ``--codes`` is honoured even for a course PurdueIO does not list (so the eval fixture can
    always be filled), by splitting the code lexically.
    """
    if codes:
        out = []
        for raw in codes:
            code = " ".join(raw.split()).upper()
            match = code.replace(" ", "")
            split = next((i for i, c in enumerate(match) if c.isdigit()), None)
            if split:
                out.append((f"{match[:split]} {match[split:]}", match[:split], match[split:]))
        return out

    clause = 'WHERE s."Abbreviation" = ANY(%s)' if subjects else ""
    params: tuple = (sorted(subjects),) if subjects else ()
    sql = f"""
        SELECT s."Abbreviation" AS subject, c."Number" AS number
        FROM "Courses" c JOIN "Subjects" s ON s."Id" = c."SubjectId"
        {clause}
        ORDER BY s."Abbreviation", c."Number"
        {"LIMIT %s" if limit else ""}
    """
    if limit:
        params = (*params, limit)
    with psycopg.connect(dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return [(f"{r['subject']} {r['number']}", r["subject"], r["number"])
                for r in cur.fetchall()]


def apply_schema(conn: psycopg.Connection) -> None:
    conn.execute((Path(__file__).parent / "schema.sql").read_text(encoding="utf-8"))


def run_sync(
    *, dsn: str, term_code: str, cache_dir: Path, delay_s: float,
    codes: list[str] | None, subjects: set[str] | None, limit: int | None,
) -> dict:
    targets = _course_list(dsn, codes, subjects, limit)
    if not targets:
        raise SystemExit("No courses matched the filter.")

    run_id = uuid.uuid4()
    fetcher = BannerFetcher(term_code=term_code, cache_dir=cache_dir, delay_s=delay_s)
    with psycopg.connect(dsn) as conn:
        apply_schema(conn)
        conn.execute(
            "INSERT INTO advisor.prereq_sync_runs (id, term_code, status, metadata) "
            "VALUES (%s, %s, 'running', %s)",
            (run_id, term_code, json.dumps(
                {"targets": len(targets),
                 "scope": "codes" if codes else ("subjects" if subjects else "all")})),
        )
        conn.commit()

        seen = parsed = cached = 0
        errors: list[str] = []
        try:
            for code, subject, number in targets:
                try:
                    page, from_cache = fetcher.fetch(subject, number)
                except Exception as exc:  # noqa: BLE001 — one bad page must not kill the crawl
                    errors.append(f"{code}: fetch failed: {exc}")
                    logger.warning("fetch failed for %s: %s", code, exc)
                    continue
                seen += 1
                cached += 1 if from_cache else 0
                result = parse_prereq_html(page)
                if result.confidence in ("high", "medium"):
                    parsed += 1

                conn.execute(
                    """
                    INSERT INTO advisor.course_prerequisites
                      (course_code, subject, number, term_code, has_prereqs, raw_text,
                       parsed_tree, prereq_groups, confidence, notes, sync_run_id, updated_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now())
                    ON CONFLICT (course_code) DO UPDATE SET
                      term_code=EXCLUDED.term_code, has_prereqs=EXCLUDED.has_prereqs,
                      raw_text=EXCLUDED.raw_text, parsed_tree=EXCLUDED.parsed_tree,
                      prereq_groups=EXCLUDED.prereq_groups, confidence=EXCLUDED.confidence,
                      notes=EXCLUDED.notes, sync_run_id=EXCLUDED.sync_run_id, updated_at=now()
                    """,
                    (code, subject, number, term_code,
                     result.confidence != "none",
                     result.raw_text or None,
                     json.dumps(result.tree) if result.tree else None,
                     json.dumps(result.groups) if result.groups is not None else None,
                     result.confidence, json.dumps(result.notes), run_id),
                )
                conn.commit()
                flag = "cache" if from_cache else "  net"
                print(f"  [{flag}] {code:12} {result.confidence:6} "
                      f"groups={result.groups if result.groups else '—'}")

            conn.execute(
                "UPDATE advisor.prereq_sync_runs SET status='completed', completed_at=now(), "
                "courses_seen=%s, courses_parsed=%s, from_cache=%s, errors=%s WHERE id=%s",
                (seen, parsed, cached, json.dumps(errors[:100]), run_id),
            )
            conn.commit()
            return {"seen": seen, "parsed": parsed, "cached": cached, "errors": errors}
        except Exception as exc:
            conn.rollback()
            conn.execute(
                "UPDATE advisor.prereq_sync_runs SET status='failed', completed_at=now(), "
                "errors=%s WHERE id=%s", (json.dumps([str(exc)]), run_id),
            )
            conn.commit()
            raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default=settings.purdueio_database_url)
    ap.add_argument("--term-code", default="202710",
                    help="Banner term whose pages are read (default 202710 = Fall 2026)")
    ap.add_argument("--cache-dir", default=str(
        Path(__file__).resolve().parents[4] / ".banner_cache"),
        help="persistent page cache; a re-run re-fetches nothing")
    ap.add_argument("--delay", type=float, default=5.0,
                    help="seconds between REAL fetches (never below 1; cache hits are free)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--codes", help="comma-separated course codes, e.g. 'CS 18000,CS 25100'")
    group.add_argument("--subjects", help="comma-separated subjects, e.g. CS,MA,STAT,PHYS")
    group.add_argument("--all", action="store_true", help="every course in PurdueIO (long)")
    ap.add_argument("--limit", type=int, help="cap the number of courses (testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_sync(
        dsn=args.dsn, term_code=args.term_code, cache_dir=Path(args.cache_dir),
        delay_s=args.delay,
        codes=[c for c in args.codes.split(",")] if args.codes else None,
        subjects={s.strip().upper() for s in args.subjects.split(",")} if args.subjects else None,
        limit=args.limit,
    )
    print(f"\nseen={result['seen']} parsed(high/med)={result['parsed']} "
          f"from_cache={result['cached']}")
    if result["errors"]:
        print(f"{len(result['errors'])} error(s); first few:")
        for err in result["errors"][:10]:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
