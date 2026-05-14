#!/usr/bin/env python3
"""Load extracted degree requirements JSON into SQLite.

This creates a separate degree requirements DB that can be joined
with the course DB (`data/purdue_catalog.db`) by `course_code`.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS degrees (
  degree_id INTEGER PRIMARY KEY AUTOINCREMENT,
  degree_name TEXT NOT NULL UNIQUE,
  award TEXT,
  level TEXT,
  confidence TEXT,
  source_page_start INTEGER,
  source_page_end INTEGER,
  requirement_pages_count INTEGER
);

CREATE TABLE IF NOT EXISTS degree_requirements (
  requirement_id INTEGER PRIMARY KEY AUTOINCREMENT,
  degree_id INTEGER NOT NULL,
  requirement_name TEXT NOT NULL,
  credits_text TEXT,
  source_page_start INTEGER,
  source_page_end INTEGER,
  UNIQUE (degree_id, requirement_name, credits_text),
  FOREIGN KEY (degree_id) REFERENCES degrees(degree_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS degree_requirement_courses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  degree_id INTEGER NOT NULL,
  requirement_id INTEGER,
  course_code TEXT NOT NULL,
  course_title_on_page TEXT,
  course_in_catalog INTEGER NOT NULL DEFAULT 0,
  catalog_course_title TEXT,
  source_page_start INTEGER,
  source_page_end INTEGER,
  UNIQUE (degree_id, requirement_id, course_code),
  FOREIGN KEY (degree_id) REFERENCES degrees(degree_id) ON DELETE CASCADE,
  FOREIGN KEY (requirement_id) REFERENCES degree_requirements(requirement_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS degree_extraction_issues (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  degree_id INTEGER,
  page INTEGER,
  issue_type TEXT NOT NULL,
  details TEXT,
  FOREIGN KEY (degree_id) REFERENCES degrees(degree_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS ingestion_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_degree_requirements_degree_id
  ON degree_requirements(degree_id);
CREATE INDEX IF NOT EXISTS idx_degree_requirement_courses_degree_id
  ON degree_requirement_courses(degree_id);
CREATE INDEX IF NOT EXISTS idx_degree_requirement_courses_requirement_id
  ON degree_requirement_courses(requirement_id);
CREATE INDEX IF NOT EXISTS idx_degree_requirement_courses_code
  ON degree_requirement_courses(course_code);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load degree extraction JSON into SQLite.")
    parser.add_argument(
        "--degrees",
        type=Path,
        default=Path("data/degree_extracted/degrees.json"),
        help="Path to degrees.json",
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=Path("data/degree_extracted/degree_requirements.json"),
        help="Path to degree_requirements.json",
    )
    parser.add_argument(
        "--requirement-courses",
        type=Path,
        default=Path("data/degree_extracted/degree_requirement_courses.json"),
        help="Path to degree_requirement_courses.json",
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=Path("data/degree_extracted/degree_extraction_issues.json"),
        help="Path to degree_extraction_issues.json",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/degree_extracted/degree_extraction_summary.json"),
        help="Path to degree_extraction_summary.json",
    )
    parser.add_argument(
        "--catalog-db",
        type=Path,
        default=Path("data/purdue_catalog.db"),
        help="Path to course catalog SQLite DB for join enrichment",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/degree_requirements.db"),
        help="Output degree requirements SQLite DB path",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)

    degrees = load_json(args.degrees)
    requirements = load_json(args.requirements)
    requirement_courses = load_json(args.requirement_courses)
    issues = load_json(args.issues)
    summary = load_json(args.summary)

    # Build lookup from catalog DB to enrich joins.
    course_title_lookup: dict[str, str] = {}
    if args.catalog_db.exists():
        catalog_conn = sqlite3.connect(args.catalog_db)
        try:
            rows = catalog_conn.execute("SELECT course_code, title FROM courses").fetchall()
            course_title_lookup = {code: title for code, title in rows}
        finally:
            catalog_conn.close()

    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.cursor()

        cur.execute("DELETE FROM degree_requirement_courses")
        cur.execute("DELETE FROM degree_requirements")
        cur.execute("DELETE FROM degree_extraction_issues")
        cur.execute("DELETE FROM degrees")
        cur.execute("DELETE FROM ingestion_metadata")

        # Insert degrees.
        for degree in degrees:
            cur.execute(
                """
                INSERT INTO degrees (
                  degree_name, award, level, confidence,
                  source_page_start, source_page_end, requirement_pages_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    degree["degree_name"],
                    degree.get("award"),
                    degree.get("level"),
                    degree.get("confidence"),
                    degree.get("source_page_start"),
                    degree.get("source_page_end"),
                    degree.get("requirement_pages_count"),
                ),
            )

        degree_id_lookup = {
            name: degree_id
            for degree_id, name in cur.execute("SELECT degree_id, degree_name FROM degrees")
        }

        # Insert requirements.
        requirement_id_lookup: dict[tuple[str, str, str], int] = {}
        for req in requirements:
            degree_name = req["degree_name"]
            degree_id = degree_id_lookup[degree_name]
            cur.execute(
                """
                INSERT INTO degree_requirements (
                  degree_id, requirement_name, credits_text, source_page_start, source_page_end
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    degree_id,
                    req["requirement_name"],
                    req.get("credits_text"),
                    req.get("source_page_start"),
                    req.get("source_page_end"),
                ),
            )
            requirement_id = cur.lastrowid
            requirement_id_lookup[
                (degree_name, req["requirement_name"], req.get("credits_text") or "")
            ] = requirement_id

        # Insert requirement-course links.
        for link in requirement_courses:
            degree_name = link["degree_name"]
            degree_id = degree_id_lookup[degree_name]
            req_key = (
                degree_name,
                link["requirement_name"],
                link.get("credits_text") or "",
            )
            requirement_id = requirement_id_lookup.get(req_key)

            course_code = link["course_code"]
            catalog_title = course_title_lookup.get(course_code)

            cur.execute(
                """
                INSERT OR IGNORE INTO degree_requirement_courses (
                  degree_id, requirement_id, course_code, course_title_on_page,
                  course_in_catalog, catalog_course_title, source_page_start, source_page_end
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    degree_id,
                    requirement_id,
                    course_code,
                    link.get("course_title_on_page"),
                    1 if course_code in course_title_lookup else 0,
                    catalog_title,
                    link.get("source_page_start"),
                    link.get("source_page_end"),
                ),
            )

        # Insert issues.
        for issue in issues:
            degree_name = issue.get("degree_name")
            degree_id = degree_id_lookup.get(degree_name) if degree_name else None
            cur.execute(
                """
                INSERT INTO degree_extraction_issues (
                  degree_id, page, issue_type, details
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    degree_id,
                    issue.get("page"),
                    issue["issue_type"],
                    issue.get("details"),
                ),
            )

        metadata = {
            "degrees_json_path": str(args.degrees),
            "requirements_json_path": str(args.requirements),
            "requirement_courses_json_path": str(args.requirement_courses),
            "issues_json_path": str(args.issues),
            "summary_json_path": str(args.summary),
            "catalog_db_path": str(args.catalog_db),
            "degrees_count": str(len(degrees)),
            "requirements_count": str(len(requirements)),
            "requirement_courses_count": str(len(requirement_courses)),
            "issues_count": str(len(issues)),
            "source_pages_scanned": str(summary.get("pages_scanned")),
        }
        cur.executemany(
            "INSERT INTO ingestion_metadata (key, value) VALUES (?, ?)",
            list(metadata.items()),
        )

        conn.commit()

        db_counts = {
            "degrees": cur.execute("SELECT COUNT(*) FROM degrees").fetchone()[0],
            "degree_requirements": cur.execute(
                "SELECT COUNT(*) FROM degree_requirements"
            ).fetchone()[0],
            "degree_requirement_courses": cur.execute(
                "SELECT COUNT(*) FROM degree_requirement_courses"
            ).fetchone()[0],
            "degree_extraction_issues": cur.execute(
                "SELECT COUNT(*) FROM degree_extraction_issues"
            ).fetchone()[0],
            "course_links_found_in_catalog": cur.execute(
                "SELECT COUNT(*) FROM degree_requirement_courses WHERE course_in_catalog = 1"
            ).fetchone()[0],
        }

        print("Degree requirements SQLite load complete.")
        print(f"Database: {args.db}")
        print(json.dumps(db_counts, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()

