#!/usr/bin/env python3
"""Build a unified SQLite database with catalog + degree tables.

This merges:
  - data/purdue_catalog.db
  - data/degree_requirements.db

Into:
  - data/purdue_academic.db

All domains remain separate, joinable tables.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS courses (
  course_code TEXT PRIMARY KEY,
  subject TEXT NOT NULL,
  catalog_number TEXT NOT NULL,
  title TEXT NOT NULL,
  credit_hours REAL,
  description TEXT,
  prerequisite_text TEXT,
  prerequisite_logic TEXT,
  source_page_start INTEGER,
  source_page_end INTEGER
);

CREATE TABLE IF NOT EXISTS course_prerequisites (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_code TEXT NOT NULL,
  prerequisite_code TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  logic_hint TEXT,
  source_text TEXT,
  FOREIGN KEY (course_code) REFERENCES courses(course_code) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS normalization_issues (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_code TEXT NOT NULL,
  issue_type TEXT NOT NULL,
  details TEXT
);

CREATE TABLE IF NOT EXISTS degree_requirement_snippets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  page INTEGER NOT NULL,
  matched_line_count INTEGER NOT NULL,
  sample_text TEXT NOT NULL
);

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
  FOREIGN KEY (requirement_id) REFERENCES degree_requirements(requirement_id) ON DELETE SET NULL,
  FOREIGN KEY (course_code) REFERENCES courses(course_code) ON DELETE NO ACTION
);

CREATE TABLE IF NOT EXISTS degree_extraction_issues (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  degree_id INTEGER,
  page INTEGER,
  issue_type TEXT NOT NULL,
  details TEXT,
  FOREIGN KEY (degree_id) REFERENCES degrees(degree_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS catalog_ingestion_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS degree_ingestion_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS v_degree_course_links AS
SELECT
  d.degree_name,
  d.award,
  d.level,
  d.confidence,
  COALESCE(r.requirement_name, 'UNASSIGNED') AS requirement_name,
  r.credits_text,
  drc.course_code,
  c.title AS course_title,
  c.credit_hours AS course_credit_hours,
  drc.source_page_start AS link_source_page_start,
  drc.source_page_end AS link_source_page_end
FROM degree_requirement_courses drc
JOIN degrees d
  ON d.degree_id = drc.degree_id
LEFT JOIN degree_requirements r
  ON r.requirement_id = drc.requirement_id
LEFT JOIN courses c
  ON c.course_code = drc.course_code;

CREATE INDEX IF NOT EXISTS idx_courses_subject ON courses(subject);
CREATE INDEX IF NOT EXISTS idx_course_prerequisites_course_code
  ON course_prerequisites(course_code);
CREATE INDEX IF NOT EXISTS idx_course_prerequisites_prerequisite_code
  ON course_prerequisites(prerequisite_code);
CREATE INDEX IF NOT EXISTS idx_degree_requirement_snippets_page
  ON degree_requirement_snippets(page);

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
    parser = argparse.ArgumentParser(description="Build unified Purdue academic SQLite DB.")
    parser.add_argument(
        "--catalog-db",
        type=Path,
        default=Path("data/purdue_catalog.db"),
        help="Path to catalog DB",
    )
    parser.add_argument(
        "--degree-db",
        type=Path,
        default=Path("data/degree_requirements.db"),
        help="Path to degree requirements DB",
    )
    parser.add_argument(
        "--output-db",
        type=Path,
        default=Path("data/purdue_academic.db"),
        help="Path to unified output DB",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.output_db)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.cursor()

        # Attach source DBs.
        cur.execute("ATTACH DATABASE ? AS catalog_src", (str(args.catalog_db),))
        cur.execute("ATTACH DATABASE ? AS degree_src", (str(args.degree_db),))

        # Clear target.
        cur.execute("DELETE FROM degree_requirement_courses")
        cur.execute("DELETE FROM degree_requirements")
        cur.execute("DELETE FROM degree_extraction_issues")
        cur.execute("DELETE FROM degrees")
        cur.execute("DELETE FROM degree_requirement_snippets")
        cur.execute("DELETE FROM normalization_issues")
        cur.execute("DELETE FROM course_prerequisites")
        cur.execute("DELETE FROM courses")
        cur.execute("DELETE FROM catalog_ingestion_metadata")
        cur.execute("DELETE FROM degree_ingestion_metadata")

        # Copy catalog domain.
        cur.execute(
            """
            INSERT INTO courses (
              course_code, subject, catalog_number, title, credit_hours,
              description, prerequisite_text, prerequisite_logic,
              source_page_start, source_page_end
            )
            SELECT
              course_code, subject, catalog_number, title, credit_hours,
              description, prerequisite_text, prerequisite_logic,
              source_page_start, source_page_end
            FROM catalog_src.courses
            """
        )

        cur.execute(
            """
            INSERT INTO course_prerequisites (
              id, course_code, prerequisite_code, sequence, logic_hint, source_text
            )
            SELECT
              id, course_code, prerequisite_code, sequence, logic_hint, source_text
            FROM catalog_src.course_prerequisites
            """
        )

        cur.execute(
            """
            INSERT INTO normalization_issues (id, course_code, issue_type, details)
            SELECT id, course_code, issue_type, details
            FROM catalog_src.normalization_issues
            """
        )

        cur.execute(
            """
            INSERT INTO degree_requirement_snippets (id, page, matched_line_count, sample_text)
            SELECT id, page, matched_line_count, sample_text
            FROM catalog_src.degree_requirement_snippets
            """
        )

        cur.execute(
            """
            INSERT INTO catalog_ingestion_metadata (key, value)
            SELECT key, value
            FROM catalog_src.ingestion_metadata
            """
        )

        # Copy degree domain.
        cur.execute(
            """
            INSERT INTO degrees (
              degree_id, degree_name, award, level, confidence,
              source_page_start, source_page_end, requirement_pages_count
            )
            SELECT
              degree_id, degree_name, award, level, confidence,
              source_page_start, source_page_end, requirement_pages_count
            FROM degree_src.degrees
            """
        )

        cur.execute(
            """
            INSERT INTO degree_requirements (
              requirement_id, degree_id, requirement_name, credits_text,
              source_page_start, source_page_end
            )
            SELECT
              requirement_id, degree_id, requirement_name, credits_text,
              source_page_start, source_page_end
            FROM degree_src.degree_requirements
            """
        )

        cur.execute(
            """
            INSERT INTO degree_requirement_courses (
              id, degree_id, requirement_id, course_code, course_title_on_page,
              course_in_catalog, catalog_course_title, source_page_start, source_page_end
            )
            SELECT
              id, degree_id, requirement_id, course_code, course_title_on_page,
              course_in_catalog, catalog_course_title, source_page_start, source_page_end
            FROM degree_src.degree_requirement_courses
            """
        )

        cur.execute(
            """
            INSERT INTO degree_extraction_issues (id, degree_id, page, issue_type, details)
            SELECT id, degree_id, page, issue_type, details
            FROM degree_src.degree_extraction_issues
            """
        )

        cur.execute(
            """
            INSERT INTO degree_ingestion_metadata (key, value)
            SELECT key, value
            FROM degree_src.ingestion_metadata
            """
        )

        # Add unified metadata.
        cur.execute(
            "INSERT OR REPLACE INTO catalog_ingestion_metadata (key, value) VALUES (?, ?)",
            ("unified_output_db", str(args.output_db)),
        )
        cur.execute(
            "INSERT OR REPLACE INTO degree_ingestion_metadata (key, value) VALUES (?, ?)",
            ("unified_output_db", str(args.output_db)),
        )

        conn.commit()

        counts = {
            "courses": conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0],
            "course_prerequisites": conn.execute(
                "SELECT COUNT(*) FROM course_prerequisites"
            ).fetchone()[0],
            "normalization_issues": conn.execute(
                "SELECT COUNT(*) FROM normalization_issues"
            ).fetchone()[0],
            "degree_requirement_snippets": conn.execute(
                "SELECT COUNT(*) FROM degree_requirement_snippets"
            ).fetchone()[0],
            "degrees": conn.execute("SELECT COUNT(*) FROM degrees").fetchone()[0],
            "degree_requirements": conn.execute(
                "SELECT COUNT(*) FROM degree_requirements"
            ).fetchone()[0],
            "degree_requirement_courses": conn.execute(
                "SELECT COUNT(*) FROM degree_requirement_courses"
            ).fetchone()[0],
            "degree_extraction_issues": conn.execute(
                "SELECT COUNT(*) FROM degree_extraction_issues"
            ).fetchone()[0],
        }

        print("Unified academic DB build complete.")
        print(f"Database: {args.output_db}")
        print(json.dumps(counts, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
