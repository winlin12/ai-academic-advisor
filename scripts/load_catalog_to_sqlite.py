#!/usr/bin/env python3
"""Load normalized catalog JSON data into SQLite.

Inputs:
  - data/normalized/courses_normalized.json
  - data/normalized/prerequisite_edges.json
  - data/normalized/normalization_issues.json
  - data/degree_requirement_snippets.json

Output:
  - data/purdue_catalog.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


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

CREATE TABLE IF NOT EXISTS ingestion_metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_courses_subject ON courses(subject);
CREATE INDEX IF NOT EXISTS idx_course_prerequisites_course_code ON course_prerequisites(course_code);
CREATE INDEX IF NOT EXISTS idx_course_prerequisites_prerequisite_code
  ON course_prerequisites(prerequisite_code);
CREATE INDEX IF NOT EXISTS idx_degree_requirement_snippets_page ON degree_requirement_snippets(page);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load normalized catalog JSON into SQLite.")
    parser.add_argument(
        "--courses",
        type=Path,
        default=Path("data/normalized/courses_normalized.json"),
        help="Path to normalized courses JSON",
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=Path("data/normalized/prerequisite_edges.json"),
        help="Path to prerequisite edges JSON",
    )
    parser.add_argument(
        "--issues",
        type=Path,
        default=Path("data/normalized/normalization_issues.json"),
        help="Path to normalization issues JSON",
    )
    parser.add_argument(
        "--snippets",
        type=Path,
        default=Path("data/degree_requirement_snippets.json"),
        help="Path to degree requirement snippets JSON",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/purdue_catalog.db"),
        help="Output SQLite database path",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)

    courses = load_json(args.courses)
    edges = load_json(args.edges)
    issues = load_json(args.issues)
    snippets = load_json(args.snippets)

    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(SCHEMA_SQL)
        cur = conn.cursor()

        # Replace existing rows to keep reruns deterministic.
        cur.execute("DELETE FROM course_prerequisites")
        cur.execute("DELETE FROM normalization_issues")
        cur.execute("DELETE FROM degree_requirement_snippets")
        cur.execute("DELETE FROM courses")
        cur.execute("DELETE FROM ingestion_metadata")

        cur.executemany(
            """
            INSERT INTO courses (
              course_code,
              subject,
              catalog_number,
              title,
              credit_hours,
              description,
              prerequisite_text,
              prerequisite_logic,
              source_page_start,
              source_page_end
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    course["course_code"],
                    course["subject"],
                    course["catalog_number"],
                    course["title"],
                    course.get("credit_hours"),
                    course.get("description"),
                    course.get("prerequisite_text"),
                    course.get("prerequisite_logic"),
                    course.get("source_page_start"),
                    course.get("source_page_end"),
                )
                for course in courses
            ],
        )

        cur.executemany(
            """
            INSERT INTO course_prerequisites (
              course_code,
              prerequisite_code,
              sequence,
              logic_hint,
              source_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    edge["course_code"],
                    edge["prerequisite_code"],
                    edge["sequence"],
                    edge.get("logic_hint"),
                    edge.get("source_text"),
                )
                for edge in edges
            ],
        )

        cur.executemany(
            """
            INSERT INTO normalization_issues (
              course_code,
              issue_type,
              details
            ) VALUES (?, ?, ?)
            """,
            [
                (
                    issue["course_code"],
                    issue["issue_type"],
                    issue.get("details"),
                )
                for issue in issues
            ],
        )

        cur.executemany(
            """
            INSERT INTO degree_requirement_snippets (
              page,
              matched_line_count,
              sample_text
            ) VALUES (?, ?, ?)
            """,
            [
                (
                    snippet["page"],
                    snippet["matched_line_count"],
                    snippet["sample_text"],
                )
                for snippet in snippets
            ],
        )

        metadata = {
            "courses_json_path": str(args.courses),
            "edges_json_path": str(args.edges),
            "issues_json_path": str(args.issues),
            "snippets_json_path": str(args.snippets),
            "courses_count": str(len(courses)),
            "prerequisite_edges_count": str(len(edges)),
            "issues_count": str(len(issues)),
            "requirement_snippets_count": str(len(snippets)),
        }
        cur.executemany(
            "INSERT INTO ingestion_metadata (key, value) VALUES (?, ?)",
            list(metadata.items()),
        )

        conn.commit()

        print("SQLite load complete.")
        print(f"Database: {args.db}")
        print(
            json.dumps(
                {
                    "courses": len(courses),
                    "prerequisite_edges": len(edges),
                    "issues": len(issues),
                    "degree_requirement_snippets": len(snippets),
                },
                indent=2,
            )
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

