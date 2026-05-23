#!/usr/bin/env python3
"""Sync Purdue API entities into separate SQLite tables.

Default entities:
  - Subjects
  - Terms
  - Classes
  - Sections
  - Meetings
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.purdue.io/odata"
DEFAULT_DB = Path("purdueio/purdue_api_academic.db")
DEFAULT_ENTITIES = ("Subjects", "Courses", "Terms", "Classes", "Sections", "Meetings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Purdue API entities into SQLite.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Purdue OData base URL")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Target SQLite DB path")
    parser.add_argument(
        "--entities",
        default=",".join(DEFAULT_ENTITIES),
        help="Comma-separated entity names to sync",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def parse_entities(value: str) -> list[str]:
    entities = [token.strip() for token in value.split(",") if token.strip()]
    if not entities:
        raise ValueError("At least one entity is required.")
    return entities


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ai-academic-advisor/1.0",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {exc.reason}\n{body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected response type for {url}: {type(payload)!r}")
    return payload


def fetch_all_rows(base_url: str, entity: str, timeout: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_url: str | None = f"{base_url.rstrip('/')}/{entity}"

    while next_url:
        payload = fetch_json(next_url, timeout=timeout)
        page = payload.get("value", [])
        if not isinstance(page, list):
            raise RuntimeError(f"Expected 'value' list for {entity}")
        rows.extend(item for item in page if isinstance(item, dict))

        next_link = payload.get("@odata.nextLink") or payload.get("odata.nextLink")
        next_url = urljoin(f"{base_url.rstrip('/')}/", str(next_link)) if next_link else None

    return rows


TABLE_DDL: dict[str, str] = {
    "Subjects": """
        CREATE TABLE IF NOT EXISTS Subjects (
          Id TEXT PRIMARY KEY,
          Name TEXT,
          Abbreviation TEXT
        );
    """,
    "Courses": """
        CREATE TABLE IF NOT EXISTS Courses (
          Id TEXT PRIMARY KEY,
          Number TEXT,
          SubjectId TEXT,
          Title TEXT,
          CreditHours REAL,
          Description TEXT
        );
    """,
    "Terms": """
        CREATE TABLE IF NOT EXISTS Terms (
          Id TEXT PRIMARY KEY,
          Code TEXT,
          Name TEXT,
          StartDate TEXT,
          EndDate TEXT
        );
    """,
    "Classes": """
        CREATE TABLE IF NOT EXISTS Classes (
          Id TEXT PRIMARY KEY,
          CourseId TEXT,
          TermId TEXT,
          CampusId TEXT
        );
    """,
    "Sections": """
        CREATE TABLE IF NOT EXISTS Sections (
          Id TEXT PRIMARY KEY,
          Crn TEXT,
          ClassId TEXT,
          Type TEXT,
          StartDate TEXT,
          EndDate TEXT
        );
    """,
    "Meetings": """
        CREATE TABLE IF NOT EXISTS Meetings (
          Id TEXT PRIMARY KEY,
          SectionId TEXT,
          Type TEXT,
          StartDate TEXT,
          EndDate TEXT,
          DaysOfWeek INTEGER,
          StartTime TEXT,
          Duration TEXT,
          RoomId TEXT
        );
    """,
}


def create_tables(conn: sqlite3.Connection, entities: list[str]) -> None:
    for entity in entities:
        ddl = TABLE_DDL.get(entity)
        if ddl:
            conn.executescript(ddl)


def sync_subjects(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (str(r.get("Id") or ""), r.get("Name"), r.get("Abbreviation"))
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Subjects (Id, Name, Abbreviation)
        VALUES (?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET Name=excluded.Name, Abbreviation=excluded.Abbreviation
        """,
        values,
    )
    return len(values)


def sync_terms(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            str(r.get("Id") or ""),
            r.get("Code"),
            r.get("Name"),
            r.get("StartDate"),
            r.get("EndDate"),
        )
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Terms (Id, Code, Name, StartDate, EndDate)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET
          Code=excluded.Code,
          Name=excluded.Name,
          StartDate=excluded.StartDate,
          EndDate=excluded.EndDate
        """,
        values,
    )
    return len(values)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sync_courses(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            str(r.get("Id") or ""),
            r.get("Number"),
            r.get("SubjectId"),
            r.get("Title"),
            as_float(r.get("CreditHours")),
            r.get("Description"),
        )
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Courses (Id, Number, SubjectId, Title, CreditHours, Description)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET
          Number=excluded.Number,
          SubjectId=excluded.SubjectId,
          Title=excluded.Title,
          CreditHours=excluded.CreditHours,
          Description=excluded.Description
        """,
        values,
    )
    return len(values)


def sync_classes(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (str(r.get("Id") or ""), r.get("CourseId"), r.get("TermId"), r.get("CampusId"))
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Classes (Id, CourseId, TermId, CampusId)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET
          CourseId=excluded.CourseId,
          TermId=excluded.TermId,
          CampusId=excluded.CampusId
        """,
        values,
    )
    return len(values)


def sync_sections(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            str(r.get("Id") or ""),
            r.get("Crn"),
            r.get("ClassId"),
            r.get("Type"),
            r.get("StartDate"),
            r.get("EndDate"),
        )
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Sections (Id, Crn, ClassId, Type, StartDate, EndDate)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET
          Crn=excluded.Crn,
          ClassId=excluded.ClassId,
          Type=excluded.Type,
          StartDate=excluded.StartDate,
          EndDate=excluded.EndDate
        """,
        values,
    )
    return len(values)


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sync_meetings(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            str(r.get("Id") or ""),
            r.get("SectionId"),
            r.get("Type"),
            r.get("StartDate"),
            r.get("EndDate"),
            as_int(r.get("DaysOfWeek")),
            r.get("StartTime"),
            r.get("Duration"),
            r.get("RoomId"),
        )
        for r in rows
        if r.get("Id")
    ]
    conn.executemany(
        """
        INSERT INTO Meetings (
          Id, SectionId, Type, StartDate, EndDate, DaysOfWeek, StartTime, Duration, RoomId
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Id)
        DO UPDATE SET
          SectionId=excluded.SectionId,
          Type=excluded.Type,
          StartDate=excluded.StartDate,
          EndDate=excluded.EndDate,
          DaysOfWeek=excluded.DaysOfWeek,
          StartTime=excluded.StartTime,
          Duration=excluded.Duration,
          RoomId=excluded.RoomId
        """,
        values,
    )
    return len(values)


SYNC_FUNCS = {
    "Subjects": sync_subjects,
    "Courses": sync_courses,
    "Terms": sync_terms,
    "Classes": sync_classes,
    "Sections": sync_sections,
    "Meetings": sync_meetings,
}


def main() -> None:
    args = parse_args()
    entities = parse_entities(args.entities)
    for entity in entities:
        if entity not in SYNC_FUNCS:
            raise ValueError(f"Unsupported entity: {entity}")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        create_tables(conn, entities)
        counts: dict[str, int] = {}

        for entity in entities:
            print(f"Fetching {entity}...")
            rows = fetch_all_rows(args.base_url, entity=entity, timeout=args.timeout)
            print(f"{entity}: fetched {len(rows)} rows")

            conn.execute(f"DELETE FROM {entity}")
            synced = SYNC_FUNCS[entity](conn, rows)
            conn.commit()
            counts[entity] = synced
            print(f"{entity}: saved {synced} rows")

        print("Sync complete.")
        print(json.dumps(counts, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
