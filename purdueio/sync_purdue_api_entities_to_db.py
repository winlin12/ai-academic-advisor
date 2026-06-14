#!/usr/bin/env python3
"""Mirror PurdueIO catalog/class entities into a local SQLite database.

The target database is intentionally limited to PurdueIO-owned entities. Degree
programs, degree requirements, audits, and advising-specific data should be
added later through their own explicit migrations.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.purdue.io/odata"
DEFAULT_DB = SCRIPT_DIR / "purdue_academic_db.db"

ENTITY_ORDER = (
    "Campuses",
    "Buildings",
    "Rooms",
    "Subjects",
    "Courses",
    "Terms",
    "Classes",
    "Sections",
    "Meetings",
    "Instructors",
)
DEFAULT_ENTITIES = ENTITY_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mirror PurdueIO catalog/class entities into local SQLite."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="PurdueIO OData base URL")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Target SQLite DB path")
    parser.add_argument(
        "--entities",
        default=",".join(DEFAULT_ENTITIES),
        help="Comma-separated PurdueIO entity names to mirror",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds")
    return parser.parse_args()


def parse_entities(value: str) -> list[str]:
    entities = [token.strip() for token in value.split(",") if token.strip()]
    if not entities:
        raise ValueError("At least one entity is required.")

    unsupported = sorted(set(entities) - set(ENTITY_ORDER))
    if unsupported:
        raise ValueError(f"Unsupported entity/entities: {', '.join(unsupported)}")

    requested = set(entities)
    return [entity for entity in ENTITY_ORDER if entity in requested]


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
    "Campuses": """
        CREATE TABLE IF NOT EXISTS Campuses (
          Id TEXT PRIMARY KEY,
          Code TEXT,
          Name TEXT,
          ZipCode TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Campuses_Code ON Campuses(Code);
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Campuses_Name ON Campuses(Name);
    """,
    "Buildings": """
        CREATE TABLE IF NOT EXISTS Buildings (
          Id TEXT PRIMARY KEY,
          CampusId TEXT,
          Name TEXT,
          ShortCode TEXT,
          FOREIGN KEY (CampusId) REFERENCES Campuses(Id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Buildings_CampusId_ShortCode
          ON Buildings(CampusId, ShortCode);
        CREATE INDEX IF NOT EXISTS IX_Buildings_Name ON Buildings(Name);
        CREATE INDEX IF NOT EXISTS IX_Buildings_ShortCode ON Buildings(ShortCode);
    """,
    "Rooms": """
        CREATE TABLE IF NOT EXISTS Rooms (
          Id TEXT PRIMARY KEY,
          Number TEXT,
          BuildingId TEXT,
          FOREIGN KEY (BuildingId) REFERENCES Buildings(Id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS IX_Rooms_BuildingId ON Rooms(BuildingId);
        CREATE INDEX IF NOT EXISTS IX_Rooms_Number ON Rooms(Number);
    """,
    "Subjects": """
        CREATE TABLE IF NOT EXISTS Subjects (
          Id TEXT PRIMARY KEY,
          Name TEXT,
          Abbreviation TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Subjects_Abbreviation ON Subjects(Abbreviation);
        CREATE INDEX IF NOT EXISTS IX_Subjects_Name ON Subjects(Name);
    """,
    "Courses": """
        CREATE TABLE IF NOT EXISTS Courses (
          Id TEXT PRIMARY KEY,
          Number TEXT,
          SubjectId TEXT,
          Title TEXT,
          CreditHours REAL,
          Description TEXT,
          FOREIGN KEY (SubjectId) REFERENCES Subjects(Id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS IX_Courses_Number ON Courses(Number);
        CREATE INDEX IF NOT EXISTS IX_Courses_SubjectId ON Courses(SubjectId);
        CREATE INDEX IF NOT EXISTS IX_Courses_Title ON Courses(Title);
    """,
    "Terms": """
        CREATE TABLE IF NOT EXISTS Terms (
          Id TEXT PRIMARY KEY,
          Code TEXT,
          Name TEXT,
          StartDate TEXT,
          EndDate TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Terms_Code ON Terms(Code);
        CREATE UNIQUE INDEX IF NOT EXISTS IX_Terms_Name ON Terms(Name);
    """,
    "Classes": """
        CREATE TABLE IF NOT EXISTS Classes (
          Id TEXT PRIMARY KEY,
          CourseId TEXT,
          TermId TEXT,
          CampusId TEXT,
          FOREIGN KEY (CourseId) REFERENCES Courses(Id) ON DELETE CASCADE,
          FOREIGN KEY (TermId) REFERENCES Terms(Id) ON DELETE CASCADE,
          FOREIGN KEY (CampusId) REFERENCES Campuses(Id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS IX_Classes_CourseId ON Classes(CourseId);
        CREATE INDEX IF NOT EXISTS IX_Classes_TermId ON Classes(TermId);
        CREATE INDEX IF NOT EXISTS IX_Classes_CampusId ON Classes(CampusId);
    """,
    "Sections": """
        CREATE TABLE IF NOT EXISTS Sections (
          Id TEXT PRIMARY KEY,
          Crn TEXT,
          ClassId TEXT,
          Type TEXT,
          StartDate TEXT,
          EndDate TEXT,
          FOREIGN KEY (ClassId) REFERENCES Classes(Id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS IX_Sections_ClassId ON Sections(ClassId);
        CREATE INDEX IF NOT EXISTS IX_Sections_Crn ON Sections(Crn);
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
          RoomId TEXT,
          FOREIGN KEY (SectionId) REFERENCES Sections(Id) ON DELETE CASCADE,
          FOREIGN KEY (RoomId) REFERENCES Rooms(Id)
        );
        CREATE INDEX IF NOT EXISTS IX_Meetings_SectionId ON Meetings(SectionId);
        CREATE INDEX IF NOT EXISTS IX_Meetings_RoomId ON Meetings(RoomId);
    """,
    "Instructors": """
        CREATE TABLE IF NOT EXISTS Instructors (
          Id TEXT PRIMARY KEY,
          Name TEXT,
          Email TEXT
        );
        CREATE INDEX IF NOT EXISTS IX_Instructors_Name ON Instructors(Name);
        CREATE INDEX IF NOT EXISTS IX_Instructors_Email ON Instructors(Email);
    """,
}


def create_tables(conn: sqlite3.Connection, entities: list[str]) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    for entity in entities:
        conn.executescript(TABLE_DDL[entity])


def clear_tables(conn: sqlite3.Connection, entities: list[str]) -> None:
    selected = set(entities)
    for entity in reversed(ENTITY_ORDER):
        if entity in selected:
            conn.execute(f'DELETE FROM "{entity}"')


def as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def as_id(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sync_campuses(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_text(r.get("Code")),
            as_text(r.get("Name")),
            as_text(r.get("ZipCode")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Campuses (Id, Code, Name, ZipCode)
        VALUES (?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_buildings(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_id(r.get("CampusId")),
            as_text(r.get("Name")),
            as_text(r.get("ShortCode")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Buildings (Id, CampusId, Name, ShortCode)
        VALUES (?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_rooms(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (as_id(r.get("Id")), as_text(r.get("Number")), as_id(r.get("BuildingId")))
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Rooms (Id, Number, BuildingId)
        VALUES (?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_subjects(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (as_id(r.get("Id")), as_text(r.get("Name")), as_text(r.get("Abbreviation")))
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Subjects (Id, Name, Abbreviation)
        VALUES (?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_courses(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_text(r.get("Number")),
            as_id(r.get("SubjectId")),
            as_text(r.get("Title")),
            as_float(r.get("CreditHours")),
            as_text(r.get("Description")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Courses (Id, Number, SubjectId, Title, CreditHours, Description)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_terms(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_text(r.get("Code")),
            as_text(r.get("Name")),
            as_text(r.get("StartDate")),
            as_text(r.get("EndDate")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Terms (Id, Code, Name, StartDate, EndDate)
        VALUES (?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_classes(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_id(r.get("CourseId")),
            as_id(r.get("TermId")),
            as_id(r.get("CampusId")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Classes (Id, CourseId, TermId, CampusId)
        VALUES (?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_sections(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_text(r.get("Crn")),
            as_id(r.get("ClassId")),
            as_text(r.get("Type")),
            as_text(r.get("StartDate")),
            as_text(r.get("EndDate")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Sections (Id, Crn, ClassId, Type, StartDate, EndDate)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_meetings(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (
            as_id(r.get("Id")),
            as_id(r.get("SectionId")),
            as_text(r.get("Type")),
            as_text(r.get("StartDate")),
            as_text(r.get("EndDate")),
            as_int(r.get("DaysOfWeek")),
            as_text(r.get("StartTime")),
            as_text(r.get("Duration")),
            as_id(r.get("RoomId")),
        )
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Meetings (
          Id, SectionId, Type, StartDate, EndDate, DaysOfWeek, StartTime, Duration, RoomId
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        values,
    )
    return len(values)


def sync_instructors(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    values = [
        (as_id(r.get("Id")), as_text(r.get("Name")), as_text(r.get("Email")))
        for r in rows
        if as_id(r.get("Id"))
    ]
    conn.executemany(
        """
        INSERT INTO Instructors (Id, Name, Email)
        VALUES (?, ?, ?)
        """,
        values,
    )
    return len(values)


SYNC_FUNCS: dict[str, Callable[[sqlite3.Connection, list[dict[str, Any]]], int]] = {
    "Campuses": sync_campuses,
    "Buildings": sync_buildings,
    "Rooms": sync_rooms,
    "Subjects": sync_subjects,
    "Courses": sync_courses,
    "Terms": sync_terms,
    "Classes": sync_classes,
    "Sections": sync_sections,
    "Meetings": sync_meetings,
    "Instructors": sync_instructors,
}


def main() -> None:
    args = parse_args()
    entities = parse_entities(args.entities)

    fetched: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        print(f"Fetching {entity}...")
        rows = fetch_all_rows(args.base_url, entity=entity, timeout=args.timeout)
        fetched[entity] = rows
        print(f"{entity}: fetched {len(rows)} rows")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        create_tables(conn, entities)
        counts: dict[str, int] = {}

        with conn:
            clear_tables(conn, entities)
            for entity in entities:
                synced = SYNC_FUNCS[entity](conn, fetched[entity])
                counts[entity] = synced
                print(f"{entity}: saved {synced} rows")

        print(f"Sync complete: {args.db}")
        print(json.dumps(counts, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
