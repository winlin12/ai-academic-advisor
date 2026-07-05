"""Populate the pgvector ``academic_rules`` table from the ingested catalog.

The advisor answers over ``academic_rules`` (semantic retrieval), but that table starts empty
— the catalog lives in the relational tables (``courses``, ``programs``, ...). This module is
the bridge: it turns each course and each program requirement block into one self-contained,
retrievable text chunk, embeds it via the local model, and upserts it.

Run it once after the relational catalog has been ingested (and re-run any time it changes):

    python -m app.services.rag.ingest_catalog            # embed courses + program rules
    python -m app.services.rag.ingest_catalog --dry-run  # build chunks, print samples, no model
    python -m app.services.rag.ingest_catalog --limit 50 # smoke-test the model path cheaply

Embedding 10k+ courses through a local model takes a while; ingestion is idempotent (chunks are
keyed by a content hash), so a re-run only embeds new or changed chunks unless you pass --force.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.models.schemas import AcademicProgramDetail, RequirementBlockDetail
from app.services.academic_db import _credits_text, fetch_program_detail, fetch_program_summaries
from app.services.ollama_client import OllamaClient
from app.services.rag import store
from app.services.rag.pipeline import ingest_rule

logger = logging.getLogger(__name__)

# Keep any single chunk within a comfortable embedding window. A handful of huge requirement
# blocks would otherwise be silently truncated by the model and retrieve poorly; clipping here
# makes that explicit. Courses are always well under this.
MAX_CHUNK_CHARS = 6000

Chunk = tuple[str, dict[str, Any]]


# --- pure chunk builders (no DB, no model — unit-testable) --------------------------------


def build_course_chunk(row: dict[str, Any]) -> Chunk:
    """One course row -> a self-contained '(what/prereqs/credits)' chunk + tags.

    This is the unit a student most often asks about ("what's in CS 251?", "prereqs for MA
    261?"), so each course is its own retrievable chunk rather than being buried in a program.
    """
    code = (row["course_code"] or "").strip() or "?"
    title = (row["title"] or "Untitled course").strip()
    credits = _credits_text(row["credit_hours_min"], row["credit_hours_max"])
    credit_text = f" ({credits} cr)" if credits else ""
    description = " ".join((row["description"] or "").split()) or "No description on file."
    prereqs = " ".join((row["prerequisites_raw"] or "").split()) or "not listed in the catalog"
    content = f"COURSE {code} — {title}{credit_text}. {description} Prerequisites: {prereqs}."
    metadata = {
        "type": "course",
        "code": code,
        "subject": (row["subject_code"] or "").strip() or None,
    }
    return content, metadata


def _render_course_option(course: Any) -> str:
    label = (course.course_code_text or (course.raw_text or "")).strip()
    if course.course_title:
        label = f"{label} — {course.course_title}" if label else course.course_title
    if course.credits_text:
        label += f" ({course.credits_text} cr)"
    return label


def build_block_chunk(
    program: AcademicProgramDetail, block: RequirementBlockDetail
) -> Chunk | None:
    """One requirement block of one program -> a chunk describing what it requires.

    A block (e.g. "Major Core", "Science Selectives") is a good retrieval unit: focused enough
    to embed well, complete enough to answer "what do I need for X?". Blocks with no courses and
    no narrative text (empty containers) yield ``None`` and are skipped.
    """
    title = block.title or "Requirements"
    body: list[str] = []
    for rule in block.rules:
        courses = [course for option in rule.options for course in option.courses]
        if courses:
            rendered = "; ".join(_render_course_option(course) for course in courses)
            label = rule.options[0].label if rule.options else None
            body.append(f"{label}: {rendered}" if label and label != title else rendered)
        elif rule.raw_text:
            body.append(" ".join(rule.raw_text.split()))

    if not body:
        return None

    header = f"PROGRAM: {program.program_title}"
    if program.degree_code:
        header += f" ({program.degree_code})"
    if program.school:
        header += f" — {program.school}"

    lines = [header, f"REQUIREMENT BLOCK: {title}"]
    if block.credits_text:
        lines.append(f"Credits: {block.credits_text}")
    content = "\n".join(lines + body)
    if len(content) > MAX_CHUNK_CHARS:
        content = content[: MAX_CHUNK_CHARS - 1].rstrip() + "…"

    metadata = {
        "type": "requirement",
        "program": program.program_title,
        "program_id": program.id,
        "degree": program.degree_code,
        "catalog_year": program.catalog_year,
        "block": title,
    }
    return content, metadata


# --- DB-backed chunk sources --------------------------------------------------------------


def iter_course_chunks() -> Iterator[Chunk]:
    """Every course in the catalog, one chunk each."""
    with psycopg.connect(settings.academic_database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT course_code, subject_code, title, description,
                       credit_hours_min, credit_hours_max, prerequisites_raw
                FROM courses
                ORDER BY course_code
                """
            )
            for row in cur.fetchall():
                yield build_course_chunk(row)


def iter_program_chunks() -> Iterator[Chunk]:
    """Every program's requirement blocks, one chunk per non-empty block."""
    for summary in fetch_program_summaries(limit=1_000_000):
        detail = fetch_program_detail(summary.id)
        if detail is None:
            continue
        for block in detail.blocks:
            chunk = build_block_chunk(detail, block)
            if chunk is not None:
                yield chunk


# --- driver -------------------------------------------------------------------------------


async def ingest_catalog(
    *,
    force: bool = False,
    include_courses: bool = True,
    include_programs: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Build catalog chunks and (unless ``dry_run``) embed + upsert them into ``academic_rules``.

    Idempotent: chunks already stored (matched by content hash) are skipped unless ``force``.
    Returns per-type counts so the CLI/tests can report what happened.
    """
    counts = {"course": 0, "requirement": 0, "skipped": 0}

    if not dry_run:
        store.ensure_schema()
    existing = set() if (force or dry_run) else store.existing_hashes()
    client = None if dry_run else OllamaClient()

    def chunk_sources() -> Iterator[Chunk]:
        if include_courses:
            yield from iter_course_chunks()
        if include_programs:
            yield from iter_program_chunks()

    processed = 0
    for content, metadata in chunk_sources():
        if not force and store._content_hash(content) in existing:
            counts["skipped"] += 1
            continue

        kind = metadata["type"]
        if dry_run:
            if counts[kind] < 2:  # show a couple of real samples per kind
                logger.info("SAMPLE [%s] %s", kind, content[:200].replace("\n", " / "))
        else:
            await ingest_rule(content, metadata, client=client)
        counts[kind] += 1

        processed += 1
        if processed % 100 == 0:
            logger.info(
                "… %d chunks (%d courses, %d requirements, %d skipped)",
                processed, counts["course"], counts["requirement"], counts["skipped"],
            )
        if limit is not None and processed >= limit:
            break

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-embed chunks even if unchanged")
    parser.add_argument("--limit", type=int, default=None, help="stop after N chunks (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="build chunks only; no model/DB writes")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--courses-only", action="store_true")
    group.add_argument("--programs-only", action="store_true")
    args = parser.parse_args()

    counts = asyncio.run(
        ingest_catalog(
            force=args.force,
            include_courses=not args.programs_only,
            include_programs=not args.courses_only,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )
    verb = "Would ingest" if args.dry_run else "Ingested"
    print(
        f"{verb} {counts['course']} course chunk(s) and {counts['requirement']} "
        f"requirement chunk(s); skipped {counts['skipped']} already-stored."
    )


if __name__ == "__main__":
    main()
