"""Best-effort course suggestions for requirements the catalog states but never enumerates.

See ``planner_db.UnresolvedRequirement`` for why these exist: the catalog does not list
courses for "University Core Requirements" or "Choose 1 course in 6 different disciplines" —
there is no ground truth to check a course against. What this module offers instead is a
semantic best guess, pulled from the same course descriptions the catalog does publish: embed
the requirement's own text, find the nearest course chunks in the ``academic_rules`` pgvector
index (already built for the advisor's RAG chat — see ``rag/ingest_catalog.py``), and hand
back whichever real courses come back closest.

THESE ARE GUESSES, NOT ANSWERS. Nothing here has checked that a suggested course actually
satisfies the requirement's rule — a nearest-neighbour match on "Human Cultures: Behavioral/
Social Science (UCC: BSS)" might land on a course that reads social-science-shaped and isn't
tagged for the requirement at all, because the catalog carries no such tag on courses in the
first place. They exist so a student has somewhere to start looking instead of nothing, and
every caller MUST present them that way — see the "not verified" language in
``plan_requirements.SuggestedCourseView`` and the checklist UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.services.rag import embeddings, store

logger = logging.getLogger(__name__)

SUGGESTION_LIMIT = 5
# The cached lookup fetches more candidates than any single call needs, so per-student
# exclusion (already completed/planned courses) can be applied AFTER the cache without
# varying the cache key — see `suggest_courses`.
_CANDIDATE_POOL = 15


@dataclass(frozen=True)
class SuggestedCourse:
    code: str
    title: str
    credits: int


def _connect() -> psycopg.Connection:
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


@lru_cache(maxsize=512)
def _nearest_course_codes(query_text: str) -> tuple[str, ...]:
    """The semantic half, cached on the requirement's own text.

    A requirement's ``name``+``raw_text`` is often IDENTICAL across every program that
    shares it — "University Core Requirements" is the same paragraph in all 295 programs
    that carry it (see ``planner_db.UNIVERSITY_REQUIREMENT_TYPES``) — so caching on the text
    itself reuses one embed+search across all of them instead of repeating it per program,
    per plan view.

    Filters to course-type chunks IN THE QUERY (``store.search``'s ``metadata_type``), not
    after. A first version ranked the whole mixed index and filtered client-side; for exactly
    the "identical across 295 programs" text above, every one of those 295 near-duplicate
    REQUIREMENT chunks outranks every course chunk, so a plain top-k came back 100%
    requirement rows and the filtered result was empty — nine of WGSS's ten unresolved
    requirements got zero suggestions. Filtering before ranking is what fixes that.
    """
    try:
        vector = embeddings.embed_query(query_text)
        matches = store.search(vector, top_k=_CANDIDATE_POOL, metadata_type="course")
    except Exception:
        # A missing academic_rules table, an unreachable embedding model, whatever — a
        # suggestion list is a nicety, not core functionality, and must never take the
        # requirements checklist down with it.
        logger.warning("Course suggestion lookup failed for %r", query_text[:80], exc_info=True)
        return ()

    codes: list[str] = []
    for match in matches:
        code = match.get("metadata", {}).get("code")
        if code and code not in codes:
            codes.append(code)
    return tuple(codes)


def suggest_courses(
    name: str,
    raw_text: str | None,
    *,
    exclude: frozenset[str] = frozenset(),
    limit: int = SUGGESTION_LIMIT,
) -> list[SuggestedCourse]:
    """Up to ``limit`` real courses whose catalog description reads closest to this
    requirement's own text. Never raises: a broken embedding model or a missing
    ``academic_rules`` table degrades to an empty list, not a failed page.
    """
    query_text = f"{name}. {raw_text or ''}".strip()[:2000]
    codes = [code for code in _nearest_course_codes(query_text) if code not in exclude][:limit]
    if not codes:
        return []

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (course_code) course_code, title, credit_hours_min
                FROM courses
                WHERE course_code = ANY(%s)
                ORDER BY course_code
                """,
                (codes,),
            )
            rows = {row["course_code"]: row for row in cur.fetchall()}
    except psycopg.Error:
        logger.warning("Course lookup for suggestions failed", exc_info=True)
        return []

    out = []
    for code in codes:
        row = rows.get(code)
        if row is not None:
            out.append(SuggestedCourse(
                code=code,
                title=row["title"] or code,
                credits=int(row["credit_hours_min"] or 0),
            ))
    return out
