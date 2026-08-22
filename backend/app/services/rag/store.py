"""Step 1 (schema) + the Postgres side of Steps 2 and 3.

This module is the *only* place that talks to Postgres/pgvector. It receives vectors as
plain ``list[float]`` and returns plain dict rows; it neither knows nor cares which model
produced those vectors. That decoupling is what let the embedding provider move from
Ollama to in-process fastembed (2026-07-10) without touching a line of SQL.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.core.config import settings

logger = logging.getLogger(__name__)


def _connect() -> psycopg.Connection:
    """A dict-row connection to the same catalog_ingestion DB the rest of the app uses.

    Deliberately local to this module (not imported from academic_db) so the RAG layer has
    no dependency on the read-only catalog code — either could be deleted without breaking
    the other.
    """
    return psycopg.connect(settings.academic_database_url, row_factory=dict_row)


def _vector_literal(embedding: list[float]) -> str:
    """Render a Python vector as the text form pgvector accepts: ``[0.1,0.2,...]``.

    We send this string as an ordinary query parameter and cast it with ``::vector`` in SQL.
    That keeps the module dependency-free (no pgvector Python adapter required) while still
    going through parameter binding, so it is injection-safe.
    """
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _content_hash(text: str) -> str:
    """Stable id for a chunk: sha256 of its whitespace-normalized, lowercased text.

    This is the UPSERT key. Re-ingesting the same rule (even with different spacing/case)
    updates the existing row instead of creating a duplicate, so ingestion is idempotent and
    safe to re-run.
    """
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def ensure_schema() -> None:
    """Step 1: enable pgvector and create ``academic_rules`` + its indexes (idempotent).

    Safe to call on every startup: every statement is ``IF NOT EXISTS``. The VECTOR width is
    pinned from config so the column, the embedding model, and ``embeddings.embed_*`` can
    never silently disagree.
    """
    dims = settings.rag_embed_dimensions
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS academic_rules (
                id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                content       TEXT        NOT NULL,
                metadata      JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
                content_hash  TEXT        NOT NULL UNIQUE,
                embedding     VECTOR({dims}) NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # HNSW graph index for fast approximate nearest-neighbour on COSINE distance. The
        # ops class MUST be vector_cosine_ops to match the ``<=>`` operator used in search().
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS academic_rules_embedding_hnsw
            ON academic_rules USING hnsw (embedding vector_cosine_ops)
            """
        )
        # GIN index so metadata filters (e.g. WHERE metadata @> '{"department":"MATH"}')
        # stay fast if you later combine semantic search with structured filtering.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS academic_rules_metadata_gin
            ON academic_rules USING gin (metadata)
            """
        )
    logger.info("academic_rules schema ready (VECTOR(%d), hnsw cosine index)", dims)


def _table_ready(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.academic_rules') AS t")
        row = cur.fetchone()
    return bool(row and row["t"])


def existing_hashes() -> set[str]:
    """Every ``content_hash`` already stored, so bulk ingestion can skip re-embedding chunks
    it has seen before. Returns an empty set (not an error) when the table doesn't exist yet,
    so a first-time ingest just treats everything as new.
    """
    with _connect() as conn:
        if not _table_ready(conn):
            return set()
        with conn.cursor() as cur:
            cur.execute("SELECT content_hash FROM academic_rules")
            return {row["content_hash"] for row in cur.fetchall()}


def upsert_rule(content: str, metadata: dict[str, Any], embedding: list[float]) -> int:
    """The DB half of Step 2: insert-or-update one chunk keyed by its content hash.

    Returns the row id. Takes an already-computed ``embedding`` so this function is pure
    persistence — no network, no model. ``metadata`` is bound as JSONB via ``Jsonb`` so the
    dict round-trips without manual json.dumps and stays queryable.
    """
    row_id_key = _content_hash(content)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO academic_rules (content, metadata, content_hash, embedding)
            VALUES (%(content)s, %(metadata)s, %(hash)s, %(embedding)s::vector)
            ON CONFLICT (content_hash) DO UPDATE
                SET content    = EXCLUDED.content,
                    metadata   = EXCLUDED.metadata,
                    embedding  = EXCLUDED.embedding,
                    updated_at = now()
            RETURNING id
            """,
            {
                "content": content,
                "metadata": Jsonb(metadata),
                "hash": row_id_key,
                "embedding": _vector_literal(embedding),
            },
        )
        row = cur.fetchone()
    return int(row["id"])


def fetch_by_course_codes(codes: list[str]) -> list[dict[str, Any]]:
    """Exact-match retrieval: course chunks whose metadata ``code`` matches one of ``codes``.

    The deterministic first tier of retrieval (see ``pipeline``): when a student names a
    course, its chunk is fetched by metadata equality instead of hoping cosine distance
    finds it. Codes are compared space-insensitively ("CS 25100" == "CS25100"). Rows come
    back in the same shape as ``search()`` with similarity pinned to 1.0 so the pipeline
    can merge both tiers into one ranked context. Returns [] on an empty/missing table.
    """
    if not codes:
        return []
    wanted = [code.replace(" ", "").upper() for code in codes]
    with _connect() as conn:
        if not _table_ready(conn):
            return []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id,
                       content,
                       metadata,
                       1.0::float8 AS similarity
                FROM academic_rules
                WHERE metadata->>'type' = 'course'
                  AND upper(replace(metadata->>'code', ' ', '')) = ANY(%(codes)s)
                ORDER BY metadata->>'code'
                """,
                {"codes": wanted},
            )
            return cur.fetchall()


def search(
    embedding: list[float], top_k: int, *, metadata_type: str | None = None,
) -> list[dict[str, Any]]:
    """The DB half of Step 3: cosine nearest-neighbour lookup.

    ``<=>`` is pgvector's *cosine distance* (0 = identical direction, 2 = opposite), so we
    ORDER BY it ascending to get the closest chunks and report ``1 - distance`` as an
    easy-to-read similarity in [-1, 1] (≈1 = very relevant). Returns [] if the table has not
    been created yet, so a fresh DB degrades gracefully instead of raising.

    ``metadata_type`` filters BEFORE ranking (via the GIN index on ``metadata``), not after.
    That distinction matters: the index holds one chunk per course AND one per requirement
    block, and requirement text about the same policy is near-duplicated across every program
    that states it — "University Core Requirements" is close to identical in ~295 programs'
    chunks. A caller wanting only courses back who post-filtered a plain top-k would find
    those ~295 near-duplicates crowding out every course chunk before ranking ever reached
    one — the since-removed prose-requirement course suggester hit exactly that.

    A second, sharper version of the same trap survives even with the filter IN the query:
    pgvector's HNSW index answers ``ORDER BY <=>`` by walking the graph and examining only
    ``hnsw.ef_search`` neighbours (40 by default) BEFORE the WHERE filter is applied. For that
    "295 near-identical chunks" case the entire 40-neighbour neighbourhood the graph walk finds
    can be non-course rows, so the filter has nothing to let through — zero rows back, not an
    error, just quietly wrong. Widening ``ef_search`` for a filtered call is what fixes it; on
    this table's size (~18k rows) Postgres typically answers by falling back to a plain
    sequential scan, which is correct and, filtered or not, still well under 50ms.
    """
    with _connect() as conn:
        if not _table_ready(conn):
            logger.warning("academic_rules does not exist yet; run ensure_schema() and ingest.")
            return []
        with conn.cursor() as cur:
            if metadata_type is not None:
                cur.execute("SET LOCAL hnsw.ef_search = 1000")
            cur.execute(
                """
                SELECT id,
                       content,
                       metadata,
                       1 - (embedding <=> %(q)s::vector) AS similarity
                FROM academic_rules
                WHERE %(mtype)s::text IS NULL OR metadata @> jsonb_build_object('type', %(mtype)s)
                ORDER BY embedding <=> %(q)s::vector
                LIMIT %(k)s
                """,
                {"q": _vector_literal(embedding), "k": top_k, "mtype": metadata_type},
            )
            return cur.fetchall()
