"""Tests for the pgvector store.

Two tiers:
  * Pure-function tests (vector literal, content hash) — no DB, always run.
  * A cosine-search round-trip using hand-made vectors — needs the catalog_ingestion Postgres
    with pgvector, so it SKIPS (never fails) when that DB isn't reachable or lacks the
    extension. That keeps CI green without a database while still exercising the real SQL
    locally.
"""

import psycopg
import pytest

from app.core.config import settings
from app.services.rag import store


# --- pure functions: no database needed ---------------------------------------------------


def test_vector_literal_formats_pgvector_text():
    assert store._vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"
    assert store._vector_literal([1, 0]) == "[1.0,0.0]"  # ints coerced to float


def test_content_hash_normalizes_case_and_whitespace():
    # Same rule written with different spacing/case must collapse to one upsert key.
    assert store._content_hash("Intro to  CS") == store._content_hash("intro to cs")
    assert store._content_hash("MATH requirement") != store._content_hash("ENGL requirement")


# --- DB round-trip: hand-made vectors, cosine ranking -------------------------------------


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.academic_database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def _unit_vector(axis: int, dims: int) -> list[float]:
    """A hand-made embedding: all zeros except a 1.0 on ``axis`` (a full-width fake vector)."""
    vec = [0.0] * dims
    vec[axis] = 1.0
    return vec


_TEST_MARKER = "rag_store_round_trip"


@pytest.mark.skipif(not _db_available(), reason="catalog_ingestion Postgres not reachable")
def test_upsert_and_cosine_search_ranks_nearest_first():
    dims = settings.rag_embed_dimensions

    try:
        store.ensure_schema()
    except psycopg.Error as exc:  # e.g. image without the vector extension
        pytest.skip(f"pgvector not available (use the pgvector/pgvector image): {exc}")

    try:
        # Two orthogonal chunks: A points along axis 0, B along axis 1.
        id_a = store.upsert_rule("TEST humanities elective", {"_test": _TEST_MARKER, "label": "A"}, _unit_vector(0, dims))
        id_b = store.upsert_rule("TEST calculus requirement", {"_test": _TEST_MARKER, "label": "B"}, _unit_vector(1, dims))

        # Idempotency: re-upserting A (same content) updates in place, no new row.
        id_a_again = store.upsert_rule("TEST humanities elective", {"_test": _TEST_MARKER, "label": "A"}, _unit_vector(0, dims))
        assert id_a_again == id_a

        # Query points mostly along axis 0 -> A (cos ~0.995) must beat B (cos ~0.1).
        query = _unit_vector(0, dims)
        query[1] = 0.1
        results = store.search(query, top_k=50)

        by_id = {r["id"]: r for r in results}
        assert id_a in by_id, "nearest chunk A should be retrieved"
        assert by_id[id_a]["similarity"] > 0.9
        assert by_id[id_a]["metadata"]["label"] == "A"  # JSONB round-trips as a dict

        # Results come back sorted by similarity (i.e. ORDER BY <=> ascending).
        sims = [r["similarity"] for r in results]
        assert sims == sorted(sims, reverse=True)

        # When both test chunks are present, A outranks B.
        ids = [r["id"] for r in results]
        if id_b in by_id:
            assert ids.index(id_a) < ids.index(id_b)
    finally:
        with psycopg.connect(settings.academic_database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM academic_rules WHERE metadata->>'_test' = %s", (_TEST_MARKER,))


@pytest.mark.skipif(not _db_available(), reason="catalog_ingestion Postgres not reachable")
def test_fetch_by_course_codes_matches_space_insensitively():
    dims = settings.rag_embed_dimensions

    try:
        store.ensure_schema()
    except psycopg.Error as exc:
        pytest.skip(f"pgvector not available (use the pgvector/pgvector image): {exc}")

    try:
        # Codes deliberately outside any real subject range (Purdue subjects are 2-4 real
        # letters), so this test is robust to running against the fully-ingested catalog —
        # a real code like "CS 25100" would collide with the actual stored row.
        store.upsert_rule(
            "TEST COURSE ZZ 90001 — Data Structures",
            {"_test": _TEST_MARKER, "type": "course", "code": "ZZ 90001"},
            _unit_vector(0, dims),
        )
        store.upsert_rule(
            "TEST COURSE ZZ 90002 — Calculus",
            {"_test": _TEST_MARKER, "type": "course", "code": "ZZ 90002"},
            _unit_vector(1, dims),
        )

        # "ZZ90001" (no space, as extract_course_codes normalizes it) must still match
        # the stored "ZZ 90001" metadata.
        matches = store.fetch_by_course_codes(["ZZ90001"])
        assert [m["metadata"]["code"] for m in matches] == ["ZZ 90001"]
        assert matches[0]["similarity"] == 1.0

        # A code with no stored match returns nothing, not an error.
        assert store.fetch_by_course_codes(["ZZ99999"]) == []

        # Empty input short-circuits without touching the DB.
        assert store.fetch_by_course_codes([]) == []
    finally:
        with psycopg.connect(settings.academic_database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM academic_rules WHERE metadata->>'_test' = %s", (_TEST_MARKER,))
