"""Tests for the unresolved-requirement course suggester.

Two tiers, mirroring test_embeddings.py / test_rag_store.py's pattern:
  * Pure-function tests (candidate filtering, exclusion) — no model, no DB, always run.
  * A real embed+search+lookup round-trip against the actual model and Postgres — SKIPS
    (never fails) when either isn't reachable, so CI/offline runs stay green while a normal
    dev machine still exercises the real path.
"""

import psycopg
import pytest

from app.core.config import settings
from app.services import plan_suggestions
from app.services.plan_suggestions import SuggestedCourse, suggest_courses
from app.services.rag import embeddings


# --- pure function: exclusion and limit apply after the (mocked) semantic lookup -------------


def test_suggest_courses_returns_empty_without_candidates(monkeypatch):
    monkeypatch.setattr(plan_suggestions, "_nearest_course_codes", lambda text: ())
    assert suggest_courses("Anything", "some text") == []


def test_suggest_courses_filters_excluded_and_respects_limit(monkeypatch):
    monkeypatch.setattr(
        plan_suggestions, "_nearest_course_codes",
        lambda text: ("CS 18000", "CS 25100", "MA 16100", "ENGL 10600"),
    )

    def fake_connect():
        class FakeCursor:
            def execute(self, query, params):
                self.codes = params[0]

            def fetchall(self):
                rows = {
                    "CS 18000": {"course_code": "CS 18000", "title": "Problem Solving",
                                 "credit_hours_min": 4},
                    "CS 25100": {"course_code": "CS 25100", "title": "Data Structures",
                                 "credit_hours_min": 3},
                    "MA 16100": {"course_code": "MA 16100", "title": "Calculus I",
                                 "credit_hours_min": 5},
                    "ENGL 10600": {"course_code": "ENGL 10600", "title": "Composition",
                                   "credit_hours_min": 4},
                }
                return [rows[code] for code in self.codes]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeConn:
            def cursor(self):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return FakeConn()

    monkeypatch.setattr(plan_suggestions, "_connect", fake_connect)

    result = suggest_courses(
        "Whatever", "text", exclude=frozenset({"CS 18000"}), limit=2,
    )

    # CS 18000 excluded (already taken), and the limit stops at 2 even though 3 remained.
    assert [c.code for c in result] == ["CS 25100", "MA 16100"]
    assert result[0] == SuggestedCourse(code="CS 25100", title="Data Structures", credits=3)


def test_nearest_course_codes_filters_to_courses_before_ranking(monkeypatch):
    """The academic_rules index also holds one chunk per requirement BLOCK, and requirement
    text is near-duplicated across every program that states it — enough of them can outrank
    every course chunk in a plain top-k (this is exactly what happened: 9 of WGSS's 10
    unresolved requirements got zero suggestions before this was a query-time filter). The
    filtering must happen IN the store.search call, not after — assert the type is requested,
    not merely that the result looks filtered."""
    plan_suggestions._nearest_course_codes.cache_clear()
    monkeypatch.setattr(plan_suggestions.embeddings, "embed_query", lambda text: [0.1, 0.2])
    captured = {}

    def fake_search(embedding, top_k, *, metadata_type=None):
        captured["metadata_type"] = metadata_type
        return [
            {"metadata": {"type": "course", "code": "CS 18000"}},
            {"metadata": {"type": "course", "code": "CS 18000"}},  # duplicate, kept once
            {"metadata": {"type": "course", "code": "MA 16100"}},
        ]

    monkeypatch.setattr(plan_suggestions.store, "search", fake_search)

    codes = plan_suggestions._nearest_course_codes("some query")
    assert captured["metadata_type"] == "course"
    assert codes == ("CS 18000", "MA 16100")
    plan_suggestions._nearest_course_codes.cache_clear()


def test_nearest_course_codes_never_raises_on_a_broken_dependency(monkeypatch):
    """A suggestion list is a nicety, not core functionality — see the module docstring."""
    plan_suggestions._nearest_course_codes.cache_clear()

    def boom(text):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(plan_suggestions.embeddings, "embed_query", boom)
    assert plan_suggestions._nearest_course_codes("some query") == ()
    plan_suggestions._nearest_course_codes.cache_clear()


# --- real model + DB round-trip ---------------------------------------------------------------


def _model_available() -> bool:
    try:
        embeddings._get_model()
        return True
    except embeddings.EmbeddingError:
        return False


def _db_available() -> bool:
    try:
        with psycopg.connect(settings.academic_database_url, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM academic_rules LIMIT 1")
        return True
    except (psycopg.Error, OSError):
        return False


@pytest.mark.skipif(not _model_available(), reason="fastembed model could not be loaded")
@pytest.mark.skipif(not _db_available(), reason="academic_rules table not reachable")
def test_suggest_courses_returns_real_courses_for_a_real_requirement():
    plan_suggestions._nearest_course_codes.cache_clear()
    result = suggest_courses(
        "World Language Courses",
        "Completion of coursework in one world language.",
    )
    assert len(result) <= plan_suggestions.SUGGESTION_LIMIT
    for course in result:
        assert course.code and course.title
        assert course.credits >= 0
    plan_suggestions._nearest_course_codes.cache_clear()
