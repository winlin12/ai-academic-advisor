"""Tests for the RAG composition layer's pure logic: no DB, no model, no network.

Covers the two-tier retrieval helpers added with the Anthropic API pivot — deterministic
course-code extraction (tier 1) and the context token budgeter that packs both tiers under
``advisor_context_token_budget`` before anything reaches the prompt.
"""

from app.services.rag.pipeline import _approx_tokens, extract_course_codes, merge_and_budget


# --- extract_course_codes -------------------------------------------------------------------


def test_extract_course_codes_handles_spacing_and_case_variants():
    assert extract_course_codes("What are the prereqs for CS 25100?") == ["CS25100"]
    assert extract_course_codes("is cs25100 hard") == ["CS25100"]
    assert extract_course_codes("MA-16100 vs MA 16200") == ["MA16100", "MA16200"]


def test_extract_course_codes_dedupes_preserving_first_occurrence_order():
    assert extract_course_codes("CS 25100 and CS25100 again, then CS 18200") == [
        "CS25100",
        "CS18200",
    ]


def test_extract_course_codes_returns_empty_for_no_match():
    assert extract_course_codes("what's left for my degree?") == []


# --- merge_and_budget ------------------------------------------------------------------------


def _match(id_: int, content: str, similarity: float | None = None) -> dict:
    return {"id": id_, "content": content, "metadata": {}, "similarity": similarity}


def test_merge_and_budget_ranks_exact_matches_before_semantic():
    exact = [_match(1, "CS 25100 exact chunk", 1.0)]
    semantic = [_match(2, "some semantically similar chunk", 0.6)]
    merged = merge_and_budget(exact, semantic, budget_tokens=1000)
    assert [m["id"] for m in merged] == [1, 2]


def test_merge_and_budget_dedupes_by_id_keeping_exact_tier():
    # Same row surfaced by both tiers (mentioned by code AND semantically close).
    exact = [_match(1, "CS 25100 exact chunk", 1.0)]
    semantic = [_match(1, "CS 25100 exact chunk", 0.83), _match(2, "other chunk", 0.5)]
    merged = merge_and_budget(exact, semantic, budget_tokens=1000)
    assert [m["id"] for m in merged] == [1, 2]


def test_merge_and_budget_stops_packing_at_the_token_budget():
    big = "x" * 4000  # ~1000 approx-tokens
    small = "y" * 40  # ~10 approx-tokens
    exact = [_match(1, big), _match(2, small)]
    # Budget only fits the first chunk; the second is skipped, not truncated.
    merged = merge_and_budget(exact, [], budget_tokens=_approx_tokens(big) + 1)
    assert [m["id"] for m in merged] == [1]


def test_merge_and_budget_skips_a_chunk_that_alone_exceeds_the_budget_but_keeps_smaller_ones():
    too_big = "z" * 40000
    fits = "w" * 40
    merged = merge_and_budget([_match(1, too_big), _match(2, fits)], [], budget_tokens=100)
    assert [m["id"] for m in merged] == [2]
