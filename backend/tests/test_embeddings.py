"""Tests for the in-process embedding provider (fastembed/ONNX).

Two tiers, mirroring test_rag_store.py's pattern for the pgvector round-trip:
  * A pure-function test for the dimension guard — no model load, always runs.
  * A real embed() round-trip against the actual ONNX model — SKIPS (never fails) if the
    model can't be loaded (e.g. first run with no network to fetch it from the HF cache),
    so CI/offline runs stay green while a normal dev machine still exercises the real path.
"""

import math

import pytest

from app.services.rag import embeddings


# --- pure function: no model needed ----------------------------------------------------------


def test_validated_rejects_wrong_dimension_vector(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "rag_embed_dimensions", 384)
    with pytest.raises(embeddings.EmbeddingError, match="384"):
        embeddings._validated([0.1, 0.2, 0.3])  # 3-d, not 384-d


def test_validated_passes_through_correct_dimension_vector(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "rag_embed_dimensions", 3)
    vector = [0.1, 0.2, 0.3]
    assert embeddings._validated(vector) is vector


# --- real model round-trip -------------------------------------------------------------------


def _model_available() -> bool:
    try:
        embeddings._get_model()
        return True
    except embeddings.EmbeddingError:
        return False


@pytest.mark.skipif(not _model_available(), reason="fastembed model could not be loaded")
def test_embed_query_and_passage_are_normalized_and_asymmetric():
    passage = embeddings.embed_passage("CS 25100 covers trees, graphs, and sorting.")
    query = embeddings.embed_query("what does CS 25100 cover?")

    assert len(passage) == embeddings.settings.rag_embed_dimensions
    assert len(query) == embeddings.settings.rag_embed_dimensions

    # BGE embeddings are L2-normalized (cosine == dot product).
    assert math.isclose(sum(x * x for x in passage) ** 0.5, 1.0, abs_tol=1e-3)
    assert math.isclose(sum(x * x for x in query) ** 0.5, 1.0, abs_tol=1e-3)

    # A related passage/query pair should be reasonably close in cosine similarity.
    similarity = sum(p * q for p, q in zip(passage, query))
    assert similarity > 0.5


@pytest.mark.skipif(not _model_available(), reason="fastembed model could not be loaded")
def test_embed_passages_batches_multiple_texts():
    vectors = embeddings.embed_passages(["first chunk", "second chunk"])
    assert len(vectors) == 2
    assert all(len(v) == embeddings.settings.rag_embed_dimensions for v in vectors)
