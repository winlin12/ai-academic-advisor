"""In-process text embeddings for RAG (fastembed / ONNX on CPU).

The Anthropic API has no embeddings endpoint, so when the chat side moved to Claude the
embedding side moved *into the process*: ``BAAI/bge-small-en-v1.5`` (384-d, normalized)
served by fastembed. No model daemon to run, no per-token cost, ~100MB of weights that
onnxruntime executes fast enough on a lightweight VPS CPU for query-time embedding.

This module is the embedding half of the RAG layering (see ``rag/__init__``): it knows the
model, not Postgres. Everything is synchronous — callers on the event loop should wrap
calls in ``asyncio.to_thread``.

BGE v1.5 is an asymmetric retrieval model: QUERIES are embedded with the model card's
instruction prefix so a question lands near the passages that answer it; PASSAGES are
embedded bare. (fastembed 0.8's ``query_embed`` does not add the prefix itself — verified
empirically — so we prepend it here.) Stored vectors and query vectors must come from the
same model: changing ``rag_embed_model`` means re-running ingest_catalog.
"""

from __future__ import annotations

import threading

from app.core.config import settings

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_model_lock = threading.Lock()


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot load or returns a wrong-sized vector."""


def _get_model():
    """Lazy, process-wide singleton. First call downloads the model to the HF cache
    (~100MB) and loads the ONNX session; both are too slow to repeat per request."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from fastembed import TextEmbedding

                    _model = TextEmbedding(settings.rag_embed_model)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Could not load embedding model {settings.rag_embed_model!r}: {exc}"
                    ) from exc
    return _model


def _validated(vector: list[float]) -> list[float]:
    if len(vector) != settings.rag_embed_dimensions:
        raise EmbeddingError(
            f"Embedding model {settings.rag_embed_model!r} returned {len(vector)} dims, but "
            f"rag_embed_dimensions is {settings.rag_embed_dimensions}. Fix the config and the "
            f"VECTOR(n) column so all three agree, then re-run ingest_catalog."
        )
    return vector


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed catalog chunks for storage (no prefix). Batched — pass many at once."""
    try:
        vectors = list(_get_model().passage_embed(texts))
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Passage embedding failed: {exc}") from exc
    return [_validated(v.tolist()) for v in vectors]


def embed_passage(text: str) -> list[float]:
    return embed_passages([text])[0]


def embed_query(text: str) -> list[float]:
    """Embed a student question with the BGE query instruction (asymmetric retrieval)."""
    try:
        vector = next(iter(_get_model().embed([f"{_QUERY_INSTRUCTION}{text}"])))
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Query embedding failed: {exc}") from exc
    return _validated(vector.tolist())
