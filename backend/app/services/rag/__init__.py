"""Semantic (pgvector) retrieval for academic rules.

Three decoupled layers, one per build step:

    store.py       -- Step 1 (schema) + the DB half of Step 2 (upsert) and Step 3 (search +
                      exact metadata match). Knows Postgres/pgvector, nothing else.
    embeddings.py  -- the embedding half: text -> vector, in-process via fastembed/ONNX
                      (kept independent of the chat model — retrieval works even if
                      OLLAMA_MODEL changes or Ollama is temporarily down). Knows the
                      embedding model, not Postgres.
    pipeline.py    -- Step 2 (ingest) + Step 3 (answer). The only layer that composes the
                      two, plus the chat model (services/ollama_client.py).
"""
