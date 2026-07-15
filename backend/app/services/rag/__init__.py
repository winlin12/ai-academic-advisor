"""Semantic (pgvector) retrieval for academic rules.

Three decoupled layers, one per build step:

    store.py       -- Step 1 (schema) + the DB half of Step 2 (upsert) and Step 3 (search +
                      exact metadata match). Knows Postgres/pgvector, nothing else.
    embeddings.py  -- the embedding half: text -> vector, in-process via fastembed/ONNX
                      (the Anthropic API has no embeddings endpoint). Knows the model,
                      not Postgres.
    pipeline.py    -- Step 2 (ingest) + Step 3 (answer). The only layer that composes the
                      two, plus the chat model (services/anthropic_client.py).
"""
