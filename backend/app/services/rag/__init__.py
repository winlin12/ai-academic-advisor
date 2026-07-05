"""Semantic (pgvector) retrieval for academic rules.

Three decoupled layers, one per build step:

    store.py     -- Step 1 (schema) + the DB half of Step 2 (upsert) and Step 3 (search).
                    Knows Postgres/pgvector. Knows nothing about Ollama.
    ollama_client.embed  -- the embedding half: text -> vector. Knows Ollama, not Postgres.
    pipeline.py  -- Step 2 (ingest) + Step 3 (answer). The only layer that composes the two.
"""
