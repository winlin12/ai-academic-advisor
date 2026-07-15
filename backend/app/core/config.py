from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env lives two levels up from this file (app/core/ -> app -> backend).
# Anchor to an absolute path so the config loads the same file whether the app is
# launched from the repo root or from backend/. Real process env vars still win.
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # Anthropic API (cloud inference — replaced the local Ollama stack on 2026-07-10).
    # The API key is deliberately NOT a field here: the SDK reads ANTHROPIC_API_KEY
    # straight from the process environment, so the secret can never leak through a
    # settings.model_dump() in a log line or debug endpoint.
    # claude-sonnet-4-5 (not haiku) so prompt caching engages: its minimum cacheable
    # prefix is 1,024 tokens vs. Haiku 4.5's 4,096, and the advisor system prompt +
    # inlined DB schema (~2.5K tokens) clears the former but not the latter.
    anthropic_model: str = "claude-sonnet-4-5"
    # Per-response output cap. Advisor answers are deliberately short; raise only with
    # a reason (this is also the per-request cost ceiling on the $15/MTok output side).
    anthropic_max_tokens: int = 2048

    # catalog_ingestion database (Docker Postgres publishes 5432 -> host 5433).
    academic_database_url: str = "postgresql://catalog:catalog@localhost:5433/catalog_ingestion"

    # RAG / pgvector semantic retrieval. Embeddings are produced IN-PROCESS by fastembed
    # (ONNX on CPU) — the Anthropic API has no embeddings endpoint, and a 384-d model is
    # small enough for a lightweight VPS. ``rag_embed_dimensions`` MUST match both the
    # model and the VECTOR(n) column, or inserts/searches raise a dimension-mismatch
    # error; changing the model means re-running ingest_catalog (stored vectors and
    # query vectors must come from the same model).
    rag_embed_model: str = "BAAI/bge-small-en-v1.5"
    rag_embed_dimensions: int = 384
    rag_top_k: int = 3  # how many nearest chunks to consider for the prompt
    # Cosine similarity floor (0..1). Chunks below this are dropped so we never inject a
    # weakly-related rule just to fill top_k. 0.0 disables the filter.
    rag_min_similarity: float = 0.45
    # Hard ceiling (approximate tokens, ~4 chars/token) on retrieved context injected
    # into an advisor prompt. Input tokens are the Anthropic bill; this is the cap.
    advisor_context_token_budget: int = 3000

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
