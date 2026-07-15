from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env lives two levels up from this file (app/core/ -> app -> backend).
# Anchor to an absolute path so the config loads the same file whether the app is
# launched from the repo root or from backend/. Real process env vars still win.
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # Ollama (local inference — re-pivoted from the Anthropic API on 2026-07-15; a
    # public-facing site can't carry unbounded per-question cloud spend). Production
    # topology: the backend process and Ollama run on the SAME box (the RTX 2060 Super
    # server, 24/7). 127.0.0.1, not "localhost": under uvicorn's uvloop, "localhost" can
    # resolve AAAA (::1) first and fail outright against an IPv4-only Ollama bind, even
    # though the same lookup falls back to IPv4 fine under the default asyncio loop —
    # hit this in local testing. A literal IP sidesteps dual-stack resolution entirely.
    ollama_base_url: str = "http://127.0.0.1:11434"
    # PLACEHOLDER pending model_eval/'s verdict (see repo root model_eval/README.md) —
    # this is not yet the chosen model, just something known-good to build the wiring
    # against. Swap once the harness names a winner for the 2060 Super's 8GB.
    ollama_model: str = "llama3.1:8b"
    # Refuses to start against a non-local/non-private base URL unless explicitly
    # lifted — a misconfiguration guard (accidentally pointing at a random public
    # endpoint), not a real security boundary.
    ollama_local_only: bool = True
    # Mirrors model_eval/config.yaml's run knobs so production uses values that were
    # actually measured, not defaults. Ollama defaults num_ctx to 2K-4K regardless of
    # what the model supports — leaving this unset is the single biggest way to silently
    # under-serve a model.
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.15
    # Keeps the model resident between requests on the same box; never restart the
    # Ollama process to get a "fresh" context — each request already starts fresh
    # server-side (no conversation carryover), this only controls VRAM residency.
    ollama_keep_alive: str = "15m"

    # Anthropic API — kept as an optional swappable backend (services/anthropic_client.py),
    # not the active path. Requires ANTHROPIC_API_KEY (read by the SDK from the process
    # environment, deliberately not a Settings field) if ever swapped back in.
    anthropic_model: str = "claude-sonnet-4-5"
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
