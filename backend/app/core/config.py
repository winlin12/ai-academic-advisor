from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env lives two levels up from this file (app/core/ -> app -> backend).
# Anchor to an absolute path so the config loads the same file whether the app is
# launched from the repo root or from backend/. Real process env vars still win.
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    # Liquid LFM2 (~8B), served locally by Ollama. The published tag on this host is
    # ``lfm2.5:latest``; ``lfm2.5:8b`` does not resolve. Override with OLLAMA_MODEL.
    ollama_model: str = "lfm2.5:latest"
    ollama_local_only: bool = True
    # catalog_ingestion database (rootless-podman Postgres publishes 5432 -> host 5433).
    academic_database_url: str = "postgresql://catalog:catalog@localhost:5433/catalog_ingestion"

    # RAG / pgvector semantic retrieval. The embedding model is *separate* from the chat
    # model (``ollama_model`` above): nomic-embed-text turns text into a 768-d vector, the
    # chat model writes the final answer. ``rag_embed_dimensions`` MUST match both the model
    # and the VECTOR(n) column, or inserts/searches raise a dimension-mismatch error.
    rag_embed_model: str = "nomic-embed-text"
    rag_embed_dimensions: int = 768
    rag_top_k: int = 3  # how many nearest chunks to inject into the prompt
    # Cosine similarity floor (0..1). Chunks below this are dropped so we never inject a
    # weakly-related rule just to fill top_k. 0.0 disables the filter.
    rag_min_similarity: float = 0.0

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
