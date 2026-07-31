from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env lives two levels up from this file (app/core/ -> app -> backend).
# Anchor to an absolute path so the config loads the same file whether the app is
# launched from the repo root or from backend/. Real process env vars still win.
_BACKEND_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # llama.cpp (local inference via `llama-server` — moved off Ollama on 2026-07-21; a
    # public-facing site can't carry unbounded per-question cloud spend). Production
    # topology: the backend process and llama-server run on the SAME box (the RTX 2060
    # Super server, 24/7). 127.0.0.1, not "localhost": under uvicorn's uvloop, "localhost"
    # can resolve AAAA (::1) first and fail outright against an IPv4-only bind, even
    # though the same lookup falls back to IPv4 fine under the default asyncio loop —
    # hit this in local testing with Ollama. A literal IP sidesteps dual-stack resolution
    # entirely. Port 8080 is llama-server's default.
    llamacpp_base_url: str = "http://127.0.0.1:8080"
    # model_eval/'s verdict for the 2060 Super's 8GB: best quality among 8GB-class models
    # tried, code-tuned pretraining measurably helped this app's SQL-shaped tasks. Must be
    # the exact gguf filename llama-server was launched with (`llama-server -m <this>`) —
    # unlike Ollama there is no tag/pull; the model is whatever file the process loaded.
    llamacpp_model: str = "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    # Refuses to start against a non-local/non-private base URL unless explicitly
    # lifted — a misconfiguration guard (accidentally pointing at a random public
    # endpoint), not a real security boundary.
    llamacpp_local_only: bool = True
    llamacpp_temperature: float = 0.15
    # Generous ceiling so truncation is a model fault, not a config one; mirrors
    # model_eval/config.yaml's max_output_tokens. Unlike Ollama's num_ctx, llama-server's
    # context size is fixed at process launch (`--ctx-size`; omitted entirely on the 2060
    # Super box, where the default `0` means "the context the model was trained with" —
    # 32768 for Qwen2.5-Coder-7B, which still fits under 8GB VRAM at Q4_K_M) and can't be
    # overridden per-request — keep this budget comfortably under whatever that launch
    # context is so a long system+user prompt still leaves room to generate. 8192 is a
    # quarter of 32768, so it is.
    #
    # RAISED FROM 1024, 2026-07-29, because 1024 was silently breaking revise-plan. The
    # eval measured what the verbose models actually write into PlanEditProposal.rationale:
    # up to 10,659 characters (~2.7k tokens) for qwen3.5-9b, ~7.2k chars for
    # qwen3.6-35b-a3b. At 1024 they never reach the closing keys, the response fails to
    # validate, and revise_plan logs "unusable proposal" and degrades to a no-op — the
    # student's feedback silently does nothing. The eval did not catch it because
    # model_eval ran Mode A at 16384 while production ran at 1024, so the harness's claim
    # to measure the shipped path was false on exactly this axis.
    #
    # Settled at 4096 once PlanEditProposal.rationale grew its max_length=400: the reason
    # this needed to be large was the rambling, and the grammar now forbids that. A capped
    # proposal is a few hundred tokens; the largest thing this budget still has to cover is
    # explain_plan, which is free text with no schema and has never exceeded ~1400 tokens.
    # Mirrors model_eval/config.yaml's run.max_output_tokens — change one, change the other.
    llamacpp_max_tokens: int = 4096

    # Anthropic API — kept as an optional swappable backend (services/anthropic_client.py),
    # not the active path. Requires ANTHROPIC_API_KEY (read by the SDK from the process
    # environment, deliberately not a Settings field) if ever swapped back in.
    anthropic_model: str = "claude-sonnet-4-5"
    anthropic_max_tokens: int = 2048

    # Summer is a real term with real offerings, but summer enrolment has cost/aid/residency
    # consequences the planner cannot reason about, so it is not scheduled into by default.
    # Flip to true when summer planning becomes a deliberate, student-facing choice.
    planner_include_summer: bool = False

    # catalog_ingestion database (Docker Postgres publishes 5432 -> host 5433).
    academic_database_url: str = "postgresql://catalog:catalog@localhost:5433/catalog_ingestion"

    # PurdueIO database. This is where the data actually is: the `advisor` schema holds the
    # crawled degree programs/requirements (1,435 programs) and, as of 002_course_offerings,
    # observed term offerings; the PurdueIO tables alongside it hold courses and the Classes
    # rows those offerings are derived from. The catalog_ingestion database above is the
    # older, currently-empty copy of the same idea — see TODO.md for the consolidation.
    # The container publishes no host port, so the default addresses it on the Docker
    # network; override for a different topology.
    purdueio_database_url: str = (
        "postgresql://purdueio:changeme_in_production@172.18.0.4:5432/purdueio"
    )

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
