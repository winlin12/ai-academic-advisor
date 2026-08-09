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
    # GEMMA 4 26B A4B, 2026-08-03, replacing Qwen2.5-Coder-7B. The 7B was chosen for an 8GB
    # budget back when the model only summarised retrieved chunks; the app now asks it to
    # EMIT THE SCHEDULE (Mode B — services/ai_planner.py), which is the task model_eval exists
    # to measure and the only one on which models separate sharply. This box has two cards
    # (RTX 3060 12GB + RTX 2060 Super 8GB = 20GB), which is what makes a 26B A4B MoE at Q4_K_M
    # affordable: ~16GB of weights across both cards, ~4B active parameters per token, so it
    # generates at small-model speed.
    #
    # Must be the exact gguf filename llama-server was launched with (`llama-server -m <this>`)
    # — unlike Ollama there is no tag/pull; the model is whatever file the process loaded, and
    # health() flags a mismatch rather than silently answering from the wrong one. See the
    # README's llama-server invocation for the matching --tensor-split and --ctx-size.
    llamacpp_model: str = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
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

    # The context llama-server was LAUNCHED with (`--ctx-size`). It cannot be changed per
    # request, so nothing here can recover from getting it wrong — the two budgets below are
    # sized against it and an overflowing prompt is a failed request, not a slow one.
    # Mirrors model_eval/config.yaml's run.num_ctx.
    llamacpp_context_tokens: int = 16384
    # Mode B's own output ceiling, separate from `llamacpp_max_tokens` because the two tasks
    # have nothing in common: a plan of study is ~470-550 tokens of `semesters` at the median
    # and never exceeded ~1200 anywhere in model_eval's corpus, while free-text explain answers
    # run longer. Mirrors model_eval/config.yaml's run.max_plan_tokens.
    llamacpp_plan_max_tokens: int = 2048
    # Ceiling on the catalog export handed to Mode B (services/plan_context.py), in the token
    # estimate that module computes. The 16384-token window has to hold ALL of: this export,
    # the ~1600-token rules prose, the ~200-token student block, the chat template's own
    # wrapping, and llamacpp_plan_max_tokens of generated plan. 16384 - 2048 - 1900 ≈ 12400, so
    # 11500 leaves a little over 900 tokens of slack for a long completed-courses list.
    #
    # A prompt that overflows does not degrade gracefully — the plan is truncated mid-JSON and
    # fails to parse, which surfaces as "the AI planner was unavailable". Raise BOTH this and
    # llamacpp_context_tokens together, never one alone.
    plan_context_token_budget: int = 11500
    # A 26B MoE reading an 11k-token export spends most of a minute on prompt processing alone
    # the first time; the eval measured 32-65 s medians per plan and allowed 600 s. The old
    # hard-coded 120 s in llamacpp_client was written for a 7B answering from three retrieved
    # chunks and would time out a perfectly healthy plan request.
    llamacpp_timeout_s: float = 600.0

    # PROCESS OWNERSHIP — added 2026-08-04 alongside model_manager.py. Before this, llama-server
    # was always started by hand outside the app; `llamacpp_base_url` only ever pointed at
    # whatever was already running. Now the backend launches it itself at startup and can stop
    # and relaunch it with a different gguf when the user picks a different model from the UI
    # (see services/model_manager.py, api/routers/models.py) — same port every time
    # (`llamacpp_base_url` never changes), so nothing else in the app needs to know a switch
    # happened.
    llamacpp_server_exe: str = "/home/wylin/ai-academic-advisor/llama.cpp/build/bin/llama-server"
    llamacpp_models_root: str = "/home/wylin/ai-academic-advisor/models"
    # How long a switch waits for the new process to answer /health before giving up and
    # reporting the switch failed. Generous: a 26-35B gguf's weights alone can take a while to
    # read off disk before CUDA init even starts, independent of GPU speed.
    llamacpp_startup_timeout_s: float = 180.0

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
