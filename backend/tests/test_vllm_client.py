"""Tests for the llama.cpp client.

All HTTP is mocked (httpx.MockTransport, swapped in for httpx.AsyncClient via monkeypatch) —
no real llama-server needed, no network. This exercises the wiring (payload shape, error
mapping, the propose() retry-on-invalid-JSON path, and the health() model-match check)
independent of any particular model's actual output quality, which is model_eval/'s job.
Async calls run via asyncio.run(), matching this repo's convention (no pytest-asyncio dep).
"""

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from app.services.vllm_client import (
    VllmClient,
    VllmConnectionError,
    ModelResponseError,
    is_local_model_endpoint,
    strip_reasoning,
)

# A SERVED MODEL NAME, not a filename. Under llama.cpp the model simply WAS the gguf the
# process was launched with, and /v1/models reported its path; vLLM takes
# `--served-model-name` and reports that string verbatim, so the app and the server agree on
# a short stable name and health() compares the two exactly.
_MODEL = "gemma4-26b"


class _Proposal(BaseModel):
    rationale: str
    max_credits_per_semester: int | None = None


def _mock_client(monkeypatch, handler) -> VllmClient:
    """A VllmClient whose internal httpx.AsyncClient is transparently backed by a
    MockTransport calling ``handler(request) -> httpx.Response``. Patches the shared
    ``httpx`` module object the client module imported, so no client code changes."""
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return VllmClient(model=_MODEL, base_url="http://localhost:8080")


# --- pure helpers ---------------------------------------------------------------------------


def test_strip_reasoning_removes_think_block():
    text = "<think>the user wants prereqs</think>CS 18000 is required."
    assert strip_reasoning(text) == "CS 18000 is required."


def test_strip_reasoning_passthrough_without_think_block():
    assert strip_reasoning("plain answer") == "plain answer"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:8080", True),
        ("http://127.0.0.1:8080", True),
        ("http://host.docker.internal:8080", True),
        ("http://10.0.0.5:8080", True),
        ("http://100.64.1.2:8080", True),  # Tailscale CGNAT
        ("http://api.example.com:8080", False),
        ("http://8.8.8.8:8080", False),
        ("not-a-url", False),
    ],
)
def test_is_local_model_endpoint(url, expected):
    assert is_local_model_endpoint(url) is expected


def test_local_only_guard_rejects_public_base_url(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "vllm_local_only", True)
    with pytest.raises(Exception, match="local or self-hosted"):
        VllmClient(base_url="http://api.example.com:8080")


# --- generate() ------------------------------------------------------------------------------


def test_generate_returns_stripped_text(monkeypatch):
    def handler(req):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  CS 18000 is required.  "}}]},
        )

    client = _mock_client(monkeypatch, handler)
    answer = asyncio.run(client.generate("system", "user"))
    assert answer == "CS 18000 is required."


def test_generate_raises_on_empty_response(monkeypatch):
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    client = _mock_client(monkeypatch, handler)
    with pytest.raises(ModelResponseError):
        asyncio.run(client.generate("system", "user"))


def test_generate_strips_reasoning_block(monkeypatch):
    def handler(req):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "<think>internal</think>final answer"}}]},
        )

    client = _mock_client(monkeypatch, handler)
    assert asyncio.run(client.generate("system", "user")) == "final answer"


def test_generate_sends_expected_payload(monkeypatch):
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = _mock_client(monkeypatch, handler)
    asyncio.run(client.generate("sys prompt", "user prompt"))

    body = captured["body"]
    assert body["model"] == _MODEL
    assert body["messages"] == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert "response_format" not in body


# --- propose() ---------------------------------------------------------------------------------


def test_propose_parses_valid_json_first_try(monkeypatch):
    def handler(req):
        body = json.loads(req.content)
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["schema"] == _Proposal.model_json_schema()
        content = json.dumps({"rationale": "ok", "max_credits_per_semester": 12})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = _mock_client(monkeypatch, handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "ok"
    assert result.max_credits_per_semester == 12


def test_propose_retries_once_on_invalid_json_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "not valid json at all"}}]}
            )
        content = json.dumps({"rationale": "fixed"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = _mock_client(monkeypatch, handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "fixed"
    assert calls["n"] == 2


def test_propose_gives_up_after_one_retry(monkeypatch):
    def handler(req):
        return httpx.Response(200, json={"choices": [{"message": {"content": "still not json"}}]})

    client = _mock_client(monkeypatch, handler)
    with pytest.raises(ModelResponseError):
        asyncio.run(client.propose("system", "user", _Proposal))


def test_propose_retries_on_schema_validation_failure(monkeypatch):
    """Valid JSON syntax but missing a required field — the JSON-schema grammar doesn't catch
    this, only Pydantic validation does, which is exactly the gap propose() exists to cover."""
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            content = json.dumps({"max_credits_per_semester": 5})
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        content = json.dumps({"rationale": "recovered"})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    client = _mock_client(monkeypatch, handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "recovered"
    assert calls["n"] == 2


def test_propose_strips_markdown_fence(monkeypatch):
    def handler(req):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '```json\n{"rationale": "fenced"}\n```'}}]},
        )

    client = _mock_client(monkeypatch, handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "fenced"


# --- error mapping -----------------------------------------------------------------------------


def test_generate_raises_model_response_error_on_bad_status(monkeypatch):
    def handler(req):
        return httpx.Response(500, json={"error": "internal error"})

    client = _mock_client(monkeypatch, handler)
    with pytest.raises(ModelResponseError):
        asyncio.run(client.generate("system", "user"))


def test_generate_raises_connection_error_when_unreachable():
    client = VllmClient(model=_MODEL, base_url="http://localhost:19999")
    with pytest.raises(VllmConnectionError):
        asyncio.run(client.generate("system", "user"))


# --- health() ------------------------------------------------------------------------------


def test_health_ok_when_loaded_model_matches(monkeypatch):
    def handler(req):
        if req.url.path == "/health":
            # vLLM answers /health with 200 and an EMPTY body; only the status code is
            # meaningful. The client must not try to parse it.
            return httpx.Response(200)
        assert req.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": _MODEL}]})

    client = _mock_client(monkeypatch, handler)
    ok, detail = asyncio.run(client.health())
    assert ok is True
    assert _MODEL in detail


def test_health_false_when_loaded_model_does_not_match(monkeypatch):
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"data": [{"id": "qwen3.8-27b"}]})

    client = _mock_client(monkeypatch, handler)
    ok, detail = asyncio.run(client.health())
    assert ok is False
    assert "doesn't match" in detail


def test_health_true_with_caveat_when_models_endpoint_fails(monkeypatch):
    def handler(req):
        if req.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, json={"error": "boom"})

    client = _mock_client(monkeypatch, handler)
    ok, detail = asyncio.run(client.health())
    assert ok is True
    assert "could not confirm" in detail


def test_health_false_when_unreachable():
    client = VllmClient(model=_MODEL, base_url="http://localhost:19999")
    ok, detail = asyncio.run(client.health())
    assert ok is False
    assert "Cannot reach" in detail
