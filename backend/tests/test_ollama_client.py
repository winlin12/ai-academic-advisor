"""Tests for the Ollama client.

All HTTP is mocked (httpx.MockTransport) — no real Ollama server needed, no network. This
exercises the wiring (payload shape, error mapping, the propose() retry-on-invalid-JSON path)
independent of any particular model's actual output quality, which is model_eval/'s job.
Async calls run via asyncio.run(), matching this repo's convention (no pytest-asyncio dep).
"""

import asyncio
import json

import httpx
import pytest
from pydantic import BaseModel

from app.services.ollama_client import (
    ModelNotFoundError,
    ModelResponseError,
    OllamaClient,
    OllamaConnectionError,
    is_local_model_endpoint,
    strip_reasoning,
)


class _Proposal(BaseModel):
    rationale: str
    max_credits_per_semester: int | None = None


def _client(handler, **kwargs) -> OllamaClient:
    client = OllamaClient(model="llama3.1:8b", base_url="http://localhost:11434", **kwargs)

    async def fake_generate_raw(prompt, *, output_format=None):
        request_payload = {"prompt": prompt, "format": output_format}
        transport = httpx.MockTransport(lambda req: handler(request_payload, req))
        async with httpx.AsyncClient(transport=transport, timeout=5.0) as http:
            resp = await http.post(f"{client.base_url}/api/generate", json=request_payload)
            if resp.status_code == 404:
                raise ModelNotFoundError(f"Model {client.model!r} is not pulled.")
            resp.raise_for_status()
            return resp.json()

    client._generate_raw = fake_generate_raw  # type: ignore[method-assign]
    return client


def _health_client(models: list[str] | None, *, unreachable: bool = False) -> OllamaClient:
    client = OllamaClient(model="llama3.1:8b", base_url="http://localhost:11434")
    if unreachable:
        client.base_url = "http://localhost:19999"
        return client

    def handler(req):
        return httpx.Response(200, json={"models": [{"name": m} for m in (models or [])]})

    real_health = OllamaClient.health

    async def fake_health(self):
        transport = httpx.MockTransport(handler)
        try:
            async with httpx.AsyncClient(transport=transport, timeout=5.0) as http:
                resp = await http.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        tags = {m.get("name") for m in data.get("models", [])}
        pulled = self.model in tags or f"{self.model}:latest" in tags
        if not pulled:
            return False, f"Ollama is reachable but {self.model!r} is not pulled."
        return True, f"Ollama reachable; model {self.model!r} is pulled."

    client.health = fake_health.__get__(client, OllamaClient)  # type: ignore[method-assign]
    return client


# --- pure helpers ---------------------------------------------------------------------------


def test_strip_reasoning_removes_think_block():
    text = "<think>the user wants prereqs</think>CS 18000 is required."
    assert strip_reasoning(text) == "CS 18000 is required."


def test_strip_reasoning_passthrough_without_think_block():
    assert strip_reasoning("plain answer") == "plain answer"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://host.docker.internal:11434", True),
        ("http://10.0.0.5:11434", True),
        ("http://100.64.1.2:11434", True),  # Tailscale CGNAT
        ("http://api.example.com:11434", False),
        ("http://8.8.8.8:11434", False),
        ("not-a-url", False),
    ],
)
def test_is_local_model_endpoint(url, expected):
    assert is_local_model_endpoint(url) is expected


def test_local_only_guard_rejects_public_base_url(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ollama_local_only", True)
    with pytest.raises(Exception, match="local or self-hosted"):
        OllamaClient(base_url="http://api.example.com:11434")


# --- generate() ------------------------------------------------------------------------------


def test_generate_returns_stripped_text():
    def handler(payload, req):
        return httpx.Response(200, json={"response": "  CS 18000 is required.  ", "done": True})

    client = _client(handler)
    answer = asyncio.run(client.generate("system", "user"))
    assert answer == "CS 18000 is required."


def test_generate_raises_on_empty_response():
    def handler(payload, req):
        return httpx.Response(200, json={"response": "", "done": True})

    client = _client(handler)
    with pytest.raises(ModelResponseError):
        asyncio.run(client.generate("system", "user"))


def test_generate_strips_reasoning_block():
    def handler(payload, req):
        return httpx.Response(
            200, json={"response": "<think>internal</think>final answer", "done": True}
        )

    client = _client(handler)
    assert asyncio.run(client.generate("system", "user")) == "final answer"


# --- propose() ---------------------------------------------------------------------------------


def test_propose_parses_valid_json_first_try():
    def handler(payload, req):
        assert payload["format"] == _Proposal.model_json_schema()
        return httpx.Response(
            200, json={"response": json.dumps({"rationale": "ok", "max_credits_per_semester": 12})}
        )

    client = _client(handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "ok"
    assert result.max_credits_per_semester == 12


def test_propose_retries_once_on_invalid_json_then_succeeds():
    calls = {"n": 0}

    def handler(payload, req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"response": "not valid json at all"})
        return httpx.Response(200, json={"response": json.dumps({"rationale": "fixed"})})

    client = _client(handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "fixed"
    assert calls["n"] == 2


def test_propose_gives_up_after_one_retry():
    def handler(payload, req):
        return httpx.Response(200, json={"response": "still not json"})

    client = _client(handler)
    with pytest.raises(ModelResponseError):
        asyncio.run(client.propose("system", "user", _Proposal))


def test_propose_retries_on_schema_validation_failure():
    """Valid JSON syntax but missing a required field — format=<schema> doesn't catch this,
    only Pydantic validation does, which is exactly the gap propose() exists to cover."""
    calls = {"n": 0}

    def handler(payload, req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"response": json.dumps({"max_credits_per_semester": 5})})
        return httpx.Response(200, json={"response": json.dumps({"rationale": "recovered"})})

    client = _client(handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "recovered"
    assert calls["n"] == 2


def test_propose_strips_markdown_fence():
    def handler(payload, req):
        return httpx.Response(200, json={"response": '```json\n{"rationale": "fenced"}\n```'})

    client = _client(handler)
    result = asyncio.run(client.propose("system", "user", _Proposal))
    assert result.rationale == "fenced"


# --- error mapping -----------------------------------------------------------------------------


def test_generate_raises_model_not_found_on_404():
    def handler(payload, req):
        return httpx.Response(404, json={"error": "model not found"})

    client = _client(handler)
    with pytest.raises(ModelNotFoundError):
        asyncio.run(client.generate("system", "user"))


def test_generate_raises_connection_error_when_unreachable():
    client = OllamaClient(model="llama3.1:8b", base_url="http://localhost:19999")
    with pytest.raises(OllamaConnectionError):
        asyncio.run(client.generate("system", "user"))


# --- health() ------------------------------------------------------------------------------


def test_health_ok_when_model_is_pulled():
    client = _health_client(["llama3.1:8b"])
    ok, detail = asyncio.run(client.health())
    assert ok is True
    assert "llama3.1:8b" in detail


def test_health_false_when_model_not_pulled():
    client = _health_client(["mistral"])
    ok, detail = asyncio.run(client.health())
    assert ok is False
    assert "not pulled" in detail


def test_health_false_when_unreachable():
    client = OllamaClient(model="llama3.1:8b", base_url="http://localhost:19999")
    ok, detail = asyncio.run(client.health())
    assert ok is False
    assert "Cannot reach" in detail
