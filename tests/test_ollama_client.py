from __future__ import annotations

import json

import pytest

from data_engineering_copilot.infrastructure.ollama_client import OllamaClient, OllamaError


class _FakeResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def _client() -> OllamaClient:
    return OllamaClient(
        base_url="http://localhost:11434",
        model="qwen3:4b",
        timeout_seconds=5,
        num_ctx=4096,
        num_predict=768,
    )


def test_prompt_disables_qwen_thinking_mode() -> None:
    prompt = _client()._format_raw_chat_prompt("What is Delta Lake?")

    assert "/no_think" in prompt
    assert "Do not show reasoning" in prompt


def test_generate_strips_thinking_block(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            {
                "response": "<think>I should reason privately.</think>\nDelta Lake supports ACID table storage.",
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("data_engineering_copilot.infrastructure.ollama_client.urlopen", fake_urlopen)

    assert _client().generate("Answer from context") == "Delta Lake supports ACID table storage."


def test_generate_reports_reasoning_only_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(*_args: object, **_kwargs: object) -> _FakeResponse:
        return _FakeResponse(
            {
                "response": "<think>I ran out before the final answer.",
                "done_reason": "length",
            }
        )

    monkeypatch.setattr("data_engineering_copilot.infrastructure.ollama_client.urlopen", fake_urlopen)

    with pytest.raises(OllamaError, match="spent its output budget on reasoning"):
        _client().generate("Answer from context")
