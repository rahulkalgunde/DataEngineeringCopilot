"""Tests for `dec ask` user/session-id flags (Langfuse session tracking)."""

from __future__ import annotations

from data_engineering_copilot import cli
from data_engineering_copilot.domain.models import Answer


def test_ask_parser_exposes_user_and_session_flags() -> None:
    args = cli.build_parser().parse_args(["ask", "--user-id", "u1", "--session-id", "s1", "What is Spark?"])
    assert args.question == "What is Spark?"
    assert args.user_id == "u1"
    assert args.session_id == "s1"


def test_ask_parser_defaults_flags_to_none() -> None:
    args = cli.build_parser().parse_args(["ask", "What is Spark?"])
    assert args.user_id is None
    assert args.session_id is None


def test_ask_forwards_user_and_session_to_service(monkeypatch, capsys) -> None:
    seen: dict[str, str | None] = {}

    class _FakeService:
        async def answer(self, question: str, user_id=None, session_id=None, **kwargs) -> Answer:
            seen["question"] = question
            seen["user_id"] = user_id
            seen["session_id"] = session_id
            return Answer(text="ok", sources=(), confidence=1.0)

    monkeypatch.setattr("data_engineering_copilot.factory.build_rag_service", lambda: _FakeService())
    cli.ask("What is Spark?", user_id="u1", session_id="s1")

    assert seen == {"question": "What is Spark?", "user_id": "u1", "session_id": "s1"}
    assert "ok" in capsys.readouterr().out
