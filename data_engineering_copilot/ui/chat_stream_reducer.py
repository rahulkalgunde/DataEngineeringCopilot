"""Pure chat SSE reducer + helpers used by the Streamlit chat tab.

Kept Streamlit-free so the turn logic is unit-testable without a session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data_engineering_copilot.ui.chat_parse import extract_streaming_answer


@dataclass
class ChatTurnState:
    full_text: str = ""
    raw_buffer: str = ""
    error_msg: str | None = None
    last_status: str = "Connecting to chat API…"
    last_event_ts: float = 0.0
    turn_started: float = 0.0
    stage_started: float = 0.0
    sources: list[dict[str, Any]] = field(default_factory=list)
    groundedness_score: float = 1.0
    groundedness_claims: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    session_id: str | None = None
    done: bool = False
    should_break: bool = False
    _alarm_shown: bool = False


def apply_chat_event(state: ChatTurnState, event: dict[str, Any], now: float) -> ChatTurnState:
    etype = event.get("type")
    if etype == "session_created":
        state.session_id = event.get("session_id") or state.session_id
    elif etype == "status":
        message = event.get("message", "Working…")
        if message != state.last_status:
            state.last_status = message
            state.stage_started = now
    elif etype == "token":
        state.raw_buffer += event.get("content", "")
        clean = extract_streaming_answer(state.raw_buffer)
        if clean is not None:
            state.full_text = clean
    elif etype == "sources":
        state.sources = event.get("sources", [])
    elif etype == "done":
        state.full_text = event.get("text", state.full_text)
        state.groundedness_score = float(event.get("groundedness_score", 1.0))
        state.groundedness_claims = event.get("groundedness_claims") or []
        state.done = True
        state.should_break = True
    elif etype == "suggestions":
        state.suggestions = event.get("suggestions", [])
    elif etype == "error":
        state.error_msg = event.get("message", "Unknown error")
        state.should_break = True
    state.last_event_ts = now
    return state


def stall_alarm_message(state: ChatTurnState, now: float, stall_seconds: float, budget_seconds: float) -> str | None:
    if state.done or state.error_msg is not None:
        return None
    age = now - state.last_event_ts
    if state.last_event_ts and age > stall_seconds:
        return (
            f"Still working on “{state.last_status}” for {age:.0f}s. "
            f"The server will auto-abort after {budget_seconds:.0f}s."
        )
    return None
