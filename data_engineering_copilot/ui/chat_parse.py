"""SSE streaming answer extraction for the chat UI.

Kept Streamlit-free so the parser is unit-testable without a session.
"""

from __future__ import annotations

import json


def extract_streaming_answer(buffer: str) -> str | None:
    """Return the clean answer if *buffer* is complete valid JSON with an answer.

    Returns ``None`` while the streamed JSON is still incomplete so the UI shows
    the progress status instead of raw ``{"status": ...}`` fragments.
    """
    if not buffer or not buffer.strip():
        return None
    import re

    text = buffer.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    answer = data.get("answer") or data.get("response") or data.get("text") or data.get("content")
    if answer is None:
        return None
    return str(answer)
