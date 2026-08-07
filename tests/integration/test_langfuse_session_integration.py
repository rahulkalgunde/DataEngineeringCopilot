"""Integration: Langfuse session groups multiple traces under one session_id.

Creates two traces with the same session/user ids via the telemetry client and
verifies the public API exposes both under the session. Auto-skips when
Langfuse is unreachable (require_langfuse).
"""

from __future__ import annotations

import json
import time
import urllib.request
import uuid

import pytest

from tests.conftest import require_langfuse


@pytest.mark.integration
def test_session_groups_multiple_traces() -> None:
    require_langfuse()

    from data_engineering_copilot.config.settings import settings
    from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

    session_id = f"test-session-{uuid.uuid4()}"
    user_id = f"test-user-{uuid.uuid4()}"

    client = get_langfuse_instance()
    assert client is not None
    for i in range(2):
        trace = client.trace(
            name="sess-integration",
            input={"q": f"question-{i}"},
            user_id=user_id,
            session_id=session_id,
        )
        trace.end()
    client.flush()

    url = f"{settings.langfuse_host.rstrip('/')}/api/public/traces?sessionId={session_id}&limit=10"
    req = urllib.request.Request(url)
    req.add_header(
        "Authorization",
        "Basic "
        + __import__("base64")
        .b64encode(
            f"{settings.langfuse_public_key.get_secret_value()}:{settings.langfuse_secret_key.get_secret_value()}".encode()
        )
        .decode(),
    )

    # Ingestion is eventually consistent (SDK batch exporter + server-side
    # ClickHouse writes), so poll briefly for both traces to appear.
    traces: list[dict] = []
    for _ in range(15):
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        traces = body.get("data", [])
        if len(traces) >= 2:
            break
        time.sleep(1)

    assert len(traces) == 2
    for trace in traces:
        assert trace["sessionId"] == session_id
        assert trace["userId"] == user_id
