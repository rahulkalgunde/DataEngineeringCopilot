"""Integration: Langfuse session groups multiple traces under one session_id.

Creates two traces with the same session/user ids via the telemetry client and
verifies both carry the session/user attributes on their OTel spans. This
checks the deterministic SDK-level contract (``propagate_attributes`` attaches
``session.id``/``user.id`` to every observation in the trace).

Note: we do NOT verify async ClickHouse persistence. In Langfuse v4
``events_only`` mode the v3 public ``/api/public/traces`` endpoint is disabled
(404), and the OTel export from a short-lived test process is not guaranteed to
drain to ClickHouse before exit — so persistence is asserted at the span
attribute level, which is what the app's telemetry adapter controls.
"""

from __future__ import annotations

import uuid

import pytest

from tests.conftest import require_langfuse


@pytest.mark.integration
def test_session_groups_multiple_traces() -> None:
    require_langfuse()

    from data_engineering_copilot.observability.langfuse_client import get_langfuse_instance

    session_id = f"test-session-{uuid.uuid4()}"
    user_id = f"test-user-{uuid.uuid4()}"

    client = get_langfuse_instance()
    if client is None:
        pytest.skip("Langfuse client could not be initialized (missing LANGFUSE_PUBLIC_KEY/SECRET_KEY)")

    # Create two traces with the same session/user ids; each returns an
    # _ObservationCompat wrapping the v4 observation with its OTel span.
    traces = []
    for i in range(2):
        trace = client.trace(
            name="sess-integration",
            input={"q": f"question-{i}"},
            user_id=user_id,
            session_id=session_id,
        )
        trace.end()
        traces.append(trace)

    # The compat adapter must have propagated session/user onto the OTel span
    # attributes synchronously (deterministic, no async persistence required).
    seen_trace_ids = set()
    for trace in traces:
        obs = getattr(trace, "_observation", None)
        assert obs is not None, "trace must wrap a real observation"
        otel_span = getattr(obs, "_otel_span", None)
        assert otel_span is not None, "observation must expose its OTel span"
        attrs = dict(otel_span.attributes or {})
        assert attrs.get("session.id") == session_id, f"session not propagated: {attrs}"
        assert attrs.get("user.id") == user_id, f"user not propagated: {attrs}"

        from opentelemetry.trace import format_trace_id

        seen_trace_ids.add(format_trace_id(otel_span.context.trace_id))

    # Two distinct traces share the session (grouping), with unique trace ids.
    assert len(seen_trace_ids) == 2, f"expected 2 distinct traces under session, got {len(seen_trace_ids)}"
