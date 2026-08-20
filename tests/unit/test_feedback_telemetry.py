from __future__ import annotations

from pathlib import Path

import pytest

from data_engineering_copilot.services.feedback_telemetry import FeedbackTelemetryService


@pytest.mark.asyncio
async def test_log_interaction_writes_jsonl(tmp_path: Path) -> None:
    log = tmp_path / "telemetry_logs.jsonl"
    svc = FeedbackTelemetryService(log_path=str(log))
    await svc.log_interaction(
        query_id="q1",
        query="What is Spark?",
        answer="Spark is ...",
        provenance=[{"chunk_id": "c1", "url": "u", "source_name": "Spark"}],
        feedback=None,
    )
    assert log.exists()
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    import json

    row = json.loads(lines[0])
    assert row["query_id"] == "q1"
    assert row["query"] == "What is Spark?"
    assert row["provenance"][0]["url"] == "u"


@pytest.mark.asyncio
async def test_log_implicit_feedback_writes_event(tmp_path: Path) -> None:
    log = tmp_path / "telemetry_logs.jsonl"
    svc = FeedbackTelemetryService(log_path=str(log))
    await svc.log_implicit_feedback(query_id="q1", click_url="https://example.com/doc")
    implicit = Path(str(log) + ".implicit")
    assert implicit.exists()
    import json

    row = json.loads(implicit.read_text().splitlines()[0])
    assert row["query_id"] == "q1"
    assert row["click_url"] == "https://example.com/doc"
    assert row["event"] == "citation_click"
