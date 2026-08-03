"""Unit tests for the ``dec ingest`` status-poll loop.

Verifies the CLI tolerates transient status-poll failures (timeouts, 5xx)
instead of crashing with a traceback while the ingestion task keeps running
server-side.
"""

from __future__ import annotations

import json
import urllib.error
from email.message import Message
from unittest.mock import patch

from data_engineering_copilot import cli

TASK_ID = "2f06443c-fe23-4f07-be98-b34f27983229"


class FakeResp:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode()

    def __enter__(self) -> FakeResp:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _status(status: str) -> str:
    return json.dumps(
        {
            "task_id": TASK_ID,
            "status": status,
            "pages_fetched": 1,
            "chunks_indexed": 5,
            "current_url": "https://example.com",
            "error": None,
        }
    )


def _patch_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_check_deps_before_dispatch", lambda *a, **k: None)


@patch("time.sleep", return_value=None)
def test_poll_retries_transient_timeout_then_completes(mock_sleep, monkeypatch, capsys):
    """A single TimeoutError must not kill the CLI; the next poll succeeds."""
    _patch_dispatch(monkeypatch)
    side_effect = [
        FakeResp(json.dumps({"task_id": TASK_ID})),
        TimeoutError("timed out"),
        FakeResp(_status("PROCESSING")),
        FakeResp(_status("COMPLETED")),
    ]
    with patch("data_engineering_copilot.cli.urllib.request.urlopen", side_effect=side_effect) as mock_open:
        cli.ingest(max_pages=10, source_names=("Spark",))

    out = capsys.readouterr().out
    assert "Ingestion completed: 5 chunks indexed." in out
    assert mock_open.call_count == 4


@patch("time.sleep", return_value=None)
def test_poll_gives_up_after_repeated_timeouts(mock_sleep, monkeypatch, capsys):
    """Persistent timeouts abort the loop with guidance, not a traceback."""
    _patch_dispatch(monkeypatch)
    side_effect = [
        FakeResp(json.dumps({"task_id": TASK_ID})),
        TimeoutError("timed out"),
        TimeoutError("timed out"),
        TimeoutError("timed out"),
    ]
    with patch("data_engineering_copilot.cli.urllib.request.urlopen", side_effect=side_effect):
        cli.ingest(max_pages=10, source_names=("Spark",))

    out = capsys.readouterr().out
    assert "still running server-side" in out
    assert "dec ingestion-monitor" in out
    assert f"dec cancel {TASK_ID}" in out


@patch("time.sleep", return_value=None)
def test_poll_gives_up_after_repeated_5xx(mock_sleep, monkeypatch, capsys):
    """Persistent HTTP 5xx responses abort the loop with guidance."""
    _patch_dispatch(monkeypatch)
    error = urllib.error.HTTPError(
        f"http://localhost:8000/api/v1/ingest/status/{TASK_ID}",
        503,
        "Service Unavailable",
        Message(),
        None,
    )
    side_effect = [
        FakeResp(json.dumps({"task_id": TASK_ID})),
        error,
        error,
        error,
    ]
    with patch("data_engineering_copilot.cli.urllib.request.urlopen", side_effect=side_effect):
        cli.ingest(max_pages=10, source_names=("Spark",))

    out = capsys.readouterr().out
    assert "still running server-side" in out
