"""Tests for cli_monitor: formatters, dashboard rendering, fetch_status retries.

All network/system effects are monkeypatched (urllib, os.system, time) — no
API or Redis is contacted. fetch_status retry semantics are pinned because
they decide whether a transient API blip kills the monitor.
"""

from __future__ import annotations

import urllib.error

import pytest

from data_engineering_copilot.cli_monitor import (
    _fmt_delta,
    _fmt_elapsed,
    fetch_status,
    render_dashboard,
)

pytestmark = pytest.mark.unit


class TestFormatters:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0"), (5, "+5"), (1234, "+1,234"), (-3, "-3")],
    )
    def test_fmt_delta(self, value, expected):
        assert _fmt_delta(value) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00:00"), (59.9, "00:00:59"), (3661, "01:01:01"), (7200, "02:00:00")],
    )
    def test_fmt_elapsed(self, seconds, expected):
        assert _fmt_elapsed(seconds) == expected


def _status(**overrides) -> dict:
    base = {
        "status": "PROCESSING",
        "task_id": "t-1",
        "source_names": ["spark"],
        "pages_fetched": 100,
        "chunks_indexed": 500,
        "pages_skipped": 2,
        "source_stats": {"spark": {"pages_fetched": 100, "chunks_indexed": 500, "errors": 0}},
        "current_url": "https://example.com/x",
        "recent_events": [{"ts": 0, "type": "page_indexed", "url": "u", "chunks": 4}],
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _quiet_screen(monkeypatch):
    monkeypatch.setattr("os.system", lambda *_a, **_k: None)


class TestRenderDashboard:
    def test_first_poll_shows_counts_without_rates(self, capsys):
        render_dashboard(_status(), None, poll_ts=10.0, first_poll_ts=5.0)
        out = capsys.readouterr().out
        assert "[>]" in out and "PROCESSING" in out
        assert f"{100:>10,}" in out  # pages count cell
        assert "Next refresh" in out  # non-terminal state footer

    def test_deltas_and_rates_from_previous_poll(self, capsys):
        prev = _status(pages_fetched=90, chunks_indexed=450)
        prev["_poll_ts"] = 7.0
        render_dashboard(_status(), prev, poll_ts=17.0, first_poll_ts=5.0)
        out = capsys.readouterr().out
        assert "+10" in out  # pages delta
        assert "+50" in out  # chunks delta
        assert "1.0 p/s" in out
        assert "5.0 c/s" in out

    def test_terminal_completed_footer(self, capsys):
        render_dashboard(_status(status="COMPLETED"), None, poll_ts=1.0, first_poll_ts=0.0)
        out = capsys.readouterr().out
        assert "DONE" in out
        assert "Next refresh" not in out

    def test_error_state_renders_error_block(self, capsys):
        render_dashboard(_status(status="FAILED", error="boom"), None, poll_ts=1.0, first_poll_ts=0.0)
        out = capsys.readouterr().out
        assert "!!! ERROR !!!" in out
        assert "boom" in out
        assert "FAILED" in out

    def test_enrichment_info_rendered(self, capsys):
        render_dashboard(_status(), None, poll_ts=1.0, first_poll_ts=0.0, enrichment_info={"queue_depth": 7})
        out = capsys.readouterr().out
        assert "Enrichment Queue" in out
        assert "\U0001f534 worker" in out  # worker_alive False -> red


class TestFetchStatus:
    def _http_error(self, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError("url", code, "msg", hdrs=None, fp=None)  # type: ignore[arg-type]

    def test_success_returns_payload(self, monkeypatch):
        import io

        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout: io.BytesIO(b'{"status": "PROCESSING"}'),
        )
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert fetch_status("http://api", "t-1") == {"status": "PROCESSING"}

    def test_404_returns_none_immediately(self, monkeypatch):
        calls = []

        def boom(req, timeout):
            calls.append(1)
            raise self._http_error(404)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert fetch_status("http://api", "missing") is None
        assert len(calls) == 1  # no retries for definitive 404

    def test_client_http_error_raises_without_retry(self, monkeypatch):
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout: (_ for _ in ()).throw(self._http_error(403)))
        with pytest.raises(urllib.error.HTTPError):
            fetch_status("http://api")

    def test_server_errors_retry_then_raise_last(self, monkeypatch):
        attempts = []

        def boom(req, timeout):
            attempts.append(1)
            raise self._http_error(503)

        monkeypatch.setattr("urllib.request.urlopen", boom)
        sleeps: list[float] = []
        monkeypatch.setattr("time.sleep", sleeps.append)
        with pytest.raises(urllib.error.HTTPError):
            fetch_status("http://api")
        assert len(attempts) == 3
        assert sleeps == [2, 4, 6]  # linear backoff 2*(attempt+1)

    def test_connection_refused_exhausts_to_none(self, monkeypatch):
        def refused(req, timeout):
            raise ConnectionRefusedError()

        monkeypatch.setattr("urllib.request.urlopen", refused)
        monkeypatch.setattr("time.sleep", lambda s: None)
        assert fetch_status("http://api") is None

    def test_latest_endpoint_when_no_task_id(self, monkeypatch):
        seen = {}

        def capture(req, timeout):
            import io

            seen["url"] = req.full_url
            return io.BytesIO(b"{}")

        monkeypatch.setattr("urllib.request.urlopen", capture)
        fetch_status("http://api")
        assert seen["url"].endswith("/api/v1/ingest/latest")
