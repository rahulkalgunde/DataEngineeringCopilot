"""Tests for Steps 5-6: reset-index and progress tracker.

Step 5: reset-index clears crawl:url_registry:* Redis keys
Step 6: Progress tracker handles page_skipped_cached event
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_engineering_copilot.domain.models import IngestionEvent
from data_engineering_copilot.workers.progress import IngestionProgressTracker

# ---------------------------------------------------------------------------
# Step 5: reset-index clears registry keys
# ---------------------------------------------------------------------------


class TestResetIndexClearsRegistry:
    """Step 5: Verify reset-index clears crawl:url_registry:* keys."""

    def test_reset_index_clears_registry_keys(self) -> None:
        from data_engineering_copilot.cli import reset_index

        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = [
            b"crawl:url_registry:Source A",
            b"crawl:url_registry:Source B",
        ]

        with patch("data_engineering_copilot.cli.urllib.request") as mock_req:
            import urllib.error

            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_req.Request.return_value = MagicMock()
            mock_req.urlopen.side_effect = urllib.error.HTTPError(
                url="",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            )

            with patch("data_engineering_copilot.workers.progress.get_redis_client", return_value=mock_redis):
                reset_index()

        mock_redis.scan_iter.assert_called_once_with("crawl:url_registry:*")
        mock_redis.delete.assert_called_once_with(
            b"crawl:url_registry:Source A",
            b"crawl:url_registry:Source B",
        )

    def test_reset_index_handles_no_registry_keys(self) -> None:
        from data_engineering_copilot.cli import reset_index

        mock_redis = MagicMock()
        mock_redis.scan_iter.return_value = []

        with patch("data_engineering_copilot.cli.urllib.request") as mock_req:
            import urllib.error

            mock_req.Request.return_value = MagicMock()
            mock_req.urlopen.side_effect = urllib.error.HTTPError(
                url="",
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,
            )

            with patch("data_engineering_copilot.workers.progress.get_redis_client", return_value=mock_redis):
                reset_index()

        mock_redis.scan_iter.assert_called_once_with("crawl:url_registry:*")
        mock_redis.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Step 6: Progress tracker handles page_skipped_cached
# ---------------------------------------------------------------------------


class TestProgressTrackerPageSkippedCached:
    """Step 6: Verify progress tracker counts page_skipped_cached events."""

    def test_page_skipped_cached_increments_pages_skipped(self) -> None:
        mock_redis = MagicMock()
        tracker = IngestionProgressTracker("task-1", mock_redis, source_names=["S"])

        event = IngestionEvent(
            event_type="page_skipped_cached",
            source_name="S",
            url="https://example.com/page.html",
            message="Cache hit",
        )
        tracker.on_event(event)

        status = tracker.get_status()
        assert status["pages_skipped"] == 1

    def test_page_skipped_cached_in_source_stats(self) -> None:
        mock_redis = MagicMock()
        tracker = IngestionProgressTracker("task-1", mock_redis, source_names=["S"])

        event = IngestionEvent(
            event_type="page_skipped_cached",
            source_name="S",
            url="https://example.com/page.html",
            message="Cache hit",
        )
        tracker.on_event(event)

        stats = tracker.get_status()["source_stats"]["S"]
        assert stats["pages_skipped"] == 1

    def test_page_skipped_cached_in_recent_events(self) -> None:
        mock_redis = MagicMock()
        tracker = IngestionProgressTracker("task-1", mock_redis, source_names=["S"])

        event = IngestionEvent(
            event_type="page_skipped_cached",
            source_name="S",
            url="https://example.com/page.html",
            message="Cache hit",
        )
        tracker.on_event(event)

        recent = tracker.get_status()["recent_events"]
        cached = [e for e in recent if e["type"] == "page_skipped_cached"]
        assert len(cached) == 1
        assert cached[0]["url"] == "https://example.com/page.html"

    def test_mixed_skip_events(self) -> None:
        mock_redis = MagicMock()
        tracker = IngestionProgressTracker("task-1", mock_redis, source_names=["S"])

        for etype in ("page_skipped", "page_skipped_cached", "page_skipped_duplicate"):
            tracker.on_event(
                IngestionEvent(
                    event_type=etype,
                    source_name="S",
                    url="https://example.com/p",
                    message="skip",
                )
            )

        status = tracker.get_status()
        assert status["pages_skipped"] == 3
