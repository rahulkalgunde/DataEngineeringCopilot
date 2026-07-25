"""Tests for Celery task exception propagation and retry config."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from celery.exceptions import SoftTimeLimitExceeded


class TestAsyncIngestTaskConfig:
    def test_autoretry_for_configured(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert async_ingest_task.autoretry_for == (ConnectionError, TimeoutError)

    def test_retry_kwargs_configured(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert async_ingest_task.retry_kwargs == {"max_retries": 3, "countdown": 10}

    def test_retry_backoff_enabled(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert async_ingest_task.retry_backoff is True

    def test_max_retries_set(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert async_ingest_task.max_retries == 3

    def test_queue_is_ingestion(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert async_ingest_task.queue == "ingestion"


class TestAsyncIngestTaskBehavior:
    @patch("data_engineering_copilot.workers.tasks.IngestionProgressTracker")
    @patch("data_engineering_copilot.factory.build_async_ingestion_service")
    @patch("data_engineering_copilot.workers.tasks.asyncio.run")
    def test_success_path_calls_mark_completed(self, mock_asyncio_run, mock_build_svc, mock_tracker_cls):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_service = MagicMock()
        mock_build_svc.return_value = mock_service

        async_ingest_task.run(["Test Source"], 10)

        mock_tracker_cls.assert_called_once()
        mock_asyncio_run.assert_called_once()
        mock_tracker.mark_completed.assert_called_once()

    @patch("data_engineering_copilot.workers.tasks.IngestionProgressTracker")
    @patch("data_engineering_copilot.factory.build_async_ingestion_service")
    @patch("data_engineering_copilot.workers.tasks.asyncio.run")
    def test_exception_re_raised_after_mark_failed(self, mock_asyncio_run, mock_build_svc, mock_tracker_cls):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_asyncio_run.side_effect = ValueError("connection refused")

        with pytest.raises(ValueError, match="connection refused"):
            async_ingest_task.run(["Test Source"], 10)

        mock_tracker.mark_failed.assert_called_once_with("connection refused")

    @patch("data_engineering_copilot.workers.tasks.IngestionProgressTracker")
    @patch("data_engineering_copilot.factory.build_async_ingestion_service")
    @patch("data_engineering_copilot.workers.tasks.asyncio.run")
    def test_soft_time_limit_raises_and_marks_failed(self, mock_asyncio_run, mock_build_svc, mock_tracker_cls):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_asyncio_run.side_effect = SoftTimeLimitExceeded()

        with pytest.raises(SoftTimeLimitExceeded):
            async_ingest_task.run(["Test Source"], 10)

        mock_tracker.mark_failed.assert_called_once_with("Task exceeded soft time limit. Execution cancelled.")

    @patch("data_engineering_copilot.workers.tasks.IngestionProgressTracker")
    @patch("data_engineering_copilot.factory.build_async_ingestion_service")
    @patch("data_engineering_copilot.workers.tasks.asyncio.run")
    def test_mark_completed_not_called_on_failure(self, mock_asyncio_run, mock_build_svc, mock_tracker_cls):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        mock_tracker = MagicMock()
        mock_tracker_cls.return_value = mock_tracker

        mock_asyncio_run.side_effect = RuntimeError("unexpected error")

        with pytest.raises(RuntimeError):
            async_ingest_task.run(["Test Source"], 10)

        mock_tracker.mark_completed.assert_not_called()
        mock_tracker.mark_failed.assert_called_once()
