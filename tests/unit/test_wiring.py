"""Integration-style wiring tests that verify real component signatures match.

These tests instantiate real classes (with minimal mocks for external services)
to catch signature mismatches, missing arguments, and wrong method names
that pure unit tests with heavy mocking would miss.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from data_engineering_copilot.domain.models import IngestionEvent


class TestTrackerWiring:
    """Verify IngestionProgressTracker can be instantiated and called
    with the exact arguments the Celery task uses."""

    def test_tracker_init_with_redis_client(self):
        """Tracker accepts (task_id, redis_client, source_names)."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        tracker = IngestionProgressTracker(
            task_id="test-123",
            redis_client=mock_redis,
            source_names=["Apache Spark Documentation"],
        )
        assert tracker.redis_key == "ingestion:status:test-123"

    def test_tracker_on_event_method_exists(self):
        """Tracker exposes on_event() — not update()."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        tracker = IngestionProgressTracker(
            task_id="test-456",
            redis_client=MagicMock(),
        )
        assert callable(getattr(tracker, "on_event", None))

    def test_tracker_mark_completed_method_exists(self):
        """Tracker exposes mark_completed() — not complete()."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        tracker = IngestionProgressTracker(
            task_id="test-789",
            redis_client=MagicMock(),
        )
        assert callable(getattr(tracker, "mark_completed", None))

    def test_tracker_mark_failed_method_exists(self):
        """Tracker exposes mark_failed() — not fail()."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        tracker = IngestionProgressTracker(
            task_id="test-fail",
            redis_client=MagicMock(),
        )
        assert callable(getattr(tracker, "mark_failed", None))

    def test_tracker_on_event_accepts_ingestion_event(self):
        """on_event() accepts an IngestionEvent instance."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        tracker = IngestionProgressTracker(
            task_id="test-evt",
            redis_client=mock_redis,
        )
        mock_redis.reset_mock()

        event = IngestionEvent(
            event_type="page_indexed",
            source_name="Spark",
            message="Indexed page",
            url="https://example.com",
            chunks_indexed=10,
            pages_fetched=1,
        )
        tracker.on_event(event)

        assert mock_redis.set.called

    def test_tracker_mark_completed_writes_to_redis(self):
        """mark_completed() writes status=COMPLETED to Redis."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        tracker = IngestionProgressTracker(
            task_id="test-done",
            redis_client=mock_redis,
        )
        mock_redis.reset_mock()

        tracker.mark_completed()

        mock_redis.set.assert_called_once()
        payload = json.loads(mock_redis.set.call_args[0][1])
        assert payload["status"] == "COMPLETED"

    def test_tracker_mark_failed_writes_error(self):
        """mark_failed() writes status=FAILED and error message."""
        from data_engineering_copilot.workers.progress import IngestionProgressTracker

        mock_redis = MagicMock()
        tracker = IngestionProgressTracker(
            task_id="test-fail2",
            redis_client=mock_redis,
        )
        mock_redis.reset_mock()

        tracker.mark_failed("Connection refused")

        payload = json.loads(mock_redis.set.call_args[0][1])
        assert payload["status"] == "FAILED"
        assert payload["error"] == "Connection refused"


class TestTaskSignature:
    """Verify the Celery task function accepts the right arguments."""

    def test_async_ingest_task_is_callable(self):
        """async_ingest_task can be imported and is callable."""
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert callable(async_ingest_task)

    def test_async_ingest_task_has_delay(self):
        """async_ingest_task has .delay() for Celery dispatch."""
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert hasattr(async_ingest_task, "delay")

    def test_tasks_module_has_app_attribute(self):
        """tasks.py exposes 'app' for celery -A discovery."""
        import data_engineering_copilot.workers.tasks as tasks_mod

        assert hasattr(tasks_mod, "app")

    def test_app_is_celery_app(self):
        """The 'app' attribute is the same object as celery_app."""
        import data_engineering_copilot.workers.tasks as tasks_mod
        from data_engineering_copilot.workers.celery_app import celery_app

        assert tasks_mod.app is celery_app


class TestApiRoutesWiring:
    """Verify API routes import correctly and endpoints match expected paths."""

    def test_routes_module_imports(self):
        """All route dependencies can be imported."""
        from data_engineering_copilot.api.routes import (
            async_ingest_task,
            celery_app,
            get_redis_client,
            router,
        )

        assert router is not None
        assert callable(async_ingest_task.delay)
        assert celery_app is not None
        assert callable(get_redis_client)

    def test_dispatch_endpoint_writes_redis(self):
        """POST /api/v1/ingest writes initial status to Redis."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from data_engineering_copilot.api.routes import router

        app = FastAPI()
        app.include_router(router)

        with (
            patch("data_engineering_copilot.api.routes.get_redis_client") as mock_get_client,
            patch("data_engineering_copilot.api.routes.async_ingest_task") as mock_task,
        ):
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_task.delay.return_value = MagicMock(id="new-task-id", state="PENDING")

            client = TestClient(app)
            response = client.post(
                "/api/v1/ingest",
                json={"source_names": ["Spark"], "max_pages": 10},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["task_id"] == "new-task-id"

            # Verify initial status was written to Redis (status key + latest_task_id key)
            assert mock_client.set.call_count == 2
            set_keys = [call.args[0] for call in mock_client.set.call_args_list]
            assert "ingestion:status:new-task-id" in set_keys
            assert "ingestion:latest_task_id" in set_keys
            status_call = mock_client.set.call_args_list[0]
            redis_key = status_call.args[0]
            assert redis_key == "ingestion:status:new-task-id"
            status_payload = json.loads(status_call.args[1])
            assert status_payload["status"] == "DISPATCHED"
            assert status_payload["source_names"] == ["Spark"]


class TestCeleryAppWiring:
    """Verify celery_app uses settings.redis_url, not hardcoded localhost."""

    def test_celery_app_uses_settings(self):
        """celery_app.broker_url comes from settings.redis_url."""
        from data_engineering_copilot.config.settings import settings
        from data_engineering_copilot.workers.celery_app import celery_app

        # The broker URL should match the settings value
        assert celery_app.conf.broker_url == settings.redis_url
        assert celery_app.conf.result_backend == settings.redis_url


class TestAsyncFactoryWiring:
    """Verify async factory functions return correctly typed objects."""

    def test_build_async_crawler_returns_crawler(self):
        from data_engineering_copilot.factory import build_async_crawler
        from data_engineering_copilot.infrastructure.async_crawler import AsyncDocumentationCrawler

        crawler = build_async_crawler()
        assert isinstance(crawler, AsyncDocumentationCrawler)

    def test_build_async_ingestion_service_returns_service(self):
        from data_engineering_copilot.factory import build_async_ingestion_service
        from data_engineering_copilot.services.async_ingestion import AsyncIngestionService

        service = build_async_ingestion_service()
        assert isinstance(service, AsyncIngestionService)

    def test_async_crawler_has_crawl_method(self):
        from data_engineering_copilot.factory import build_async_crawler

        crawler = build_async_crawler()
        assert hasattr(crawler, "crawl")
        import inspect

        assert inspect.isasyncgenfunction(crawler.crawl)

    def test_async_crawler_has_shutdown_method(self):
        from data_engineering_copilot.factory import build_async_crawler

        crawler = build_async_crawler()
        assert hasattr(crawler, "shutdown")
        assert callable(crawler.shutdown)
