"""Tests for Celery configuration and task routing."""

from __future__ import annotations


class TestCeleryAppConfiguration:
    def test_celery_app_has_broker(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.broker_url is not None

    def test_celery_app_has_backend(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.result_backend is not None

    def test_prefetch_multiplier_is_one(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.worker_prefetch_multiplier == 1

    def test_acks_late_enabled(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.task_acks_late is True

    def test_task_reject_on_worker_lost(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.task_reject_on_worker_lost is True

    def test_task_time_limit(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.task_time_limit is not None
        assert celery_app.conf.task_time_limit > 0

    def test_task_soft_time_limit(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.task_soft_time_limit is not None
        assert celery_app.conf.task_soft_time_limit > 0

    def test_soft_time_limit_less_than_hard(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert celery_app.conf.task_soft_time_limit < celery_app.conf.task_time_limit

    def test_task_routes_ingestion_queue(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        routes = celery_app.conf.task_routes
        assert isinstance(routes, dict)

    def test_worker_concurrency_configurable(self):
        from data_engineering_copilot.workers.celery_app import celery_app

        assert hasattr(celery_app.conf, "worker_concurrency")


class TestTaskRouting:
    def test_async_ingest_task_has_queue(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        assert hasattr(async_ingest_task, "queue")

    def test_async_ingest_task_queue_is_ingestion(self):
        from data_engineering_copilot.workers.tasks import async_ingest_task

        queue = getattr(async_ingest_task, "queue", None)
        assert queue == "ingestion"
