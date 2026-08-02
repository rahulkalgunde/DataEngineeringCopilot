import sys

from celery import Celery, signals

from data_engineering_copilot.config.settings import settings
from data_engineering_copilot.infrastructure.dep_check import check_deps

# Celery configuration – broker and backend both use Redis from settings
celery_app = Celery(
    "data_engineering_copilot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)


def _enforce_fresh_deps(sender, **kwargs) -> None:
    """Refuse to run the worker on a stale image (deps changed since build)."""
    if not check_deps(fail_fast=False):
        sys.exit(1)


signals.worker_ready.connect(_enforce_fresh_deps)

# Production tuning: fair task distribution, late ack, time limits
celery_app.conf.update(
    # Fair distribution: fetch one task at a time per worker
    worker_prefetch_multiplier=1,
    # Late ack: requeue task if worker dies mid-execution
    task_acks_late=True,
    # Reject task on worker lost for requeue
    task_reject_on_worker_lost=True,
    # Hard time limit: kill task after 12 hours
    task_time_limit=43200,
    # Soft time limit: raise SoftTimeLimitExceeded after 10 hours
    task_soft_time_limit=36000,
    # Task routing: ingestion tasks go to dedicated queue
    task_routes={
        "data_engineering_copilot.workers.tasks.async_ingest_task": {"queue": "ingestion"},
    },
    # Worker concurrency: let worker auto-detect, but cap at 4
    worker_concurrency=4,
)
