from celery import Celery

from data_engineering_copilot.config.settings import settings

# Celery configuration – broker and backend both use Redis from settings
celery_app = Celery(
    "data_engineering_copilot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# Production tuning: fair task distribution, late ack, time limits
celery_app.conf.update(
    # Fair distribution: fetch one task at a time per worker
    worker_prefetch_multiplier=1,
    # Late ack: requeue task if worker dies mid-execution
    task_acks_late=True,
    # Reject task on worker lost for requeue
    task_reject_on_worker_lost=True,
    # Hard time limit: kill task after 30 minutes
    task_time_limit=1800,
    # Soft time limit: raise SoftTimeLimitExceeded after 25 minutes
    task_soft_time_limit=1500,
    # Task routing: ingestion tasks go to dedicated queue
    task_routes={
        "data_engineering_copilot.workers.tasks.async_ingest_task": {"queue": "ingestion"},
        "data_engineering_copilot.workers.tasks.execute_background_ingestion": {"queue": "ingestion"},
    },
    # Worker concurrency: let worker auto-detect, but cap at 4
    worker_concurrency=4,
)
