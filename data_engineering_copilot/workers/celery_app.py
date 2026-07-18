from celery import Celery

from data_engineering_copilot.config.settings import settings

# Celery configuration – broker and backend both use Redis from settings
celery_app = Celery(
    "data_engineering_copilot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
