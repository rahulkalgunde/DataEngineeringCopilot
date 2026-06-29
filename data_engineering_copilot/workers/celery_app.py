from celery import Celery

# Celery configuration – broker and backend both use local Redis
celery_app = Celery(
    "data_engineering_copilot",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)