from celery import Celery

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.observability import init_sentry

configure_logging()
init_sentry()

celery_app = Celery(
    "docflow",
    broker=settings.broker_host,
    backend=settings.broker_backend,
    include=["app.tasks.document_task", "app.tasks.index_task"],
)
celery_app.conf.worker_hijack_root_logger = False
