from celery import Celery

from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()

celery_app = Celery('docflow', broker=settings.broker_host,
             backend=settings.broker_backend)
celery_app.conf.worker_hijack_root_logger = False

celery_app.autodiscover_tasks(
    ["app.tasks"]
)
