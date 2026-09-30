from celery import Celery

from app.core.config import settings
from app.core.logging_config import configure_logging

configure_logging()

celery_app = Celery('docflow', broker=settings.broker_host,
             backend=settings.broker_backend,
             include=["app.tasks.document_task"])
celery_app.conf.worker_hijack_root_logger = False
