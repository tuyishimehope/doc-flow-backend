import asyncio
import logging
from datetime import datetime, timezone
from io import BytesIO

import openai
from sqlalchemy import Select

from app.db.session import SyncSession
from app.core.config import settings
from app.models.schema import (
    Document,
    Extracted_Result,
    File,
    Processing_Job,
    Processing_Request,
    User,
)
from app.service.document.schema import (
    Processing_Job_Status,
    Processing_Type,
    Processing_status,
)
from app.service.file.file import get_file
from app.service.openai.service import OpenaiService
from app.tasks.celery_app import celery_app
from app.utils.document import (
    extract_text_from_doc,
    extract_text_from_image,
    extract_text_from_pdf,
)
from app.utils.email_utils import send_processing_finished_email

logger = logging.getLogger(__name__)

# Retrying cannot fix these: unusable input, or a request the AI provider rejects outright.
# Our own checks raise ValueError with a message that is safe to show to the user.
PERMANENT_ERRORS = (
    ValueError,
    openai.AuthenticationError,
    openai.BadRequestError,
    openai.PermissionDeniedError,
)


def describe_failure(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, openai.OpenAIError):
        return "The AI service could not process this document"
    return "Unexpected error while processing the document"


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    dont_autoretry_for=PERMANENT_ERRORS,
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def start_processing(self, processing_request_id: int):
    logger.info(
        "Starting document processing",
        extra={
            "processing_request_id": processing_request_id,
            "celery_task_id": getattr(self.request, "id", None),
        },
    )
    try:
        asyncio.run(process_document(self, processing_request_id))
    except Exception:
        logger.exception(
            "Document processing task failed",
            extra={
                "processing_request_id": processing_request_id,
                "celery_task_id": getattr(self.request, "id", None),
                "attempt_number": self.request.retries + 1,
            },
        )
        # Let Celery's autoretry handler observe the failure.
        raise


async def notify_owner(db_session, processing_request: Processing_Request, succeeded: bool, failure_reason: str | None = None):
    """Email the document owner. Never fails the task: the result is already saved."""
    if not settings.processing_email_enabled:
        return
    try:
        document = db_session.get(Document, processing_request.document_id)
        user = db_session.get(User, document.user_id) if document else None
        if user is None or user.deleted_at is not None:
            return
        await send_processing_finished_email(
            to_email=user.email,
            first_name=user.first_name,
            document_name=document.name,
            processing_type=processing_request.processing_type.value,
            succeeded=succeeded,
            failure_reason=failure_reason,
        )
    except Exception:
        logger.exception(
            "Could not send processing email",
            extra={"processing_request_id": processing_request.id},
        )


async def process_document(self, processing_request_id: int):
    db_session = SyncSession()
    file_object = None
    processing_request = None
    job = None

    try:
        # Lock the row so a concurrent cancel either lands first or sees PROCESSING.
        processing_request = db_session.execute(
            Select(Processing_Request)
            .where(Processing_Request.id == processing_request_id)
            .with_for_update()
        ).scalar_one_or_none()
        if processing_request is None:
            logger.warning("Processing request %s was not found", processing_request_id)
            return
        if processing_request.status == Processing_status.CANCELLED:
            logger.info(
                "Skipping cancelled processing request",
                extra={"processing_request_id": processing_request_id},
            )
            return

        job = Processing_Job(
            processing_request_id=processing_request_id,
            attempt_number=self.request.retries + 1,
            status=Processing_Job_Status.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        processing_request.status = Processing_status.PROCESSING
        db_session.add(job)
        db_session.commit()

        document = db_session.execute(
            Select(Document).where(Document.id == processing_request.document_id)
        ).scalar_one_or_none()
        if document is None:
            raise ValueError("Document for processing request was not found")

        file_record = db_session.execute(
            Select(File).where(File.id == document.file_id)
        ).scalar_one_or_none()
        if file_record is None:
            raise ValueError("File record for document was not found")

        file_object = get_file(file_id=file_record.id)
        file_bytes = file_object.read()

        if file_record.content_type == "application/pdf":
            extracted_content = extract_text_from_pdf(BytesIO(file_bytes))
        elif file_record.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_content = extract_text_from_doc(BytesIO(file_bytes))
        elif file_record.content_type and file_record.content_type.startswith("image/"):
            extracted_content = extract_text_from_image(file_bytes)
        else:
            raise ValueError(f"Unsupported file content type: {file_record.content_type}")

        if not extracted_content.strip():
            raise ValueError("No text could be extracted from the uploaded file")

        if len(extracted_content) > settings.OPENAI_MAX_INPUT_CHARS:
            raise ValueError(
                f"Extracted text exceeds the configured {settings.OPENAI_MAX_INPUT_CHARS} character limit"
            )

        service = OpenaiService()
        try:
            if processing_request.processing_type == Processing_Type.DOCUMENT_SUMMARY:
                result = await service.generate_summary(
                    content=extracted_content,
                    instructions=processing_request.instructions or "",
                )
            elif processing_request.processing_type == Processing_Type.INVOICE_EXTRACTION:
                result = await service.get_invoice_metadata(
                    content=extracted_content,
                    instructions=processing_request.instructions or "",
                )
            elif processing_request.processing_type == Processing_Type.CONTRACT_METADATA:
                result = await service.get_contract_metadata(
                    content=extracted_content,
                    instructions=processing_request.instructions or "",
                )
            else:
                raise ValueError(
                    f"Unsupported processing type: {processing_request.processing_type}"
                )
        finally:
            await service.close()

        if not result:
            raise RuntimeError("AI processing returned an empty result")

        db_session.add(
            Extracted_Result(
                processing_request_id=processing_request.id,
                result_type=processing_request.processing_type.value,
                content_json={"result": result, "confidence_score": None},
                confidence_score=None,
            )
        )
        processing_request.status = Processing_status.COMPLETED
        job.status = Processing_Job_Status.COMPLETED
        job.completed_at = datetime.now(timezone.utc)
        db_session.commit()
        await notify_owner(db_session, processing_request, succeeded=True)
    except Exception as exc:
        db_session.rollback()
        if processing_request is not None:
            will_retry = (
                not isinstance(exc, PERMANENT_ERRORS)
                and self.request.retries < (self.max_retries or 0)
            )
            failure_reason = describe_failure(exc)
            processing_request.status = (
                Processing_status.QUEUED if will_retry else Processing_status.FAILED
            )
            if job is not None:
                job.status = (
                    Processing_Job_Status.RETRYING if will_retry else Processing_Job_Status.FAILED
                )
                job.failure_reason = failure_reason
                job.completed_at = datetime.now(timezone.utc)
            db_session.commit()
            if not will_retry:
                await notify_owner(
                    db_session, processing_request, succeeded=False, failure_reason=failure_reason)
        raise
    finally:
        if file_object is not None:
            file_object.close()
            file_object.release_conn()
        db_session.close()
