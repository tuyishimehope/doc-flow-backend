import asyncio
import logging
from datetime import datetime, timezone
from io import BytesIO

from sqlalchemy import Select

from app.db.session import SyncSession
from app.core.config import settings
from app.models.schema import (
    Document,
    Extracted_Result,
    File,
    Processing_Job,
    Processing_Request,
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

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
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


async def process_document(self, processing_request_id: int):
    db_session = SyncSession()
    file_object = None
    processing_request = None
    job = None

    try:
        processing_request = db_session.execute(
            Select(Processing_Request).where(
                Processing_Request.id == processing_request_id
            )
        ).scalar_one_or_none()
        if processing_request is None:
            logger.warning("Processing request %s was not found", processing_request_id)
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
            processing_request.status = Processing_status.FAILED
            job.status = Processing_Job_Status.FAILED
            job.failure_reason = "No text could be extracted from the uploaded file"
            job.completed_at = datetime.now(timezone.utc)
            db_session.commit()
            return

        if len(extracted_content) > settings.OPENAI_MAX_INPUT_CHARS:
            processing_request.status = Processing_status.FAILED
            job.status = Processing_Job_Status.FAILED
            job.failure_reason = (
                f"Extracted text exceeds the configured {settings.OPENAI_MAX_INPUT_CHARS} character limit"
            )
            job.completed_at = datetime.now(timezone.utc)
            db_session.commit()
            logger.warning(
                "Document text exceeded AI input limit",
                extra={
                    "processing_request_id": processing_request_id,
                    "character_count": len(extracted_content),
                    "character_limit": settings.OPENAI_MAX_INPUT_CHARS,
                },
            )
            return

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
    except Exception:
        db_session.rollback()
        if processing_request is not None:
            retries = self.request.retries
            max_retries = self.max_retries or 0
            processing_request.status = (
                Processing_status.QUEUED
                if retries < max_retries
                else Processing_status.FAILED
            )
            if job is not None:
                job.status = (
                    Processing_Job_Status.RETRYING
                    if retries < max_retries
                    else Processing_Job_Status.FAILED
                )
                job.failure_reason = "Document processing failed; see worker logs"
                job.completed_at = datetime.now(timezone.utc)
            db_session.commit()
        raise
    finally:
        if file_object is not None:
            file_object.close()
            file_object.release_conn()
        db_session.close()
