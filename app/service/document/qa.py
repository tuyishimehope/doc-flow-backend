"""Keep a PDF so that it can be asked about again without uploading it again."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.models.schema import Document, File
from app.service.document.pages import save_pages
from app.service.file.file import delete_file, post_bytes
from app.utils.pdf import Page
from app.utils.upload import CheckedUpload

logger = logging.getLogger(__name__)

NAME_MAX_CHARS = 255  # Document.name


async def store_pdf(
    *, user_id: int, upload: CheckedUpload, pages: list[Page], db_session: AsyncSession
) -> Document:
    """Save the file in MinIO, a ``Document`` row, and one row per page. All or nothing.

    The caller has already checked, scanned and read the PDF. The content type comes from the
    type table, never from the client. If anything fails the rows are rolled back and the
    stored object is removed again.
    """
    file_id: int | None = None
    stored = False
    try:
        file_object = File(
            name=upload.display_name[:NAME_MAX_CHARS],
            content_type=upload.mime,
            extension=upload.extension,
        )
        db_session.add(file_object)
        await db_session.flush()
        file_id = file_object.id

        await post_bytes(upload.content, str(file_id), upload.mime)
        stored = True

        document = Document(user_id=user_id, name=file_object.name, file=file_object)
        db_session.add(document)
        await db_session.flush()
        await save_pages(document.id, pages, db_session)
        await db_session.commit()
    except Exception:
        await db_session.rollback()
        if stored and file_id is not None:
            try:
                await run_in_threadpool(delete_file, file_id)
            except Exception:
                logger.exception("Could not remove a stored PDF after a failed save")
        raise

    await db_session.refresh(document)
    logger.info(
        "Document stored for questions",
        extra={"document_id": document.id, "page_count": len(pages), "user_id": user_id},
    )
    return document
