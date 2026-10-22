"""Chunks of a stored PDF: build and embed them (in the worker), and find the nearest ones.

``index_document`` is synchronous because the Celery worker uses a synchronous database
session; the only asynchronous step, the embedding call, is handed in as a function. The API side
(``retrieve_chunks``, ``delete_chunks``) is asynchronous like the rest of the API.

Privacy: logs carry ids, counts and distances, never chunk text.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.schema import Document, DocumentChunk, DocumentPage
from app.service.document.schema import Document_Status, Index_Status
from app.service.llm_client.embedding import EmbeddingResult
from app.utils.chunking import chunk_pages
from app.utils.pdf import Page

logger = logging.getLogger(__name__)

# Turns texts into vectors (see ``embed_texts``); a function so a test can hand in a fake.
Embedder = Callable[[list[str]], EmbeddingResult]


@dataclass(frozen=True)
class RetrievedChunk:
    id: int
    page: int
    position: int
    text: str
    distance: float  # cosine distance: 0 is the same direction, 1 unrelated, 2 opposite


def _alive(document: Document) -> bool:
    return document.deleted_at is None and document.status != Document_Status.DELETED


def index_document(document_id: int, session: Session, embed: Embedder) -> int | None:
    """Split the stored pages into chunks, embed them and save them. Returns the chunk count.

    ``index_status`` goes PROCESSING, then READY, or FAILED when anything raises (the error is
    raised again so the task can retry). Chunks of an earlier run are replaced, never added to.
    Nothing happens, and None is returned, for a document that is gone or deleted.
    """
    started = time.monotonic()
    document = session.get(Document, document_id)
    if document is None or not _alive(document):
        return None
    user_id = document.user_id
    document.index_status = Index_Status.PROCESSING.value
    session.commit()

    state = "ok"
    chunk_count = 0
    input_tokens = None
    try:
        rows = session.execute(
            select(DocumentPage.page, DocumentPage.text)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page)
        ).all()
        chunks = chunk_pages([Page(number=page, text=text) for page, text in rows])
        if not chunks:
            raise ValueError("The document has no text to index")
        result = embed([chunk.text for chunk in chunks])
        input_tokens = result.input_tokens

        session.refresh(document)  # it may have been deleted while we were embedding
        if not _alive(document):
            state = "deleted"
            return None
        session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        session.add_all(
            DocumentChunk(
                document_id=document_id,
                user_id=user_id,
                page=chunk.page,
                position=chunk.position,
                text=chunk.text,
                embedding=vector,
                embedding_model=result.model,
            )
            for chunk, vector in zip(chunks, result.vectors, strict=True)
        )
        document.index_status = Index_Status.READY.value
        session.commit()
        chunk_count = len(chunks)
        return chunk_count
    except Exception:
        state = "failed"
        session.rollback()
        document = session.get(Document, document_id)
        if document is not None:
            document.index_status = Index_Status.FAILED.value
            session.commit()
        raise
    finally:
        logger.log(
            logging.INFO if state in ("ok", "deleted") else logging.WARNING,
            "Document indexed",
            extra={
                "document_id": document_id,
                "user_id": user_id,
                "status": state,
                "chunk_count": chunk_count,
                "input_tokens": input_tokens,
                "latency_ms": int((time.monotonic() - started) * 1000),
            },
        )


async def retrieve_chunks(
    db_session: AsyncSession,
    *,
    user_id: int,
    document_id: int,
    query_vector: list[float],
    limit: int,
) -> list[RetrievedChunk]:
    """The ``limit`` chunks of this user's document nearest to the vector, nearest first.

    The owner is part of the query, not only checked before it: another user's chunks cannot be
    returned even when the caller passes someone else's document id. Chunks embedded with another
    model than the current one are skipped, since their vectors cannot be compared.
    """
    distance = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
    result = await db_session.execute(
        select(
            DocumentChunk.id,
            DocumentChunk.page,
            DocumentChunk.position,
            DocumentChunk.text,
            distance,
        )
        .where(
            DocumentChunk.user_id == user_id,
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding_model == settings.EMBEDDING_MODEL,
        )
        .order_by(distance, DocumentChunk.id)
        .limit(limit)
    )
    return [
        RetrievedChunk(id=id_, page=page, position=position, text=text, distance=float(dist))
        for id_, page, position, text, dist in result.all()
    ]


async def delete_chunks(document_id: int, db_session: AsyncSession) -> None:
    """Remove the chunks in the open transaction (nothing happens if there are none)."""
    await db_session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))


async def count_chunks(document_id: int, db_session: AsyncSession) -> int:
    result = await db_session.execute(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
    )
    return int(result.scalar_one())
