"""Background task: split a stored PDF into chunks and embed them (plan_4, C4)."""

import asyncio
import logging

from app.db.session import SyncSession
from app.service.document.chunks import index_document
from app.service.llm_client.embedding import EmbeddingResult, embed_texts
from app.service.llm_client.gemini import new_gemini_client
from app.service.llm_client.llm_client import LLMError, LLMNotConfiguredError
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


async def _embed(texts: list[str]) -> EmbeddingResult:
    client = new_gemini_client()
    try:
        return await embed_texts(texts, kind="document", client=client)
    finally:
        await client.close()


def embed_in_new_loop(texts: list[str]) -> EmbeddingResult:
    """The worker is synchronous: run the async embedding call in its own event loop, with a
    client made for this call (an HTTP client must not outlive its loop)."""
    return asyncio.run(_embed(texts))


@celery_app.task(
    bind=True,
    autoretry_for=(LLMError,),
    dont_autoretry_for=(LLMNotConfiguredError, ValueError),  # retrying cannot fix these
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def index_document_task(self, document_id: int) -> int | None:
    """Embed the document's chunks. The document ends READY, or FAILED after the last retry."""
    session = SyncSession()
    try:
        return index_document(document_id, session, embed_in_new_loop)
    finally:
        session.close()
