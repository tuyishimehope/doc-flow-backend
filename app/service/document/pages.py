"""The text of a stored PDF, page by page (table ``document_page``)."""

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import DocumentPage
from app.utils.pdf import Page


def _clean(text: str) -> str:
    # Postgres refuses a NUL character inside text, and some PDFs produce one.
    return text.replace("\x00", "")


async def save_pages(document_id: int, pages: list[Page], db_session: AsyncSession) -> None:
    """Add the pages in the open transaction. The caller commits."""
    db_session.add_all(
        DocumentPage(document_id=document_id, page=page.number, text=_clean(page.text))
        for page in pages
    )
    await db_session.flush()


async def get_pages(document_id: int, db_session: AsyncSession) -> list[Page]:
    """All pages of the document, in order. Empty when it was not stored for questions."""
    result = await db_session.execute(
        select(DocumentPage.page, DocumentPage.text)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page)
    )
    return [Page(number=number, text=text) for number, text in result.all()]


async def delete_pages(document_id: int, db_session: AsyncSession) -> None:
    """Remove the pages in the open transaction (nothing happens if there are none)."""
    await db_session.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))


async def count_pages(document_id: int, db_session: AsyncSession) -> int:
    result = await db_session.execute(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_id == document_id)
    )
    return int(result.scalar_one())
