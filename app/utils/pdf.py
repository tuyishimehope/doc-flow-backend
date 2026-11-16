"""PDF uploads for the question endpoints: the upload policy and the page-by-page read.

``pdf_policy`` plugs into ``read_checked_upload`` (``app/utils/upload.py``), the same way
``image_policy`` does for pictures. The order for a document endpoint is: ``read_checked_upload``
(type, size, first bytes), then ``scan_upload`` (``app/utils/scan.py``), then ``extract_pages``,
because the parser is what a hostile file attacks and must see the file last.

The reading itself is Docling's (``app/utils/parser.py``), the one parser of the project. This
module only turns its errors into ``UploadRejectedError`` with the HTTP status the file deserves.
Call ``extract_pages`` with ``await asyncio.to_thread(extract_pages, content, max_pages)``.
"""

from app.core.config import settings
from app.utils.chat_utils import convert_mb_to_bytes
from app.utils.parser import PDF, Page, ParseError, parse
from app.utils.upload import FileType, UploadPolicy, UploadRejectedError

__all__ = ["PDF_TYPES", "Page", "extract_pages", "pdf_policy"]

# A real PDF starts with "%PDF-". The extension and the browser's content type prove nothing.
_PDF = FileType(PDF, lambda b: b.startswith(b"%PDF-"))
PDF_TYPES = {"pdf": _PDF}


def pdf_policy() -> UploadPolicy:
    return UploadPolicy(
        types=PDF_TYPES, max_bytes=convert_mb_to_bytes(settings.DOCUMENT_QA_MAX_MB)
    )


def extract_pages(content: bytes, max_pages: int) -> list[Page]:
    """Return every page in order. An empty page is kept, so the numbers still match the viewer.

    Raises ``UploadRejectedError``: 413 above ``max_pages`` (checked before reading any text),
    422 for an encrypted or damaged PDF, or one with no text on any page (even after OCR).
    """
    try:
        pages = parse(content, PDF, max_pages=max_pages)
    except ParseError as exc:
        if exc.too_many_pages is not None:
            raise UploadRejectedError(
                413, f"This PDF has {exc.too_many_pages} pages; the limit is {max_pages} pages."
            ) from exc
        raise UploadRejectedError(
            422, "This PDF is damaged or encrypted and cannot be read."
        ) from exc

    if not any(page.text for page in pages):
        raise UploadRejectedError(422, "This PDF has no readable text.")
    return pages
