"""Read a PDF page by page, so that an answer can point to a page.

``extract_text_from_pdf`` in ``app/utils/document.py`` joins every page into one string and the
page numbers are lost; the processing pipeline needs it that way, so it stays as it is. This
module keeps the page number next to its text.

``extract_pages`` is plain blocking code (parsing is CPU work). Call it from an endpoint with
``await asyncio.to_thread(extract_pages, content, max_pages)`` so the server stays responsive.
Problems are raised as ``UploadRejectedError`` with the HTTP status the file deserves.
"""

import io
from dataclasses import dataclass

import pypdf

from app.utils.upload import UploadRejectedError


@dataclass(frozen=True)
class Page:
    number: int  # starts at 1, the same number the PDF viewer shows
    text: str  # empty when the page has no text (a blank page, or only an image)


def extract_pages(content: bytes, max_pages: int) -> list[Page]:
    """Return every page in order. An empty page is kept, so the numbers still match the viewer.

    Raises ``UploadRejectedError``: 413 above ``max_pages`` (checked before reading any text),
    422 for an encrypted, damaged or scanned PDF (no text on any page).
    """
    try:
        pages = _read(content, max_pages)
    except UploadRejectedError:
        raise
    except Exception as exc:  # pypdf raises many kinds of errors on bad files
        raise UploadRejectedError(422, "This PDF is damaged and cannot be read.") from exc

    if not any(page.text for page in pages):
        raise UploadRejectedError(422, "This PDF has no text; scanned PDFs are not supported yet.")
    return pages


def _read(content: bytes, max_pages: int) -> list[Page]:
    reader = pypdf.PdfReader(io.BytesIO(content))
    if reader.is_encrypted:
        raise UploadRejectedError(422, "This PDF is encrypted; remove the password and try again.")

    total = len(reader.pages)
    if total > max_pages:
        raise UploadRejectedError(
            413, f"This PDF has {total} pages; the limit is {max_pages} pages."
        )

    return [
        Page(number=number, text=(page.extract_text() or "").strip())
        for number, page in enumerate(reader.pages, start=1)
    ]
