"""Read the content of an uploaded file with Docling. The only parser in the project.

Every feature that needs the text of a file comes here: the processing worker (PDF, DOCX and
images, as one string) and the question endpoints (PDF, page by page, so an answer can cite a
page). Docling reads the layout (reading order, headings, tables) and runs OCR on pages that
are pictures, so a scanned PDF or a photo of an invoice gives text too.

``parse`` is plain blocking code (layout models and OCR are CPU work). From an endpoint, call it
with ``await asyncio.to_thread(...)``. Run it only after ``read_checked_upload`` and
``scan_upload``: a parser is what a hostile file attacks, so it must see the file last.

One ``DocumentConverter`` is built per process and kept, because building it loads the models.
The models are downloaded once (``docling-tools models download``, done in the Docker image).
OCR is Docling's ``auto`` choice, RapidOCR here (it comes with Docling).
"""

import io
import logging
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff"}

# Docling picks the reader from the file name's extension, so each type gets a fixed name.
_NAMES = {
    PDF: "upload.pdf",
    DOCX: "upload.docx",
    "image/jpeg": "upload.jpg",
    "image/png": "upload.png",
    "image/tiff": "upload.tiff",
}


class ParseError(ValueError):
    """The file could not be read. ``reason`` says why, in words safe to show to the user."""

    def __init__(self, reason: str, too_many_pages: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.too_many_pages = too_many_pages  # the page count, when that was the problem


@dataclass(frozen=True)
class Page:
    number: int  # starts at 1, the same number the PDF viewer shows
    text: str  # empty when the page has no text (a blank page)


def parse(content: bytes, content_type: str, max_pages: int | None = None) -> list[Page]:
    """Return every page in order. An empty page is kept, so the numbers still match the viewer.

    A DOCX has no fixed pages, so it comes back as one page. Raises ``ParseError`` for an
    unsupported type, a file above ``max_pages`` (counted before anything is read), or a file
    that cannot be read (damaged, encrypted).
    """
    name = _NAMES.get(content_type)
    if name is None:
        raise ParseError(f"Unsupported file content type: {content_type}")
    if not content:
        raise ParseError("The file is empty.")

    from docling.datamodel.base_models import ConversionStatus, DocumentStream

    kwargs = {"max_num_pages": max_pages} if max_pages else {}
    try:
        result = _converter().convert(
            DocumentStream(name=name, stream=io.BytesIO(content)),
            raises_on_error=False,
            **kwargs,
        )
    except Exception as exc:  # docling raises many kinds of errors on bad files
        raise ParseError("The file is damaged and cannot be read.") from exc

    page_count = result.input.page_count or 0
    if max_pages and page_count > max_pages:
        raise ParseError(
            f"The file has {page_count} pages; the limit is {max_pages} pages.",
            too_many_pages=page_count,
        )
    if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
        logger.info(
            "Docling could not read a file",
            extra={"content_type": content_type, "status": str(result.status)},
        )
        raise ParseError("The file is damaged or encrypted and cannot be read.")

    by_page = _blocks_by_page(result.document)
    last = max([*result.document.pages, *by_page, 1])
    return [Page(n, "\n\n".join(by_page.get(n, []))) for n in range(1, last + 1)]


def parse_text(content: bytes, content_type: str) -> str:
    """The whole text of a file, pages separated by a blank line."""
    return "\n\n".join(page.text for page in parse(content, content_type) if page.text).strip()


def _blocks_by_page(document) -> dict[int, list[str]]:
    """The text blocks of each page, in reading order.

    Plain text, one block per paragraph, heading or list item; the caller puts a blank line
    between blocks, which is where the chunker splits. No markdown marks, so a quote the model
    copies is found again by the citation check. Tables are the exception: a markdown table keeps
    its rows and columns, which matters for invoices.

    Docling joins a paragraph that runs over a page break into one item with one provenance per
    page, each with the span of the text on that page. The text is split along those spans, so
    every page gets only its own words (``export_to_text(page_no=...)`` would put the whole
    paragraph on its first page). An item with no provenance (a DOCX) belongs to page 1.
    """
    from docling_core.types.doc import TableItem, TextItem

    by_page: dict[int, list[str]] = {}
    for item, _level in document.iterate_items():
        if isinstance(item, TableItem):
            parts = [
                (prov.page_no, item.export_to_markdown(doc=document)) for prov in item.prov[:1]
            ]
            parts = parts or [(1, item.export_to_markdown(doc=document))]
        elif isinstance(item, TextItem):
            if len(item.prov) > 1:
                parts = [(p.page_no, item.text[p.charspan[0] : p.charspan[1]]) for p in item.prov]
            else:
                parts = [(item.prov[0].page_no if item.prov else 1, item.text)]
        else:
            continue
        for page_no, text in parts:
            if text.strip():
                by_page.setdefault(page_no, []).append(text.strip())
    return by_page


@lru_cache(maxsize=1)
def _converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
        WordFormatOption,
    )

    options = PdfPipelineOptions(
        do_ocr=settings.DOCUMENT_OCR_ENABLED,
        do_table_structure=True,
        document_timeout=settings.DOCUMENT_PARSE_TIMEOUT_SECONDS,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.IMAGE],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
            InputFormat.DOCX: WordFormatOption(),
        },
    )
