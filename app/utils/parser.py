"""Read the content of an uploaded file with Docling. The only parser in the project.

Every feature that needs the text of a file comes here: the processing worker (PDF, DOCX and
images, as one string) and the question endpoints (PDF, page by page, so an answer can cite a
page). Docling reads the layout (reading order, headings, tables) and runs OCR on pages that
are pictures, so a scanned PDF or a photo of an invoice gives text too.

``parse`` is plain blocking code (layout models and OCR are CPU work). From an endpoint, call it
with ``await asyncio.to_thread(...)``. Run it only after ``read_checked_upload`` and
``scan_upload``: a parser is what a hostile file attacks, so it must see the file last.

One ``DocumentConverter`` is built per process and kept, because building it loads the models.
``_LOCK`` lets one thread at a time build or use it: endpoints call ``parse`` from a thread pool,
and two parses at once would only fight for the same CPU (and could load the models twice).
The models are downloaded once (``docling-tools models download``, done in the Docker image).
OCR is Docling's ``auto`` choice, RapidOCR here (it comes with Docling).
"""

import io
import logging
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Docling picks the reader from the file name's extension, so each type gets a fixed name.
_NAMES = {
    PDF: "upload.pdf",
    DOCX: "upload.docx",
    "image/jpeg": "upload.jpg",
    "image/png": "upload.png",
    "image/tiff": "upload.tiff",
}


class ParseError(ValueError):
    """The file could not be read. The message says why, in words safe to show to the user."""

    def __init__(self, reason: str, too_many_pages: int | None = None):
        super().__init__(reason)
        self.too_many_pages = too_many_pages  # the page count, when that was the problem


@dataclass(frozen=True)
class Page:
    number: int  # starts at 1, the same number the PDF viewer shows
    text: str  # empty when the page has no text (a blank page)


def parse(content: bytes, content_type: str, max_pages: int | None = None) -> list[Page]:
    """Return every page in order. An empty page is kept, so the numbers still match the viewer.

    A DOCX has no fixed pages, so it comes back as one page. Raises ``ParseError`` for an
    unsupported type, a file above ``max_pages`` (counted before anything is read), a file that
    cannot be read (damaged, encrypted), or one that was read only in part (the time limit ran
    out): half a document is never returned as if it were whole.
    """
    name = _NAMES.get(content_type)
    if name is None:
        raise ParseError(f"Unsupported file content type: {content_type}")
    if not content:
        raise ParseError("The file is empty.")

    from docling.datamodel.base_models import ConversionStatus, DocumentStream

    try:
        with _LOCK:
            result = _converter().convert(
                DocumentStream(name=name, stream=io.BytesIO(content)),
                raises_on_error=False,
                max_num_pages=max_pages or sys.maxsize,
            )
    except Exception as exc:  # docling raises many kinds of errors on bad files
        raise ParseError("The file is damaged and cannot be read.") from exc

    page_count = result.input.page_count or 0
    if max_pages and page_count > max_pages:
        raise ParseError(
            f"The file has {page_count} pages; the limit is {max_pages} pages.",
            too_many_pages=page_count,
        )
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        logger.warning(
            "Docling read a file only in part",
            extra={"content_type": content_type, "page_count": page_count},
        )
        raise ParseError("The file took too long to read, or some of its pages failed.")
    if result.status != ConversionStatus.SUCCESS:
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

    Page headers and footers and the text inside figures are included. Docling keeps them apart
    from the body, but an invoice number in a header or a total in a stamp is content here, and
    a repeated footer costs less than a missing one.

    An item with no provenance (a DOCX has no pages) belongs to page 1.
    """
    from docling_core.types.doc import TableItem, TextItem
    from docling_core.types.doc.common.content_layer import ContentLayer

    by_page: dict[int, list[str]] = {}
    items = document.iterate_items(
        traverse_pictures=True,
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE},
    )
    for item, _level in items:
        first_page = item.prov[0].page_no if getattr(item, "prov", None) else 1
        if isinstance(item, TableItem):
            parts = [(first_page, item.export_to_markdown(doc=document))]
        elif isinstance(item, TextItem):
            parts = _split_by_page(item) or [(first_page, item.text)]
        else:
            continue
        for page_no, text in parts:
            if text.strip():
                by_page.setdefault(page_no, []).append(text.strip())
    return by_page


def _split_by_page(item) -> list[tuple[int, str]]:
    """The parts of a paragraph that runs over a page break, each with its own page.

    Docling joins such a paragraph into one item with one provenance per page, each with the
    span of the text on that page (``export_to_text(page_no=...)`` would put it all on the first
    page). Returns nothing for an item on one page, or when the spans do not line up with the
    text; the caller then keeps the item whole on its first page.
    """
    if len(item.prov) < 2:
        return []
    spans = [prov.charspan for prov in item.prov]
    in_order = all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    if spans[0][0] != 0 or spans[-1][1] != len(item.text) or not in_order:
        return []
    return [(prov.page_no, item.text[prov.charspan[0] : prov.charspan[1]]) for prov in item.prov]


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
