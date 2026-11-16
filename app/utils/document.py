"""Uploads for the processing route: PDF, DOCX and pictures of documents.

``document_policy`` plugs into ``read_checked_upload`` (``app/utils/upload.py``), like
``pdf_policy`` and ``image_policy``. The order in the route is: ``read_checked_upload`` (type,
size, first bytes), ``scan_upload``, then ``check_docx`` for a DOCX, which opens the ZIP and so
runs after the scan. The content itself is read later, by Docling in the worker.
"""

import io
from zipfile import BadZipFile, ZipFile

from app.core.config import settings
from app.utils.parser import DOCX, PDF
from app.utils.upload import CheckedUpload, FileType, UploadPolicy, UploadRejectedError

MAX_DOCX_ENTRIES = 10_000
MAX_DOCX_UNPACKED_BYTES = 50 * 1024 * 1024

_JPEG = FileType("image/jpeg", lambda b: b.startswith(b"\xff\xd8\xff"))
_TIFF = FileType("image/tiff", lambda b: b.startswith((b"II*\x00", b"MM\x00*")))
DOCUMENT_TYPES = {
    "pdf": FileType(PDF, lambda b: b.startswith(b"%PDF-")),
    "docx": FileType(DOCX, lambda b: b.startswith(b"PK\x03\x04")),  # a ZIP container
    "jpg": _JPEG,
    "jpeg": _JPEG,
    "png": FileType("image/png", lambda b: b.startswith(b"\x89PNG\r\n\x1a\n")),
    "tif": _TIFF,
    "tiff": _TIFF,
}


def document_policy() -> UploadPolicy:
    return UploadPolicy(types=DOCUMENT_TYPES, max_bytes=settings.MAX_UPLOAD_SIZE_BYTES)


def check_docx(upload: CheckedUpload) -> None:
    """Refuse a DOCX that is not a Word file, or that unpacks to far more than it weighs.

    Does nothing for the other types. Raises ``UploadRejectedError`` (422).
    """
    if upload.mime != DOCX:
        return
    try:
        with ZipFile(io.BytesIO(upload.content)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            ok = (
                len(entries) <= MAX_DOCX_ENTRIES
                and sum(entry.file_size for entry in entries) <= MAX_DOCX_UNPACKED_BYTES
                and {"[Content_Types].xml", "word/document.xml"} <= names
                and archive.testzip() is None
            )
    except (BadZipFile, OSError):
        ok = False
    if not ok:
        raise UploadRejectedError(422, "This is not a valid Word document.")
