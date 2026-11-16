import io
from pathlib import Path

from fastapi import UploadFile
from PIL import Image


def get_file_extension(file: UploadFile) -> str:
    if file.filename is None:
        raise ValueError("Uploaded file has no filename")

    return Path(file.filename).suffix.lstrip(".")


ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/png",
    "image/tiff",
}


def valid_type_document(file: UploadFile) -> bool:
    return file.content_type in ALLOWED_CONTENT_TYPES


def validate_document_content(file: UploadFile, content: bytes) -> bool:
    """Check that uploaded bytes match the declared supported document type."""
    content_type = file.content_type
    if not content:
        return False

    if content_type == "application/pdf":
        # Only the signature here: the file is read once, by Docling in the worker, which
        # fails the request with a clear reason if it is damaged or encrypted.
        return content.startswith(b"%PDF-")
    if content_type in {"image/jpeg", "image/png", "image/tiff"}:
        signatures = {
            "image/jpeg": content.startswith(b"\xff\xd8\xff"),
            "image/png": content.startswith(b"\x89PNG\r\n\x1a\n"),
            "image/tiff": content.startswith((b"II*\x00", b"MM\x00*")),
        }
        if not signatures[content_type]:
            return False
        try:
            with Image.open(io.BytesIO(content)) as image:
                image.verify()
            return True
        except Exception:
            return False
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # DOCX is a ZIP container; verify its signature and required document entry.
        from zipfile import BadZipFile, ZipFile

        try:
            with ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if (
                    len(entries) > 10_000
                    or sum(entry.file_size for entry in entries) > 50 * 1024 * 1024
                ):
                    return False
                if (
                    "[Content_Types].xml" not in archive.namelist()
                    or "word/document.xml" not in archive.namelist()
                ):
                    return False
                return archive.testzip() is None
        except (BadZipFile, OSError):
            return False

    return False
