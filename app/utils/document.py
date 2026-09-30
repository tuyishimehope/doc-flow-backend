import io
import logging
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError
import pytesseract

logger = logging.getLogger(__name__)



def get_file_extension(file: UploadFile) -> str:
    if file.filename is None:
        raise ValueError("Uploaded file has no filename")

    return Path(file.filename).suffix.lstrip(".")

def valid_type_document(file: UploadFile) -> bool:
    ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/png",
    "image/tiff",
    }
    return file.content_type in ALLOWED_CONTENT_TYPES


def validate_document_content(file: UploadFile, content: bytes) -> bool:
    """Check that uploaded bytes match the declared supported document type."""
    content_type = file.content_type
    if not content:
        return False

    if content_type == "application/pdf":
        if not content.startswith(b"%PDF-"):
            return False
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            return len(reader.pages) > 0 and not reader.is_encrypted
        except Exception:
            return False
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
                if len(entries) > 10_000 or sum(entry.file_size for entry in entries) > 50 * 1024 * 1024:
                    return False
                if "[Content_Types].xml" not in archive.namelist() or "word/document.xml" not in archive.namelist():
                    return False
                return archive.testzip() is None
        except (BadZipFile, OSError):
            return False

    return False

def extract_text_from_pdf(file_stream) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_stream.read()))

    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    text = "\n".join(pages).strip()
    if text:
        return text

    # Scanned PDFs contain page images rather than an embedded text layer.
    from pdf2image import convert_from_bytes

    images = convert_from_bytes(file_stream.getvalue())
    return "\n".join(pytesseract.image_to_string(ImageOps.grayscale(page)) for page in images).strip()


def extract_text_from_doc(file_stream) -> str:
    from docx import Document

    doc = Document(io.BytesIO(file_stream.read()))

    text = [para.text for para in doc.paragraphs]
    full_text = "\n".join(text)

    return full_text


def extract_text_from_image(file_stream) -> str:
    try:
        image = Image.open(io.BytesIO(file_stream))
        image = ImageOps.exif_transpose(image)
        image = ImageOps.grayscale(image)
        logger.debug(
            "Image prepared for OCR",
            extra={"image_width": image.width, "image_height": image.height},
        )
        return pytesseract.image_to_string(image)

    except UnidentifiedImageError:
        raise ValueError("Invalid image file")
