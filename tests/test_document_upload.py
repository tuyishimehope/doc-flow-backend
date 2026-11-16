"""The upload checks of the processing route (app/utils/document.py), without a database."""

import io
from zipfile import ZipFile

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.utils.document import check_docx, document_policy
from app.utils.parser import DOCX, PDF
from app.utils.upload import UploadRejectedError, read_checked_upload
from tests.test_parser import make_docx
from tests.test_pdf import make_pdf


async def check(content: bytes, filename: str):
    upload = await read_checked_upload(
        UploadFile(file=io.BytesIO(content), filename=filename), document_policy()
    )
    check_docx(upload)
    return upload


@pytest.mark.anyio
async def test_the_type_comes_from_the_bytes_and_the_extension():
    assert (await check(make_pdf(["x"]), "Invoice.PDF")).mime == PDF
    assert (await check(make_docx(["x"]), "letter.docx")).mime == DOCX
    assert (await check(b"\xff\xd8\xff" + b"0" * 20, "scan.jpeg")).mime == "image/jpeg"
    assert (await check(b"II*\x00" + b"0" * 20, "scan.tif")).mime == "image/tiff"


@pytest.mark.anyio
@pytest.mark.parametrize("filename", ["notes.txt", "sheet.xlsx", "no_extension"])
async def test_another_extension_is_415(filename):
    with pytest.raises(UploadRejectedError) as caught:
        await check(b"hello", filename)

    assert caught.value.status_code == 415


@pytest.mark.anyio
async def test_content_that_is_not_what_the_extension_says_is_415():
    with pytest.raises(UploadRejectedError) as caught:
        await check(make_pdf(["x"]), "photo.png")

    assert caught.value.status_code == 415


@pytest.mark.anyio
async def test_a_file_over_the_limit_is_413(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)

    with pytest.raises(UploadRejectedError) as caught:
        await check(b"%PDF-" + b"0" * 200, "big.pdf")

    assert caught.value.status_code == 413


@pytest.mark.anyio
async def test_a_zip_that_is_not_a_word_file_is_422():
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        archive.writestr("hello.txt", "not a document")

    with pytest.raises(UploadRejectedError) as caught:
        await check(out.getvalue(), "fake.docx")

    assert caught.value.status_code == 422


@pytest.mark.anyio
async def test_a_docx_that_unpacks_too_big_is_422(monkeypatch):
    monkeypatch.setattr("app.utils.document.MAX_DOCX_UNPACKED_BYTES", 10)

    with pytest.raises(UploadRejectedError) as caught:
        await check(make_docx(["a long enough paragraph"]), "bomb.docx")

    assert caught.value.status_code == 422
