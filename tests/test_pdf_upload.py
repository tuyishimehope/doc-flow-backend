"""A PDF upload goes through the shared checks in the order a document endpoint uses them:
``read_checked_upload`` (type, size, first bytes), ``scan_upload``, then ``extract_pages``.

The virus scan is tested against a tiny fake ``clamd`` that speaks the same INSTREAM protocol, so
no ClamAV is needed. It proves the wiring (an infected file ends as a 422 and the PDF reader never
sees it), not ClamAV's own detection.
"""

import asyncio
import io
import struct

import pypdf
import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.utils.pdf import extract_pages, pdf_policy
from app.utils.scan import scan_upload
from app.utils.upload import UploadRejectedError, read_checked_upload
from tests.test_pdf import make_pdf

# The EICAR test file is harmless, and every antivirus flags it. It is written in two halves so
# this source file does not trip an antivirus on a developer's machine.
EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$" + b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE" + b"!$H+H*"
MB = 1024 * 1024


def upload(content: bytes, filename: str = "report.pdf") -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def check(content: bytes, filename: str = "report.pdf"):
    """What the endpoint does with a file, minus the model call."""
    checked = await read_checked_upload(upload(content, filename), pdf_policy())
    await scan_upload(checked.content)
    return extract_pages(checked.content, max_pages=settings.DOCUMENT_QA_MAX_PAGES)


@pytest.fixture
def ten_mb_limit(monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_QA_MAX_MB", 10)


@pytest.fixture
async def clamd(monkeypatch):
    """A fake scanner on a free local port. Switches scanning on for the test."""
    seen = []  # one entry per scanned file, so a test can tell whether the scanner was asked

    async def handle(reader, writer):
        await reader.readexactly(len(b"zINSTREAM\0"))
        data = bytearray()
        while True:
            (size,) = struct.unpack("!I", await reader.readexactly(4))
            if size == 0:
                break
            data += await reader.readexactly(size)
        seen.append(len(data))
        infected = EICAR in data
        writer.write(b"stream: Eicar-Test-Signature FOUND\0" if infected else b"stream: OK\0")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    monkeypatch.setattr(settings, "CLAMAV_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "CLAMAV_PORT", server.sockets[0].getsockname()[1])
    try:
        yield seen
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.anyio
async def test_a_real_pdf_passes_every_step(ten_mb_limit, clamd):
    pages = await check(make_pdf(["alpha", "bravo"]))

    assert [(p.number, p.text) for p in pages] == [(1, "alpha"), (2, "bravo")]
    assert len(clamd) == 1  # the scanner was asked once


@pytest.mark.anyio
async def test_policy_reports_the_pdf_type_and_the_configured_size(ten_mb_limit):
    checked = await read_checked_upload(upload(make_pdf(["x"]), "Report.PDF"), pdf_policy())

    assert checked.extension == "pdf"
    assert checked.mime == "application/pdf"
    assert pdf_policy().max_bytes == 10 * MB


@pytest.mark.anyio
async def test_an_exe_renamed_to_pdf_is_415(ten_mb_limit):
    fake = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 200  # starts like a Windows program

    with pytest.raises(UploadRejectedError) as caught:
        await check(fake, "invoice.pdf")

    assert caught.value.status_code == 415
    assert "not a real .pdf" in caught.value.detail


@pytest.mark.anyio
async def test_a_pdf_renamed_to_exe_is_415(ten_mb_limit):
    with pytest.raises(UploadRejectedError) as caught:
        await check(make_pdf(["x"]), "invoice.exe")

    assert caught.value.status_code == 415
    assert "pdf" in caught.value.detail  # tells the user what is allowed


@pytest.mark.anyio
async def test_eleven_mb_is_413_and_exactly_ten_mb_is_not(ten_mb_limit):
    header = b"%PDF-1.4\n"
    with pytest.raises(UploadRejectedError) as caught:
        await read_checked_upload(upload(header + b"0" * (11 * MB)), pdf_policy())
    assert caught.value.status_code == 413
    assert "10 MB" in caught.value.detail

    at_the_limit = header + b"0" * (10 * MB - len(header))
    checked = await read_checked_upload(upload(at_the_limit), pdf_policy())
    assert checked.size == 10 * MB


@pytest.mark.anyio
async def test_an_empty_file_is_422(ten_mb_limit):
    with pytest.raises(UploadRejectedError) as caught:
        await check(b"")

    assert caught.value.status_code == 422


@pytest.mark.anyio
async def test_the_eicar_file_is_422_when_scanning_is_on(ten_mb_limit, clamd):
    with pytest.raises(UploadRejectedError) as caught:
        await check(b"%PDF-1.4\n" + EICAR)  # passes the type check, so the scan is what stops it

    assert caught.value.status_code == 422
    assert "virus" in caught.value.detail


@pytest.mark.anyio
async def test_an_infected_file_never_reaches_the_pdf_reader(ten_mb_limit, clamd, monkeypatch):
    real_reader = pypdf.PdfReader
    opened = []

    def spy(*args, **kwargs):  # records the call; an exception here would be turned into a 422
        opened.append(True)
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(pypdf, "PdfReader", spy)

    with pytest.raises(UploadRejectedError) as caught:
        await check(b"%PDF-1.4\n" + EICAR)

    assert caught.value.status_code == 422
    assert "virus" in caught.value.detail
    assert opened == []  # the reader was never opened


@pytest.mark.anyio
async def test_when_the_scanner_is_on_but_down_the_upload_is_503(ten_mb_limit, monkeypatch):
    monkeypatch.setattr(settings, "CLAMAV_HOST", "127.0.0.1")
    monkeypatch.setattr(settings, "CLAMAV_PORT", 1)  # nothing listens here

    with pytest.raises(UploadRejectedError) as caught:
        await check(make_pdf(["x"]))

    assert caught.value.status_code == 503


@pytest.mark.anyio
async def test_the_eicar_file_is_not_flagged_when_scanning_is_off(ten_mb_limit, monkeypatch):
    # With CLAMAV_HOST empty nothing scans. The file is not a readable PDF, so it fails later, at
    # the reader (422 "damaged"), not at the scan. This pins the off switch.
    monkeypatch.setattr(settings, "CLAMAV_HOST", "")

    with pytest.raises(UploadRejectedError) as caught:
        await check(b"%PDF-1.4\n" + EICAR)

    assert caught.value.status_code == 422
    assert "damaged" in caught.value.detail  # stopped by the PDF reader, not by a scan
