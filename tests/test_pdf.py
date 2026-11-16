"""The PDF wrapper of the question endpoints (app/utils/pdf.py).

Docling itself is faked here, so these tests are fast and need no models: they pin how the
parser's answers become HTTP statuses. ``tests/test_parser.py`` runs the real Docling.
"""

import pytest

from app.utils import pdf
from app.utils.parser import ParseError
from app.utils.pdf import Page, extract_pages
from app.utils.upload import UploadRejectedError


def make_pdf(page_texts: list[str]) -> bytes:
    """A small valid PDF with one page per text ("" gives a page with no text).

    Built by hand, so the tests need no PDF library. Texts must not contain parentheses.
    """
    count = len(page_texts)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(count))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> " + f"/Contents {5 + 2 * i} 0 R >>".encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


@pytest.fixture
def parsed(monkeypatch):
    """Make the parser return the given pages (or raise), and record what it was asked."""
    calls = []

    def use(result):
        def fake(content, content_type, max_pages=None):
            calls.append((content_type, max_pages))
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(pdf, "parse", fake)
        return calls

    return use


def test_pages_come_back_from_the_parser_in_order(parsed):
    pages = [Page(1, "alpha one"), Page(2, ""), Page(3, "charlie three")]
    calls = parsed(pages)

    assert extract_pages(make_pdf(["x"]), max_pages=40) == pages
    assert calls == [("application/pdf", 40)]  # the cap goes to the parser, which counts first


def test_too_many_pages_is_413_and_names_the_count(parsed):
    parsed(ParseError("too many", too_many_pages=41))

    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(make_pdf(["x"]), max_pages=40)

    assert caught.value.status_code == 413
    assert "41" in caught.value.detail
    assert "40" in caught.value.detail


def test_a_pdf_the_parser_cannot_read_is_422(parsed):
    parsed(ParseError("damaged"))

    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(b"%PDF-1.4\n garbage", max_pages=40)

    assert caught.value.status_code == 422
    assert "damaged or encrypted" in caught.value.detail


def test_a_pdf_with_no_text_on_any_page_is_422(parsed):
    parsed([Page(1, ""), Page(2, "")])

    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(make_pdf(["", ""]), max_pages=40)

    assert caught.value.status_code == 422
    assert "no readable text" in caught.value.detail
