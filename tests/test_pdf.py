"""Reading a PDF page by page (app/utils/pdf.py)."""

import io

import pypdf
import pytest

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


def encrypted(pdf: bytes) -> bytes:
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(pdf))
    writer.encrypt("secret", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_three_pages_come_back_in_order_with_their_own_text():
    pages = extract_pages(make_pdf(["alpha one", "bravo two", "charlie three"]), max_pages=40)

    assert pages == [
        Page(1, "alpha one"),
        Page(2, "bravo two"),
        Page(3, "charlie three"),
    ]


def test_an_empty_page_keeps_its_number():
    pages = extract_pages(make_pdf(["first", "", "third"]), max_pages=40)

    assert [p.number for p in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert pages[2].text == "third"


def test_exactly_the_cap_is_allowed():
    assert len(extract_pages(make_pdf(["x"] * 40), max_pages=40)) == 40


def test_one_page_over_the_cap_is_413():
    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(make_pdf(["x"] * 41), max_pages=40)

    assert caught.value.status_code == 413
    assert "41" in caught.value.detail


def test_the_cap_is_checked_before_any_text_is_read(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("text was read before the page count was checked")

    monkeypatch.setattr(pypdf.PageObject, "extract_text", boom)

    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(make_pdf(["x"] * 5), max_pages=4)

    assert caught.value.status_code == 413


def test_a_pdf_with_no_text_on_any_page_is_422_and_says_scanned():
    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(make_pdf(["", "", ""]), max_pages=40)

    assert caught.value.status_code == 422
    assert "scanned PDFs are not supported yet" in caught.value.detail


def test_an_encrypted_pdf_is_422():
    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(encrypted(make_pdf(["secret text"])), max_pages=40)

    assert caught.value.status_code == 422
    assert "encrypted" in caught.value.detail


@pytest.mark.parametrize(
    "content",
    [b"", b"not a pdf at all", b"%PDF-1.4\n garbage with no structure", make_pdf(["ok"])[:60]],
    ids=["empty", "text", "header-only", "cut-off"],
)
def test_a_broken_pdf_is_422(content):
    with pytest.raises(UploadRejectedError) as caught:
        extract_pages(content, max_pages=40)

    assert caught.value.status_code == 422
