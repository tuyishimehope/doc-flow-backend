"""The real Docling parser (app/utils/parser.py) on small files built in the test.

The first run downloads Docling's layout model (cached in ~/.cache/docling afterwards). OCR is
switched off so the tests do not depend on an OCR engine being installed.
"""

import io

import pytest

pytest.importorskip("docling")

from app.core.config import settings  # noqa: E402
from app.utils import parser  # noqa: E402
from app.utils.parser import DOCX, PDF, Page, ParseError, parse, parse_text  # noqa: E402
from tests.test_pdf import make_pdf  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def no_ocr():
    old = settings.DOCUMENT_OCR_ENABLED
    settings.DOCUMENT_OCR_ENABLED = False
    parser._converter.cache_clear()
    yield
    settings.DOCUMENT_OCR_ENABLED = old
    parser._converter.cache_clear()


def encrypted(pdf: bytes) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(pdf))
    writer.encrypt("secret", algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def make_docx(paragraphs: list[str]) -> bytes:
    """The smallest DOCX Word and Docling accept: content types, a relationship, the body."""
    from zipfile import ZipFile

    body = "".join(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>" for text in paragraphs)
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" '
            'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.'
            'openxmlformats.org/officeDocument/2006/relationships/officeDocument"/>'
            "</Relationships>"
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}</w:body></w:document>"
        ),
    }
    out = io.BytesIO()
    with ZipFile(out, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return out.getvalue()


def test_pdf_pages_come_back_in_order_with_their_own_text():
    pages = parse(make_pdf(["alpha one", "bravo two", "charlie three"]), PDF, max_pages=40)

    assert pages == [Page(1, "alpha one"), Page(2, "bravo two"), Page(3, "charlie three")]


def test_an_empty_page_keeps_its_number():
    pages = parse(make_pdf(["first", "", "third"]), PDF, max_pages=40)

    assert [p.number for p in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert pages[2].text == "third"


def test_exactly_the_cap_is_allowed():
    assert len(parse(make_pdf(["x"] * 5), PDF, max_pages=5)) == 5


def test_one_page_over_the_cap_names_the_count():
    with pytest.raises(ParseError) as caught:
        parse(make_pdf(["x"] * 6), PDF, max_pages=5)

    assert caught.value.too_many_pages == 6


def test_an_encrypted_pdf_cannot_be_read():
    with pytest.raises(ParseError) as caught:
        parse(encrypted(make_pdf(["secret text"])), PDF, max_pages=40)

    assert caught.value.too_many_pages is None


@pytest.mark.parametrize(
    "content",
    [b"", b"not a pdf at all", b"%PDF-1.4\n garbage with no structure", make_pdf(["ok"])[:60]],
    ids=["empty", "text", "header-only", "cut-off"],
)
def test_a_broken_pdf_cannot_be_read(content):
    with pytest.raises(ParseError):
        parse(content, PDF, max_pages=40)


def test_a_docx_is_one_page_with_every_paragraph():
    text = parse_text(make_docx(["Invoice 42", "Total due: 100 EUR"]), DOCX)

    assert "Invoice 42" in text
    assert "Total due: 100 EUR" in text


def test_parse_text_joins_the_pages_of_a_pdf():
    assert parse_text(make_pdf(["alpha", "", "charlie"]), PDF) == "alpha\n\ncharlie"


def test_an_unsupported_type_is_refused_before_docling_runs():
    with pytest.raises(ParseError, match="Unsupported"):
        parse(b"hello", "text/plain")
