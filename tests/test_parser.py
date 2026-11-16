"""The real Docling parser (app/utils/parser.py) on small files built in the test.

Marked ``slow``: they load Docling's models (downloaded on the first run), so the default
``pytest`` run skips them. Run them with ``pytest -m slow``. OCR is switched off to keep them
quick. The tests at the end use a fake converter and always run.
"""

import io

import pytest

pytest.importorskip("docling")
slow = pytest.mark.slow

from app.core.config import settings  # noqa: E402
from app.utils import parser  # noqa: E402
from app.utils.parser import DOCX, PDF, Page, ParseError, parse, parse_text  # noqa: E402
from tests.test_pdf import make_pdf  # noqa: E402


@pytest.fixture(scope="module")
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


@slow
def test_pdf_pages_come_back_in_order_with_their_own_text(no_ocr):
    pages = parse(make_pdf(["alpha one", "bravo two", "charlie three"]), PDF, max_pages=40)

    assert pages == [Page(1, "alpha one"), Page(2, "bravo two"), Page(3, "charlie three")]


@slow
def test_an_empty_page_keeps_its_number(no_ocr):
    pages = parse(make_pdf(["first", "", "third"]), PDF, max_pages=40)

    assert [p.number for p in pages] == [1, 2, 3]
    assert pages[1].text == ""
    assert pages[2].text == "third"


@slow
def test_exactly_the_cap_is_allowed(no_ocr):
    assert len(parse(make_pdf(["x"] * 5), PDF, max_pages=5)) == 5


@slow
def test_one_page_over_the_cap_names_the_count(no_ocr):
    with pytest.raises(ParseError) as caught:
        parse(make_pdf(["x"] * 6), PDF, max_pages=5)

    assert caught.value.too_many_pages == 6


@slow
def test_an_encrypted_pdf_cannot_be_read(no_ocr):
    with pytest.raises(ParseError) as caught:
        parse(encrypted(make_pdf(["secret text"])), PDF, max_pages=40)

    assert caught.value.too_many_pages is None


@slow
@pytest.mark.parametrize(
    "content",
    [b"", b"not a pdf at all", b"%PDF-1.4\n garbage with no structure", make_pdf(["ok"])[:60]],
    ids=["empty", "text", "header-only", "cut-off"],
)
def test_a_broken_pdf_cannot_be_read(no_ocr, content):
    with pytest.raises(ParseError):
        parse(content, PDF, max_pages=40)


@slow
def test_a_docx_is_one_page_with_every_paragraph(no_ocr):
    text = parse_text(make_docx(["Invoice 42", "Total due: 100 EUR"]), DOCX)

    assert "Invoice 42" in text
    assert "Total due: 100 EUR" in text


@slow
def test_parse_text_joins_the_pages_of_a_pdf(no_ocr):
    assert parse_text(make_pdf(["alpha", "", "charlie"]), PDF) == "alpha\n\ncharlie"


def test_an_unsupported_type_is_refused_before_docling_runs():
    with pytest.raises(ParseError, match="Unsupported"):
        parse(b"hello", "text/plain")


class FakeConverter:
    """Stands in for Docling: returns a result with the given status and items."""

    def __init__(self, status, items=(), page_count=1):
        from types import SimpleNamespace

        document = SimpleNamespace(
            pages={n: None for n in range(1, page_count + 1)},
            iterate_items=lambda **kwargs: [(item, 0) for item in items],
        )
        self.result = SimpleNamespace(
            status=status, document=document, input=SimpleNamespace(page_count=page_count)
        )

    def convert(self, *args, **kwargs):
        return self.result


def text_item(text: str, spans: list[tuple[int, tuple[int, int]]]):
    from docling_core.types.doc import BoundingBox, DocItemLabel, ProvenanceItem, TextItem

    box = BoundingBox(l=0, t=0, r=1, b=1)
    prov = [ProvenanceItem(page_no=page, bbox=box, charspan=span) for page, span in spans]
    return TextItem(self_ref="#/texts/0", label=DocItemLabel.TEXT, text=text, orig=text, prov=prov)


def test_a_file_read_only_in_part_is_refused(monkeypatch):
    from docling.datamodel.base_models import ConversionStatus

    fake = FakeConverter(ConversionStatus.PARTIAL_SUCCESS, page_count=2)
    monkeypatch.setattr(parser, "_converter", lambda: fake)

    with pytest.raises(ParseError, match="too long"):
        parse(make_pdf(["x", "y"]), PDF)


def test_a_paragraph_over_a_page_break_is_split_along_its_spans(monkeypatch):
    from docling.datamodel.base_models import ConversionStatus

    item = text_item("alpha one bravo two", [(1, (0, 9)), (2, (10, 19))])
    fake = FakeConverter(ConversionStatus.SUCCESS, [item], page_count=2)
    monkeypatch.setattr(parser, "_converter", lambda: fake)

    assert parse(b"%PDF-", PDF) == [Page(1, "alpha one"), Page(2, "bravo two")]


def test_spans_that_do_not_fit_the_text_keep_the_paragraph_on_its_first_page(monkeypatch):
    from docling.datamodel.base_models import ConversionStatus

    item = text_item("alpha one bravo two", [(1, (0, 9)), (2, (10, 40))])
    fake = FakeConverter(ConversionStatus.SUCCESS, [item], page_count=2)
    monkeypatch.setattr(parser, "_converter", lambda: fake)

    assert parse(b"%PDF-", PDF) == [Page(1, "alpha one bravo two"), Page(2, "")]
