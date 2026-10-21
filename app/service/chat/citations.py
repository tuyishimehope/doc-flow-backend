"""Check, in code, that the citations of a document answer are real.

A model can invent a page number or a quote that sounds right. So an answer is shown as backed
by the document only when a citation survives this check: its page exists *and* its quote is
found on that page. The model's own claim ("found: true") is never enough.

Matching is deliberately plain: lower case, and every run of whitespace becomes one space (PDF
text has odd line breaks). It does not fix hyphenated line ends, ligatures or typographic
quotes. If the "citations dropped" number in the log is high for answers that look right, that
is the trigger to make it smarter; until then a stricter check only costs a few good citations.
A very short quote matches almost anywhere, so it proves little; that is a known limit.

Nothing here logs the question, the answer, a quote or the page text; only counts.
"""

import logging
from dataclasses import dataclass

from app.service.llm_client.document_qa import Citation, DocAnswer
from app.utils.pdf import Page

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckedCitations:
    citations: list[Citation]  # only the ones that were verified, in the model's order
    grounded: bool  # the answer was found and at least one citation was verified
    dropped: int  # how many the model gave that did not survive (or all, when not found)


def verify_citations(citations: list[Citation], pages: list[Page]) -> list[Citation]:
    """Keep a citation only if its page exists and its quote is on that page.

    An empty quote is dropped (it is "found" everywhere). A repeat of a page and quote already
    kept is dropped. The order the model gave is kept.
    """
    text_by_page = {page.number: _normalize(page.text) for page in pages}
    kept: list[Citation] = []
    seen: set[tuple[int, str]] = set()
    for citation in citations:
        page_text = text_by_page.get(citation.page)
        quote = _normalize(citation.quote)
        if page_text is None or not quote or quote not in page_text:
            continue
        if (citation.page, quote) in seen:
            continue
        seen.add((citation.page, quote))
        kept.append(citation)
    return kept


def check_answer(answer: DocAnswer, pages: list[Page]) -> CheckedCitations:
    """Verify the citations of an answer and decide whether it is grounded.

    ``found: false`` means no citations and not grounded, whatever the model listed. Found with
    every citation dropped is still returned, but not grounded, so the caller can warn the user
    instead of looking sure.
    """
    if not answer.found:
        checked = CheckedCitations(citations=[], grounded=False, dropped=len(answer.citations))
    else:
        kept = verify_citations(answer.citations, pages)
        checked = CheckedCitations(
            citations=kept, grounded=bool(kept), dropped=len(answer.citations) - len(kept)
        )
    logger.log(
        logging.WARNING if answer.found and not checked.grounded else logging.INFO,
        "Document citations checked",
        extra={
            "citations_kept": len(checked.citations),
            "citations_dropped": checked.dropped,
            "grounded": checked.grounded,
        },
    )
    return checked


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
