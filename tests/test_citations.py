"""The citation check: a citation counts only if its page exists and its quote is on that page."""

import json
import logging

import pytest

from app.core.logging_config import JsonLogFormatter
from app.service.chat import citations as citations_module
from app.service.chat.citations import check_answer, verify_citations
from app.service.llm_client.document_qa import Citation, DocAnswer
from app.utils.pdf import Page

PAGES = [
    Page(1, "The notice period is 30 days.\nEither side may end the contract."),
    Page(2, "Salary is paid monthly,   on the last working day."),
    Page(3, ""),  # an empty page still exists
]


def cite(page, quote):
    return Citation(page=page, quote=quote)


def answer(*cites, found=True):
    return DocAnswer(found=found, answer="SECRET ANSWER TEXT", citations=list(cites))


# --- verify_citations -----------------------------------------------------------------------


def test_a_real_quote_on_its_page_is_kept():
    good = cite(1, "The notice period is 30 days.")

    assert verify_citations([good], PAGES) == [good]


def test_a_citation_to_a_page_that_does_not_exist_is_dropped():
    assert verify_citations([cite(99, "The notice period is 30 days.")], PAGES) == []
    assert verify_citations([cite(0, "The notice period is 30 days.")], PAGES) == []
    assert verify_citations([cite(-1, "The notice period is 30 days.")], PAGES) == []


def test_a_made_up_quote_on_a_real_page_is_dropped():
    assert verify_citations([cite(1, "The notice period is 90 days.")], PAGES) == []


def test_a_real_quote_cited_on_the_wrong_page_is_dropped():
    assert verify_citations([cite(2, "The notice period is 30 days.")], PAGES) == []


def test_different_spacing_and_case_are_still_a_match():
    spaced = cite(2, "SALARY is paid monthly, on the\nlast working   day.")
    broken_line = cite(1, "notice period is 30 days. either side")

    assert verify_citations([spaced, broken_line], PAGES) == [spaced, broken_line]


@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_an_empty_quote_is_dropped_even_though_it_is_inside_every_page(quote):
    assert verify_citations([cite(1, quote)], PAGES) == []


def test_a_citation_to_an_empty_page_is_dropped():
    assert verify_citations([cite(3, "anything")], PAGES) == []


def test_repeats_are_removed_and_the_models_order_is_kept():
    first = cite(2, "paid monthly")
    second = cite(1, "30 days")
    again = cite(2, "PAID   monthly")  # same page, same quote after normalizing

    assert verify_citations([first, second, again, first], PAGES) == [first, second]


def test_the_same_quote_on_two_different_pages_is_not_a_repeat():
    pages = [Page(1, "see the annex"), Page(2, "see the annex")]

    kept = verify_citations([cite(1, "the annex"), cite(2, "the annex")], pages)

    assert [c.page for c in kept] == [1, 2]


def test_no_citations_in_no_citations_out():
    assert verify_citations([], PAGES) == []


# --- check_answer ---------------------------------------------------------------------------


def test_found_with_a_verified_citation_is_grounded():
    good, fake = cite(1, "30 days"), cite(99, "invented")

    checked = check_answer(answer(good, fake), PAGES)

    assert checked.citations == [good]
    assert checked.grounded is True
    assert checked.dropped == 1


def test_found_but_every_citation_dropped_is_not_grounded():
    checked = check_answer(answer(cite(99, "x"), cite(1, "never written")), PAGES)

    assert checked.citations == []
    assert checked.grounded is False
    assert checked.dropped == 2


def test_found_with_no_citations_at_all_is_not_grounded():
    checked = check_answer(answer(), PAGES)

    assert (checked.citations, checked.grounded, checked.dropped) == ([], False, 0)


def test_not_found_gives_no_citations_and_is_not_grounded_even_if_the_model_listed_some():
    checked = check_answer(answer(cite(1, "30 days"), found=False), PAGES)

    assert checked.citations == []
    assert checked.grounded is False


# --- the log line ---------------------------------------------------------------------------


def checked_records(caplog):
    return [r for r in caplog.records if r.getMessage() == "Document citations checked"]


def test_one_log_line_with_counts_and_never_the_text(caplog):
    with caplog.at_level(logging.INFO, logger=citations_module.__name__):
        check_answer(answer(cite(1, "30 days"), cite(99, "SECRET QUOTE")), PAGES)

    (record,) = checked_records(caplog)
    assert record.levelno == logging.INFO
    assert (record.citations_kept, record.citations_dropped, record.grounded) == (1, 1, True)
    line = JsonLogFormatter().format(record)  # what is really written to stdout
    data = json.loads(line)
    assert (data["citations_kept"], data["citations_dropped"], data["grounded"]) == (1, 1, True)
    for private in ("SECRET QUOTE", "SECRET ANSWER TEXT", "30 days", "notice period"):
        assert private not in line


def test_found_but_not_grounded_is_a_warning_so_it_stands_out(caplog):
    with caplog.at_level(logging.INFO, logger=citations_module.__name__):
        check_answer(answer(cite(99, "x")), PAGES)

    (record,) = checked_records(caplog)
    assert record.levelno == logging.WARNING


def test_not_found_is_not_a_warning(caplog):
    with caplog.at_level(logging.INFO, logger=citations_module.__name__):
        check_answer(answer(found=False), PAGES)

    (record,) = checked_records(caplog)
    assert record.levelno == logging.INFO
