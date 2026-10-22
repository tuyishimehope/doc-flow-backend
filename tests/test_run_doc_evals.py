"""The document eval runner. The API is a fake, so no model and no server is needed."""

import json
from pathlib import Path

import httpx
import pytest

from scripts import run_doc_evals
from scripts.run_doc_evals import (
    compare_modes,
    expected_pages,
    load_doc_cases,
    percentile,
    run_case,
    run_modes,
    save_doc_run,
    score_reply,
    summarize,
    upload_pdf,
    wait_until_indexed,
)
from scripts.run_evals import EvalError

FACT = {"id": "fact", "question": "How long?", "page": 12, "checks": {"contains_any": ["30"]}}
TWO = {"id": "two", "question": "Which two?", "pages": [18, 19], "checks": {}}
NOPE = {"id": "nope", "question": "Sabbaticals?", "page": None, "checks": {}}
DATA = {"version": 1, "cases": [FACT, TWO, NOPE]}


def reply(*, found=True, answer="30 days.", pages=(12,), tokens=1000, mode="chunks"):
    return {
        "answer": answer,
        "found": found,
        "grounded": found and bool(pages),
        "citations": [{"page": p, "quote": "q"} for p in pages],
        "input_tokens": tokens,
        "output_tokens": 20,
        "mode": mode,
        "pages": 80,
        "model": "m",
        "conversation_id": 1,
    }


# ---- cases ------------------------------------------------------------------------------------


def test_expected_pages_for_one_many_and_none():
    assert expected_pages(FACT) == {12}
    assert expected_pages(TWO) == {18, 19}
    assert expected_pages(NOPE) == set()


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    return path


def test_the_example_file_loads_and_has_one_unanswerable_case():
    data = load_doc_cases(Path("evals/documents.example.json"))

    assert [expected_pages(c) == set() for c in data["cases"]].count(True) == 1
    assert all(c["question"].strip() for c in data["cases"])


@pytest.mark.parametrize(
    "data",
    [
        {"cases": []},
        {"cases": [{"id": "x", "question": "  ", "page": 1}]},
        {"cases": [{"question": "q", "page": 1}]},
        {"cases": [{"id": "x", "question": "q"}]},  # no page and no "page": null
        {
            "cases": [
                {"id": "x", "question": "q", "page": 1},
                {"id": "x", "question": "r", "page": 2},
            ]
        },
    ],
)
def test_bad_case_files_are_a_clear_error(tmp_path, data):
    with pytest.raises(EvalError):
        load_doc_cases(write(tmp_path, data))


def test_a_missing_case_file_is_a_clear_error(tmp_path):
    with pytest.raises(EvalError):
        load_doc_cases(tmp_path / "nope.json")


# ---- scoring ----------------------------------------------------------------------------------


def test_a_right_answer_on_the_right_page_passes():
    scored = score_reply(FACT, reply())

    assert scored["passed"] and scored["right_page"] and scored["cited_pages"] == [12]


def test_a_right_answer_with_the_wrong_page_fails_and_says_which_pages():
    scored = score_reply(FACT, reply(pages=(3,)))

    assert not scored["passed"] and not scored["right_page"]
    assert "[12]" in scored["failures"][0] and "[3]" in scored["failures"][0]


def test_an_answer_with_no_citation_fails_when_the_document_has_the_answer():
    assert not score_reply(FACT, reply(pages=()))["passed"]


def test_saying_not_found_when_it_is_there_fails():
    scored = score_reply(FACT, reply(found=False, answer="Could not find it.", pages=()))

    assert not scored["passed"]
    assert "not found" in scored["failures"][0]


def test_the_keyword_check_still_applies():
    assert not score_reply(FACT, reply(answer="Six months."))["passed"]


def test_either_page_of_a_two_page_answer_is_right():
    assert score_reply(TWO, reply(pages=(19,)))["passed"]
    assert score_reply(TWO, reply(pages=(18,)))["passed"]
    assert not score_reply(TWO, reply(pages=(20,)))["passed"]


def test_an_unanswerable_question_passes_only_with_not_found_and_no_citation():
    assert score_reply(NOPE, reply(found=False, answer="Not in the parts given.", pages=()))[
        "passed"
    ]
    assert not score_reply(NOPE, reply(found=True, pages=()))["passed"]  # it made something up
    assert not score_reply(NOPE, reply(found=False, pages=(5,)))["passed"]


# ---- numbers ----------------------------------------------------------------------------------


def test_percentile_is_nearest_rank():
    assert percentile([], 0.5) is None
    assert percentile([5.0], 0.95) == 5.0
    values = [float(n) for n in range(1, 21)]
    assert percentile(values, 0.5) == 10.0
    assert percentile(values, 0.95) == 19.0


def result(passed=True, tokens=1000, seconds=1.0, answerable=True, right_page=True, failed=False):
    return {
        "passed": passed,
        "input_tokens": None if failed else tokens,
        "output_tokens": None if failed else 10,
        "seconds": seconds,
        "answerable": answerable,
        "right_page": right_page,
        "request_failed": failed,
    }


def test_summary_counts_pages_refusals_tokens_and_time():
    summary = summarize(
        [
            result(tokens=1000, seconds=1),
            result(passed=False, tokens=3000, seconds=3, right_page=False),
            result(answerable=False, tokens=2000, seconds=2),
            result(failed=True, passed=False, seconds=60),
        ]
    )

    assert (summary["passed"], summary["total"], summary["score"]) == (2, 4, 0.5)
    assert (summary["right_page"], summary["answerable"]) == (1, 3)
    assert (summary["refused_correctly"], summary["unanswerable"]) == (1, 1)
    assert summary["request_failures"] == 1
    assert summary["input_tokens_per_question"] == 2000  # failed requests are not averaged in
    assert summary["input_tokens_total"] == 6000
    assert summary["seconds_p50"] == 2.0  # the 60 s of the failed request is not counted


def test_summary_of_nothing_does_not_divide_by_zero():
    summary = summarize([])

    assert summary["total"] == 0 and summary["input_tokens_per_question"] is None


def summary_of(score_passed, total, tokens):
    return summarize([result(passed=i < score_passed, tokens=tokens) for i in range(total)])


def test_the_c7_rule_keeps_chunks_when_tokens_fall_a_lot_and_the_score_holds():
    text = compare_modes(
        {"pages": summary_of(12, 15, 40_000), "chunks": summary_of(12, 15, 3_000)}
    )

    assert "40000 -> 3000 (-92%)" in text
    assert "KEEP chunks" in text


@pytest.mark.parametrize(
    ("pages", "chunks"),
    [
        ((12, 40_000), (11, 3_000)),  # cheaper but worse
        ((12, 40_000), (12, 30_000)),  # not cheap enough
    ],
)
def test_the_c7_rule_does_not_keep_chunks_otherwise(pages, chunks):
    text = compare_modes(
        {
            "pages": summary_of(pages[0], 15, pages[1]),
            "chunks": summary_of(chunks[0], 15, chunks[1]),
        }
    )

    assert "do NOT keep chunks" in text


def test_one_mode_alone_gives_a_table_and_no_verdict():
    text = compare_modes({"pages": summary_of(10, 15, 40_000)})

    assert "pages" in text and "KEEP" not in text and "do NOT" not in text


# ---- talking to a fake API --------------------------------------------------------------------


class FakeApi:
    """Answers /ask from a table of (question, mode) -> reply, and serves the index status."""

    def __init__(self, replies, *, statuses=("READY",), index_start=202):
        self.replies, self.statuses, self.index_start = replies, list(statuses), index_start
        self.requests: list[httpx.Request] = []
        self.rate_limited = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if request.method == "POST" and path.endswith("/ask"):
            if self.rate_limited:
                self.rate_limited -= 1
                return httpx.Response(429, json={"detail": "slow"}, headers={"Retry-After": "7"})
            body = json.loads(request.content)
            found = self.replies.get((body["question"], body["mode"]))
            if found is None:
                return httpx.Response(409, json={"detail": "It is not indexed"})
            return httpx.Response(200, json=found)
        if request.method == "POST" and path.endswith("/index"):
            return httpx.Response(self.index_start, json={})
        if request.method == "POST" and path.endswith("/qa"):
            return httpx.Response(201, json={"document_id": 5, "pages": 80})
        if request.method == "GET":
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json={"index_status": status})
        raise AssertionError(f"unexpected {request.method} {path}")


def client_for(api) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(api), base_url="http://test")


REPLIES = {
    ("How long?", "pages"): reply(tokens=40_000, mode="pages"),
    ("How long?", "chunks"): reply(tokens=3_000, mode="chunks"),
    ("Which two?", "pages"): reply(pages=(18,), tokens=40_000, mode="pages"),
    ("Which two?", "chunks"): reply(pages=(3,), tokens=3_000, mode="chunks"),  # wrong page
    ("Sabbaticals?", "pages"): reply(
        found=False, answer="Not found.", pages=(), tokens=40_000, mode="pages"
    ),
    ("Sabbaticals?", "chunks"): reply(
        found=False, answer="Not found.", pages=(), tokens=3_000, mode="chunks"
    ),
}


def test_every_question_runs_in_every_mode_and_each_mode_is_summarised():
    api = FakeApi(REPLIES)

    run = run_modes(client_for(api), 5, DATA, ["pages", "chunks"])

    assert run["summary"]["pages"]["passed"] == 3
    assert run["summary"]["chunks"]["passed"] == 2  # the wrong page on "two"
    assert run["summary"]["chunks"]["input_tokens_per_question"] == 3000
    sent = [json.loads(r.content) for r in api.requests]
    assert {(b["question"], b["mode"]) for b in sent} == set(REPLIES)
    assert all("conversation_id" not in b for b in sent)  # each question is a fresh conversation
    assert run["document_id"] == 5 and run["modes"] == ["pages", "chunks"]


def test_a_reply_in_another_mode_than_asked_for_is_a_failure():
    api = FakeApi({("How long?", "chunks"): reply(mode="pages")})

    scored = run_case(client_for(api), 5, FACT, "chunks")

    assert not scored["passed"]
    assert "answered in 'pages'" in scored["failures"][-1]


def test_an_http_error_is_a_failed_case_with_the_reason_not_a_crash():
    scored = run_case(client_for(FakeApi({})), 5, FACT, "chunks")

    assert scored["passed"] is False and scored["request_failed"] is True
    assert "HTTP 409" in scored["failures"][0] and "not indexed" in scored["failures"][0]
    assert scored["answerable"] is True


def test_a_rate_limit_is_waited_out_with_retry_after():
    api = FakeApi(REPLIES)
    api.rate_limited = 2
    waits = []

    scored = run_case(client_for(api), 5, FACT, "chunks", sleep=waits.append)

    assert scored["passed"] and waits == [7.0, 7.0]


def test_the_results_are_saved_under_a_docs_prefix(tmp_path):
    run = run_modes(client_for(FakeApi(REPLIES)), 5, DATA, ["chunks"])

    path = save_doc_run(run, tmp_path / "results")

    assert path.name.startswith("docs-") and path.suffix == ".json"
    assert json.loads(path.read_text())["summary"]["chunks"]["total"] == 3


# ---- upload and index -------------------------------------------------------------------------


def test_the_pdf_is_uploaded_and_the_document_id_comes_back(tmp_path):
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    api = FakeApi({})

    stored = upload_pdf(client_for(api), pdf)

    assert stored["document_id"] == 5
    assert b"book.pdf" in api.requests[0].content


def test_a_missing_pdf_and_a_refused_upload_are_clear_errors(tmp_path):
    with pytest.raises(EvalError, match="Cannot read"):
        upload_pdf(client_for(FakeApi({})), tmp_path / "nope.pdf")

    def too_long(request):
        return httpx.Response(413, json={"detail": "Too many pages"})

    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"x")
    client = httpx.Client(transport=httpx.MockTransport(too_long), base_url="http://test")
    with pytest.raises(EvalError, match="Too many pages.*DOCUMENT_QA_MAX_PAGES"):
        upload_pdf(client, pdf)


def test_an_index_that_is_already_ready_starts_nothing():
    api = FakeApi({}, statuses=["READY"])

    wait_until_indexed(client_for(api), 5, timeout=5, sleep=lambda s: None)

    assert [r.method for r in api.requests] == ["GET"]


def test_a_missing_index_is_started_once_and_waited_for():
    api = FakeApi({}, statuses=["NONE", "PROCESSING", "PROCESSING", "READY"])
    naps = []

    wait_until_indexed(client_for(api), 5, timeout=60, sleep=naps.append)

    assert [r.method for r in api.requests].count("POST") == 1
    assert len(naps) == 3


def test_an_index_that_fails_after_starting_is_an_error():
    api = FakeApi({}, statuses=["NONE", "FAILED"])

    with pytest.raises(EvalError, match="FAILED"):
        wait_until_indexed(client_for(api), 5, timeout=60, sleep=lambda s: None)


def test_when_retrieval_is_off_the_runner_says_so(monkeypatch):
    api = FakeApi({}, statuses=["NONE"], index_start=409)

    with pytest.raises(EvalError, match="Cannot start indexing"):
        wait_until_indexed(client_for(api), 5, timeout=60, sleep=lambda s: None)


def test_an_index_that_never_gets_ready_times_out(monkeypatch):
    clock = iter([0.0, 1.0, 2.0, 1000.0, 1001.0])
    monkeypatch.setattr(run_doc_evals.time, "monotonic", lambda: next(clock))
    api = FakeApi({}, statuses=["PROCESSING"])

    with pytest.raises(EvalError, match="not ready after"):
        wait_until_indexed(client_for(api), 5, timeout=10, sleep=lambda s: None)


# ---- the command line -------------------------------------------------------------------------


def test_main_stops_with_2_and_a_message_on_bad_arguments(capsys, tmp_path):
    assert run_doc_evals.main(["--modes", "nope", "--cases", str(tmp_path / "x.json")]) == 2
    assert "--modes" in capsys.readouterr().err
    cases = write(tmp_path, {"cases": [FACT]})
    assert run_doc_evals.main(["--cases", str(cases), "--modes", "pages"]) == 2
    assert "--pdf" in capsys.readouterr().err
