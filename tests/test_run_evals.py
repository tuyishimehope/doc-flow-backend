"""The eval runner. The API is replaced by a fake, so no model is called."""

import json
from pathlib import Path

import httpx
import pytest

from scripts import run_evals
from scripts.run_evals import EvalError, check_answer, compare_runs, load_cases, run_case, run_cases, save_run

# ---- checks ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, checks, ok",
    [
        ("The capital is PARIS.", {"contains_all": ["paris"]}, True),  # case is ignored
        ("Paris, France", {"contains_all": ["Paris", "Rome"]}, False),
        ("Rome", {"contains_any": ["Paris", "Rome"]}, True),
        ("Madrid", {"contains_any": ["Paris", "Rome"]}, False),
        ("a safe answer", {"not_contains": ["password"]}, True),
        ("My Password is 1234", {"not_contains": ["password"]}, False),
        ("one two three", {"max_words": 3}, True),
        ("one two three four", {"max_words": 3}, False),
        ("anything", {}, True),
        ("Paris", {"contains_all": ["Paris"], "max_words": 1, "not_contains": ["Rome"]}, True),
    ],
)
def test_check_answer(answer, checks, ok):
    assert (check_answer(answer, checks) == []) is ok


def test_every_failed_check_is_named():
    failures = check_answer("Rome is big and long and wordy", {"contains_all": ["Paris"], "max_words": 3})
    assert len(failures) == 2 and "Paris" in failures[0] and "3" in failures[1]


def test_the_real_cases_file_only_uses_checks_the_runner_understands():
    data = load_cases(Path("evals/chat.json"))
    known = {"contains_all", "contains_any", "not_contains", "max_words"}
    assert len(data["cases"]) == 10
    for case in data["cases"]:
        assert set(case["checks"]) <= known, case["id"]
        assert case["message"].strip()


def test_missing_or_empty_case_files_are_a_clear_error(tmp_path):
    with pytest.raises(EvalError):
        load_cases(tmp_path / "nope.json")
    empty = tmp_path / "empty.json"
    empty.write_text('{"cases": []}')
    with pytest.raises(EvalError):
        load_cases(empty)


# ---- talking to a fake API -----------------------------------------------------------------------

CASES = {
    "version": 1,
    "prompt": "chat_v1",
    "cases": [
        {"id": "a", "category": "fact", "message": "Capital of France?", "checks": {"contains_all": ["Paris"]}},
        {"id": "b", "category": "fact", "message": "Capital of Italy?", "checks": {"contains_all": ["Rome"]}},
    ],
}


class FakeApi:
    def __init__(self, answers: dict[str, str], *, rate_limited_first: int = 0, status: int = 200) -> None:
        self.answers = answers
        self.rate_limited = rate_limited_first
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/token"):
            ok = b"password=right" in request.content
            return httpx.Response(200, json={"access_token": "T"}) if ok else httpx.Response(401, json={})
        if self.rate_limited:
            self.rate_limited -= 1
            return httpx.Response(429, json={"detail": "slow down"}, headers={"Retry-After": "7"})
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "provider down"})
        question = json.loads(request.content)["message"]
        return httpx.Response(200, json={"answer": self.answers[question], "model": "fake/model"})


def client_for(api: FakeApi) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(api), base_url="http://test")


def test_a_run_scores_each_case_and_records_the_answers():
    api = FakeApi({"Capital of France?": "Paris.", "Capital of Italy?": "Milan."})
    run = run_cases(client_for(api), CASES, None)
    assert (run["passed"], run["total"], run["score"]) == (1, 2, 0.5)
    assert [r["passed"] for r in run["results"]] == [True, False]
    assert run["results"][1]["answer"] == "Milan." and run["results"][1]["model"] == "fake/model"
    assert run["prompt"] == "chat_v1" and run["model_choice"] == "auto"


def test_the_model_choice_is_sent_only_when_given():
    api = FakeApi({"Capital of France?": "Paris", "Capital of Italy?": "Rome"})
    run_cases(client_for(api), CASES, "groq")
    assert all(json.loads(r.content)["model"] == "groq" for r in api.requests)
    api = FakeApi({"Capital of France?": "Paris", "Capital of Italy?": "Rome"})
    run_cases(client_for(api), CASES, None)
    assert all("model" not in json.loads(r.content) for r in api.requests)


def test_a_rate_limit_is_waited_out_using_retry_after():
    api = FakeApi({"Capital of France?": "Paris", "Capital of Italy?": "Rome"}, rate_limited_first=2)
    waits: list[float] = []
    run = run_cases(client_for(api), CASES, None, sleep=waits.append)
    assert waits == [7.0, 7.0] and run["score"] == 1.0


def test_an_api_error_fails_that_case_and_the_run_carries_on():
    result = run_case(client_for(FakeApi({}, status=502)), CASES["cases"][0], None)
    assert result["passed"] is False and result["answer"] is None
    assert "502" in result["failures"][0] and "provider down" in result["failures"][0]


def test_a_pause_is_kept_between_questions_but_not_before_the_first():
    api = FakeApi({"Capital of France?": "Paris", "Capital of Italy?": "Rome"})
    waits: list[float] = []
    run_cases(client_for(api), CASES, None, pause=3, sleep=waits.append)
    assert waits == [3]


# ---- saving and comparing ----------------------------------------------------------------------------


def make_run(passes: dict[str, bool], **extra) -> dict:
    results = [{"id": i, "passed": ok, "failures": [], "answer": "x", "model": "m"} for i, ok in passes.items()]
    passed = sum(passes.values())
    return {
        "started_at": "2026-10-19T12:00:00+00:00", "prompt": "chat_v1", "model_choice": "auto",
        "passed": passed, "total": len(passes), "score": passed / len(passes), "results": results, **extra,
    }


def test_a_run_is_saved_under_its_start_time(tmp_path):
    path = save_run(make_run({"a": True}), tmp_path / "results")
    assert path.name == "20261019T120000Z.json"
    assert json.loads(path.read_text())["score"] == 1.0


def test_compare_shows_the_score_and_which_cases_changed():
    old = make_run({"a": True, "b": True, "c": False, "gone": True})
    new = make_run({"a": True, "b": False, "c": True, "fresh": True}, prompt="chat_v2")
    report = compare_runs(old, new)
    assert "75%" in report and "chat_v1 -> chat_v2" in report
    assert "got worse: b" in report and "got better: c" in report
    assert "new cases: fresh" in report and "removed cases: gone" in report


def test_compare_says_so_when_nothing_changed():
    same = make_run({"a": True, "b": False})
    assert "no case changed" in compare_runs(same, same)


# ---- the command line ----------------------------------------------------------------------------------


@pytest.fixture
def cases_file(tmp_path) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(CASES))
    return path


def use_fake_api(monkeypatch, api: FakeApi) -> None:
    real_client = httpx.Client
    monkeypatch.setattr(
        run_evals.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(api), **kw)
    )


def test_main_logs_in_runs_saves_and_exits_with_the_score_gate(monkeypatch, tmp_path, cases_file, capsys):
    api = FakeApi({"Capital of France?": "Paris", "Capital of Italy?": "Milan"})
    use_fake_api(monkeypatch, api)
    args = ["--cases", str(cases_file), "--out", str(tmp_path / "out"), "--email", "a@b.c", "--password", "right"]

    assert run_evals.main(args) == 0  # no gate by default
    assert run_evals.main(args + ["--min-score", "0.9"]) == 1  # 50% is below 90%
    assert run_evals.main(args + ["--min-score", "0.5"]) == 0
    shown = capsys.readouterr().out
    assert "PASS  a" in shown and "FAIL  b" in shown
    chat_calls = [r for r in api.requests if r.url.path.endswith("/chat")]
    assert {r.headers["authorization"] for r in chat_calls} == {"Bearer T"}
    assert len(list((tmp_path / "out").glob("*.json"))) >= 1


def test_main_reports_problems_with_exit_code_2(monkeypatch, tmp_path, cases_file, capsys):
    use_fake_api(monkeypatch, FakeApi({}))
    base = ["--cases", str(cases_file), "--out", str(tmp_path)]
    assert run_evals.main(base) == 2  # no token, no login
    assert run_evals.main(base + ["--email", "a@b.c", "--password", "wrong"]) == 2  # bad login
    assert run_evals.main(["--cases", str(tmp_path / "missing.json"), "--token", "T"]) == 2
    errors = capsys.readouterr().err
    assert "Give --token" in errors and "Login failed" in errors and "Cannot read" in errors


def test_main_compares_two_saved_runs(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text(json.dumps(make_run({"a": True})))
    new.write_text(json.dumps(make_run({"a": False})))
    assert run_evals.main(["--compare", str(old), str(new)]) == 0
    assert "got worse: a" in capsys.readouterr().out
