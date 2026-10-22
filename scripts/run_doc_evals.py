"""Ask the same questions about one PDF two ways and compare them (plan_4, C0 and C7).

    # the server needs the page cap raised for a long PDF and retrieval switched on:
    #   DOCUMENT_QA_MAX_PAGES=120 DOCUMENT_RETRIEVAL_ENABLED=true
    python -m scripts.run_doc_evals --email me@example.com --password '...' \\
        --cases evals/documents.json
    python -m scripts.run_doc_evals --token "$TOKEN" --document-id 12 --modes pages   # C0 baseline
    python -m scripts.run_doc_evals --compare evals/results/docs-<old>.json

``pages`` reads the whole document for every question (the Part A/B way). ``chunks`` embeds the
question and reads only the nearest chunks (Part C). The runner calls the real
``POST /api/v1/documents/{id}/ask`` for each question and mode, so it measures the whole path,
and it costs real model calls: questions x modes. Run it once for the baseline and again after
any change to chunking, the embedding model, ``DOCUMENT_RETRIEVAL_TOP_K`` or a prompt.

The case file is described in evals/README.md. It needs a PDF and questions written from it,
so the repository ships only ``evals/documents.example.json``.

Scoring, per question and mode:

* answerable question (``"page": 12`` or ``"pages": [12, 13]``): it passes when the model said
  ``found``, the keyword checks pass, **and** at least one verified citation is on a listed page;
* unanswerable question (``"page": null``): it passes when the model said it was not found and
  gave no citation. This is the check that a model does not make something up.

Exit code: 0 when every mode reaches ``--min-score`` (default 0), 1 when one does not, 2 when the
runner could not start (login, file, upload, an index that never gets ready).
"""

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from scripts.run_evals import EvalError, check_answer, login

DEFAULT_CASES = Path("evals/documents.json")
DEFAULT_OUT = Path("evals/results")
API = "/api/v1/documents"
MODES = ("pages", "chunks")
MAX_RATE_LIMIT_WAITS = 3


# ---- cases ----


def load_doc_cases(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"Cannot read the cases in {path}: {exc}") from None
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvalError(f"{path} has no cases")
    seen: set[str] = set()
    for case in cases:
        if not case.get("id") or not str(case.get("question", "")).strip():
            raise EvalError(f"Every case needs an id and a question: {case}")
        if case["id"] in seen:
            raise EvalError(f"Two cases have the id {case['id']!r}")
        seen.add(case["id"])
        if "page" not in case and "pages" not in case:
            raise EvalError(f'{case["id"]}: give the page(s) of the answer, or "page": null')
    return data


def expected_pages(case: dict[str, Any]) -> set[int]:
    """The pages that count as right. Empty means the answer is not in the document."""
    if case.get("pages"):
        return {int(page) for page in case["pages"]}
    return set() if case.get("page") is None else {int(case["page"])}


# ---- scoring ----


def score_reply(case: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    """Judge one reply from ``/ask``. Returns the facts and why it failed, if it did."""
    wanted = expected_pages(case)
    cited = sorted({int(c["page"]) for c in reply.get("citations", [])})
    failures: list[str] = []
    if wanted:
        right_page = bool(wanted & set(cited))
        if not reply.get("found"):
            failures.append("said it was not found, but it is in the document")
        failures += check_answer(reply.get("answer") or "", case.get("checks", {}))
        if reply.get("found") and not right_page:
            failures.append(f"cited page(s) {cited or 'none'}, expected one of {sorted(wanted)}")
    else:
        right_page = not cited
        if reply.get("found"):
            failures.append("answered a question the document cannot answer")
        if cited:
            failures.append(f"cited page(s) {cited} for an unanswerable question")
    return {
        "passed": not failures,
        "failures": failures,
        "answerable": bool(wanted),
        "right_page": right_page,
        "found": reply.get("found"),
        "grounded": reply.get("grounded"),
        "cited_pages": cited,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers C0 and C7 ask for, for the results of one mode."""
    answered = [r for r in results if r.get("input_tokens") is not None]
    tokens = [r["input_tokens"] for r in answered]
    seconds = [r["seconds"] for r in results if not r.get("request_failed")]
    passed = sum(1 for r in results if r["passed"])
    answerable = [r for r in results if r.get("answerable")]
    return {
        "passed": passed,
        "total": len(results),
        "score": round(passed / len(results), 4) if results else 0.0,
        "right_page": sum(1 for r in answerable if r["right_page"] and r["passed"]),
        "answerable": len(answerable),
        "refused_correctly": sum(1 for r in results if not r.get("answerable") and r["passed"]),
        "unanswerable": sum(1 for r in results if not r.get("answerable")),
        "request_failures": sum(1 for r in results if r.get("request_failed")),
        "input_tokens_per_question": round(sum(tokens) / len(tokens)) if tokens else None,
        "input_tokens_total": sum(tokens),
        "output_tokens_total": sum(r.get("output_tokens") or 0 for r in answered),
        "seconds_p50": percentile(seconds, 0.5),
        "seconds_p95": percentile(seconds, 0.95),
    }


def compare_modes(summaries: dict[str, dict[str, Any]], min_token_drop: float = 0.5) -> str:
    """A table of the modes, and the C7 rule: keep chunks if tokens dropped a lot and the
    score is not worse. Only the rule is applied here; reading the failures is still yours."""

    def cell(value: Any) -> str:
        return (
            "-" if value is None else (f"{value:.1f}" if isinstance(value, float) else str(value))
        )

    rows = [
        ("passed", lambda s: f"{s['passed']}/{s['total']}"),
        ("right page", lambda s: f"{s['right_page']}/{s['answerable']}"),
        ("refused correctly", lambda s: f"{s['refused_correctly']}/{s['unanswerable']}"),
        ("request failures", lambda s: s["request_failures"]),
        ("input tokens / question", lambda s: s["input_tokens_per_question"]),
        ("seconds p50", lambda s: s["seconds_p50"]),
        ("seconds p95", lambda s: s["seconds_p95"]),
    ]
    modes = list(summaries)
    lines = [f"{'':<26}" + "".join(f"{mode:>14}" for mode in modes)]
    for label, get in rows:
        lines.append(f"{label:<26}" + "".join(f"{cell(get(summaries[m])):>14}" for m in modes))
    if "pages" in summaries and "chunks" in summaries:
        base, new = summaries["pages"], summaries["chunks"]
        if base["input_tokens_per_question"] and new["input_tokens_per_question"]:
            drop = 1 - new["input_tokens_per_question"] / base["input_tokens_per_question"]
            not_worse = new["score"] >= base["score"]
            lines.append("")
            lines.append(
                f"tokens per question: {base['input_tokens_per_question']} -> "
                f"{new['input_tokens_per_question']} ({-drop:+.0%}); "
                f"score: {base['score']:.0%} -> {new['score']:.0%}"
            )
            keep = drop >= min_token_drop and not_worse
            lines.append(
                f"rule (tokens down at least {min_token_drop:.0%}, score not worse): "
                + ("KEEP chunks" if keep else "do NOT keep chunks yet - read the failures")
            )
    return "\n".join(lines)


# ---- talking to the API ----


def _detail(response: httpx.Response) -> str:
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            return str(response.json().get("detail") or response.text[:200])
        except (ValueError, AttributeError):
            pass
    return response.text[:200]


def _send(
    send: Callable[[], httpx.Response], sleep: Callable[[float], None] = time.sleep
) -> httpx.Response:
    """Run a request; wait and try again when the per-user rate limit (429) is hit."""
    response = send()
    for _ in range(MAX_RATE_LIMIT_WAITS):
        if response.status_code != 429:
            break
        sleep(float(response.headers.get("Retry-After", "5")))
        response = send()
    return response


def upload_pdf(client: httpx.Client, path: Path) -> dict[str, Any]:
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"Cannot read the PDF {path}: {exc}") from None
    response = _send(
        lambda: client.post(f"{API}/qa", files={"file": (path.name, content, "application/pdf")})
    )
    if response.status_code != 201:
        raise EvalError(
            f"Upload failed ({response.status_code}): {_detail(response)}. "
            "A long PDF needs DOCUMENT_QA_MAX_PAGES raised (or retrieval on)."
        )
    return response.json()


def wait_until_indexed(
    client: httpx.Client,
    document_id: int,
    *,
    timeout: float,
    sleep: Callable[[float], None] = time.sleep,
    poll: float = 3.0,
) -> None:
    """Chunks mode needs a READY index. Start one when there is none, then wait for it."""
    deadline = time.monotonic() + timeout
    started = False
    while True:
        response = client.get(f"{API}/{document_id}")
        if response.status_code != 200:
            raise EvalError(f"Cannot read document {document_id}: {_detail(response)}")
        status = response.json().get("index_status")
        if status == "READY":
            return
        if status in ("NONE", "FAILED"):
            if started:
                raise EvalError(f"Indexing ended as {status}. Check the worker log.")
            queued = _send(lambda: client.post(f"{API}/{document_id}/index"))
            if queued.status_code != 202:
                raise EvalError(f"Cannot start indexing: {_detail(queued)}")
            started = True
        if time.monotonic() > deadline:
            raise EvalError(f"The index is not ready after {timeout:.0f} s (status {status}).")
        sleep(poll)


def run_case(
    client: httpx.Client,
    document_id: int,
    case: dict[str, Any],
    mode: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {"id": case["id"], "mode": mode, "answer": None}
    try:
        response = _send(
            lambda: client.post(
                f"{API}/{document_id}/ask", json={"question": case["question"], "mode": mode}
            ),
            sleep,
        )
        if response.status_code != 200:
            raise EvalError(f"HTTP {response.status_code}: {_detail(response)}")
        reply = response.json()
        result.update(
            answer=reply.get("answer"),
            input_tokens=reply.get("input_tokens"),
            output_tokens=reply.get("output_tokens"),
            served_mode=reply.get("mode"),
            citations=reply.get("citations", []),
            **score_reply(case, reply),
        )
        if reply.get("mode") != mode:
            result.update(passed=False)
            result["failures"].append(
                f"asked for mode {mode!r}, answered in {reply.get('mode')!r}"
            )
    except (EvalError, httpx.HTTPError) as exc:
        result.update(
            passed=False,
            failures=[f"request failed: {exc}"],
            request_failed=True,
            answerable=bool(expected_pages(case)),
            right_page=False,
        )
    result["seconds"] = round(time.monotonic() - started, 2)
    return result


def run_modes(
    client: httpx.Client,
    document_id: int,
    data: dict[str, Any],
    modes: list[str],
    *,
    pause: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    on_result: Callable[[dict[str, Any]], None] = lambda result: None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    by_mode: dict[str, list[dict[str, Any]]] = {mode: [] for mode in modes}
    for mode in modes:
        for index, case in enumerate(data["cases"]):
            if index and pause:
                sleep(pause)
            by_mode[mode].append(run_case(client, document_id, case, mode, sleep=sleep))
            on_result(by_mode[mode][-1])
    return {
        "started_at": started.isoformat(timespec="seconds"),
        "base_url": str(client.base_url),
        "document_id": document_id,
        "cases_version": data.get("version"),
        "modes": modes,
        "summary": {mode: summarize(results) for mode, results in by_mode.items()},
        "results": by_mode,
    }


def save_doc_run(run: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(run["started_at"]).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"docs-{stamp}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---- command line ----


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reading whole vs retrieved chunks.")
    parser.add_argument("--base-url", default=os.getenv("EVALS_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--token", default=os.getenv("EVALS_TOKEN"))
    parser.add_argument("--email", default=os.getenv("EVALS_EMAIL"))
    parser.add_argument("--password", default=os.getenv("EVALS_PASSWORD"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--pdf", type=Path, help='the PDF (default: "document" in the cases)')
    parser.add_argument("--document-id", type=int, help="use a PDF that is already stored")
    parser.add_argument(
        "--modes", default=",".join(MODES), help="pages, chunks or both (default: both)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--min-token-drop", type=float, default=0.5)
    parser.add_argument("--pause", type=float, default=0.0, help="seconds between questions")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds for one answer")
    parser.add_argument("--index-timeout", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        if not modes or any(m not in MODES for m in modes):
            raise EvalError(f"--modes must be some of {', '.join(MODES)}")
        data = load_doc_cases(args.cases)
        pdf = args.pdf or (args.cases.parent / data["document"] if data.get("document") else None)
        if args.document_id is None and pdf is None:
            raise EvalError('Give --pdf (or "document" in the cases), or --document-id.')
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            token = args.token or (
                login(client, args.email, args.password) if args.email and args.password else None
            )
            if not token:
                raise EvalError("Give --token, or --email and --password.")
            client.headers["Authorization"] = f"Bearer {token}"
            document_id = args.document_id
            if document_id is None:
                assert pdf is not None
                stored = upload_pdf(client, pdf)
                document_id = stored["document_id"]
                print(f"stored {pdf.name}: document {document_id}, {stored['pages']} pages")
            if "chunks" in modes:
                wait_until_indexed(client, document_id, timeout=args.index_timeout)
            run = run_modes(
                client,
                document_id,
                data,
                modes,
                pause=args.pause,
                on_result=lambda r: print(
                    f"{'PASS' if r['passed'] else 'FAIL'}  {r['mode']:<7} {r['id']:<16} "
                    f"{'; '.join(r['failures'])}"
                ),
            )
    except (EvalError, OSError, json.JSONDecodeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = save_doc_run(run, args.out)
    print()
    print(compare_modes(run["summary"], args.min_token_drop))
    print(f"\nsaved: {path}")
    worst = min(s["score"] for s in run["summary"].values())
    return 0 if worst >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
