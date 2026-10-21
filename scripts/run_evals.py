"""Run the chat evals against a running API and keep a scored result for each run.

    python -m scripts.run_evals --email me@example.com --password '...'
    python -m scripts.run_evals --token "$TOKEN" --model groq --min-score 0.9
    python -m scripts.run_evals --compare evals/results/<old>.json evals/results/<new>.json

It calls the real ``POST /api/v1/chat`` like any client would, so it measures the whole
path: prompt, provider, fallback and parsing. That costs a few model calls per run. Run it
after every change of the prompt (``app/prompts/chat_*.md``) or of the model settings, then
compare the new result with the one before. The case format is described in evals/README.md.

Exit code: 0 when the score reaches ``--min-score`` (default 0, which never fails on score),
1 when it does not, 2 when the runner itself could not start (bad login, file not found).
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CASES = Path("evals/chat.json")
DEFAULT_OUT = Path("evals/results")
CHAT_PATH = "/api/v1/chat"
LOGIN_PATH = "/api/v1/users/token"
MAX_RATE_LIMIT_WAITS = 3


class EvalError(Exception):
    """The runner cannot continue (not a failing case: those are just recorded)."""


# ---- cases and checks --------------------------------------------------------------------------


def load_cases(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"Cannot read the cases in {path}: {exc}") from None
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise EvalError(f"{path} has no cases")
    return data


def check_answer(answer: str, checks: dict[str, Any]) -> list[str]:
    """Every check that is present must pass. Returns what failed (empty list: all passed).

    Text is compared ignoring case. Checks: contains_all, contains_any, not_contains, max_words.
    """
    text = answer.lower()
    failures: list[str] = []
    missing = [s for s in checks.get("contains_all", []) if s.lower() not in text]
    if missing:
        failures.append(f"missing: {', '.join(missing)}")
    options = checks.get("contains_any")
    if options and not any(s.lower() in text for s in options):
        failures.append(f"none of: {', '.join(options)}")
    forbidden = [s for s in checks.get("not_contains", []) if s.lower() in text]
    if forbidden:
        failures.append(f"must not contain: {', '.join(forbidden)}")
    limit = checks.get("max_words")
    if limit is not None and len(answer.split()) > limit:
        failures.append(f"{len(answer.split())} words, at most {limit}")
    return failures


# ---- talking to the API ------------------------------------------------------------------------


def login(client: httpx.Client, email: str, password: str) -> str:
    response = client.post(LOGIN_PATH, data={"username": email, "password": password})
    if response.status_code != 200:
        raise EvalError(f"Login failed ({response.status_code}). Check the e-mail and password.")
    return response.json()["access_token"]


def ask(
    client: httpx.Client,
    message: str,
    model: str | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """One question. Waits and retries when the per-user rate limit (429) is hit."""
    body: dict[str, Any] = {"message": message}
    if model:
        body["model"] = model
    for attempt in range(MAX_RATE_LIMIT_WAITS + 1):
        response = client.post(CHAT_PATH, json=body)
        if response.status_code != 429 or attempt == MAX_RATE_LIMIT_WAITS:
            break
        sleep(float(response.headers.get("Retry-After", "5")))
    if response.status_code != 200:
        detail = (
            response.json().get("detail")
            if response.headers.get("content-type", "").startswith("application/json")
            else None
        )
        raise EvalError(f"HTTP {response.status_code}: {detail or response.text[:200]}")
    return response.json()


def run_case(
    client: httpx.Client,
    case: dict[str, Any],
    model: str | None,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "id": case["id"],
        "category": case.get("category", ""),
        "answer": None,
        "model": None,
    }
    try:
        reply = ask(client, case["message"], model, sleep=sleep)
        result.update(answer=reply["answer"], model=reply.get("model"))
        failures = check_answer(reply["answer"], case.get("checks", {}))
    except (EvalError, httpx.HTTPError) as exc:
        failures = [f"request failed: {exc}"]
    result.update(
        passed=not failures, failures=failures, seconds=round(time.monotonic() - started, 2)
    )
    return result


def run_cases(
    client: httpx.Client,
    data: dict[str, Any],
    model: str | None,
    *,
    pause: float = 0,
    sleep: Callable[[float], None] = time.sleep,
    on_result: Callable[[dict[str, Any]], None] = lambda result: None,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    results = []
    for index, case in enumerate(data["cases"]):
        if index and pause:
            sleep(pause)
        results.append(run_case(client, case, model, sleep=sleep))
        on_result(results[-1])
    passed = sum(1 for r in results if r["passed"])
    return {
        "started_at": started.isoformat(timespec="seconds"),
        "base_url": str(client.base_url),
        "prompt": data.get("prompt"),
        "cases_version": data.get("version"),
        "model_choice": model or "auto",
        "passed": passed,
        "total": len(results),
        "score": round(passed / len(results), 4),
        "results": results,
    }


# ---- results on disk and comparison -------------------------------------------------------------


def save_run(run: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(run["started_at"]).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def compare_runs(old: dict[str, Any], new: dict[str, Any]) -> str:
    """A short report: the score change, and which cases got worse or better."""
    before = {r["id"]: r["passed"] for r in old["results"]}
    after = {r["id"]: r["passed"] for r in new["results"]}
    regressed = sorted(i for i in after if before.get(i) is True and after[i] is False)
    fixed = sorted(i for i in after if before.get(i) is False and after[i] is True)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    lines = [
        f"score: {old['score']:.0%} ({old['passed']}/{old['total']}) -> {new['score']:.0%} ({new['passed']}/{new['total']})",
        f"prompt: {old.get('prompt')} -> {new.get('prompt')}   model: {old.get('model_choice')} -> {new.get('model_choice')}",
    ]
    for title, ids in (
        ("got worse", regressed),
        ("got better", fixed),
        ("new cases", added),
        ("removed cases", removed),
    ):
        if ids:
            lines.append(f"{title}: {', '.join(ids)}")
    if not (regressed or fixed or added or removed):
        lines.append("no case changed")
    return "\n".join(lines)


# ---- command line ------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the chat evals against a running API.")
    parser.add_argument("--base-url", default=os.getenv("EVALS_BASE_URL", "http://localhost:8000"))
    parser.add_argument(
        "--token", default=os.getenv("EVALS_TOKEN"), help="a bearer token (or EVALS_TOKEN)"
    )
    parser.add_argument(
        "--email",
        default=os.getenv("EVALS_EMAIL"),
        help="log in with this e-mail (or EVALS_EMAIL)",
    )
    parser.add_argument(
        "--password", default=os.getenv("EVALS_PASSWORD"), help="its password (or EVALS_PASSWORD)"
    )
    parser.add_argument(
        "--model", help="auto, openrouter, gemini, groq or openai (default: automatic)"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="where scored runs are saved"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0, help="exit with 1 below this score (0 to 1)"
    )
    parser.add_argument(
        "--pause", type=float, default=0.0, help="seconds to wait between questions"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="seconds to wait for one answer"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("OLD", "NEW"),
        help="compare two saved runs and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.compare:
            old, new = (json.loads(path.read_text(encoding="utf-8")) for path in args.compare)
            print(compare_runs(old, new))
            return 0
        data = load_cases(args.cases)
        with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
            token = args.token or (
                login(client, args.email, args.password) if args.email and args.password else None
            )
            if not token:
                raise EvalError(
                    "Give --token, or --email and --password (or EVALS_TOKEN / EVALS_EMAIL / EVALS_PASSWORD)."
                )
            client.headers["Authorization"] = f"Bearer {token}"
            run = run_cases(
                client,
                data,
                args.model,
                pause=args.pause,
                on_result=lambda r: print(
                    f"{'PASS' if r['passed'] else 'FAIL'}  {r['id']:<16} {'; '.join(r['failures'])}"
                ),
            )
    except (EvalError, OSError, json.JSONDecodeError, httpx.HTTPError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = save_run(run, args.out)
    print(f"\nscore: {run['score']:.0%} ({run['passed']}/{run['total']})  saved: {path}")
    return 0 if run["score"] >= args.min_score else 1


if __name__ == "__main__":
    sys.exit(main())
