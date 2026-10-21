"""Logging: JSON on stdout, the queue, the request id, the request line.

No database and no network. Each test builds its own logging setup writing into a StringIO, and the
fixture takes it away again, so other tests see the logging they had before.
"""

import io
import json
import logging
import threading

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from app.core import logging_config
from app.core.logging_config import configure_logging, dropped_records, shutdown_logging
from app.core.request_log import REQUEST_ID_HEADER, RequestLogMiddleware, request_id_var

logger = logging.getLogger("tests.logging")


@pytest.fixture(autouse=True)
def clean_logging():
    root = logging.getLogger()
    before_level = root.level
    uvicorn_state = {
        name: (list(logging.getLogger(name).handlers), logging.getLogger(name).propagate)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access")
    }
    yield
    shutdown_logging(keep_output=False)
    root.setLevel(before_level)
    for name, (handlers, propagate) in uvicorn_state.items():
        logging.getLogger(name).handlers[:] = handlers
        logging.getLogger(name).propagate = propagate


def lines(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


class BlockedStream:
    """A stdout that cannot be written to until ``release`` is called (a full pipe)."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.text = io.StringIO()

    def write(self, data: str) -> int:
        self.released.wait(timeout=10)
        return self.text.write(data)

    def flush(self) -> None:
        pass


@pytest.fixture
def stuck():
    stream = BlockedStream()
    yield stream
    stream.released.set()  # never leave a thread waiting, even when the test failed


def log_lines(count: int) -> bool:
    """Log from another thread. False if it is still stuck after 3 seconds (so a bug fails, not hangs)."""
    worker = threading.Thread(target=lambda: [logger.info("line %s", n) for n in range(count)], daemon=True)
    worker.start()
    worker.join(timeout=3)
    return not worker.is_alive()


# ---- the JSON line -----------------------------------------------------------------------------


@pytest.mark.parametrize("use_queue", [False, True])
def test_json_line_has_whitelisted_fields_and_the_traceback(use_queue):
    out = io.StringIO()
    configure_logging(use_queue=use_queue, stream=out)
    logger.info("hello %s", "world", extra={"user_id": 7, "latency_ms": 12, "question": "secret text"})
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("it failed")
    shutdown_logging()

    first, second = lines(out)
    assert first["message"] == "hello world"
    assert first["level"] == "INFO" and first["logger"] == "tests.logging"
    assert first["user_id"] == 7 and first["latency_ms"] == 12
    assert "question" not in first and "secret" not in out.getvalue()  # not on the whitelist
    assert second["level"] == "ERROR" and "ZeroDivisionError" in second["exception"]


def test_log_level_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    out = io.StringIO()
    configure_logging(stream=out)
    logger.info("quiet")
    logger.warning("loud")
    assert [line["message"] for line in lines(out)] == ["loud"]


def test_noisy_libraries_are_quiet():
    out = io.StringIO()
    configure_logging(stream=out)
    logging.getLogger("httpx").info("HTTP Request: POST https://example.test/v1 200 OK")
    logging.getLogger("httpx").warning("real problem")
    assert [line["message"] for line in lines(out)] == ["real problem"]


# ---- the queue ---------------------------------------------------------------------------------


def test_logging_does_not_wait_for_a_stuck_output(stuck):
    configure_logging(use_queue=True, stream=stuck)  # type: ignore[arg-type]
    assert log_lines(20), "logging waited for the stuck output"
    assert stuck.text.getvalue() == ""

    stuck.released.set()
    shutdown_logging()
    assert [line["message"] for line in lines(stuck.text)] == [f"line {n}" for n in range(20)]


def test_a_full_queue_drops_records_instead_of_blocking(stuck):
    configure_logging(use_queue=True, stream=stuck, max_queue_size=5)  # type: ignore[arg-type]
    assert log_lines(50), "logging waited for room in a full queue"
    dropped = dropped_records()
    assert dropped >= 40  # one record is held by the writer, five wait in the queue

    stuck.released.set()
    shutdown_logging()
    assert len(lines(stuck.text)) == 50 - dropped


def test_configuring_twice_keeps_one_handler():
    out = io.StringIO()
    configure_logging(stream=out)
    configure_logging(use_queue=True, stream=out)
    configure_logging(use_queue=True, stream=out)
    logger.info("once")
    shutdown_logging()
    assert [line["message"] for line in lines(out)] == ["once"]


def test_a_line_logged_during_shutdown_is_not_lost():
    out = io.StringIO()
    configure_logging(use_queue=True, stream=out)
    shutdown_logging()  # what the exit hook does
    logger.info("late")
    assert [line["message"] for line in lines(out)] == ["late"]


def test_queue_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("LOG_QUEUE", "false")
    assert logging_config.queue_logs_enabled() is False
    monkeypatch.delenv("LOG_QUEUE")
    assert logging_config.queue_logs_enabled() is True


# ---- uvicorn -----------------------------------------------------------------------------------


def test_uvicorn_lines_use_our_handler_and_the_access_log_is_off():
    out = io.StringIO()
    old_handler = logging.StreamHandler(io.StringIO())  # what uvicorn installs before the app loads
    for name in ("uvicorn", "uvicorn.access"):
        logging.getLogger(name).handlers[:] = [old_handler]
        logging.getLogger(name).propagate = False

    configure_logging(stream=out)
    logging.getLogger("uvicorn.error").info("Started server process [1]")
    logging.getLogger("uvicorn.access").info('127.0.0.1 - "GET /x?token=abc HTTP/1.1" 200')

    result = lines(out)
    assert [(line["logger"], line["message"]) for line in result] == [
        ("uvicorn.error", "Started server process [1]")
    ]
    assert "token=abc" not in out.getvalue()


# ---- the request id and the request line -------------------------------------------------------


def make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLogMiddleware)

    @app.get("/items/{item_id}")
    def item(item_id: int):
        logger.info("inside the handler")
        return {"id": item_id}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/down")
    def down():
        raise HTTPException(status_code=503, detail="try later")

    @app.get("/stream")
    def stream():
        def pieces():
            for piece in ("a", "b", "c"):
                logger.info("piece %s", piece)
                yield piece

        return StreamingResponse(pieces(), media_type="text/plain")

    return app


@pytest.fixture
def web():
    out = io.StringIO()
    configure_logging(stream=out)
    return TestClient(make_app()), out


def request_lines(out: io.StringIO) -> list[dict]:
    return [line for line in lines(out) if line["message"] == "Request finished"]


def test_every_response_has_a_request_id_and_it_is_in_the_handler_log(web):
    client, out = web
    response = client.get("/items/42?secret=abc")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 32

    inside, finished = lines(out)
    assert inside["message"] == "inside the handler" and inside["request_id"] == request_id
    assert finished["request_id"] == request_id
    assert request_id_var.get() is None  # nothing leaks into the next request


def test_the_request_line_has_the_route_template_and_no_url_details(web):
    client, out = web
    client.get("/items/42?secret=abc")
    (finished,) = request_lines(out)
    assert finished["method"] == "GET" and finished["status"] == 200
    assert finished["path"] == "/items/{item_id}"
    assert isinstance(finished["latency_ms"], int)
    assert "/items/42" not in out.getvalue()  # the real URL is never written
    assert "secret" not in out.getvalue()  # nor its query string


def test_a_sane_id_from_the_caller_is_kept(web):
    client, out = web
    response = client.get("/items/1", headers={REQUEST_ID_HEADER: "frontend-req_2026.10.21"})
    assert response.headers[REQUEST_ID_HEADER] == "frontend-req_2026.10.21"
    assert all(line["request_id"] == "frontend-req_2026.10.21" for line in lines(out))


@pytest.mark.parametrize("bad", ["short", "has space in it 12345", "x" * 65, "semi;colon-12345"])
def test_a_bad_id_from_the_caller_is_replaced(web, bad):
    client, out = web
    response = client.get("/items/1", headers={REQUEST_ID_HEADER: bad})
    assert response.headers[REQUEST_ID_HEADER] != bad
    assert bad not in out.getvalue()


def test_unknown_urls_are_logged_as_unmatched(web):
    client, out = web
    assert client.get("/no/such/page/12345").status_code == 404
    (finished,) = request_lines(out)
    assert finished["path"] == "unmatched" and finished["status"] == 404 and finished["level"] == "INFO"
    assert "/no/such/page" not in out.getvalue()


def test_a_503_is_a_warning_not_an_error(web):
    client, out = web
    client.get("/down")
    (finished,) = request_lines(out)
    assert finished["status"] == 503 and finished["level"] == "WARNING"  # an ERROR would reach Sentry


def test_health_checks_are_quiet(web):
    client, out = web
    assert client.get("/health").status_code == 200
    assert request_lines(out) == []


def test_a_stream_is_logged_once_at_the_end_and_keeps_the_id(web):
    client, out = web
    response = client.get("/stream")
    assert response.text == "abc"
    result = lines(out)
    assert [line["message"] for line in result] == ["piece a", "piece b", "piece c", "Request finished"]
    assert {line["request_id"] for line in result} == {response.headers[REQUEST_ID_HEADER]}


def test_the_request_id_reaches_the_log_line_when_the_queue_writes_it():
    out = io.StringIO()
    configure_logging(use_queue=True, stream=out)
    response = TestClient(make_app()).get("/items/1")
    shutdown_logging()
    result = lines(out)
    assert len(result) == 2
    assert {line["request_id"] for line in result} == {response.headers[REQUEST_ID_HEADER]}
