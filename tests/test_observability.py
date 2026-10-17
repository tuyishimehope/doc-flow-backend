"""Sentry set-up. Events are captured by a fake transport, nothing is sent anywhere."""

import json
import logging

import pytest
import sentry_sdk
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from sentry_sdk.transport import Transport

from app.core.config import settings
from app.core.observability import _scrub, init_sentry, set_tag, set_user_id
from app.service.llm_client.llm_client import LLMTimeoutError
from tests.test_llm_client import MESSAGES, FakeOpenAI, completion, make_client
from tests.test_llm_fallback import Scripted, chain

SECRET = "TOP SECRET USER TEXT"


class CaptureTransport(Transport):
    """Collects the events Sentry would have sent."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []
        self.logs: list[dict] = []

    def capture_envelope(self, envelope) -> None:
        for item in envelope.items:
            if item.type == "event":
                self.events.append(item.payload.json)
            elif item.type == "log":
                self.logs.extend(item.payload.json["items"])


@pytest.fixture
def sentry(monkeypatch):
    monkeypatch.setattr(settings, "SENTRY_DSN", "https://public@o0.ingest.sentry.io/1")
    monkeypatch.setattr(settings, "SENTRY_TRACES_SAMPLE_RATE", 0)
    transport = CaptureTransport()
    assert init_sentry(transport=transport)
    yield transport
    sentry_sdk.get_current_scope().clear()
    sentry_sdk.get_isolation_scope().clear()
    sentry_sdk.init(dsn=None)  # switch it off again for the other tests


def make_app() -> FastAPI:
    """Created after init, because only apps created afterwards are instrumented."""
    app = FastAPI()

    @app.post("/crash")
    async def crash(body: dict):
        message = SECRET  # a local variable that must never reach Sentry  # noqa: F841
        set_user_id(42)
        raise RuntimeError("unexpected bug")

    @app.post("/status/{code}")
    async def status(code: int):
        raise HTTPException(code, detail="on purpose")

    return app


@pytest.fixture
async def http():
    async with AsyncClient(
        transport=ASGITransport(app=make_app(), raise_app_exceptions=False), base_url="http://test"
    ) as client:
        yield client


# ---- switched on / off ------------------------------------------------------


def test_it_stays_off_without_a_dsn(monkeypatch):
    monkeypatch.setattr(settings, "SENTRY_DSN", "")
    assert init_sentry() is False
    assert not sentry_sdk.get_client().is_active()


def test_it_starts_with_privacy_options_locked_down(sentry):
    client = sentry_sdk.get_client()
    assert client.is_active()
    assert client.options["send_default_pii"] is False
    assert client.options["max_request_body_size"] == "never"
    assert client.options["include_local_variables"] is False


# ---- what is reported -------------------------------------------------------


@pytest.mark.anyio
async def test_an_unexpected_exception_is_reported_once(sentry, http):
    response = await http.post("/crash", json={"message": SECRET})
    assert response.status_code == 500
    assert len(sentry.events) == 1
    assert sentry.events[0]["exception"]["values"][0]["type"] == "RuntimeError"


@pytest.mark.anyio
async def test_user_text_headers_and_locals_never_leave_the_server(sentry, http):
    await http.post(
        "/crash", json={"message": SECRET}, headers={"Authorization": "Bearer abc.def.ghi", "Cookie": "session=xyz"}
    )
    payload = json.dumps(sentry.events[0])
    assert SECRET not in payload  # neither from the request body nor from a local variable
    assert "abc.def.ghi" not in payload and "session=xyz" not in payload


@pytest.mark.anyio
async def test_only_the_numeric_user_id_is_attached(sentry, http):
    await http.post("/crash", json={})
    assert sentry.events[0]["user"] == {"id": "42"}


@pytest.mark.anyio
@pytest.mark.parametrize("code", [400, 401, 404, 422, 429, 502, 503, 504])
async def test_expected_failures_are_not_events(sentry, http, code):
    """Provider outages (502/503/504) are answered on purpose, and logged elsewhere."""
    await http.post(f"/status/{code}")
    assert sentry.events == []


@pytest.mark.anyio
async def test_a_real_500_is_an_event(sentry, http):
    await http.post("/status/500")
    assert len(sentry.events) == 1


# ---- LLM specifics ----------------------------------------------------------


@pytest.mark.anyio
async def test_running_out_of_providers_is_an_event_but_failing_over_is_not(sentry):
    a = Scripted("a", error=LLMTimeoutError("a"))
    b = Scripted("b", error=LLMTimeoutError("b"))
    with pytest.raises(LLMTimeoutError):
        await chain(a, b).complete(MESSAGES)

    assert len(sentry.events) == 1
    event = sentry.events[0]
    assert event["level"] == "error"
    assert "no providers left" in json.dumps(event)


@pytest.mark.anyio
async def test_a_successful_failover_creates_no_event(sentry):
    await chain(Scripted("a", error=LLMTimeoutError("a")), Scripted("b")).complete(MESSAGES)
    assert sentry.events == []


@pytest.mark.anyio
async def test_events_say_which_provider_handled_the_request(sentry):
    await make_client(FakeOpenAI(completion())).complete(MESSAGES)
    sentry_sdk.capture_message("check the tags")
    assert sentry.events[0]["tags"]["llm.provider"] == "openrouter"


def test_helpers_are_harmless_when_sentry_is_off(monkeypatch):
    monkeypatch.setattr(settings, "SENTRY_DSN", "")
    set_user_id(1)
    set_tag("chat.model_choice", "auto")  # must not raise


# ---- the safety net ---------------------------------------------------------


def test_scrub_removes_user_data_even_if_an_option_is_changed_later():
    event = {
        "request": {"data": {"message": SECRET}, "cookies": {"s": "1"}, "headers": {"Authorization": "x", "Accept": "*/*"}},
        "user": {"id": "7", "email": "a@b.c", "ip_address": "1.2.3.4"},
    }
    cleaned = _scrub(event, None)
    assert "data" not in cleaned["request"] and "cookies" not in cleaned["request"]
    assert cleaned["request"]["headers"] == {"Accept": "*/*"}
    assert cleaned["user"] == {"id": "7"}


# ---- logs from the logger config --------------------------------------------


def sent_logs(transport) -> list[dict]:
    sentry_sdk.flush()
    return transport.logs


def attr(log: dict, key: str):
    value = log["attributes"].get(key)
    return value["value"] if isinstance(value, dict) else value


def test_logs_written_through_logging_reach_sentry_with_their_extra_fields(sentry):
    logging.getLogger("app.test").warning(
        "LLM provider failed", extra={"provider": "groq", "status": 503, "latency_ms": 12}
    )
    (log,) = [x for x in sent_logs(sentry) if x["body"] == "LLM provider failed"]
    assert log["level"] == "warn"
    assert attr(log, "provider") == "groq"
    assert attr(log, "status") == 503
    assert attr(log, "latency_ms") == 12


def test_an_error_log_is_both_a_log_and_an_event(sentry):
    logging.getLogger("app.test").error("every provider failed")
    assert any(x["body"] == "every provider failed" for x in sent_logs(sentry))
    assert any(e.get("logentry", {}).get("message") == "every provider failed" for e in sentry.events)


def test_uvicorn_access_lines_are_not_sent_because_they_contain_query_strings(sentry):
    logging.getLogger("uvicorn.access").info('GET /file?token=SECRET HTTP/1.1')
    logging.getLogger("app.test").info("marker")
    bodies = [x["body"] for x in sent_logs(sentry)]
    assert "marker" in bodies
    assert not any("SECRET" in b for b in bodies)


def test_logs_can_be_switched_off_without_switching_errors_off(monkeypatch, sentry):
    sentry_sdk.init(dsn=None)
    monkeypatch.setattr(settings, "SENTRY_ENABLE_LOGS", False)
    transport = CaptureTransport()
    assert init_sentry(transport=transport)
    logging.getLogger("app.test").warning("should stay local")
    logging.getLogger("app.test").error("still an event")
    sentry_sdk.flush()
    assert transport.logs == []
    assert any(e.get("logentry", {}).get("message") == "still an event" for e in transport.events)
