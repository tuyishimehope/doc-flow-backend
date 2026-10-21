"""POST /api/v1/chat/stream. No database, no network: the LLM is a fake."""

import asyncio
import json

import pytest
from httpx import AsyncClient

from app.api.v1.chat.router import get_llm_picker, get_session_factory
from app.main import app
from app.prompts import load_prompt
from app.service.chat import conversation as conversation_module
from app.service.chat.crud import get_messages
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
    StreamDelta,
    StreamDone,
)

URL = "/api/v1/chat/stream"


def done_event() -> StreamDone:
    return StreamDone(
        model="fake/model",
        input_tokens=10,
        output_tokens=2,
        latency_ms=50,
        first_piece_ms=20,
        attempts=1,
    )


class FakeStreamLLM:
    def __init__(
        self,
        pieces=("Par", "is."),
        *,
        fail_before: Exception | None = None,
        fail_after: Exception | None = None,
        hang_after_first: bool = False,
        send_done: bool = True,
    ) -> None:
        self.pieces = pieces
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.hang_after_first = hang_after_first
        self.send_done = send_done
        self.calls: list[list[Message]] = []
        self.closed = False

    async def complete(self, messages):  # not used by the stream endpoint
        raise NotImplementedError

    async def stream(self, messages):
        self.calls.append(messages)
        try:
            if self.fail_before:
                raise self.fail_before
            for index, text in enumerate(self.pieces):
                yield StreamDelta(text)
                if index == 0 and self.hang_after_first:
                    await asyncio.sleep(30)
            if self.fail_after:
                raise self.fail_after
            if self.send_done:
                yield done_event()
        finally:
            self.closed = True


def parse_events(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        assert len(lines) == 2, f"an event must be exactly two lines, got: {block!r}"
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ")
        events.append(
            (lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: ")))
        )
    return events


class SameSession:
    """Lets the 'new session' used for saving be the test's own (rolled-back) session."""

    def __init__(self, session) -> None:
        self.session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, *exc_info) -> bool:
        return False


@pytest.fixture
def fake_llm() -> FakeStreamLLM:
    return FakeStreamLLM()


@pytest.fixture
async def chat_client(client: AsyncClient, fake_llm: FakeStreamLLM, db_session, auth_headers):
    """The shared test client, signed in, with the model replaced by a fake."""
    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: fake_llm)
    app.dependency_overrides[get_session_factory] = lambda: SameSession(db_session)
    client.headers.update(auth_headers)
    return client


@pytest.mark.anyio
async def test_streams_tokens_then_done(chat_client: AsyncClient):
    response = await chat_client.post(URL, json={"message": "Capital of France?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    events = parse_events(response.text)
    assert events[0][0] == "conversation"
    assert isinstance(events[0][1]["conversation_id"], int)
    assert events[1:] == [
        ("token", {"text": "Par"}),
        ("token", {"text": "is."}),
        (
            "done",
            {
                "provider": "",
                "model": "fake/model",
                "input_tokens": 10,
                "output_tokens": 2,
                "latency_ms": 50,
                "first_piece_ms": 20,
            },
        ),
    ]


@pytest.mark.anyio
async def test_sends_the_same_prompt_as_the_non_streaming_endpoint(
    chat_client: AsyncClient, fake_llm
):
    await chat_client.post(URL, json={"message": "  Capital of France?  "})
    assert fake_llm.calls == [
        [
            {"role": "system", "content": load_prompt("chat_v1")},
            {"role": "user", "content": "Capital of France?"},
        ]
    ]


@pytest.mark.anyio
async def test_newlines_and_quotes_in_the_text_cannot_break_the_event_format(
    chat_client, fake_llm
):
    fake_llm.pieces = ('line one\n\nline "two"', "ünï ✓ \\ end")
    response = await chat_client.post(URL, json={"message": "hi"})
    texts = [data["text"] for name, data in parse_events(response.text) if name == "token"]
    assert texts == ['line one\n\nline "two"', "ünï ✓ \\ end"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [{}, {"message": ""}, {"message": "   "}, {"message": "x" * 4001}],
    ids=["missing", "empty", "whitespace", "too-long"],
)
async def test_invalid_bodies_are_rejected_before_streaming_starts(chat_client, fake_llm, body):
    response = await chat_client.post(URL, json=body)
    assert response.status_code == 422
    assert fake_llm.calls == []


@pytest.mark.anyio
async def test_requires_authentication(client: AsyncClient, fake_llm):
    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: fake_llm)
    response = await client.post(URL, json={"message": "hi"})
    assert response.status_code == 401
    assert fake_llm.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (LLMTimeoutError("secret internal detail"), 504),
        (LLMRateLimitError("secret internal detail", retry_after=2.2), 503),
        (LLMUnavailableError("secret internal detail"), 503),
        (LLMError("secret internal detail"), 502),
    ],
    ids=["timeout", "rate-limited", "unavailable", "other"],
)
async def test_failure_before_the_first_piece_is_a_real_http_status(
    chat_client, fake_llm, error, expected_status
):
    fake_llm.fail_before = error
    response = await chat_client.post(URL, json={"message": "hi"})

    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/json")
    assert "secret" not in response.text
    if isinstance(error, LLMRateLimitError):
        assert response.headers["retry-after"] == "3"


@pytest.mark.anyio
async def test_a_stream_with_no_events_is_a_bad_gateway(chat_client, fake_llm):
    fake_llm.pieces = ()
    fake_llm.send_done = False
    response = await chat_client.post(URL, json={"message": "hi"})
    assert response.status_code == 502


@pytest.mark.anyio
async def test_failure_after_the_first_piece_becomes_an_error_event(chat_client, fake_llm):
    fake_llm.fail_after = LLMUnavailableError("secret internal detail")
    response = await chat_client.post(URL, json={"message": "hi"})

    assert response.status_code == 200  # the status line was already sent
    events = parse_events(response.text)
    assert [name for name, _ in events] == ["conversation", "token", "token", "error"]
    assert events[-1][1]["code"] == "unavailable"
    assert events[-1][1]["message"]
    assert "secret" not in response.text


@pytest.mark.anyio
async def test_an_unexpected_crash_mid_stream_still_ends_with_an_error_event(
    chat_client, fake_llm
):
    fake_llm.fail_after = RuntimeError("secret internal detail")
    response = await chat_client.post(URL, json={"message": "hi"})

    events = parse_events(response.text)
    assert events[-1][0] == "error" and events[-1][1]["code"] == "error"
    assert "secret" not in response.text


@pytest.mark.anyio
async def test_closing_the_stream_normally_closes_the_llm_stream(chat_client, fake_llm):
    await chat_client.post(URL, json={"message": "hi"})
    assert fake_llm.closed


@pytest.mark.anyio
async def test_client_disconnect_stops_the_llm_stream_and_keeps_the_partial_answer(
    chat_client, db_session, auth_headers
):
    """Drive the app by hand: send the request, read one piece, then 'close the tab'."""
    fake = FakeStreamLLM(pieces=("Hello", " world"), hang_after_first=True)
    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: fake)

    request_body = json.dumps({"message": "hi"}).encode()
    inbox = [{"type": "http.request", "body": request_body, "more_body": False}]
    tab_closed = asyncio.Event()
    sent: list[dict] = []

    async def receive():
        if inbox:
            return inbox.pop(0)
        await tab_closed.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and b"Hello" in message.get("body", b""):
            tab_closed.set()  # the first piece reached the browser, then the tab closes

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": URL,
        "raw_path": URL.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode()),
            (b"authorization", auth_headers["Authorization"].encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    if conversation_module._background:  # the save runs as its own task
        await asyncio.wait_for(asyncio.gather(*conversation_module._background), timeout=5)

    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 200
    assert fake.closed, "the model call kept running after the client left"

    # The conversation event is the first thing sent, so the id is in the first body chunk.
    first_chunk = next(
        m["body"] for m in sent if m["type"] == "http.response.body" and m.get("body")
    )
    conversation_id = json.loads(first_chunk.decode().split("data: ")[1].split("\n")[0])[
        "conversation_id"
    ]
    saved = await get_messages(conversation_id, db_session)
    assert [(m.role, m.content, m.status) for m in saved] == [
        ("user", "hi", "complete"),
        ("assistant", "Hello", "cancelled"),
    ]


@pytest.mark.anyio
async def test_the_chosen_model_reaches_the_picker_and_the_provider_is_reported(
    chat_client, fake_llm
):
    asked: list = []

    def picker(model):
        asked.append(model)
        return fake_llm

    app.dependency_overrides[get_llm_picker] = lambda: picker
    response = await chat_client.post(URL, json={"message": "hi", "model": "groq"})

    assert asked == ["groq"]
    assert "provider" in parse_events(response.text)[-1][1]


@pytest.mark.anyio
async def test_a_model_that_is_not_available_creates_no_conversation(chat_client, monkeypatch):
    from pydantic import SecretStr

    from app.core.config import settings
    from app.service.llm_client.registry import get_llm_client

    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", SecretStr("secret-key"))
    monkeypatch.setattr(settings, "GROQ_AI_API_KEY", SecretStr(""))
    get_llm_client.cache_clear()
    del app.dependency_overrides[get_llm_picker]  # use the real picker

    response = await chat_client.post(URL, json={"message": "hi", "model": "groq"})

    assert response.status_code == 400
    listing = await chat_client.get("/api/v1/chat/conversations")
    assert listing.json()["total"] == 0
    get_llm_client.cache_clear()
