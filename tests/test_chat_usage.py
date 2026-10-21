"""Token usage: reading it from providers that differ, and storing it on the answer.

The first half needs no database. The second half stores answers and reads them back.
"""

import logging
from types import SimpleNamespace

import pytest

from app.models.schema import Conversation
from app.service.chat.conversation import save_while_streaming
from app.service.chat.crud import add_message, get_messages
from app.service.chat.schema import CallUsage, MessageRole, MessageStatus
from app.service.llm_client.llm_client import StreamDelta, StreamDone
from tests.test_chat_stream import SameSession
from tests.test_llm_client import MESSAGES, FakeOpenAI, make_client
from tests.test_llm_client_stream import MODEL, FakeStream, piece

# ---- reading usage from different providers ------------------------------------------------


def usage(**parts) -> SimpleNamespace:
    """A ``usage`` object; leave a part out and the provider "did not send it"."""
    return SimpleNamespace(**parts)


def full_usage() -> SimpleNamespace:
    return usage(
        prompt_tokens=100,
        completion_tokens=40,
        prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
    )


def completion_with(usage_object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Paris"))],
        usage=usage_object,
        model=MODEL,
    )


def usage_chunk(usage_object) -> SimpleNamespace:
    return SimpleNamespace(model=MODEL, usage=usage_object, choices=[])


async def stream_done(usage_object) -> StreamDone:
    fake = FakeOpenAI(FakeStream([piece("Paris"), usage_chunk(usage_object)]))
    events = [event async for event in make_client(fake).stream(MESSAGES)]
    assert isinstance(events[-1], StreamDone)
    return events[-1]


@pytest.mark.anyio
async def test_complete_reads_cached_and_reasoning_tokens_from_their_own_places():
    result = await make_client(FakeOpenAI(completion_with(full_usage()))).complete(MESSAGES)
    assert (result.input_tokens, result.output_tokens) == (100, 40)
    assert result.cached_tokens == 64  # prompt_tokens_details.cached_tokens
    assert result.reasoning_tokens == 25  # completion_tokens_details.reasoning_tokens


@pytest.mark.anyio
async def test_stream_reads_cached_and_reasoning_tokens_too():
    done = await stream_done(full_usage())
    assert (done.input_tokens, done.output_tokens, done.cached_tokens, done.reasoning_tokens) == (
        100,
        40,
        64,
        25,
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "usage_object",
    [
        usage(prompt_tokens=7, completion_tokens=2),  # no details at all
        usage(
            prompt_tokens=7,
            completion_tokens=2,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
        usage(
            prompt_tokens=7,
            completion_tokens=2,
            prompt_tokens_details=SimpleNamespace(),  # details without the number we want
            completion_tokens_details=SimpleNamespace(),
        ),
    ],
    ids=["missing", "none", "empty"],
)
async def test_a_provider_that_leaves_out_the_details_does_not_break_either_call(usage_object):
    result = await make_client(FakeOpenAI(completion_with(usage_object))).complete(MESSAGES)
    done = await stream_done(usage_object)
    for outcome in (result, done):
        assert (outcome.input_tokens, outcome.output_tokens) == (7, 2)
        assert (outcome.cached_tokens, outcome.reasoning_tokens) == (None, None)


@pytest.mark.anyio
async def test_a_provider_with_no_usage_at_all_gives_unknown_not_zero():
    result = await make_client(FakeOpenAI(completion_with(None))).complete(MESSAGES)
    assert (
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        result.reasoning_tokens,
    ) == (None,) * 4


@pytest.mark.anyio
async def test_numbers_that_are_not_whole_numbers_are_treated_as_unknown():
    odd = usage(
        prompt_tokens="7",
        completion_tokens=True,
        prompt_tokens_details=SimpleNamespace(cached_tokens=2.5),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
    )
    result = await make_client(FakeOpenAI(completion_with(odd))).complete(MESSAGES)
    assert (
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        result.reasoning_tokens,
    ) == (None,) * 4


@pytest.mark.anyio
async def test_the_call_log_carries_the_new_counts_but_never_the_text(caplog):
    caplog.set_level(logging.INFO)
    await make_client(FakeOpenAI(completion_with(full_usage()))).complete(MESSAGES)
    record = next(r for r in caplog.records if r.getMessage() == "LLM call finished")
    assert (record.cached_tokens, record.reasoning_tokens) == (64, 25)  # type: ignore[attr-defined]
    assert "secret user text" not in caplog.text


# ---- storing usage on the answer ------------------------------------------------------------


@pytest.fixture
async def conversation(db_session, client, auth_headers) -> Conversation:
    me = (await client.get("/api/v1/users/me", headers=auth_headers)).json()
    row = Conversation(user_id=me["id"], title="usage")
    db_session.add(row)
    await db_session.flush()
    return row


async def store(
    db_session, conversation, usage_object: CallUsage | None, status=MessageStatus.COMPLETE
):
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content="Paris.",
        status=status,
        model="m",
        usage=usage_object,
        db_session=db_session,
    )
    return (await get_messages(conversation.id, db_session))[-1]


@pytest.mark.anyio
async def test_every_part_of_the_usage_is_stored_as_a_column(db_session, conversation):
    message = await store(
        db_session,
        conversation,
        CallUsage(
            "groq",
            input_tokens=100,
            output_tokens=40,
            cached_tokens=64,
            reasoning_tokens=25,
            latency_ms=830,
        ),
    )
    assert (message.provider, message.input_tokens, message.output_tokens) == ("groq", 100, 40)
    assert (message.cached_tokens, message.reasoning_tokens, message.latency_ms) == (64, 25, 830)


@pytest.mark.anyio
async def test_an_answer_from_a_provider_without_usage_still_saves(db_session, conversation):
    message = await store(db_session, conversation, CallUsage(provider="gemini", latency_ms=410))
    assert message.content == "Paris." and message.status == "complete"
    assert message.provider == "gemini" and message.latency_ms == 410
    assert (message.input_tokens, message.output_tokens) == (None, None)
    assert (message.cached_tokens, message.reasoning_tokens) == (None, None)


@pytest.mark.anyio
async def test_a_message_with_no_usage_at_all_saves_with_empty_columns(db_session, conversation):
    message = await store(db_session, conversation, None)
    columns = (
        message.provider,
        message.input_tokens,
        message.output_tokens,
        message.cached_tokens,
    )
    assert columns == (None,) * 4 and message.latency_ms is None


@pytest.mark.anyio
async def test_an_empty_provider_label_is_stored_as_unknown(db_session, conversation):
    assert (await store(db_session, conversation, CallUsage(provider=""))).provider is None


@pytest.mark.anyio
async def test_the_finished_stream_stores_usage_and_a_cut_off_one_does_not(
    db_session, conversation
):
    finished = StreamDone(
        model="m",
        input_tokens=9,
        output_tokens=3,
        latency_ms=70,
        first_piece_ms=20,
        attempts=2,
        provider="openrouter",
        cached_tokens=4,
    )

    async def events(*items):
        for item in items:
            yield item

    source = events(StreamDelta("Pa"), StreamDelta("ris."), finished)
    first = await source.__anext__()
    async for _ in save_while_streaming(
        first, source, conversation_id=conversation.id, session_factory=SameSession(db_session)
    ):
        pass
    cut = events(StreamDelta("Lon"))  # the stream ends without a ``done`` event
    first = await cut.__anext__()
    async for _ in save_while_streaming(
        first, cut, conversation_id=conversation.id, session_factory=SameSession(db_session)
    ):
        pass

    complete, partial = await get_messages(conversation.id, db_session)
    assert (complete.provider, complete.input_tokens, complete.output_tokens) == (
        "openrouter",
        9,
        3,
    )
    assert (complete.cached_tokens, complete.reasoning_tokens, complete.latency_ms) == (
        4,
        None,
        70,
    )
    assert complete.meta == {"first_piece_ms": 20, "attempts": 2}
    assert partial.status == "error"
    assert (partial.provider, partial.input_tokens, partial.latency_ms) == (None, None, None)
