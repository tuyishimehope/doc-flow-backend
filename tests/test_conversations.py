"""Conversations: CRUD, ownership, and memory through the stream endpoint."""

from datetime import datetime

import pytest
from httpx import AsyncClient
from sqlalchemy import update

from app.api.v1.chat.router import get_llm_picker
from app.core.config import settings
from app.main import app
from app.models.schema import Conversation
from app.service.chat.crud import add_message, get_messages
from app.service.chat.schema import MessageRole, MessageStatus
from app.service.llm_client.llm_client import LLMTimeoutError
from tests.conftest import auth_header, create_test_user, login_user
from tests.test_chat_stream import FakeStreamLLM, parse_events

BASE = "/api/v1/chat/conversations"
STREAM = "/api/v1/chat/stream"


@pytest.fixture
async def other_headers(client: AsyncClient) -> dict[str, str]:
    await create_test_user(client, email="other@example.com")
    return auth_header(await login_user(client, email="other@example.com"))


async def make_conversation(client: AsyncClient, headers, title: str | None = None) -> dict:
    response = await client.post(BASE, json={"title": title} if title else {}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


async def ask(client: AsyncClient, headers, message: str, conversation_id: int | None = None):
    body: dict = {"message": message}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return await client.post(STREAM, json=body, headers=headers)


# ---- create / list / get --------------------------------------------------


@pytest.mark.anyio
async def test_create_with_and_without_a_title(client, auth_headers):
    assert (await make_conversation(client, auth_headers, "Trip plans"))["title"] == "Trip plans"
    assert (await make_conversation(client, auth_headers))["title"] == "New conversation"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body", [{"title": ""}, {"title": "   "}, {"title": "x" * 101}, {"name": "typo"}],
    ids=["empty", "blank", "too-long", "unknown-field"],
)
async def test_create_rejects_bad_titles(client, auth_headers, body):
    assert (await client.post(BASE, json=body, headers=auth_headers)).status_code == 422


@pytest.mark.anyio
async def test_every_endpoint_requires_authentication(client):
    assert (await client.post(BASE, json={})).status_code == 401
    assert (await client.get(BASE)).status_code == 401
    assert (await client.get(f"{BASE}/1")).status_code == 401
    assert (await client.patch(f"{BASE}/1", json={"title": "x"})).status_code == 401
    assert (await client.delete(f"{BASE}/1")).status_code == 401


async def set_last_active(db_session, conversation_id: int, day: int) -> None:
    # Inside one test transaction now() never changes, so the order is set by hand.
    await db_session.execute(
        update(Conversation).where(Conversation.id == conversation_id).values(updated_at=datetime(2000, 1, day))
    )


@pytest.mark.anyio
async def test_list_is_paginated_newest_activity_first_and_private(client, auth_headers, other_headers, db_session):
    first = await make_conversation(client, auth_headers, "first")
    second = await make_conversation(client, auth_headers, "second")
    third = await make_conversation(client, auth_headers, "third")
    await make_conversation(client, other_headers, "not yours")
    for day, conversation in enumerate((first, second, third), start=1):
        await set_last_active(db_session, conversation["id"], day)
    # Talking in the oldest conversation makes it the most recently active.
    await add_message(conversation_id=first["id"], role=MessageRole.USER, content="hi", db_session=db_session)

    page = (await client.get(BASE, params={"limit": 2}, headers=auth_headers)).json()
    assert [c["title"] for c in page["conversations"]] == ["first", "third"]
    assert (page["total"], page["has_more"], page["skip"], page["limit"]) == (3, True, 0, 2)

    page = (await client.get(BASE, params={"limit": 2, "skip": 2}, headers=auth_headers)).json()
    assert [c["title"] for c in page["conversations"]] == ["second"]
    assert page["has_more"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 51}, {"skip": -1}])
async def test_list_validates_paging(client, auth_headers, params):
    assert (await client.get(BASE, params=params, headers=auth_headers)).status_code == 422


@pytest.mark.anyio
async def test_get_returns_the_messages_oldest_first(client, auth_headers, db_session):
    conversation = await make_conversation(client, auth_headers, "chat")
    await add_message(conversation_id=conversation["id"], role=MessageRole.USER, content="Q", db_session=db_session)
    await add_message(
        conversation_id=conversation["id"], role=MessageRole.ASSISTANT, content="A",
        model="vendor/model", db_session=db_session,
    )

    body = (await client.get(f"{BASE}/{conversation['id']}", headers=auth_headers)).json()
    assert body["title"] == "chat"
    assert [(m["role"], m["content"], m["status"], m["model"]) for m in body["messages"]] == [
        ("user", "Q", "complete", None),
        ("assistant", "A", "complete", "vendor/model"),
    ]


# ---- rename / delete -------------------------------------------------------


@pytest.mark.anyio
async def test_rename(client, auth_headers):
    conversation = await make_conversation(client, auth_headers, "old")
    response = await client.patch(f"{BASE}/{conversation['id']}", json={"title": "  new name  "}, headers=auth_headers)
    assert response.status_code == 200 and response.json()["title"] == "new name"
    assert (await client.patch(f"{BASE}/{conversation['id']}", json={"title": ""}, headers=auth_headers)).status_code == 422


@pytest.mark.anyio
async def test_delete_removes_the_conversation_and_its_messages(client, auth_headers, db_session):
    conversation = await make_conversation(client, auth_headers)
    await add_message(conversation_id=conversation["id"], role=MessageRole.USER, content="Q", db_session=db_session)

    assert (await client.delete(f"{BASE}/{conversation['id']}", headers=auth_headers)).status_code == 204
    assert (await client.get(f"{BASE}/{conversation['id']}", headers=auth_headers)).status_code == 404
    assert await get_messages(conversation["id"], db_session) == []
    assert (await client.delete(f"{BASE}/{conversation['id']}", headers=auth_headers)).status_code == 404


@pytest.mark.anyio
async def test_other_users_conversations_look_like_they_do_not_exist(client, auth_headers, other_headers, db_session):
    mine = await make_conversation(client, auth_headers, "private")
    await add_message(conversation_id=mine["id"], role=MessageRole.USER, content="secret", db_session=db_session)
    url = f"{BASE}/{mine['id']}"

    assert (await client.get(url, headers=other_headers)).status_code == 404
    assert (await client.patch(url, json={"title": "hacked"}, headers=other_headers)).status_code == 404
    assert (await client.delete(url, headers=other_headers)).status_code == 404
    assert (await client.get(url, headers=auth_headers)).json()["title"] == "private"  # untouched
    assert len(await get_messages(mine["id"], db_session)) == 1


# ---- memory through the stream endpoint ------------------------------------


@pytest.fixture
def fake_llm(chat_client_llm) -> FakeStreamLLM:
    return chat_client_llm


@pytest.fixture
def chat_client_llm() -> FakeStreamLLM:
    return FakeStreamLLM(pieces=("Par", "is."))


@pytest.fixture
async def chat(client, chat_client_llm, db_session, auth_headers):
    from app.api.v1.chat.router import get_session_factory
    from tests.test_chat_stream import SameSession

    app.dependency_overrides[get_llm_picker] = lambda: (lambda model: chat_client_llm)
    app.dependency_overrides[get_session_factory] = lambda: SameSession(db_session)
    return client


@pytest.mark.anyio
async def test_a_new_question_creates_a_titled_conversation_and_stores_both_messages(
    chat, auth_headers, db_session
):
    response = await ask(chat, auth_headers, "  What is the capital of France?  ")
    events = parse_events(response.text)
    conversation_id = events[0][1]["conversation_id"]

    detail = (await chat.get(f"{BASE}/{conversation_id}", headers=auth_headers)).json()
    assert detail["title"] == "What is the capital of France?"
    assert [(m["role"], m["content"], m["status"]) for m in detail["messages"]] == [
        ("user", "What is the capital of France?", "complete"),
        ("assistant", "Paris.", "complete"),
    ]
    assert detail["messages"][1]["model"] == "fake/model"
    stored = await get_messages(conversation_id, db_session)
    assert stored[1].meta == {"first_piece_ms": 20, "attempts": 1}
    assert (stored[1].input_tokens, stored[1].output_tokens, stored[1].latency_ms) == (10, 2, 50)
    assert stored[1].provider is None  # the fake sends no provider label
    assert (stored[1].cached_tokens, stored[1].reasoning_tokens) == (None, None)
    assert stored[0].input_tokens is None  # the user's own message has no usage


@pytest.mark.anyio
async def test_the_model_remembers_earlier_turns(chat, chat_client_llm, auth_headers):
    first = await ask(chat, auth_headers, "My name is Hope.")
    conversation_id = parse_events(first.text)[0][1]["conversation_id"]

    await ask(chat, auth_headers, "What is my name?", conversation_id)

    sent = chat_client_llm.calls[1]
    assert [(m["role"], m["content"]) for m in sent[1:]] == [
        ("user", "My name is Hope."),
        ("assistant", "Paris."),
        ("user", "What is my name?"),
    ]
    assert sent[0]["role"] == "system"


@pytest.mark.anyio
async def test_only_the_latest_messages_are_sent_back(chat, chat_client_llm, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 2)
    first = await ask(chat, auth_headers, "one")
    conversation_id = parse_events(first.text)[0][1]["conversation_id"]
    await ask(chat, auth_headers, "two", conversation_id)
    await ask(chat, auth_headers, "three", conversation_id)

    sent = chat_client_llm.calls[2]
    assert [m["content"] for m in sent[1:]] == ["two", "Paris.", "three"]  # "one" is forgotten


@pytest.mark.anyio
async def test_history_leaves_out_answers_that_were_cut_off(chat, chat_client_llm, auth_headers, db_session):
    conversation = await make_conversation(chat, auth_headers, "chat")
    cid = conversation["id"]
    await add_message(conversation_id=cid, role=MessageRole.USER, content="Q1", db_session=db_session)
    await add_message(
        conversation_id=cid, role=MessageRole.ASSISTANT, content="half an ans",
        status=MessageStatus.CANCELLED, db_session=db_session,
    )
    await add_message(conversation_id=cid, role=MessageRole.USER, content="Q2", db_session=db_session)
    await add_message(conversation_id=cid, role=MessageRole.ASSISTANT, content="A2", db_session=db_session)

    await ask(chat, auth_headers, "Q3", cid)

    assert [m["content"] for m in chat_client_llm.calls[0][1:]] == ["Q1", "Q2", "A2", "Q3"]


@pytest.mark.anyio
async def test_an_empty_untitled_conversation_is_named_after_its_first_message(chat, auth_headers):
    conversation = await make_conversation(chat, auth_headers)
    await ask(chat, auth_headers, "Summarise my contract", conversation["id"])
    assert (await chat.get(f"{BASE}/{conversation['id']}", headers=auth_headers)).json()["title"] == "Summarise my contract"

    await ask(chat, auth_headers, "Another question", conversation["id"])
    assert (await chat.get(f"{BASE}/{conversation['id']}", headers=auth_headers)).json()["title"] == "Summarise my contract"


@pytest.mark.anyio
async def test_a_chosen_title_is_never_replaced(chat, auth_headers):
    conversation = await make_conversation(chat, auth_headers, "My title")
    await ask(chat, auth_headers, "Hello there", conversation["id"])
    assert (await chat.get(f"{BASE}/{conversation['id']}", headers=auth_headers)).json()["title"] == "My title"


@pytest.mark.anyio
async def test_unknown_or_foreign_conversations_are_404_and_the_model_is_not_called(
    chat, chat_client_llm, auth_headers, other_headers
):
    theirs = await make_conversation(chat, other_headers, "theirs")
    assert (await ask(chat, auth_headers, "hi", 999999)).status_code == 404
    assert (await ask(chat, auth_headers, "hi", theirs["id"])).status_code == 404
    assert chat_client_llm.calls == []


@pytest.mark.anyio
async def test_a_failure_before_the_first_piece_keeps_the_question_but_stores_no_answer(
    chat, chat_client_llm, auth_headers, db_session
):
    chat_client_llm.fail_before = LLMTimeoutError("slow")
    conversation = await make_conversation(chat, auth_headers, "chat")

    response = await ask(chat, auth_headers, "hello?", conversation["id"])

    assert response.status_code == 504
    assert [(m.role, m.content) for m in await get_messages(conversation["id"], db_session)] == [("user", "hello?")]


@pytest.mark.anyio
async def test_a_failure_mid_answer_stores_the_text_so_far_as_an_error(chat, chat_client_llm, auth_headers, db_session):
    chat_client_llm.fail_after = LLMTimeoutError("slow")
    response = await ask(chat, auth_headers, "hello?")
    conversation_id = parse_events(response.text)[0][1]["conversation_id"]

    stored = await get_messages(conversation_id, db_session)
    assert [(m.role, m.content, m.status) for m in stored] == [
        ("user", "hello?", "complete"),
        ("assistant", "Paris.", "error"),
    ]
