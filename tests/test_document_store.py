"""Part B of plan_4: keep a PDF once, ask about it many times, open it again, delete it.

Covers ``POST /documents/qa`` (B1), ``GET /documents/{id}/file`` (B2), ``POST /documents/{id}/ask``
(B3), follow-ups in the same conversation (B4) and what a delete removes (B5). Real routes and
database; MinIO and the Gemini client are fakes.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from minio.error import S3Error
from openai import APITimeoutError
from sqlalchemy import func, select

from app.core.config import settings
from app.models.schema import Conversation, Document, DocumentPage, File, Message
from app.service.document.pages import delete_pages, get_pages, save_pages
from app.service.llm_client import document_qa
from app.service.llm_client.document_qa import Citation, DocAnswer
from app.utils.pdf import Page
from tests.conftest import auth_header, create_test_user, login_user
from tests.test_document_qa import REQUEST, FakeGemini, completion
from tests.test_pdf import make_pdf
from tests.test_pdf_upload import EICAR, clamd  # noqa: F401  (a fixture, used via usefixtures)

API = "/api/v1/documents"
STORE_URL = f"{API}/qa"
QUESTION = "How long is the notice period?"
PAGE_TEXTS = ["The notice period is 30 days.", "Salary is paid monthly."]
PDF = make_pdf(PAGE_TEXTS)
GOOD_CITE = Citation(page=1, quote="The notice period is 30 days.")


def answer(text="30 days.", *cites, found=True):
    return DocAnswer(found=found, answer=text, citations=list(cites))


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini(completion(answer("30 days.", GOOD_CITE)))
    monkeypatch.setattr(document_qa, "gemini_client", lambda: fake)
    return fake


@pytest.fixture
async def other_headers(client) -> dict[str, str]:
    await create_test_user(client, email="other@example.com")
    return auth_header(await login_user(client, email="other@example.com"))


async def store(client, headers, *, pdf=PDF, name="contract.pdf", content_type="application/pdf"):
    return await client.post(STORE_URL, headers=headers, files={"file": (name, pdf, content_type)})


async def stored_id(client, headers) -> int:
    response = await store(client, headers)
    assert response.status_code == 201, response.text
    return response.json()["document_id"]


async def ask(client, headers, document_id, question=QUESTION, **extra):
    return await client.post(
        f"{API}/{document_id}/ask", headers=headers, json={"question": question, **extra}
    )


async def count(db_session, model):
    return (await db_session.execute(select(func.count()).select_from(model))).scalar_one()


async def saved(db_session):
    return (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()


# --- B1: store the PDF once -----------------------------------------------------------------


@pytest.mark.anyio
async def test_storing_a_pdf_keeps_the_file_the_document_and_every_page(
    client, auth_headers, mocked_minio, db_session
):
    response = await store(client, auth_headers)

    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 2
    assert body["name"] == "contract.pdf"

    mocked_minio.put_object.assert_called_once()
    put = mocked_minio.put_object.call_args.kwargs
    assert put["object_name"] == str(body["file_id"])
    assert put["length"] == len(PDF)
    assert put["data"].read() == PDF

    pages = await get_pages(body["document_id"], db_session)
    assert [(p.number, p.text) for p in pages] == [
        (1, "The notice period is 30 days."),
        (2, "Salary is paid monthly."),
    ]


@pytest.mark.anyio
async def test_the_stored_content_type_comes_from_us_not_from_the_client(
    client, auth_headers, mocked_minio, db_session
):
    response = await store(client, auth_headers, content_type="text/html")

    assert response.status_code == 201
    file_row = await db_session.get(File, response.json()["file_id"])
    assert (file_row.content_type, file_row.extension) == ("application/pdf", "pdf")
    assert mocked_minio.put_object.call_args.kwargs["content_type"] == "application/pdf"


@pytest.mark.anyio
async def test_a_stored_document_shows_in_the_normal_document_list(client, auth_headers):
    document_id = await stored_id(client, auth_headers)

    listing = (await client.get(API, headers=auth_headers)).json()

    assert [d["id"] for d in listing["documents"]] == [document_id]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "pdf", "code"),
    [
        ("notes.txt", PDF, 415),
        ("fake.pdf", b"this is not a pdf", 415),
        ("empty.pdf", b"", 422),
        ("scanned.pdf", make_pdf(["", ""]), 422),
    ],
)
async def test_a_bad_file_is_refused_and_nothing_is_kept(
    client, auth_headers, mocked_minio, db_session, name, pdf, code
):
    response = await store(client, auth_headers, pdf=pdf, name=name)

    assert response.status_code == code
    mocked_minio.put_object.assert_not_called()
    assert await count(db_session, File) == await count(db_session, Document) == 0
    assert await count(db_session, DocumentPage) == 0


@pytest.mark.anyio
async def test_too_many_pages_is_413_and_nothing_is_kept(
    client, auth_headers, mocked_minio, db_session, monkeypatch
):
    monkeypatch.setattr(settings, "DOCUMENT_QA_MAX_PAGES", 1)

    response = await store(client, auth_headers)

    assert response.status_code == 413
    mocked_minio.put_object.assert_not_called()
    assert await count(db_session, Document) == 0


@pytest.mark.anyio
@pytest.mark.usefixtures("clamd")
async def test_an_infected_pdf_is_refused_before_it_is_stored(
    client, auth_headers, mocked_minio, db_session
):
    infected = make_pdf(PAGE_TEXTS) + b"\n%" + EICAR

    response = await store(client, auth_headers, pdf=infected)

    assert response.status_code == 422
    mocked_minio.put_object.assert_not_called()
    assert await count(db_session, Document) == 0


@pytest.mark.anyio
async def test_storing_needs_a_signed_in_user(client):
    assert (await store(client, {})).status_code == 401


@pytest.mark.anyio
async def test_if_saving_the_pages_fails_the_rows_and_the_object_are_removed(
    client, auth_headers, mocked_minio, db_session, monkeypatch
):
    async def boom(*args, **kwargs):
        raise RuntimeError("database went away")

    monkeypatch.setattr("app.service.document.qa.save_pages", boom)

    with pytest.raises(RuntimeError):
        await store(client, auth_headers)

    assert await count(db_session, File) == await count(db_session, Document) == 0
    stored_name = mocked_minio.put_object.call_args.kwargs["object_name"]
    mocked_minio.remove_object.assert_called_once_with(settings.MINIO_BUCKET, stored_name)


@pytest.mark.anyio
async def test_if_the_upload_to_storage_fails_no_rows_are_kept(
    client, auth_headers, mocked_minio, db_session
):
    mocked_minio.put_object.side_effect = RuntimeError("minio is down")

    with pytest.raises(RuntimeError):
        await store(client, auth_headers)

    assert await count(db_session, File) == await count(db_session, Document) == 0
    mocked_minio.remove_object.assert_not_called()  # nothing was stored, so nothing to remove


@pytest.mark.anyio
async def test_a_nul_character_in_page_text_is_dropped_because_postgres_refuses_it(
    client, auth_headers, db_session
):
    document_id = await stored_id(client, auth_headers)
    await delete_pages(document_id, db_session)

    await save_pages(document_id, [Page(1, "pay\x00ment due")], db_session)

    assert [p.text for p in await get_pages(document_id, db_session)] == ["payment due"]


# --- B2: get the original file back ---------------------------------------------------------


def serve(mocked_minio, chunks=(b"%PDF-1.4 ", b"rest")):
    mocked_minio.get_object.return_value = MagicMock(**{"stream.return_value": list(chunks)})


@pytest.mark.anyio
async def test_the_owner_gets_the_original_file_back(client, auth_headers, mocked_minio):
    document_id = await stored_id(client, auth_headers)
    serve(mocked_minio)

    response = await client.get(f"{API}/{document_id}/file", headers=auth_headers)

    assert response.status_code == 200
    assert response.content == b"%PDF-1.4 rest"
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert "contract.pdf" in response.headers["content-disposition"]


@pytest.mark.anyio
async def test_nobody_else_can_get_the_file(client, auth_headers, other_headers, mocked_minio):
    document_id = await stored_id(client, auth_headers)
    serve(mocked_minio)
    url = f"{API}/{document_id}/file"

    assert (await client.get(url, headers=other_headers)).status_code == 404
    assert (await client.get(url)).status_code == 401
    assert (await client.get(f"{API}/9999/file", headers=auth_headers)).status_code == 404


@pytest.mark.anyio
async def test_a_file_missing_from_storage_is_404_not_a_crash(client, auth_headers, mocked_minio):
    document_id = await stored_id(client, auth_headers)
    mocked_minio.get_object.side_effect = S3Error(
        "NoSuchKey",
        "missing",
        "/x",
        "request",
        "host",
        None,  # type: ignore[arg-type]
    )

    response = await client.get(f"{API}/{document_id}/file", headers=auth_headers)

    assert response.status_code == 404


# --- B3: ask about the stored pages ---------------------------------------------------------


@pytest.mark.anyio
async def test_asking_answers_from_the_stored_pages_with_checked_citations(
    client, auth_headers, gemini
):
    document_id = await stored_id(client, auth_headers)

    response = await ask(client, auth_headers, document_id)

    assert response.status_code == 200
    body = response.json()
    assert body.pop("conversation_id") > 0
    assert body == {
        "answer": "30 days.",
        "found": True,
        "grounded": True,
        "citations": [{"page": 1, "quote": "The notice period is 30 days."}],
        "pages": 2,
        "model": "gemini-test",
        "input_tokens": 900,
        "output_tokens": 40,
        "mode": "pages",
    }
    (call,) = gemini.calls
    system, user = call["messages"]
    assert system["role"] == "system"
    assert '<page number="1">\nThe notice period is 30 days.\n</page>' in user["content"]
    assert '<page number="2">\nSalary is paid monthly.\n</page>' in user["content"]
    assert QUESTION in user["content"]


@pytest.mark.anyio
async def test_a_new_question_starts_a_conversation_linked_to_the_document(
    client, auth_headers, gemini, db_session
):
    document_id = await stored_id(client, auth_headers)

    body = (await ask(client, auth_headers, document_id)).json()

    conversation = await db_session.get(Conversation, body["conversation_id"])
    assert conversation.document_id == document_id
    question, reply = await saved(db_session)
    assert question.meta == {"document": {"id": document_id, "pages": 2}}
    assert (reply.role, reply.content, reply.prompt_version) == (
        "assistant",
        "30 days.",
        "document_qa_v1",
    )
    assert reply.meta == {
        "citations": [{"page": 1, "quote": "The notice period is 30 days."}],
        "grounded": True,
        "found": True,
    }
    assert (reply.input_tokens, reply.output_tokens) == (900, 40)


@pytest.mark.anyio
async def test_a_citation_the_model_made_up_is_dropped(client, auth_headers, gemini):
    document_id = await stored_id(client, auth_headers)
    gemini.result = completion(answer("Free lunch.", Citation(page=2, quote="Lunch is free.")))

    body = (await ask(client, auth_headers, document_id)).json()

    assert (body["grounded"], body["citations"]) == (False, [])


@pytest.mark.anyio
async def test_unknown_deleted_and_other_users_documents_are_404(
    client, auth_headers, other_headers, gemini
):
    document_id = await stored_id(client, auth_headers)

    assert (await ask(client, auth_headers, 9999)).status_code == 404
    assert (await ask(client, other_headers, document_id)).status_code == 404
    assert (await client.delete(f"{API}/{document_id}", headers=auth_headers)).status_code == 200
    assert (await ask(client, auth_headers, document_id)).status_code == 404
    assert gemini.calls == []


@pytest.mark.anyio
async def test_a_document_uploaded_the_old_way_has_no_pages_so_it_is_409(
    client, auth_headers, gemini
):
    pdf_bytes = (Path(__file__).parent / "assets/sample-pdf-invoice.pdf").read_bytes()
    with patch("app.service.document.document.start_processing.delay"):
        uploaded = await client.post(
            API,
            files={"file": ("invoice.pdf", BytesIO(pdf_bytes), "application/pdf")},
            data={"processing_type": "DOCUMENT_SUMMARY"},
            headers=auth_headers,
        )
    assert uploaded.status_code == 201

    response = await ask(client, auth_headers, uploaded.json()["document_id"])

    assert response.status_code == 409
    assert "POST /api/v1/documents/qa" in response.json()["detail"]
    assert gemini.calls == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 4001},
        {},
        {"question": "q", "x": 1},
    ],
)
async def test_a_bad_question_is_422(client, auth_headers, gemini, payload):
    document_id = await stored_id(client, auth_headers)

    response = await client.post(f"{API}/{document_id}/ask", headers=auth_headers, json=payload)

    assert response.status_code == 422
    assert gemini.calls == []


@pytest.mark.anyio
async def test_asking_needs_a_signed_in_user(client, auth_headers, gemini):
    document_id = await stored_id(client, auth_headers)

    assert (await ask(client, {}, document_id)).status_code == 401


@pytest.mark.anyio
async def test_a_failed_call_saves_nothing_and_is_a_safe_error(
    client, auth_headers, gemini, db_session
):
    document_id = await stored_id(client, auth_headers)
    gemini.error = APITimeoutError(request=REQUEST)

    response = await ask(client, auth_headers, document_id)

    assert response.status_code == 504
    assert "Traceback" not in response.text
    assert await count(db_session, Conversation) == await count(db_session, Message) == 0


@pytest.mark.anyio
async def test_over_the_daily_budget_is_429_before_the_model_is_called(
    client, auth_headers, gemini, monkeypatch
):
    monkeypatch.setattr(settings, "CHAT_DAILY_TOKEN_BUDGET", 1000)
    document_id = await stored_id(client, auth_headers)

    assert (await ask(client, auth_headers, document_id)).status_code == 200  # 0 used
    assert (await ask(client, auth_headers, document_id)).status_code == 200  # 940 used
    refused = await ask(client, auth_headers, document_id)  # 1880 used

    assert refused.status_code == 429
    assert len(gemini.calls) == 2


# --- B4: follow-up questions in the same conversation ---------------------------------------


@pytest.mark.anyio
async def test_a_follow_up_joins_the_conversation_and_sends_the_earlier_turns(
    client, auth_headers, gemini, db_session
):
    document_id = await stored_id(client, auth_headers)
    first = (await ask(client, auth_headers, document_id)).json()
    gemini.result = completion(answer("Monthly.", Citation(page=2, quote="paid monthly")))

    second = await ask(
        client,
        auth_headers,
        document_id,
        "And how is salary paid?",
        conversation_id=first["conversation_id"],
    )

    assert second.status_code == 200
    assert second.json()["conversation_id"] == first["conversation_id"]
    assert [m.role for m in await saved(db_session)] == ["user", "assistant"] * 2

    messages = gemini.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == QUESTION  # earlier turns are plain text, no page tags
    assert messages[2]["content"] == "30 days."
    assert "The notice period is 30 days" not in messages[2]["content"]  # no citation text
    assert '<page number="2">' in messages[3]["content"]  # the pages go in again
    assert "And how is salary paid?" in messages[3]["content"]


@pytest.mark.anyio
async def test_a_first_question_sends_no_history(client, auth_headers, gemini):
    document_id = await stored_id(client, auth_headers)

    await ask(client, auth_headers, document_id)

    assert [m["role"] for m in gemini.calls[0]["messages"]] == ["system", "user"]


@pytest.mark.anyio
async def test_history_length_follows_the_chat_setting(client, auth_headers, gemini, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 0)
    document_id = await stored_id(client, auth_headers)
    first = (await ask(client, auth_headers, document_id)).json()

    await ask(client, auth_headers, document_id, "again", conversation_id=first["conversation_id"])

    assert [m["role"] for m in gemini.calls[1]["messages"]] == ["system", "user"]


@pytest.mark.anyio
async def test_a_failed_follow_up_adds_nothing_to_the_conversation(
    client, auth_headers, gemini, db_session
):
    document_id = await stored_id(client, auth_headers)
    first = (await ask(client, auth_headers, document_id)).json()
    gemini.error = APITimeoutError(request=REQUEST)

    failed = await ask(
        client, auth_headers, document_id, "more?", conversation_id=first["conversation_id"]
    )

    assert failed.status_code == 504
    assert len(await saved(db_session)) == 2


@pytest.mark.anyio
async def test_a_conversation_must_exist_belong_to_you_and_be_about_this_document(
    client, auth_headers, other_headers, gemini
):
    document_id = await stored_id(client, auth_headers)
    other_document_id = await stored_id(client, auth_headers)
    mine = (await ask(client, auth_headers, document_id)).json()["conversation_id"]
    calls_before = len(gemini.calls)

    unknown = await ask(client, auth_headers, document_id, conversation_id=9999)
    theirs = await stored_id(client, other_headers)
    someone_elses = await ask(client, other_headers, theirs, conversation_id=mine)
    wrong_document = await ask(client, auth_headers, other_document_id, conversation_id=mine)
    plain_chat = (
        await client.post("/api/v1/chat/conversations", json={}, headers=auth_headers)
    ).json()
    not_a_document_chat = await ask(
        client, auth_headers, document_id, conversation_id=plain_chat["id"]
    )

    assert unknown.status_code == 404
    assert someone_elses.status_code == 404  # on their own document, but not their conversation
    assert wrong_document.status_code == 409
    assert not_a_document_chat.status_code == 409
    assert len(gemini.calls) == calls_before


# --- what a reopened conversation shows -----------------------------------------------------


@pytest.mark.anyio
async def test_a_reopened_conversation_shows_its_document_and_the_sources(
    client, auth_headers, gemini
):
    document_id = await stored_id(client, auth_headers)
    conversation_id = (await ask(client, auth_headers, document_id)).json()["conversation_id"]

    detail = (
        await client.get(f"/api/v1/chat/conversations/{conversation_id}", headers=auth_headers)
    ).json()

    assert detail["document_id"] == document_id
    question, reply = detail["messages"]
    assert (question["citations"], question["grounded"]) == (None, None)
    assert reply["citations"] == [{"page": 1, "quote": "The notice period is 30 days."}]
    assert reply["grounded"] is True
    assert "meta" not in question and "sha256" not in str(detail)  # the raw meta stays private


@pytest.mark.anyio
async def test_a_plain_chat_conversation_has_no_document(client, auth_headers):
    created = (
        await client.post("/api/v1/chat/conversations", json={}, headers=auth_headers)
    ).json()

    assert created["document_id"] is None


# --- B5: deleting ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deleting_a_document_deletes_its_pages_and_its_file(
    client, auth_headers, mocked_minio, gemini, db_session
):
    response = await store(client, auth_headers)
    document_id, file_id = response.json()["document_id"], response.json()["file_id"]
    conversation_id = (await ask(client, auth_headers, document_id)).json()["conversation_id"]
    assert await count(db_session, DocumentPage) == 2

    deleted = await client.delete(f"{API}/{document_id}", headers=auth_headers)

    assert deleted.status_code == 200
    assert await count(db_session, DocumentPage) == 0  # a copy of the content: really gone
    mocked_minio.remove_object.assert_called_once_with(settings.MINIO_BUCKET, str(file_id))
    document = await db_session.get(Document, document_id)
    await db_session.refresh(document)
    assert document.deleted_at is not None  # the row itself is only hidden, as before
    assert (await client.get(f"{API}/{document_id}/file", headers=auth_headers)).status_code == 404
    assert (await ask(client, auth_headers, document_id)).status_code == 404
    # the person's own conversation is theirs to keep or delete
    kept = await client.get(f"/api/v1/chat/conversations/{conversation_id}", headers=auth_headers)
    assert kept.status_code == 200 and len(kept.json()["messages"]) == 2


@pytest.mark.anyio
async def test_deleting_twice_is_safe(client, auth_headers, mocked_minio, db_session):
    document_id = await stored_id(client, auth_headers)

    assert (await client.delete(f"{API}/{document_id}", headers=auth_headers)).status_code == 200
    assert (await client.delete(f"{API}/{document_id}", headers=auth_headers)).status_code == 200
    assert await count(db_session, DocumentPage) == 0


@pytest.mark.anyio
async def test_deleting_the_file_also_deletes_the_pages(
    client, auth_headers, mocked_minio, db_session
):
    response = await store(client, auth_headers)
    file_id = response.json()["file_id"]

    deleted = await client.delete(f"/api/v1/files/{file_id}", headers=auth_headers)

    assert deleted.status_code == 200
    assert await count(db_session, DocumentPage) == 0


@pytest.mark.anyio
async def test_another_users_delete_changes_nothing(
    client, auth_headers, other_headers, mocked_minio, db_session
):
    document_id = await stored_id(client, auth_headers)

    assert (await client.delete(f"{API}/{document_id}", headers=other_headers)).status_code == 404
    assert await count(db_session, DocumentPage) == 2
    mocked_minio.remove_object.assert_not_called()


# --- the migration --------------------------------------------------------------------------


def test_the_migrations_form_one_chain_ending_in_the_page_table():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = Path(__file__).parent.parent
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))

    assert ScriptDirectory.from_config(config).get_heads() == ["8b2e4f6a1c35"]
