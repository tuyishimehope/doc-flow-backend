import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.schema import Conversation, Transcription
from app.prompts import load_prompt
from app.service.chat.citations import check_answer
from app.service.chat.crud import (
    add_message,
    create_conversation,
    get_recent_messages,
    get_transcription_by_id,
    save_transcription,
)
from app.service.chat.schema import AUTO_TITLE_CHARS, CallUsage, MessageRole
from app.service.document.chunks import retrieve_chunks
from app.service.llm_client.document_qa import PROVIDER as DOCUMENT_PROVIDER
from app.service.llm_client.document_qa import (
    Citation,
    answer_from_prompt,
    chunks_to_prompt,
    pages_to_prompt,
)
from app.service.llm_client.embedding import embed_texts
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message, StreamEvent
from app.service.llm_client.transcriber import transcribe
from app.service.llm_client.vision import PROVIDER as VISION_PROVIDER
from app.service.llm_client.vision import answer_about_image
from app.utils.image import CleanImage
from app.utils.pdf import Page
from app.utils.upload import CheckedUpload

logger = logging.getLogger(__name__)

CHAT_PROMPT = "chat_v1"
VISION_PROMPT = "vision_v1"
DOCUMENT_PROMPT = "document_qa_v1"
RAG_PROMPT = "document_rag_v1"  # the same job when only retrieved chunks are given


def build_messages(message: str, history: Sequence[Message] = ()) -> list[Message]:
    """System prompt, then the earlier turns (oldest first), then the new message."""
    return [
        {"role": "system", "content": load_prompt(CHAT_PROMPT)},
        *history,
        {"role": "user", "content": message},
    ]


def trim_history(history: Sequence[Message], max_chars: int) -> list[Message]:
    """Keep the newest messages that fit in ``max_chars``; whole messages only.

    The conversation also must not start with an assistant message, because some
    providers reject a prompt whose first turn after the system prompt is not the user's.
    """
    kept: list[Message] = []
    total = 0
    for item in reversed(history):
        total += len(item["content"])
        if total > max_chars:
            break
        kept.append(item)
    kept.reverse()
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def auto_title(message: str) -> str:
    """A title from the first message: its first words, on one line."""
    text = " ".join(message.split())
    if len(text) <= AUTO_TITLE_CHARS:
        return text
    return text[:AUTO_TITLE_CHARS].rstrip() + "…"


async def answer_message(message: str, llm: LLMClient) -> LLMResponse:
    """Send the system prompt plus one user message and return the whole answer."""
    return await llm.complete(build_messages(message))


def stream_answer(
    message: str, llm: LLMClient, history: Sequence[Message] = ()
) -> AsyncGenerator[StreamEvent, None]:
    """Same request as ``answer_message``, but the answer arrives piece by piece."""
    return llm.stream(build_messages(message, history))


async def transcribe_audio_file(
    *,
    user_id: int,
    filename: str,
    content: bytes,
    content_type: str | None,
    db_session: AsyncSession,
) -> Transcription:
    """Send the audio to the provider, then keep the transcript. The audio is not kept."""
    result = await transcribe(
        user_id=user_id, filename=filename, content=content, content_type=content_type
    )
    return await save_transcription(
        user_id=user_id,
        text=result.text,
        model=result.model,
        audio_bytes=result.audio_bytes,
        latency_ms=result.latency_ms,
        db_session=db_session,
    )


async def get_transcription(
    user_id: int, transcription_id: int, db_session: AsyncSession
) -> Transcription | None:
    return await get_transcription_by_id(user_id, transcription_id, db_session)


@dataclass(frozen=True)
class ImageAnswer:
    conversation_id: int
    answer: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


async def answer_image_question(
    *,
    user_id: int,
    question: str,
    upload: CheckedUpload,
    image: CleanImage,
    db_session: AsyncSession,
) -> ImageAnswer:
    """Ask the vision model, then keep the question and the answer in a new conversation.

    The image itself is not kept: only its size, type, size in pixels and SHA-256 are written to
    the question's ``meta``. Saving the answer as a normal assistant message means the daily
    token budget and ``/chat/stats`` count it like any other answer. If the provider fails,
    nothing is saved.
    """
    result = await answer_about_image(
        user_id=user_id,
        question=question,
        image=image.content,
        mime=image.mime,
        system_prompt=load_prompt(VISION_PROMPT),
    )
    conversation = await create_conversation(user_id, auto_title(question), db_session)
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=question,
        meta={
            "image": {
                "bytes": upload.size,
                "type": upload.mime,
                "sha256": upload.sha256,
                "width": image.width,
                "height": image.height,
            }
        },
        db_session=db_session,
    )
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=result.text,
        model=result.model,
        usage=CallUsage(
            provider=VISION_PROVIDER,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            latency_ms=result.latency_ms,
        ),
        prompt_version=VISION_PROMPT,
        db_session=db_session,
    )
    return ImageAnswer(
        conversation_id=conversation.id,
        answer=result.text,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


class RetrievalUnavailableError(Exception):
    """The document is marked ready but has no chunks for the current embedding model."""


@dataclass(frozen=True)
class DocumentAnswer:
    conversation_id: int
    answer: str
    found: bool
    grounded: bool
    citations: list[Citation]  # only the ones that passed the check
    pages: int
    model: str
    input_tokens: int | None
    output_tokens: int | None
    mode: str = "pages"  # "pages": the whole text was read; "chunks": only the retrieved chunks


async def answer_document_question(
    *,
    user_id: int,
    question: str,
    upload: CheckedUpload,
    pages: list[Page],
    db_session: AsyncSession,
) -> DocumentAnswer:
    """Ask about the pages, check the citations, then keep the question and the answer.

    The PDF and its text are not kept: only its size, page count and SHA-256 are written to the
    question's ``meta``. The answer's ``meta`` holds the citations that passed the check, and
    whether the answer is grounded. Saving the answer as a normal assistant message means the
    daily token budget and ``/chat/stats`` count it like any other answer. If the provider
    fails, nothing is saved.
    """
    return await _answer_and_keep(
        user_id=user_id,
        question=question,
        content=pages_to_prompt(pages, question),
        check_pages=pages,
        prompt_version=DOCUMENT_PROMPT,
        mode="pages",
        text_blocks=len(pages),
        total_pages=len(pages),
        question_meta={
            "document": {"bytes": upload.size, "pages": len(pages), "sha256": upload.sha256}
        },
        conversation=None,
        document_id=None,
        db_session=db_session,
    )


async def ask_stored_document(
    *,
    user_id: int,
    document_id: int,
    question: str,
    pages: list[Page],
    page_count: int,
    conversation: Conversation | None,
    use_retrieval: bool,
    db_session: AsyncSession,
) -> DocumentAnswer:
    """Same as ``answer_document_question``, for a PDF that was stored earlier.

    With ``use_retrieval`` the question is embedded and only the nearest chunks of this user's
    document go to the model (``pages`` is not needed then); otherwise the model reads every page
    in ``pages``. Either way the citations are checked against exactly the text the model saw.

    With a ``conversation`` the question is a follow-up: the last few questions and answers (the
    answer text only, never the citations) go to the model too. Without one, a new conversation
    is started and linked to the document. Nothing is saved if the provider fails.
    """
    history: list[Message] = []
    if conversation is not None:
        earlier = await get_recent_messages(
            conversation.id, settings.CHAT_HISTORY_MESSAGES, db_session
        )
        history = trim_history(
            [{"role": m.role, "content": m.content} for m in earlier],  # type: ignore[misc]
            settings.CHAT_HISTORY_MAX_CHARS,
        )

    question_meta = {"document": {"id": document_id, "pages": page_count}}
    if not use_retrieval:
        return await _answer_and_keep(
            user_id=user_id,
            question=question,
            content=pages_to_prompt(pages, question),
            check_pages=pages,
            prompt_version=DOCUMENT_PROMPT,
            mode="pages",
            text_blocks=len(pages),
            total_pages=page_count,
            question_meta=question_meta,
            conversation=conversation,
            document_id=document_id,
            history=history,
            db_session=db_session,
        )

    embedded = await embed_texts([question], kind="query", user_id=user_id)
    chunks = await retrieve_chunks(
        db_session,
        user_id=user_id,
        document_id=document_id,
        query_vector=embedded.vectors[0],
        limit=settings.DOCUMENT_RETRIEVAL_TOP_K,
    )
    if not chunks:
        raise RetrievalUnavailableError
    logger.info(
        "Chunks retrieved",
        extra={
            "document_id": document_id,
            "user_id": user_id,
            "chunk_count": len(chunks),
            "chunk_ids": [chunk.id for chunk in chunks],
            "distances": [round(chunk.distance, 4) for chunk in chunks],
        },
    )
    return await _answer_and_keep(
        user_id=user_id,
        question=question,
        content=chunks_to_prompt(chunks, question),
        check_pages=[Page(number=chunk.page, text=chunk.text) for chunk in chunks],
        prompt_version=RAG_PROMPT,
        mode="chunks",
        text_blocks=len(chunks),
        total_pages=page_count,
        question_meta=question_meta,
        # ids, pages and distances only: the text is in document_chunk, not in the message
        answer_meta={
            "retrieval": [
                {"id": c.id, "page": c.page, "distance": round(c.distance, 4)} for c in chunks
            ]
        },
        conversation=conversation,
        document_id=document_id,
        history=history,
        db_session=db_session,
    )


async def _answer_and_keep(
    *,
    user_id: int,
    question: str,
    content: str,
    check_pages: list[Page],
    prompt_version: str,
    mode: str,
    text_blocks: int,
    total_pages: int,
    question_meta: dict,
    conversation: Conversation | None,
    document_id: int | None,
    db_session: AsyncSession,
    history: Sequence[Message] = (),
    answer_meta: dict | None = None,
) -> DocumentAnswer:
    result = await answer_from_prompt(
        user_id=user_id,
        content=content,
        system_prompt=load_prompt(prompt_version),
        history=history,
        mode=mode,
        text_blocks=text_blocks,
    )
    checked = check_answer(result.answer, check_pages)
    if conversation is None:
        conversation = await create_conversation(
            user_id, auto_title(question), db_session, document_id=document_id
        )
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=question,
        meta=question_meta,
        db_session=db_session,
    )
    await add_message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=result.answer.answer,
        model=result.model,
        usage=CallUsage(
            provider=DOCUMENT_PROVIDER,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.reasoning_tokens,
            latency_ms=result.latency_ms,
        ),
        prompt_version=prompt_version,
        meta={
            "citations": [citation.model_dump() for citation in checked.citations],
            "grounded": checked.grounded,
            "found": result.answer.found,
            **(answer_meta or {}),
        },
        db_session=db_session,
    )
    return DocumentAnswer(
        conversation_id=conversation.id,
        answer=result.answer.answer,
        found=result.answer.found,
        grounded=checked.grounded,
        citations=checked.citations,
        pages=total_pages,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        mode=mode,
    )
