from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Transcription
from app.prompts import load_prompt
from app.service.chat.crud import (
    add_message,
    create_conversation,
    get_transcription_by_id,
    save_transcription,
)
from app.service.chat.schema import AUTO_TITLE_CHARS, CallUsage, MessageRole
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message, StreamEvent
from app.service.llm_client.transcriber import transcribe
from app.service.llm_client.vision import PROVIDER as VISION_PROVIDER, answer_about_image
from app.utils.image import CleanImage
from app.utils.upload import CheckedUpload

CHAT_PROMPT = "chat_v1"
VISION_PROMPT = "vision_v1"


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

    The image itself is not kept: only its size, type, size in pixels and SHA-256 are written to the
    question's ``meta``. Saving the answer as a normal assistant message means the daily token budget
    and ``/chat/stats`` count it like any other answer. If the provider fails, nothing is saved.
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
