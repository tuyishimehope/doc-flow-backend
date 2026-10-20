from collections.abc import AsyncGenerator, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Transcription
from app.prompts import load_prompt
from app.service.chat.crud import get_transcription_by_id, save_transcription
from app.service.chat.schema import AUTO_TITLE_CHARS
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message, StreamEvent
from app.service.llm_client.transcriber import transcribe

CHAT_PROMPT = "chat_v1"


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
