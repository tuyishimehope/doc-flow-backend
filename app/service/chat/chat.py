from collections.abc import AsyncGenerator, Sequence

from app.prompts import load_prompt
from app.service.chat.schema import AUTO_TITLE_CHARS
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message, StreamEvent

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
