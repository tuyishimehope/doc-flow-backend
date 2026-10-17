from collections.abc import AsyncGenerator

from app.prompts import load_prompt
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message, StreamEvent

CHAT_PROMPT = "chat_v1"


def build_messages(message: str) -> list[Message]:
    return [
        {"role": "system", "content": load_prompt(CHAT_PROMPT)},
        {"role": "user", "content": message},
    ]


async def answer_message(message: str, llm: LLMClient) -> LLMResponse:
    """Send the system prompt plus one user message and return the whole answer."""
    return await llm.complete(build_messages(message))


def stream_answer(message: str, llm: LLMClient) -> AsyncGenerator[StreamEvent, None]:
    """Same request as ``answer_message``, but the answer arrives piece by piece."""
    return llm.stream(build_messages(message))
