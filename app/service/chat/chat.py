from app.prompts import load_prompt
from app.service.llm_client.llm_client import LLMClient, LLMResponse, Message

CHAT_PROMPT = "chat_v1"


async def answer_message(message: str, llm: LLMClient) -> LLMResponse:
    """Send the system prompt plus one user message and return the model's answer."""
    messages: list[Message] = [
        {"role": "system", "content": load_prompt(CHAT_PROMPT)},
        {"role": "user", "content": message},
    ]
    return await llm.complete(messages)
