"""Questions about one image, through Gemini's OpenAI-compatible endpoint.

Same rules as the transcriber: callers only see ``LLMError`` subclasses, never an SDK exception
or a provider message, and one log line per call carries metadata only. The question, the
answer, the image and its file name are never logged.

This is separate from the chat clients on purpose. The chat fallback chain also holds models
that cannot read images, and an image sent to one of them is an error, not an answer. No retries
and no fallback provider yet: add them when the log shows a real need.
"""

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError

from app.core.config import settings
from app.service.llm_client.gemini import gemini_client, token_count
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

PROVIDER = "gemini"


@dataclass(frozen=True)
class VisionResult:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int


async def answer_about_image(
    *, user_id: int, question: str, image: bytes, mime: str, system_prompt: str
) -> VisionResult:
    started = time.monotonic()
    state = "ok"
    input_tokens = output_tokens = None
    try:
        data_uri = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        completion = await gemini_client().chat.completions.create(
            model=settings.VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
            timeout=settings.VISION_TIMEOUT_SECONDS,
        )
        usage = getattr(completion, "usage", None)
        input_tokens = token_count(getattr(usage, "prompt_tokens", None))
        output_tokens = token_count(getattr(usage, "completion_tokens", None))
        text = _answer_text(completion)
        if not text:
            raise LLMError("The provider returned no answer")
    except LLMError as exc:
        state = exc.status
        raise
    except APITimeoutError:  # must come before APIConnectionError, its parent class
        state = "timeout"
        raise LLMTimeoutError("The image question timed out") from None
    except APIConnectionError:
        state = "unavailable"
        raise LLMUnavailableError("Could not reach the vision provider") from None
    except APIStatusError as exc:
        if exc.status_code == 429:
            state = "rate_limited"
            raise LLMRateLimitError("The vision provider is rate limiting us") from None
        state = "error"
        # Our own bad key or request (401, 400) must not become the user's 401 or 400.
        raise LLMError(f"The vision provider answered {exc.status_code}") from None
    finally:
        logger.log(
            logging.INFO if state == "ok" else logging.WARNING,
            "Image question finished",
            extra={
                "provider": PROVIDER,
                "model": settings.VISION_MODEL,
                "status": state,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "image_bytes": len(image),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "user_id": user_id,
            },
        )
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return VisionResult(
        text=text,
        model=getattr(completion, "model", None) or settings.VISION_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=token_count(getattr(prompt_details, "cached_tokens", None)),
        reasoning_tokens=token_count(getattr(completion_details, "reasoning_tokens", None)),
        latency_ms=int((time.monotonic() - started) * 1000),
    )


def _answer_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
    return content.strip() if isinstance(content, str) else ""
