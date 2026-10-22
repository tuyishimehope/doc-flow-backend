"""Questions about one document, answered from its pages with citations.

The whole text goes into one request, each page inside ``<page number="N">``, so the model can
point to a page. The model is asked for a structured answer (``DocAnswer``): whether the answer
was found, the answer, and a list of citations, each a page number and a short word-for-word
quote. Whether a quote really is on that page is *not* decided here; the caller checks it in
code, because a model can invent a quote.

Same rules as ``vision.py``: callers only see ``LLMError`` subclasses, never an SDK exception or a
provider message, and one log line per call carries metadata only. The question, the answer, the
pages and the file name are never logged. No retries and no fallback provider yet: add them when
the log shows a real need.
"""

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    ContentFilterFinishReasonError,
    LengthFinishReasonError,
)
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.service.llm_client.gemini import gemini_client, token_count
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
    Message,
)
from app.utils.pdf import Page

logger = logging.getLogger(__name__)

PROVIDER = "gemini"
_PAGE_TAG = re.compile(r"<(/?)page")  # an opening or closing page tag
_CHUNK_TAG = re.compile(r"<(/?)chunk")


class RetrievedLike(Protocol):
    """What the prompt needs from a retrieved chunk (``RetrievedChunk`` has more)."""

    @property
    def id(self) -> int: ...

    @property
    def page(self) -> int: ...

    @property
    def position(self) -> int: ...

    @property
    def text(self) -> str: ...


class Citation(BaseModel):
    page: int = Field(description="The N of the <page number='N'> the quote comes from")
    quote: str = Field(description="A short quote copied word for word from that page")


class DocAnswer(BaseModel):
    found: bool = Field(description="False when the pages do not contain the answer")
    answer: str
    citations: list[Citation]


@dataclass(frozen=True)
class DocumentAnswerResult:
    answer: DocAnswer
    model: str
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int


def pages_to_prompt(pages: list[Page], question: str) -> str:
    """The user message: every page with its number, then the question.

    A page that contains a page tag could end its own block or fake another page, so such tags
    are broken inside the text. This only changes what the model sees; citations are checked
    against the original page text.
    """
    blocks = [f'<page number="{page.number}">\n{_defuse(page.text)}\n</page>' for page in pages]
    return "\n".join(blocks) + f"\n\n<question>\n{question}\n</question>"


def _defuse(text: str) -> str:
    return _PAGE_TAG.sub(r"<\\\1page", text)


def chunks_to_prompt(chunks: Sequence["RetrievedLike"], question: str) -> str:
    """The user message for retrieval: the chosen chunks in reading order, each with its page,
    then the question. Same protection as ``pages_to_prompt`` against text that looks like a tag.
    """
    blocks = [
        f'<chunk id="{chunk.id}" page="{chunk.page}">\n{_defuse_chunk(chunk.text)}\n</chunk>'
        for chunk in sorted(chunks, key=lambda item: (item.page, item.position))
    ]
    return "\n".join(blocks) + f"\n\n<question>\n{question}\n</question>"


def _defuse_chunk(text: str) -> str:
    return _CHUNK_TAG.sub(r"<\\\1chunk", text)


async def answer_from_pages(
    *,
    user_id: int,
    question: str,
    pages: list[Page],
    system_prompt: str,
    history: Sequence[Message] = (),
) -> DocumentAnswerResult:
    """Ask about the pages. ``history`` is the earlier questions and answers, oldest first,
    as plain ``role``/``content`` messages: only the answer text, never the citations. The
    pages go in again with the new question, so the model always reads the real text."""
    return await answer_from_prompt(
        user_id=user_id,
        content=pages_to_prompt(pages, question),
        system_prompt=system_prompt,
        history=history,
        mode="pages",
        text_blocks=len(pages),
    )


async def answer_from_prompt(
    *,
    user_id: int,
    content: str,
    system_prompt: str,
    history: Sequence[Message] = (),
    mode: str,
    text_blocks: int,
) -> DocumentAnswerResult:
    """The model call behind both ways of answering: the whole text (``pages_to_prompt``) or
    only the retrieved chunks (``chunks_to_prompt``). ``text_blocks`` and ``mode`` are only
    for the log line."""
    started = time.monotonic()
    state = "ok"
    input_tokens = output_tokens = None
    try:
        completion = await gemini_client().beta.chat.completions.parse(
            model=settings.DOCUMENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *history,  # type: ignore[arg-type]  # plain role/content dicts
                {"role": "user", "content": content},
            ],
            response_format=DocAnswer,
            max_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
            timeout=settings.DOCUMENT_QA_TIMEOUT_SECONDS,
        )
        usage = getattr(completion, "usage", None)
        input_tokens = token_count(getattr(usage, "prompt_tokens", None))
        output_tokens = token_count(getattr(usage, "completion_tokens", None))
        choices = getattr(completion, "choices", None) or []
        parsed = getattr(getattr(choices[0], "message", None), "parsed", None) if choices else None
        if not isinstance(parsed, DocAnswer) or not parsed.answer.strip():
            # a refusal, or no answer text at all
            raise LLMError("The provider returned no usable answer")
    except LLMError as exc:
        state = exc.status
        raise
    except (ValidationError, LengthFinishReasonError, ContentFilterFinishReasonError):
        # Not valid JSON for the schema, cut off, or blocked. Nothing here is safe to use.
        state = "error"
        raise LLMError("The provider returned an answer that could not be read") from None
    except APITimeoutError:  # must come before APIConnectionError, its parent class
        state = "timeout"
        raise LLMTimeoutError("The document question timed out") from None
    except APIConnectionError:
        state = "unavailable"
        raise LLMUnavailableError("Could not reach the document provider") from None
    except APIStatusError as exc:
        if exc.status_code == 429:
            state = "rate_limited"
            raise LLMRateLimitError("The document provider is rate limiting us") from None
        state = "error"
        # Our own bad key or request (401, 400) must not become the user's 401 or 400.
        raise LLMError(f"The document provider answered {exc.status_code}") from None
    finally:
        logger.log(
            logging.INFO if state == "ok" else logging.WARNING,
            "Document question finished",
            extra={
                "provider": PROVIDER,
                "model": settings.DOCUMENT_MODEL,
                "status": state,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "mode": mode,
                "chunk_count" if mode == "chunks" else "page_count": text_blocks,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "user_id": user_id,
            },
        )
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return DocumentAnswerResult(
        answer=parsed,
        model=getattr(completion, "model", None) or settings.DOCUMENT_MODEL,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=token_count(getattr(prompt_details, "cached_tokens", None)),
        reasoning_tokens=token_count(getattr(completion_details, "reasoning_tokens", None)),
        latency_ms=int((time.monotonic() - started) * 1000),
    )
