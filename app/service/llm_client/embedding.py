"""Turn text into vectors, through Gemini's OpenAI-compatible ``/embeddings`` endpoint.

Same rules as the other direct Gemini clients: callers only see ``LLMError`` subclasses, never an
SDK exception or a provider message, and one log line per call carries counts only. The text
that is embedded (document chunks, the user's question) is never logged.

What this module guarantees about a vector:

* it has exactly ``EMBEDDING_DIMENSIONS`` numbers, the size of the database column. The model's
  vectors are longer (3072 by default), so each one is cut to that size and scaled to length 1.
  Cutting is how these models are meant to be shortened; scaling is required for
  ``gemini-embedding-001`` and does no harm for the others. Doing it here does not depend on the
  endpoint accepting a ``dimensions`` field.
* it belongs to ``settings.EMBEDDING_MODEL``. Vectors of two models cannot be compared, so the
  model name is stored with every vector and a model change means indexing everything again.

Not verified against the live endpoint yet (the live check in plan_4, C2, is yours): that a list
of texts in one request is accepted, and the real rate limit. If the list form is refused, set
``EMBEDDING_BATCH_SIZE=1``.
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

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
EMBEDDING_DIMENSIONS = 768  # the size of document_chunk.embedding; changing it needs a migration

Kind = Literal["document", "query"]


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]  # one per text, in the order given
    model: str
    input_tokens: int | None
    latency_ms: int
    batch_count: int


def _format(kind: Kind, text: str) -> str:
    """``gemini-embedding-2`` has no task type: the task goes into the text. Others: as is."""
    if not settings.EMBEDDING_MODEL.startswith("gemini-embedding-2"):
        return text
    if kind == "query":
        return f"task: search result | query: {text}"
    return f"title: none | text: {text}"


def fit_vector(vector: list[float]) -> list[float]:
    """Cut to ``EMBEDDING_DIMENSIONS`` numbers and scale to length 1."""
    if len(vector) < EMBEDDING_DIMENSIONS:
        raise LLMError("The embedding provider returned a vector that is too short")
    cut = [float(value) for value in vector[:EMBEDDING_DIMENSIONS]]
    norm = math.sqrt(sum(value * value for value in cut))
    if not norm or not math.isfinite(norm):
        raise LLMError("The embedding provider returned an unusable vector")
    return [value / norm for value in cut]


async def embed_texts(
    texts: list[str],
    *,
    kind: Kind,
    client: AsyncOpenAI | None = None,
    user_id: int | None = None,
) -> EmbeddingResult:
    """Embed every text, ``EMBEDDING_BATCH_SIZE`` per request. No retries and no fallback yet.

    Pass ``client`` from a Celery task (a fresh one per task); the API uses the shared client.
    """
    started = time.monotonic()
    state = "ok"
    input_tokens: int | None = 0
    vectors: list[list[float]] = []
    batch_count = 0
    try:
        if texts:
            client = client or gemini_client()
        size = settings.EMBEDDING_BATCH_SIZE
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            response = await client.embeddings.create(  # type: ignore[union-attr]
                model=settings.EMBEDDING_MODEL,
                input=[_format(kind, text) for text in batch],
                timeout=settings.EMBEDDING_TIMEOUT_SECONDS,
            )
            batch_count += 1
            rows = sorted(getattr(response, "data", None) or [], key=lambda row: row.index)
            if len(rows) != len(batch):
                raise LLMError("The embedding provider returned the wrong number of vectors")
            vectors.extend(fit_vector(list(row.embedding)) for row in rows)
            used = token_count(getattr(getattr(response, "usage", None), "prompt_tokens", None))
            input_tokens = None if used is None or input_tokens is None else input_tokens + used
    except LLMError as exc:
        state = exc.status
        raise
    except APITimeoutError:  # must come before APIConnectionError, its parent class
        state = "timeout"
        raise LLMTimeoutError("The embedding request timed out") from None
    except APIConnectionError:
        state = "unavailable"
        raise LLMUnavailableError("Could not reach the embedding provider") from None
    except APIStatusError as exc:
        if exc.status_code == 429:
            state = "rate_limited"
            raise LLMRateLimitError("The embedding provider is rate limiting us") from None
        state = "error"
        raise LLMError(f"The embedding provider answered {exc.status_code}") from None
    finally:
        extra = {
            "provider": PROVIDER,
            "model": settings.EMBEDDING_MODEL,
            "status": state,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "chunk_count": len(texts),
            "batch_count": batch_count,
            "input_tokens": input_tokens,
            "mode": kind,
        }
        if user_id is not None:
            extra["user_id"] = user_id
        logger.log(
            logging.INFO if state == "ok" else logging.WARNING, "Embedding finished", extra=extra
        )
    return EmbeddingResult(
        vectors=vectors,
        model=settings.EMBEDDING_MODEL,
        input_tokens=input_tokens,
        latency_ms=int((time.monotonic() - started) * 1000),
        batch_count=batch_count,
    )
