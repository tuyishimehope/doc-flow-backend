"""The embedding client: vector shape, batching, error mapping, and what is (not) logged.

The provider is a fake with an ``embeddings.create`` method; nothing here touches the network.
"""

import logging
import math
from types import SimpleNamespace

import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

from app.core.config import settings
from app.service.llm_client import embedding
from app.service.llm_client.embedding import EMBEDDING_DIMENSIONS, embed_texts, fit_vector
from app.service.llm_client.llm_client import (
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from tests.test_document_qa import REQUEST, status_error

SECRET = "SECRET CHUNK TEXT about salaries"


def raw_vector(size=3072, first=1.0):
    return [first] + [0.5] * (size - 1)


def length(vector):
    return math.sqrt(sum(value * value for value in vector))


class FakeEmbeddings:
    """Stands in for AsyncOpenAI. One vector per input text; ``order`` can scramble the rows."""

    def __init__(self, *, size=3072, error=None, usage=7, order=None, drop=0):
        self.size, self.error, self.usage, self.order, self.drop = size, error, usage, order, drop
        self.calls: list[dict] = []
        self.embeddings = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        # the first number of each vector is the position of the text in the whole request
        base = sum(len(call["input"]) for call in self.calls[:-1])
        rows = [
            SimpleNamespace(index=i, embedding=raw_vector(self.size, first=base + i + 1))
            for i in range(len(kwargs["input"]))
        ]
        if self.order:
            rows = [rows[i] for i in self.order]
        rows = rows[: len(rows) - self.drop]
        usage = None if self.usage is None else SimpleNamespace(prompt_tokens=self.usage)
        return SimpleNamespace(data=rows, usage=usage)


# --- fit_vector ------------------------------------------------------------------------------


def test_a_long_vector_is_cut_to_the_column_size_and_scaled_to_length_one():
    vector = fit_vector(raw_vector())

    assert len(vector) == EMBEDDING_DIMENSIONS
    assert length(vector) == pytest.approx(1.0)
    # cut, not resampled: the first numbers keep their proportions
    assert vector[0] / vector[1] == pytest.approx(2.0)


def test_a_vector_of_exactly_the_right_size_is_only_scaled():
    vector = fit_vector([3.0, 4.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2))

    assert vector[:2] == pytest.approx([0.6, 0.8])


def test_a_short_vector_is_refused_not_padded():
    with pytest.raises(LLMError):
        fit_vector([1.0] * (EMBEDDING_DIMENSIONS - 1))


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_a_vector_that_cannot_be_scaled_is_refused(bad):
    with pytest.raises(LLMError):
        fit_vector([bad] * EMBEDDING_DIMENSIONS)


# --- embed_texts -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_every_text_gets_one_fitted_vector_in_the_order_given():
    fake = FakeEmbeddings()

    result = await embed_texts(["a", "b", "c"], kind="document", client=fake)

    assert len(result.vectors) == 3
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in result.vectors)
    assert all(length(vector) == pytest.approx(1.0) for vector in result.vectors)
    assert result.model == settings.EMBEDDING_MODEL
    assert fake.calls[0]["model"] == settings.EMBEDDING_MODEL
    assert fake.calls[0]["input"] == ["a", "b", "c"]
    assert fake.calls[0]["timeout"] == settings.EMBEDDING_TIMEOUT_SECONDS


@pytest.mark.anyio
async def test_rows_sent_back_in_another_order_are_put_back_by_index():
    fake = FakeEmbeddings(order=[2, 0, 1])

    result = await embed_texts(["a", "b", "c"], kind="document", client=fake)

    # raw_vector(first=n) scaled: the text at position n has the largest first number for n=3
    firsts = [vector[0] / vector[1] for vector in result.vectors]
    assert firsts == pytest.approx([2.0, 4.0, 6.0])  # 2 * (position + 1)


@pytest.mark.anyio
async def test_texts_are_sent_in_batches_and_the_vectors_stay_in_order(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2)
    fake = FakeEmbeddings()

    result = await embed_texts(["a", "b", "c", "d", "e"], kind="document", client=fake)

    assert [call["input"] for call in fake.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert result.batch_count == 3
    firsts = [vector[0] / vector[1] for vector in result.vectors]
    assert firsts == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0])
    assert result.input_tokens == 21  # 7 per request


@pytest.mark.anyio
async def test_no_texts_means_no_request_and_no_client():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("the client must not be used")

    result = await embed_texts([], kind="document", client=Boom())

    assert result.vectors == [] and result.batch_count == 0


@pytest.mark.anyio
async def test_a_provider_that_sends_no_usage_gives_unknown_tokens_not_zero():
    result = await embed_texts(["a"], kind="query", client=FakeEmbeddings(usage=None))

    assert result.input_tokens is None


@pytest.mark.anyio
async def test_the_shared_client_is_used_when_none_is_passed(monkeypatch):
    fake = FakeEmbeddings()
    monkeypatch.setattr(embedding, "gemini_client", lambda: fake)

    await embed_texts(["a"], kind="query")

    assert len(fake.calls) == 1


@pytest.mark.anyio
async def test_the_wrong_number_of_vectors_is_an_error_not_a_silent_shift():
    with pytest.raises(LLMError, match="wrong number"):
        await embed_texts(["a", "b", "c"], kind="document", client=FakeEmbeddings(drop=1))


@pytest.mark.anyio
async def test_a_too_short_vector_from_the_provider_is_an_error():
    with pytest.raises(LLMError):
        await embed_texts(["a"], kind="document", client=FakeEmbeddings(size=100))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("error", "expected", "status"),
    [
        (APITimeoutError(request=REQUEST), LLMTimeoutError, "timeout"),
        (APIConnectionError(request=REQUEST), LLMUnavailableError, "unavailable"),
        (status_error(RateLimitError, 429), LLMRateLimitError, "rate_limited"),
        (status_error(AuthenticationError, 401), LLMError, "error"),
    ],
)
async def test_provider_failures_become_our_errors_without_the_provider_message(
    error, expected, status
):
    with pytest.raises(expected) as caught:
        await embed_texts(["a"], kind="document", client=FakeEmbeddings(error=error))

    assert caught.value.status == status
    assert "SECRET PROVIDER BODY" not in str(caught.value)
    assert caught.value.__cause__ is None  # nothing from the SDK leaks through the chain


# --- the task type for the second model family ----------------------------------------------


@pytest.mark.anyio
async def test_the_default_model_gets_the_text_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "gemini-embedding-001")
    fake = FakeEmbeddings()

    await embed_texts(["hello"], kind="query", client=fake)

    assert fake.calls[0]["input"] == ["hello"]


@pytest.mark.anyio
async def test_embedding_2_gets_the_task_in_the_text_and_it_differs_for_query_and_document(
    monkeypatch,
):
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "gemini-embedding-2-preview")
    fake = FakeEmbeddings()

    await embed_texts(["hello"], kind="query", client=fake)
    await embed_texts(["hello"], kind="document", client=fake)

    query, document = (call["input"][0] for call in fake.calls)
    assert query == "task: search result | query: hello"
    assert document == "title: none | text: hello"


# --- logs ------------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_one_log_line_with_counts_and_never_the_text(caplog):
    with caplog.at_level(logging.INFO, logger=embedding.__name__):
        await embed_texts([SECRET, SECRET], kind="document", client=FakeEmbeddings(), user_id=7)

    (record,) = [r for r in caplog.records if r.getMessage() == "Embedding finished"]
    assert record.status == "ok"
    assert record.chunk_count == 2
    assert record.batch_count == 1
    assert record.user_id == 7
    assert record.input_tokens == 7
    assert "SECRET" not in repr(record.__dict__)


@pytest.mark.anyio
async def test_a_failure_is_logged_as_a_warning_with_its_status(caplog):
    error = APITimeoutError(request=REQUEST)
    with caplog.at_level(logging.INFO, logger=embedding.__name__):
        with pytest.raises(LLMTimeoutError):
            await embed_texts([SECRET], kind="query", client=FakeEmbeddings(error=error))

    (record,) = [r for r in caplog.records if r.getMessage() == "Embedding finished"]
    assert (record.levelno, record.status) == (logging.WARNING, "timeout")
    assert "SECRET" not in repr(record.__dict__)
