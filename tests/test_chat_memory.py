"""History trimming, titles, and saving the answer while it streams. No database:
the save itself is replaced by a recorder."""

import asyncio

import anyio
import pytest

from app.service.chat import conversation as conversation_module
from app.service.chat.chat import auto_title, build_messages, trim_history
from app.service.chat.conversation import save_while_streaming
from app.service.chat.schema import MessageRole, MessageStatus
from app.service.llm_client.llm_client import StreamDelta, StreamDone


def turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---- history ---------------------------------------------------------------


def test_build_messages_puts_history_between_the_system_prompt_and_the_new_message():
    history = [turn("user", "Hi"), turn("assistant", "Hello!")]
    messages = build_messages("And now?", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"] == "And now?"
    assert messages[1:3] == history


def test_build_messages_without_history_is_unchanged():
    assert [m["role"] for m in build_messages("Hi")] == ["system", "user"]


def test_trim_history_drops_the_oldest_messages_first():
    history = [turn("user", "a" * 10), turn("assistant", "b" * 10), turn("user", "c" * 10), turn("assistant", "d" * 10)]
    kept = trim_history(history, max_chars=25)
    # The two newest fit (20 chars), but the conversation must start with the user.
    assert kept == history[2:]


def test_trim_history_never_starts_with_an_assistant_message():
    history = [turn("user", "a" * 10), turn("assistant", "b" * 10), turn("user", "c" * 10), turn("assistant", "d" * 10)]
    assert trim_history(history, max_chars=15) == []  # only the last assistant message fits


def test_trim_history_keeps_everything_when_it_fits():
    history = [turn("user", "hi"), turn("assistant", "hello")]
    assert trim_history(history, max_chars=1000) == history
    assert trim_history([], max_chars=1000) == []


# ---- titles ----------------------------------------------------------------


def test_auto_title_uses_the_first_words_on_one_line():
    assert auto_title("  What is\n  SSE?  ") == "What is SSE?"


def test_auto_title_is_cut_at_fifty_characters():
    title = auto_title("word " * 30)
    assert title.endswith("…") and len(title) <= 51


# ---- saving while streaming ------------------------------------------------


class Recorder:
    """Stands in for ``_save``: remembers what would have been stored."""

    def __init__(self) -> None:
        self.saved: list[dict] = []

    async def __call__(self, session_factory, conversation_id, text, status, model=None, meta=None):
        self.saved.append({"conversation_id": conversation_id, "text": text, "status": status, "model": model, "meta": meta})


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(conversation_module, "_save", recorder)
    return recorder


def done() -> StreamDone:
    return StreamDone(model="m", input_tokens=5, output_tokens=2, latency_ms=30, first_piece_ms=10, attempts=1)


class Source:
    """A fake provider stream we can make fail or hang, and see whether it was closed."""

    def __init__(self, events, *, error: Exception | None = None, hang: bool = False) -> None:
        self.events = list(events)
        self.error = error
        self.hang = hang
        self.closed = False

    async def __call__(self):
        try:
            for event in self.events:
                yield event
            if self.error:
                raise self.error
            if self.hang:
                await asyncio.sleep(30)
        finally:
            self.closed = True


def run(source: Source):
    generator = source()
    first = generator.__anext__()
    return generator, first


@pytest.mark.anyio
async def test_saves_the_full_answer_before_the_done_event_goes_out(recorder):
    generator, first = run(Source([StreamDelta("Par"), StreamDelta("is."), done()]))
    seen = []
    async for event in save_while_streaming(await first, generator, conversation_id=7, session_factory=None):
        seen.append(type(event).__name__)
        if isinstance(event, StreamDone):
            assert len(recorder.saved) == 1  # already stored when the client sees "done"
    assert seen == ["StreamDelta", "StreamDelta", "StreamDone"]
    saved = recorder.saved[0]
    assert (saved["conversation_id"], saved["text"], saved["status"], saved["model"]) == (7, "Paris.", MessageStatus.COMPLETE, "m")
    assert saved["meta"] == {"provider": "", "input_tokens": 5, "output_tokens": 2, "latency_ms": 30, "first_piece_ms": 10}


@pytest.mark.anyio
async def test_a_failure_after_some_text_saves_what_was_written_as_an_error(recorder):
    generator, first = run(Source([StreamDelta("Par")], error=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        async for _ in save_while_streaming(await first, generator, conversation_id=7, session_factory=None):
            pass
    assert [(s["text"], s["status"]) for s in recorder.saved] == [("Par", MessageStatus.ERROR)]


@pytest.mark.anyio
async def test_closing_early_saves_the_partial_answer_as_cancelled_and_stops_the_model(recorder):
    source = Source([StreamDelta("Par"), StreamDelta("is.")], hang=True)
    generator, first = run(source)
    saving = save_while_streaming(await first, generator, conversation_id=7, session_factory=None)

    assert await saving.__anext__() == StreamDelta("Par")
    await saving.aclose()

    assert source.closed
    assert [(s["text"], s["status"]) for s in recorder.saved] == [("Par", MessageStatus.CANCELLED)]


@pytest.mark.anyio
async def test_the_partial_answer_is_saved_even_when_the_request_is_cancelled(recorder):
    """Cancelling a scope cancels every await inside it, including the save itself."""
    slow_saved = anyio.Event()

    async def slow_save(*args, **kwargs):
        await asyncio.sleep(0.05)  # an await that a cancelled scope would normally interrupt
        await recorder(*args, **kwargs)
        slow_saved.set()

    conversation_module._save = slow_save  # restored by monkeypatch in the fixture teardown
    source = Source([StreamDelta("Par")], hang=True)
    got_first = anyio.Event()

    async def listen():
        generator, first = run(source)
        async for _ in save_while_streaming(await first, generator, conversation_id=7, session_factory=None):
            got_first.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(listen)
        await got_first.wait()
        task_group.cancel_scope.cancel()  # what a closed browser tab looks like

    with anyio.fail_after(2):
        await slow_saved.wait()
    assert [(s["text"], s["status"]) for s in recorder.saved] == [("Par", MessageStatus.CANCELLED)]
    assert source.closed


def test_roles_and_statuses_match_the_database_constraints():
    assert {r.value for r in MessageRole} == {"user", "assistant"}
    assert {s.value for s in MessageStatus} == {"complete", "cancelled", "error"}
