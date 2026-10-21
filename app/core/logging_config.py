"""Where the logs go and what they look like.

* Every record is one JSON line on **stdout**. In Docker nothing else is needed: the container
  runtime collects stdout (``docker compose logs``). Where it is kept and for how long is decided
  by the platform, not by the app.
* The API writes through a **queue**. ``logger.info(...)`` only puts the record in an in-memory
  queue and returns; a background thread formats and writes it. A slow or stuck stdout (a full
  pipe, a slow log collector) then cannot freeze the event loop. If the queue is full the newest
  records are dropped and counted, because losing a log line is better than stopping requests.
* The Celery worker does **not** use the queue (``configure_logging()`` default). Its pool forks
  child processes, and a thread started before a fork does not exist in the children, so their
  queued records would never be written. A worker has no event loop to block, so this costs
  nothing.
* Only the fields in ``JsonLogFormatter.context_fields`` are printed from ``extra``. This is on
  purpose: it keeps user text (questions, answers, file names) out of the logs even by mistake.

Settings (environment): ``LOG_LEVEL`` (default INFO), ``LOG_QUEUE`` (``false`` turns the queue off,
useful when debugging a crash where the last lines must appear immediately),
``LOG_QUEUE_SIZE`` (default 10000).
"""

import atexit
import copy
import json
import logging
import os
import queue
import sys
import time
from datetime import datetime, timezone
from logging.handlers import QueueHandler, QueueListener
from typing import TextIO

from app.core.request_log import request_id_var

DEFAULT_QUEUE_SIZE = 10_000
# Libraries that log every call at INFO. Their lines add noise (and full URLs), not information.
QUIET_LOGGERS = ("httpx", "httpcore", "urllib3")


class JsonLogFormatter(logging.Formatter):
    """Emit consistent JSON records with selected request and task context."""

    context_fields = (
        "request_id",
        "document_id",
        "processing_request_id",
        "file_id",
        "stored_file_id",
        "user_id",
        "celery_task_id",
        "attempt_number",
        "character_count",
        "character_limit",
        "image_width",
        "image_height",
        "bucket_name",
        "provider",
        "model",
        "status",
        "latency_ms",
        "audio_bytes",
        "image_bytes",
        "page_count",
        "citations_kept",
        "citations_dropped",
        "grounded",
        "first_piece_ms",
        "fallback_to",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "attempts",
        "method",
        "path",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class RequestIdFilter(logging.Filter):
    """Give every record the id of the request being served, if there is one.

    It must run where the log call happens (it reads a context variable), which is why it sits on
    the handler the application writes to and not on the background writer.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            request_id = request_id_var.get()
            if request_id is not None:
                record.request_id = request_id
        return True


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stdout`` is at that moment.

    This is the same trick as ``logging.lastResort`` uses.

    Looking it up on every write keeps it working when stdout is replaced, for example by pytest.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self) -> TextIO:  # type: ignore[override]
        return sys.stdout


class NonBlockingQueueHandler(QueueHandler):
    """Puts records in a bounded queue. A full queue drops the new record instead of waiting."""

    def __init__(self, log_queue: queue.Queue) -> None:
        super().__init__(log_queue)
        self.dropped = 0

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        # The standard version turns the traceback into plain text and removes it from the record.
        # Keep it, so the JSON line can carry it in its own "exception" field. The queue never
        # leaves this process, so nothing has to be pickled.
        record = copy.copy(record)
        record.message = record.getMessage()
        record.msg = record.message
        record.args = None
        return record

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1  # handlers run one at a time (the handler lock), so this is safe


class _Listener(QueueListener):
    def stop(self) -> None:
        # The standard stop() puts an "end" marker in the queue and raises queue.Full when the
        # queue is full. The writer is still running, so it makes room soon: retry for a few
        # seconds.
        deadline = time.monotonic() + 5
        while True:
            try:
                super().stop()
                return
            except queue.Full:
                if time.monotonic() >= deadline:
                    return  # the writer is stuck; its thread is a daemon and ends with the process
                time.sleep(0.01)


_writer: logging.Handler | None = None  # the handler that really writes (stdout)
_handler: logging.Handler | None = None  # what the application writes to (queue or writer)
_listener: _Listener | None = None
_exit_hook_registered = False


def queue_logs_enabled() -> bool:
    """``LOG_QUEUE=false`` turns the queue off. Anything else (or nothing) keeps it on."""
    return os.getenv("LOG_QUEUE", "true").strip().lower() != "false"


def configure_logging(
    *, use_queue: bool = False, stream: TextIO | None = None, max_queue_size: int | None = None
) -> None:
    """Set up logging for this process. Safe to call again: the new setup replaces the old one.

    The API calls it with ``use_queue=queue_logs_enabled()``. The Celery app imports it too, with
    the default (no queue), so the API process first gets a plain setup and is then switched to the
    queue by ``main``. ``stream`` is for tests; by default the output goes to stdout.
    """
    global _writer, _handler, _listener, _exit_hook_registered
    root_logger = logging.getLogger()
    shutdown_logging(keep_output=False)  # remove our previous setup, flushing what is waiting

    writer = logging.StreamHandler(stream) if stream is not None else _StdoutHandler()
    writer.setFormatter(JsonLogFormatter())

    if use_queue:
        size = max_queue_size or int(os.getenv("LOG_QUEUE_SIZE", DEFAULT_QUEUE_SIZE))
        handler: logging.Handler = NonBlockingQueueHandler(queue.Queue(maxsize=size))
        # respect_handler_level: the level of the writer still counts.
        _listener = _Listener(handler.queue, writer, respect_handler_level=True)  # type: ignore[attr-defined]
        _listener.start()
    else:
        handler = writer
    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)
    _writer, _handler = writer, handler

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger.setLevel(getattr(logging, level_name, logging.INFO))
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    _route_uvicorn_logs()

    if not _exit_hook_registered:
        atexit.register(shutdown_logging)
        _exit_hook_registered = True


def dropped_records() -> int:
    """How many records were dropped because the queue was full (0 without a queue)."""
    return getattr(_handler, "dropped", 0)


def shutdown_logging(*, keep_output: bool = True) -> None:
    """Write everything that is still waiting, then stop the background thread.

    With ``keep_output`` (the default, used at exit) the root logger keeps writing directly, so a
    line logged during shutdown is not lost. ``keep_output=False`` removes our handler completely.
    """
    global _writer, _handler, _listener
    if _listener is not None:
        _listener.stop()  # returns after the queue is empty
        _listener = None
    dropped = dropped_records()
    if dropped:
        print(
            f"{dropped} log records were dropped because the log queue was full", file=sys.stderr
        )
    root_logger = logging.getLogger()
    if _handler is not None:
        root_logger.removeHandler(_handler)
        if keep_output and _writer is not None:
            if _handler is not _writer:  # the queue handler had the filter, the writer did not
                _writer.addFilter(RequestIdFilter())
            root_logger.addHandler(_writer)
            _handler = _writer
            return
    _writer = _handler = None


def _route_uvicorn_logs() -> None:
    """Make uvicorn's own lines go through the same handler, and drop its access log.

    uvicorn sets up its loggers with plain-text handlers before it imports the app. Take them over
    so "Started server process" and errors come out as JSON like everything else. The access log
    is switched off: it prints full URLs with query strings, and ``RequestLogMiddleware`` already
    writes one line per request with the route template instead.
    """
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False
