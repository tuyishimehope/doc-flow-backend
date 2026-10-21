"""A request id for every request, and one log line when it is finished.

* The id comes from the ``X-Request-ID`` header if the caller sent a sane one (so a frontend or a
  proxy can pass its own), otherwise a new one is made. It goes back in the ``X-Request-ID``
  response header, and into **every** log line written while the request is handled, because
  ``RequestIdFilter`` (see ``logging_config``) reads it from a context variable. Search the logs
  for one id and you see the whole request.
* The finishing line has ``method``, ``path``, ``status`` and ``latency_ms``. ``path`` is the route
  *template* (``/api/v1/chat/conversations/{id}``), never the real URL: real URLs hold ids and
  query strings, and they would make one route look like thousands. It replaces ``uvicorn.access``.
* Plain ASGI middleware, not ``BaseHTTPMiddleware``: that one buffers and cancels streaming
  responses, and ``/chat/stream`` is one.
* Levels: 5xx is a WARNING, not an ERROR, because an ERROR log becomes a Sentry event and a
  provider outage (502/503/504) is answered on purpose and must not page anyone. ``/health`` is
  DEBUG while it succeeds: the orchestrator calls it every few seconds.

Known limit: when the app crashes with an unhandled exception, Starlette's outermost layer sends
the 500, so that response has no ``X-Request-ID`` header; the finishing line is still written.
"""

import logging
import re
import time
import uuid
from contextvars import ContextVar

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
# Letters, digits and . _ - only, 8 to 64 characters. Anything else could be used to forge log lines.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
QUIET_PATHS = {"/health"}

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def _incoming_request_id(scope: Scope) -> str | None:
    value = Headers(scope=scope).get(REQUEST_ID_HEADER)
    return value if value and _VALID_REQUEST_ID.match(value) else None


class RequestLogMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500  # what the client gets if the app fails before it answers

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            self._log_finished(scope, status_code, started)
            request_id_var.reset(token)

    @staticmethod
    def _log_finished(scope: Scope, status_code: int, started: float) -> None:
        # FastAPI puts the matched route in the scope; a mounted app (static files) only sets
        # root_path; a URL that matched nothing has neither.
        route = scope.get("route")
        path = getattr(route, "path", None) or scope.get("root_path") or "unmatched"
        if status_code >= 500:
            level = logging.WARNING
        elif path in QUIET_PATHS and status_code < 400:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logger.log(
            level,
            "Request finished",
            extra={
                "method": scope["method"],
                "path": path,
                "status": status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )
