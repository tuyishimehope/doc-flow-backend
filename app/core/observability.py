"""Error and performance monitoring with Sentry. Off unless ``SENTRY_DSN`` is set.

What is deliberately NOT sent, because it is user data:
* request bodies (they contain chat messages),
* local variables of stack frames (they can hold the same text),
* prompts and answers of LLM calls,
* cookies, the Authorization header, e-mail addresses and IP addresses.
Only the numeric user id is attached to an event.

Logs: everything the app writes through ``logging`` (see ``logging_config``) is also sent to
Sentry Logs, at the level ``LOG_LEVEL`` lets through, with the same extra fields as the JSON
line. ``uvicorn.access`` is left out: it prints full URLs including query strings, and the
performance traces already cover every request. ERROR logs also become events, as before.

What is deliberately not an event: a provider outage. The chat endpoints answer those
with 502/503/504 on purpose, and the fallback chain already logs them. Only real server
errors (unhandled exceptions, HTTP 500) and "every provider failed" are reported.
"""

import logging
from typing import Any

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration, ignore_logger_for_sentry_logs
from sentry_sdk.integrations.openai import OpenAIIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration

from app.core.config import settings

logger = logging.getLogger(__name__)

# Starlette and FastAPI capture 500-599 HTTP errors by default; keep only 500.
REPORTED_STATUS_CODES = {500}


def init_sentry(*, transport: Any = None) -> bool:
    """Start Sentry if a DSN is configured. Returns whether it is now active.

    Call it before the FastAPI app is created, so the app is instrumented.
    ``transport`` exists for tests, which capture events instead of sending them.
    """
    if transport is None and sentry_sdk.get_client().is_active():
        return True  # already started: the API imports Celery, and both call this
    dsn = settings.SENTRY_DSN.strip()
    if not dsn:
        logger.info("Sentry is off: SENTRY_DSN is empty")
        return False
    sentry_sdk.init(
        dsn=dsn,
        environment=settings.SENTRY_ENVIRONMENT,
        release=settings.SENTRY_RELEASE or None,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        enable_logs=settings.SENTRY_ENABLE_LOGS,
        debug=settings.SENTRY_DEBUG,  # prints why events are dropped or fail to send
        send_default_pii=False,
        max_request_body_size="never",
        include_local_variables=False,
        before_send=_scrub,
        transport=transport,
        integrations=[
            StarletteIntegration(failed_request_status_codes=REPORTED_STATUS_CODES),
            FastApiIntegration(failed_request_status_codes=REPORTED_STATUS_CODES),
            CeleryIntegration(),
            # level: breadcrumbs from INFO up; event_level: ERROR logs are events.
            # sentry_logs_level: the lowest level sent to Sentry Logs. INFO by default,
            # because the Celery worker runs with --loglevel=DEBUG and celery/kombu are chatty.
            # Whether logs are sent at all is decided by enable_logs above (works on older SDKs
            # too).
            LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
                sentry_logs_level=_level(settings.SENTRY_LOGS_LEVEL),
            ),
            OpenAIIntegration(include_prompts=False),
        ],
    )
    ignore_logger_for_sentry_logs("uvicorn.access")
    logger.info(
        "Sentry is on",
        extra={"environment": settings.SENTRY_ENVIRONMENT},
    )
    return True


def _level(name: str) -> int:
    level = logging.getLevelName(name.strip().upper())
    return level if isinstance(level, int) else logging.INFO


def _scrub(event: Any, hint: Any) -> Any:
    """A last safety net: remove user data even if an option above is changed later."""
    request = event.get("request")
    if isinstance(request, dict):
        for key in ("data", "cookies"):
            request.pop(key, None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for name in list(headers):
                if name.lower() in ("authorization", "cookie", "x-api-key"):
                    headers.pop(name)
    user = event.get("user")
    if isinstance(user, dict):
        event["user"] = {"id": user["id"]} if "id" in user else {}
    return event


def set_user_id(user_id: int) -> None:
    """Attach the numeric user id, and nothing else about the user, to later events."""
    sentry_sdk.set_user({"id": str(user_id)})


def set_tag(key: str, value: str) -> None:
    """A searchable label on later events. Harmless when Sentry is off."""
    sentry_sdk.set_tag(key, value)
