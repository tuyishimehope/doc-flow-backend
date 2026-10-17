"""Which providers exist, in what order they are tried, and how to build them.

The default order is OpenRouter, Gemini, Groq and finally OpenAI, which is the last
resort. A provider whose API key is not set is simply left out. A request may name the
provider it prefers; that one goes first and the others still follow as fallbacks.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from openai import AsyncOpenAI

from app.core.config import settings
from app.service.llm_client.fallback import FallbackLLMClient
from app.service.llm_client.llm_client import LLMClient, LLMNotConfiguredError
from app.service.llm_client.openai_compatible import OpenAICompatibleClient

AUTO = "auto"
ModelChoice = Literal["auto", "openrouter", "gemini", "groq", "openai"]

# Order matters: this is the order providers are tried. OpenAI stays last on purpose.
DEFAULT_ORDER = ("openrouter", "gemini", "groq", "openai")


class ModelUnavailableError(Exception):
    """The requested provider is not configured on this server."""


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    base_url: str
    api_key: str
    model: str


def configured_providers() -> dict[str, Provider]:
    """Every provider that has an API key, read from the current settings."""
    candidates = [
        Provider("openrouter", "OpenRouter", settings.OPENROUTER_BASE_URL,
                 settings.OPENROUTER_API_KEY.get_secret_value(), settings.CHAT_MODEL),
        Provider("gemini", "Google Gemini", settings.GEMINI_BASE_URL,
                 settings.GEMINI_AI_API_KEY.get_secret_value(), settings.CHAT_GEMINI_MODEL),
        Provider("groq", "Groq", settings.GROQ_BASE_URL,
                 settings.GROQ_AI_API_KEY.get_secret_value(), settings.CHAT_GROQ_MODEL),
        Provider("openai", "OpenAI", settings.OPENAI_BASE_URL,
                 settings.OPENAI_API_KEY.get_secret_value(), settings.CHAT_OPENAI_MODEL),
    ]
    return {p.id: p for p in candidates if p.api_key}


def chain_order(choice: str | None, available: dict[str, Provider]) -> list[str]:
    """Provider ids in the order they are tried for this request."""
    default = [name for name in DEFAULT_ORDER if name in available]
    if choice in (None, AUTO):
        return default
    if choice not in available:
        raise ModelUnavailableError(choice)
    return [choice] + [name for name in default if name != choice]


@lru_cache(maxsize=None)
def _http_client(provider: Provider) -> AsyncOpenAI:
    """One HTTP client per provider, so connections are reused between requests."""
    return AsyncOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        timeout=settings.CHAT_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _build(provider: Provider, *, max_retries: int) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        provider=provider.id,
        client=_http_client(provider),
        model=provider.model,
        timeout=settings.CHAT_TIMEOUT_SECONDS,
        total_timeout=settings.CHAT_TOTAL_TIMEOUT_SECONDS,
        max_retries=max_retries,
        max_output_tokens=settings.CHAT_MAX_OUTPUT_TOKENS,
        stream_timeout=settings.CHAT_STREAM_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=None)
def get_llm_client(choice: str | None = None) -> LLMClient:
    """The client for a request: a single provider, or a fallback chain."""
    available = configured_providers()
    if not available:
        raise LLMNotConfiguredError("No LLM provider API key is set")
    order = chain_order(choice, available)
    clients: list[LLMClient] = []
    for position, name in enumerate(order):
        is_last = position == len(order) - 1
        retries = settings.CHAT_MAX_RETRIES if is_last else min(
            settings.CHAT_MAX_RETRIES, settings.CHAT_FALLBACK_MAX_RETRIES
        )
        clients.append(_build(available[name], max_retries=retries))
    if len(clients) == 1:
        return clients[0]
    return FallbackLLMClient(clients, total_timeout=settings.CHAT_FALLBACK_TOTAL_TIMEOUT_SECONDS)


def list_models() -> list[dict[str, str | None]]:
    """What the frontend may offer: automatic mode plus every configured provider."""
    available = configured_providers()
    models: list[dict[str, str | None]] = []
    if available:
        models.append({"id": AUTO, "label": "Automatic (switches provider if one fails)", "model": None})
    for name in DEFAULT_ORDER:
        if name in available:
            models.append({"id": name, "label": available[name].label, "model": available[name].model})
    return models
