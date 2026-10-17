"""Which providers are used, in what order, and what the frontend may choose."""

import json

import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.service.llm_client.fallback import FallbackLLMClient
from app.service.llm_client.llm_client import LLMNotConfiguredError
from app.service.llm_client.openai_compatible import OpenAICompatibleClient
from app.service.llm_client.registry import (
    ModelUnavailableError,
    _http_client,
    chain_order,
    configured_providers,
    get_llm_client,
    list_models,
)

KEYS = {
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_AI_API_KEY",
    "groq": "GROQ_AI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@pytest.fixture
def keys(monkeypatch):
    """Set exactly these providers' keys; every other provider has none."""

    def configure(*providers: str) -> None:
        for name, setting in KEYS.items():
            monkeypatch.setattr(settings, setting, SecretStr(f"secret-{name}" if name in providers else ""))
        get_llm_client.cache_clear()

    yield configure
    get_llm_client.cache_clear()
    _http_client.cache_clear()


def order(choice, keys_set, keys):
    keys(*keys_set)
    return chain_order(choice, configured_providers())


ALL = ("openrouter", "gemini", "groq", "openai")


def test_default_order_ends_with_openai(keys):
    assert order(None, ALL, keys) == ["openrouter", "gemini", "groq", "openai"]
    assert order("auto", ALL, keys) == ["openrouter", "gemini", "groq", "openai"]


def test_providers_without_a_key_are_left_out(keys):
    assert order(None, ("groq", "openai"), keys) == ["groq", "openai"]


def test_a_chosen_provider_goes_first_and_the_others_still_follow(keys):
    assert order("groq", ALL, keys) == ["groq", "openrouter", "gemini", "openai"]  # openai stays last


def test_choosing_openai_explicitly_puts_it_first(keys):
    assert order("openai", ALL, keys) == ["openai", "openrouter", "gemini", "groq"]


def test_choosing_a_provider_that_is_not_configured_is_refused(keys):
    keys("openrouter")
    with pytest.raises(ModelUnavailableError):
        chain_order("gemini", configured_providers())


def test_no_keys_at_all_means_chat_is_not_configured(keys):
    keys()
    with pytest.raises(LLMNotConfiguredError):
        get_llm_client()


def test_one_provider_needs_no_fallback_wrapper(keys):
    keys("openrouter")
    assert isinstance(get_llm_client(), OpenAICompatibleClient)


def test_several_providers_are_wrapped_in_a_fallback_chain(keys):
    keys(*ALL)
    client = get_llm_client()
    assert isinstance(client, FallbackLLMClient)
    assert client.provider == "openrouter+gemini+groq+openai"
    assert get_llm_client("groq").provider == "groq+openrouter+gemini+openai"


def test_only_the_last_provider_gets_the_full_retry_budget(keys, monkeypatch):
    monkeypatch.setattr(settings, "CHAT_MAX_RETRIES", 3)
    monkeypatch.setattr(settings, "CHAT_FALLBACK_MAX_RETRIES", 1)
    keys(*ALL)
    clients = get_llm_client()._clients  # type: ignore[attr-defined]
    assert [c._max_retries for c in clients] == [1, 1, 1, 3]


def test_the_client_is_cached_per_choice(keys):
    keys(*ALL)
    assert get_llm_client() is get_llm_client()
    assert get_llm_client("groq") is not get_llm_client()


def test_models_list_offers_auto_plus_each_configured_provider(keys):
    keys("openrouter", "openai")
    models = list_models()
    assert [m["id"] for m in models] == ["auto", "openrouter", "openai"]
    assert models[2]["model"] == settings.CHAT_OPENAI_MODEL
    assert "secret" not in json.dumps(models)  # never expose a key


def test_models_list_is_empty_when_nothing_is_configured(keys):
    keys()
    assert list_models() == []
