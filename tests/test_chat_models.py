"""Choosing a model: the allowlist, GET /models, and what happens with a bad choice."""

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from app.api.v1.chat.router import get_llm_picker
from app.core.config import settings
from app.main import app
from app.service.auth.auth import get_current_user
from app.service.llm_client.registry import _http_client, get_llm_client
from tests.test_chat import FakeLLM, FakeUser

MODELS = "/api/v1/chat/models"
CHAT = "/api/v1/chat"


def set_keys(monkeypatch, *providers: str) -> None:
    names = {"openrouter": "OPENROUTER_API_KEY", "gemini": "GEMINI_AI_API_KEY",
             "groq": "GROQ_AI_API_KEY", "openai": "OPENAI_API_KEY"}
    for provider, setting in names.items():
        monkeypatch.setattr(settings, setting, SecretStr("secret-key" if provider in providers else ""))
    get_llm_client.cache_clear()


@pytest.fixture(autouse=True)
def clean_caches():
    yield
    get_llm_client.cache_clear()
    _http_client.cache_clear()
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    app.dependency_overrides[get_current_user] = lambda: FakeUser()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_models_lists_automatic_mode_and_each_configured_provider(client, monkeypatch):
    set_keys(monkeypatch, "openrouter", "groq", "openai")
    response = await client.get(MODELS)
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "auto"
    assert [m["id"] for m in body["models"]] == ["auto", "openrouter", "groq", "openai"]
    assert "secret-key" not in response.text


@pytest.mark.anyio
async def test_models_requires_authentication(monkeypatch):
    set_keys(monkeypatch, "openrouter")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as anonymous:
        assert (await anonymous.get(MODELS)).status_code == 401


@pytest.mark.anyio
async def test_the_chosen_model_reaches_the_picker(client):
    asked: list = []
    fake = FakeLLM()

    def picker(model):
        asked.append(model)
        return fake

    app.dependency_overrides[get_llm_picker] = lambda: picker
    await client.post(CHAT, json={"message": "hi"})
    await client.post(CHAT, json={"message": "hi", "model": "groq"})
    await client.post(CHAT, json={"message": "hi", "model": "auto"})
    assert asked == [None, "groq", "auto"]


@pytest.mark.anyio
@pytest.mark.parametrize("model", ["gpt-4", "openrouter/free", "", "GROQ", 5])
async def test_only_names_from_the_allowlist_are_accepted(client, model):
    fake = FakeLLM()
    app.dependency_overrides[get_llm_picker] = lambda: (lambda m: fake)
    response = await client.post(CHAT, json={"message": "hi", "model": model})
    assert response.status_code == 422
    assert fake.calls == []


@pytest.mark.anyio
async def test_a_provider_that_is_not_configured_is_a_clear_400(client, monkeypatch):
    set_keys(monkeypatch, "openrouter")  # the real picker, no fake
    response = await client.post(CHAT, json={"message": "hi", "model": "groq"})
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


@pytest.mark.anyio
async def test_no_provider_keys_at_all_is_a_503_without_details(client, monkeypatch):
    set_keys(monkeypatch)
    response = await client.post(CHAT, json={"message": "hi"})
    assert response.status_code == 503
    assert "KEY" not in response.text and "key" not in response.text.lower()
