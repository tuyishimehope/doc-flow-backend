"""ENVIRONMENT: how it is read, and what it changes (docs, logs, Sentry)."""

import io
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Environment, Settings, settings
from app.core.docs import docs_settings, register_docs
from app.core.logging_config import configure_logging, shutdown_logging


def make_settings(**values) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def required(monkeypatch):
    """The settings without a default, so a Settings() can be built without .env."""
    for name, value in {
        "secret_key": "x",
        "app_name": "DocFlow",
        "broker_host": "redis://x",
        "broker_backend": "redis://x",
        "OPENAI_API_KEY": "x",
        "DATABASE_NAME": "x",
        "DATABASE_PASSWORD": "x",
        "DATABASE_HOST": "x",
        "DATABASE_PORT": "5432",
        "DATABASE_USER": "x",
        "MINIO_ENDPOINT": "x:9000",
        "MINIO_ACCESS_KEY": "x",
        "MINIO_SECRET_KEY": "x",
        "MINIO_BUCKET": "x",
        "MINIO_SECURE": "false",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("ENVIRONMENT", "SENTRY_ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)


def test_production_is_the_default(required):
    assert make_settings().ENVIRONMENT is Environment.PRODUCTION


@pytest.mark.parametrize("value", ["DEVELOPMENT", " Development ", "development"])
def test_the_environment_is_read_in_any_case(required, value):
    assert make_settings(ENVIRONMENT=value).ENVIRONMENT is Environment.DEVELOPMENT


def test_an_unknown_environment_stops_the_app_instead_of_guessing(required):
    with pytest.raises(ValidationError):
        make_settings(ENVIRONMENT="prod")


def test_sentry_uses_the_environment_unless_told_otherwise(required):
    assert make_settings(ENVIRONMENT="staging").sentry_environment == "staging"
    assert (
        make_settings(ENVIRONMENT="staging", SENTRY_ENVIRONMENT="eu-staging").sentry_environment
        == "eu-staging"
    )


def app_in(monkeypatch, environment: Environment) -> TestClient:
    monkeypatch.setattr(settings, "ENVIRONMENT", environment)
    app = FastAPI(**docs_settings())
    register_docs(app)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return TestClient(app)


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_production_serves_no_documentation(monkeypatch, path):
    client = app_in(monkeypatch, Environment.PRODUCTION)

    assert client.get(path).status_code == 404
    assert client.get("/ping").status_code == 200  # the API itself still works


@pytest.mark.parametrize("environment", [Environment.DEVELOPMENT, Environment.STAGING])
@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_other_environments_serve_the_documentation(monkeypatch, environment, path):
    assert app_in(monkeypatch, environment).get(path).status_code == 200


def test_every_log_line_says_which_environment_wrote_it(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", Environment.STAGING)
    out = io.StringIO()
    configure_logging(stream=out)
    logging.getLogger("tests.environment").warning("hello")
    shutdown_logging(keep_output=False)

    (line,) = [json.loads(text) for text in out.getvalue().splitlines()]
    assert line["environment"] == "staging"
