"""API documentation: Swagger UI at /docs, ReDoc at /redoc, the schema at /openapi.json.

None of them exist in production (``settings.docs_enabled``): the schema lists every route and
field, which is a map for an attacker and nothing a client of the API needs there.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

from app.core.config import settings

OPENAPI_URL = "/openapi.json"
DOCS_URL = "/docs"
REDOC_URL = "/redoc"


def docs_settings() -> dict[str, Any]:
    """The FastAPI(...) arguments for the documentation URLs. /docs is added by register_docs."""
    if not settings.docs_enabled:
        return {"openapi_url": None, "docs_url": None, "redoc_url": None}
    return {"openapi_url": OPENAPI_URL, "docs_url": None, "redoc_url": REDOC_URL}


def register_docs(app: FastAPI) -> None:
    """Swagger UI with the dark theme. Does nothing when the documentation is off."""
    if app.openapi_url is None:
        return

    @app.get(
        DOCS_URL,
        include_in_schema=False,
    )
    async def swagger_ui():
        return get_swagger_ui_html(
            openapi_url=OPENAPI_URL,
            title="Docflow API Docs",
            swagger_css_url="/static/swagger-dark.css",
        )
