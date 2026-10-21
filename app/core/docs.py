from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

OPENAPI_URL = "/openapi.json"
DOCS_URL = "/docs"


def register_docs(app: FastAPI) -> None:
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
