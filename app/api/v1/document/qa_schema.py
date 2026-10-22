from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.api.v1.chat.schema import MAX_MESSAGE_CHARS


class StoredDocumentResponse(BaseModel):
    document_id: int
    file_id: int
    name: str
    pages: int
    # NONE unless retrieval is on; then PROCESSING until the worker has embedded the chunks
    index_status: str = "NONE"


class AskDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS),
    ]
    # Leave it out to start a new conversation about this document.
    conversation_id: int | None = None

    # "auto": read only the nearest chunks when the document is indexed and retrieval is switched
    # on, otherwise the whole text. "pages" and "chunks" force one way, to compare them.
    mode: Literal["auto", "pages", "chunks"] = "auto"


class IndexResponse(BaseModel):
    document_id: int
    index_status: str
