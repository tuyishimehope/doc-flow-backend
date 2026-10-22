from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

from app.api.v1.chat.schema import MAX_MESSAGE_CHARS


class StoredDocumentResponse(BaseModel):
    document_id: int
    file_id: int
    name: str
    pages: int


class AskDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS),
    ]
    # Leave it out to start a new conversation about this document.
    conversation_id: int | None = None
