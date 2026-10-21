from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    content_type: str
    extension: str
    created_at: datetime


class PaginatedFileResponse(BaseModel):
    files: list[FileResponse]
    total: int
    skip: int
    limit: int
    has_more: bool
