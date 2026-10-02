from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
class Processing_Type(str, Enum):
    DOCUMENT_SUMMARY = "DOCUMENT_SUMMARY"
    INVOICE_EXTRACTION = "INVOICE_EXTRACTION"
    CONTRACT_METADATA = "CONTRACT_METADATA"
    

class Processing_status(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    

class Document_Status(str, Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"

class Processing_Job_Status(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    RETRYING = "RETRYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEAD_LETTER = "DEAD_LETTER"
    
class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes= True)
    
    id: int
    name: str
    status: Document_Status
    file_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    
class PaginatedDocumentResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    skip: int
    limit: int
    has_more: bool


class DocumentProcessingResponse(BaseModel):
    document_id: int
    processing_request_id: int
    status: Processing_status


class ProcessingStatusResponse(BaseModel):
    id: int
    status: Processing_status


class ProcessingRequestResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    document_id: int
    processing_type: Processing_Type
    instructions: str | None
    status: Processing_status
    version: int
    created_at: datetime
    updated_at: datetime | None


class ProcessingJobResponse(BaseModel):
    attempt: int
    status: str
    created_at: datetime | None
    completed_at: datetime | None
    failure_reason: str | None = None


class DocumentUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str | None = Field(default=None, min_length=1, max_length=255)
    status: Literal[Document_Status.ACTIVE, Document_Status.ARCHIVED] | None = None

    @model_validator(mode="after")
    def require_a_change(self):
        if self.name is None and self.status is None:
            raise ValueError("Provide a name, a status, or both")
        return self


class ProcessRequest(BaseModel):
    processing_type: Processing_Type
    instructions: str | None = None


class ExportFormat(str, Enum):
    JSON = "json"
    CSV = "csv"


class InvoiceLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: float | None
    unit_price: float | None
    amount: float | None


class InvoiceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None
    vendor_name: str | None
    customer_name: str | None
    currency: str | None
    subtotal: float | None
    tax: float | None
    total: float | None
    line_items: list[InvoiceLineItem]


class ContractMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_type: str | None
    title: str | None
    parties: list[str]
    effective_date: str | None
    expiration_date: str | None
    governing_law: str | None
    renewal_terms: str | None
    termination_terms: str | None
    key_obligations: list[str]


class ProcessingResultResponse(BaseModel):
    result: str | InvoiceExtraction | ContractMetadata
    confidence_score: float | None = Field(
        default=None,
        description="No calibrated confidence score is provided by the current model workflow.",
    )


# Requests in these states have stopped and can be run again.
RETRYABLE_STATUSES = (Processing_status.FAILED, Processing_status.CANCELLED)


class DocumentStats(BaseModel):
    total: int
    active: int
    archived: int


class ProcessingStats(BaseModel):
    total: int
    by_status: dict[Processing_status, int]
    by_type: dict[Processing_Type, int]


class RecentFailure(BaseModel):
    processing_request_id: int
    document_id: int
    document_name: str
    processing_type: Processing_Type
    failure_reason: str | None
    failed_at: datetime | None


class UserStatsResponse(BaseModel):
    documents: DocumentStats
    processing_requests: ProcessingStats
    recent_failures: list[RecentFailure]
