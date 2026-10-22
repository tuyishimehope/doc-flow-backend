from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    and_,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base
from app.service.document.schema import (
    Document_Status,
    Processing_Job_Status,
    Processing_status,
    Processing_Type,
)


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[Document_Status] = mapped_column(default=Document_Status.ACTIVE, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=True
    )

    processing_requests: Mapped[List["Processing_Request"]] = relationship(
        "Processing_Request", back_populates="document", cascade="all, delete-orphan"
    )
    file_id: Mapped[int] = mapped_column(
        ForeignKey("file.id"),
        unique=True,
        nullable=False,
    )

    file: Mapped["File"] = relationship(
        "File",
        back_populates="document",
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)

    user: Mapped["User"] = relationship("User", back_populates="documents")

    @classmethod
    def is_active(cls):
        """SQL filter for documents that have not been soft-deleted."""
        return and_(cls.deleted_at.is_(None), cls.status != Document_Status.DELETED)


class DocumentPage(Base):
    """The text of one page of a PDF kept for questions. Page numbers start at 1.

    A copy of the file's content, so it is deleted (not just hidden) with the document.
    """

    __tablename__ = "document_page"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # The unique pair also serves "all pages of one document, in order".
    __table_args__ = (
        UniqueConstraint("document_id", "page", name="uq_document_page_document_id_page"),
    )


class File(Base):
    __tablename__ = "file"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(String(255), nullable=True)
    extension: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="file",
        uselist=False,
    )


class Processing_Request(Base):
    __tablename__ = "processing_request"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True
    )
    processing_type: Mapped[Processing_Type] = mapped_column(
        default=Processing_Type.DOCUMENT_SUMMARY
    )
    instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Processing_status] = mapped_column(
        default=Processing_status.PENDING, index=True
    )

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    document: Mapped["Document"] = relationship("Document", back_populates="processing_requests")
    extracted_results: Mapped[List["Extracted_Result"]] = relationship(
        "Extracted_Result", back_populates="processing_request", cascade="all, delete-orphan"
    )
    processing_jobs: Mapped[List["Processing_Job"]] = relationship(
        "Processing_Job", back_populates="processing_request", cascade="all, delete-orphan"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class Extracted_Result(Base):
    __tablename__ = "extracted_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processing_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("processing_request.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    result_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    processing_request: Mapped["Processing_Request"] = relationship(
        "Processing_Request", back_populates="extracted_results"
    )


class Processing_Job(Base):
    __tablename__ = "processing_job"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    processing_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("processing_request.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    attempt_number: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True
    )
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worker_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    status: Mapped[Processing_Job_Status] = mapped_column(
        default=Processing_Job_Status.QUEUED, index=True
    )

    processing_request: Mapped["Processing_Request"] = relationship(
        "Processing_Request", back_populates="processing_jobs"
    )

    __table_args__ = (Index("ix_processing_job_status_attempt", "status", "attempt_number"),)


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(30), nullable=False)
    last_name: Mapped[str] = mapped_column(String(30), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    documents: Mapped[List["Document"]] = relationship(
        "Document", back_populates="user", cascade="all, delete-orphan"
    )

    reset_tokens: Mapped[list[PasswordResetToken]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    user: Mapped[User] = relationship(back_populates="reset_tokens")


class Conversation(Base):
    """One chat thread. Only its owner may read or change it."""

    __tablename__ = "conversation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    # Set when the conversation is about one stored PDF (POST /documents/{id}/ask).
    document_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("document.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Bumped whenever a message is added, so "most recent chat first" works.
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )

    __table_args__ = (Index("ix_conversation_user_id_updated_at", "user_id", "updated_at"),)


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversation.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # complete | cancelled | error. Only "complete" messages are sent back to the model.
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="complete", server_default="complete"
    )
    model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # What the call cost, so usage can be summed and timed in SQL. All of them are NULL for
    # user messages and for answers cut off before the provider sent its usage; a provider
    # may also leave out cached or reasoning tokens.
    provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cached_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reasoning_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # Everything else: time to first piece, attempts, and later document citations.
    meta: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    # Which prompt file wrote this answer (for example ``chat_v1``). NULL for the user's own
    # messages and for answers saved before this column existed. No default on purpose: the
    # caller must say which prompt it used.
    prompt_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")

    @property
    def citations(self) -> Optional[list]:
        """The checked citations of a document answer (None for every other message)."""
        return (self.meta or {}).get("citations")

    @property
    def grounded(self) -> Optional[bool]:
        return (self.meta or {}).get("grounded")

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_message_role"),
        CheckConstraint("status IN ('complete', 'cancelled', 'error')", name="ck_message_status"),
        # "The last N messages of a conversation" is the hot query.
        Index("ix_message_conversation_id_id", "conversation_id", "id"),
    )


class Transcription(Base):
    __tablename__ = "transcription"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    audio_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
