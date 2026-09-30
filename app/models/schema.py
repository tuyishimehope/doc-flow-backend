from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Text, JSON, Float, Integer, ForeignKey, Index, DateTime, and_, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base_class import Base
from app.service.document.schema import Document_Status, Processing_Type, Processing_status, Processing_Job_Status


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[Document_Status] = mapped_column(
        default=Document_Status.ACTIVE, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)

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


class File(Base):
    __tablename__ = "file"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=True)
    content_type: Mapped[str] = mapped_column(String(255), nullable=True)
    extension: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="file",
        uselist=False,
    )


class Processing_Request(Base):
    __tablename__ = "processing_request"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True)

    document_id: Mapped[int] = mapped_column(Integer, ForeignKey(
        "document.id", ondelete="CASCADE"), nullable=False, index=True)
    processing_type: Mapped[Processing_Type] = mapped_column(
        default=Processing_Type.DOCUMENT_SUMMARY)
    instructions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Processing_status] = mapped_column(
        default=Processing_status.PENDING, index=True)

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    document: Mapped["Document"] = relationship(
        "Document", back_populates="processing_requests")
    extracted_results: Mapped[List["Extracted_Result"]] = relationship(
        "Extracted_Result", back_populates="processing_request", cascade="all, delete-orphan"
    )
    processing_jobs: Mapped[List["Processing_Job"]] = relationship(
        "Processing_Job", back_populates="processing_request", cascade="all, delete-orphan"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)


class Extracted_Result(Base):
    __tablename__ = "extracted_result"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True)
    processing_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_request.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    result_type: Mapped[str] = mapped_column(String(100), nullable=False)
    content_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    processing_request: Mapped["Processing_Request"] = relationship(
        "Processing_Request", back_populates="extracted_results")


class Processing_Job(Base):
    __tablename__ = "processing_job"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True)
    processing_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("processing_request.id", ondelete="CASCADE"), nullable=False, index=True
    )
    attempt_number: Mapped[int] = mapped_column(
        Integer, default=1, nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=True
    )
    failure_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worker_name: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True)
    status: Mapped[Processing_Job_Status] = mapped_column(
        default=Processing_Job_Status.QUEUED, index=True)

    processing_request: Mapped["Processing_Request"] = relationship(
        "Processing_Request", back_populates="processing_jobs")

    __table_args__ = (
        Index("ix_processing_job_status_attempt", "status", "attempt_number"),
    )


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(30), nullable=False)
    last_name: Mapped[str] = mapped_column(String(30), nullable=False)
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(
        Text(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    deleted_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    
    documents: Mapped[List["Document"]] = relationship("Document", back_populates="user",cascade="all, delete-orphan")
    
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
