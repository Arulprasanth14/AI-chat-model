"""
app/domain/conversation/models.py
───────────────────────────────────
SQLAlchemy async ORM models for conversation session persistence.

These models are infrastructure-facing — they map to Postgres tables.
The domain layer works with ``ConversationState`` (state.py), not these
ORM objects directly. The repository layer is responsible for translating
between the two.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base shared across all ORM models."""
    pass


class ConversationSessionORM(Base):
    """Persisted conversation session.

    Stores the full mutable state of a single user conversation including
    history, extracted answers, and session metadata.

    Table: ``conversation_sessions``
    """

    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="Session UUID, returned to client as sessionId",
    )
    profile_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Active profile at session creation time (e.g. 'picasso_fusion')",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="active",
        comment="Session lifecycle status: active | complete | abandoned",
    )

    # ── Conversation history ───────────────────────────────────────────────────
    # Stored as a JSONB array of {role: str, content: str, timestamp: str}
    conversation_history: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        comment="Ordered list of conversation turns [{role, content, timestamp}]",
    )

    # ── Extraction state ───────────────────────────────────────────────────────
    # Stored as a JSONB dict: {field_code: {value, confidence, turn_index}}
    extracted_answers: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Best extraction per field: {field_code: {value, confidence, turn}}",
    )

    # ── Model advisory flags ───────────────────────────────────────────────────
    model_believes_complete: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        comment="Advisory flag from LLM — does not gate completion, ledger does",
    )
    suggested_next_topic: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="LLM's suggested next topic (last turn) — for debugging/UX only",
    )

    # ── Timestamps ─────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationSessionORM id={self.id!r} "
            f"profile={self.profile_id!r} status={self.status!r}>"
        )


class KnowledgeChunkORM(Base):
    """Persisted vector knowledge chunk for RAG retrieval.

    Each row represents a single embedded text chunk from a profile's
    knowledge_docs/ folder. The ``embedding`` column is populated by the
    pgvector extension.

    Table: ``knowledge_chunks``
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    profile_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Profile namespace — used as primary retrieval filter",
    )
    doc_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Document type tag: question_guidance | domain_fact | example",
    )
    industry: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Optional industry tag for narrowed retrieval",
    )
    brief_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Optional brief type tag (e.g. brand_identity, social_campaign)",
    )
    field_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Optional field code this chunk specifically guides (nullable)",
    )
    source_file: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Originating filename for traceability",
    )
    chunk_index: Mapped[int] = mapped_column(
        nullable=False,
        comment="Zero-based index of this chunk within the source document",
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Raw text content of the chunk",
    )
    # embedding column is added dynamically by pgvector_client.py after
    # the pgvector extension registers the Vector type with SQLAlchemy
    if TYPE_CHECKING:
        embedding: Any

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<KnowledgeChunkORM id={self.id!r} "
            f"profile={self.profile_id!r} doc_type={self.doc_type!r}>"
        )
