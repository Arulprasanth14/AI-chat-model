"""
app/infrastructure/persistence/postgres_session_repo.py
─────────────────────────────────────────────────────────
PostgreSQL-backed SessionRepository implementation (for Neon Postgres).

Translates between ConversationState (domain object) and
ConversationSessionORM (SQLAlchemy model) using async SQLAlchemy sessions.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.conversation.models import ConversationSessionORM
from app.domain.conversation.state import CapturedField, ConversationState

logger = logging.getLogger(__name__)


class PostgresSessionRepository:
    """Production session repository backed by Neon Postgres via asyncpg.

    Args:
        session_factory: SQLAlchemy async_sessionmaker bound to the DB engine.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_session(self, session_id: str) -> ConversationState | None:
        """Load a session by ID.

        Args:
            session_id: UUID string.

        Returns:
            ConversationState or None if not found.
        """
        async with self._session_factory() as session:
            row = await session.get(ConversationSessionORM, session_id)
            if row is None:
                return None
            return self._orm_to_state(row)

    async def create_session(self, profile_id: str) -> ConversationState:
        """Create and persist a new empty session.

        Args:
            profile_id: Profile active at creation time.

        Returns:
            New ConversationState with a fresh session_id.
        """
        state = ConversationState(profile_id=profile_id)
        orm = self._state_to_orm(state)

        async with self._session_factory() as session:
            async with session.begin():
                session.add(orm)

        logger.info("Session created", extra={"session_id": state.session_id, "profile_id": profile_id})
        return state

    async def save_session(self, state: ConversationState) -> None:
        """Upsert the current session state.

        Args:
            state: ConversationState to persist.
        """
        row_data = {
            "id": state.session_id,
            "profile_id": state.profile_id,
            "status": state.status,
            "conversation_history": state.conversation_history,
            "extracted_answers": {
                code: {
                    "value": cf.value,
                    "confidence": cf.confidence,
                    "turn_index": cf.turn_index,
                    "extracted_at": cf.extracted_at,
                }
                for code, cf in state.captured.items()
            },
            "model_believes_complete": state.model_believes_complete,
            "suggested_next_topic": state.suggested_next_topic,
            "resolved_vertical": state.resolved_vertical,
            "resolved_template_key": state.resolved_template_key,
            "updated_at": datetime.now(timezone.utc),
        }

        async with self._session_factory() as session:
            async with session.begin():
                stmt = pg_insert(ConversationSessionORM).values(**row_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "status": stmt.excluded.status,
                        "conversation_history": stmt.excluded.conversation_history,
                        "extracted_answers": stmt.excluded.extracted_answers,
                        "model_believes_complete": stmt.excluded.model_believes_complete,
                        "suggested_next_topic": stmt.excluded.suggested_next_topic,
                        "resolved_vertical": stmt.excluded.resolved_vertical,
                        "resolved_template_key": stmt.excluded.resolved_template_key,
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                await session.execute(stmt)

        logger.debug("Session saved", extra={"session_id": state.session_id})

    async def list_sessions(
        self,
        profile_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationState]:
        """List sessions ordered by most recently updated.

        Args:
            profile_id: Optional profile filter.
            limit:      Maximum number of results.

        Returns:
            List of ConversationState objects.
        """
        async with self._session_factory() as session:
            stmt = (
                select(ConversationSessionORM)
                .order_by(ConversationSessionORM.updated_at.desc())
                .limit(limit)
            )
            if profile_id:
                stmt = stmt.where(ConversationSessionORM.profile_id == profile_id)

            rows = (await session.execute(stmt)).scalars().all()

        return [self._orm_to_state(row) for row in rows]

    # ── Serialisation ──────────────────────────────────────────────────────────

    @staticmethod
    def _orm_to_state(row: ConversationSessionORM) -> ConversationState:
        """Convert an ORM row to a domain ConversationState object."""
        captured = {
            code: CapturedField(
                field_code=code,
                value=data["value"],
                confidence=data["confidence"],
                turn_index=data["turn_index"],
                extracted_at=data.get("extracted_at", ""),
            )
            for code, data in (row.extracted_answers or {}).items()
        }

        return ConversationState(
            session_id=row.id,
            profile_id=row.profile_id,
            status=row.status,
            conversation_history=row.conversation_history or [],
            captured=captured,
            model_believes_complete=row.model_believes_complete,
            suggested_next_topic=row.suggested_next_topic,
            resolved_vertical=row.resolved_vertical,
            resolved_template_key=row.resolved_template_key,
        )

    @staticmethod
    def _state_to_orm(state: ConversationState) -> ConversationSessionORM:
        """Convert a domain ConversationState to an ORM object (for insert)."""
        return ConversationSessionORM(
            id=state.session_id,
            profile_id=state.profile_id,
            status=state.status,
            conversation_history=state.conversation_history,
            extracted_answers={
                code: {
                    "value": cf.value,
                    "confidence": cf.confidence,
                    "turn_index": cf.turn_index,
                    "extracted_at": cf.extracted_at,
                }
                for code, cf in state.captured.items()
            },
            model_believes_complete=state.model_believes_complete,
            suggested_next_topic=state.suggested_next_topic,
            resolved_vertical=state.resolved_vertical,
            resolved_template_key=state.resolved_template_key,
        )
