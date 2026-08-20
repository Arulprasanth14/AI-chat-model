"""
app/infrastructure/persistence/session_repository.py
──────────────────────────────────────────────────────
Abstract SessionRepository interface.

The orchestrator depends only on this protocol, not on any concrete
Postgres/Redis/in-memory implementation. This makes the persistence layer
fully swappable and allows tests to use simple in-memory implementations
without spinning up a database.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.domain.conversation.state import ConversationState


@runtime_checkable
class SessionRepository(Protocol):
    """Protocol for conversation session persistence backends.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      Concrete implementations:
        - PostgresSessionRepository  (production — Neon Postgres)
        - InMemorySessionRepository  (testing — no database needed)

      The orchestrator never references a concrete class.
      Inject via app/api/deps.py.
    ─────────────────────────────────────────────────────────────────────
    """

    async def get_session(self, session_id: str) -> ConversationState | None:
        """Load a session by ID.

        Args:
            session_id: UUID string of the session.

        Returns:
            ConversationState if found, None otherwise.
        """
        ...

    async def create_session(self, profile_id: str) -> ConversationState:
        """Create and persist a new empty session.

        Args:
            profile_id: Profile active at session creation time.

        Returns:
            Newly created ConversationState with a fresh session_id.
        """
        ...

    async def save_session(self, state: ConversationState) -> None:
        """Persist the current state of an existing session.

        This is an upsert — it creates if not exists, updates if it does.

        Args:
            state: ConversationState to persist.
        """
        ...

    async def list_sessions(
        self,
        profile_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationState]:
        """List sessions, optionally filtered by profile.

        Args:
            profile_id: Optional filter.
            limit:      Maximum results.

        Returns:
            List of ConversationState (without full history for performance).
        """
        ...


class InMemorySessionRepository:
    """In-memory session repository for unit tests and local dev.

    Does not require a database connection. State is lost on process exit.
    Thread-safety is not guaranteed — suitable for single-threaded tests.
    """

    def __init__(self) -> None:
        self._store: dict[str, ConversationState] = {}

    async def get_session(self, session_id: str) -> ConversationState | None:
        return self._store.get(session_id)

    async def create_session(self, profile_id: str) -> ConversationState:
        state = ConversationState(profile_id=profile_id)
        self._store[state.session_id] = state
        return state

    async def save_session(self, state: ConversationState) -> None:
        self._store[state.session_id] = state

    async def list_sessions(
        self,
        profile_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationState]:
        sessions = list(self._store.values())
        if profile_id:
            sessions = [s for s in sessions if s.profile_id == profile_id]
        return sessions[:limit]
