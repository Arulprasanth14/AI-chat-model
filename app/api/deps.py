"""
app/api/deps.py
────────────────
FastAPI dependency injection functions.

All concrete implementations are wired here. To swap a provider (e.g., LLM
vendor, vector store, session backend), update only this file — the
orchestrator, routes, and domain code are unaffected.

Reusability note: The active profile is loaded once at startup and cached
as a module-level singleton. Changing ACTIVE_PROFILE in .env and restarting
the server switches the entire domain configuration.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.domain.conversation.field_set_loader import load_field_set
from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.conversation.structural_resolver import StructuralResolver
from app.domain.conversation.template_resolver import resolve
from app.domain.llm.provider import LLMProvider
from app.domain.llm.prompt_builder import PromptBuilder
from app.domain.rag.embedder import Embedder
from app.domain.rag.retriever import RAGRetriever
from app.infrastructure.llm.openai_provider import OpenAIProvider
from app.infrastructure.rag.openai_embedder import OpenAIEmbedder
from app.infrastructure.persistence.postgres_session_repo import PostgresSessionRepository
from app.infrastructure.vector_db.pgvector_client import PgVectorClient
from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)

# ── Database engine (module-level singleton) ───────────────────────────────────

_engine = create_async_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False,
)

_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    _engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ── Profile (loaded once at import time, cached forever) ───────────────────────

@lru_cache(maxsize=1)
def get_profile() -> BaseProfile:
    """Load the active project profile from YAML.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This is where ACTIVE_PROFILE env var maps to a concrete profile.yaml.
      New profile = new folder in app/project_profiles/ + env var change.
      Zero code changes needed.
    ─────────────────────────────────────────────────────────────────────

    Returns:
        Validated BaseProfile instance.

    Raises:
        FileNotFoundError: If the profile YAML does not exist.
    """
    profile_dir = (
        Path(__file__).parent.parent
        / "project_profiles"
        / settings.active_profile
    )
    yaml_path = profile_dir / "profile.yaml"

    logger.info(
        "Loading project profile",
        extra={"profile": settings.active_profile, "path": str(yaml_path)},
    )

    return BaseProfile.from_yaml(yaml_path)


# ── Provider singletons ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    """Return a cached LLM provider instance.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This is the single point where the concrete LLM implementation is
      wired up.  To switch from OpenAI to another vendor:
        1. Create a new provider class in app/infrastructure/llm/.
        2. Replace ``OpenAIProvider(...)`` below with the new class.
        3. Zero changes to orchestrator, prompts, or any domain code.
    ─────────────────────────────────────────────────────────────────────
    """
    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model=settings.chat_model,
    )


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a cached embedder instance.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This is the single point where the concrete embedding implementation
      is wired up.  To switch embedding providers, replace ``OpenAIEmbedder``
      below with a new class from app/infrastructure/rag/.
    ─────────────────────────────────────────────────────────────────────
    """
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PgVectorClient:
    """Return a cached pgvector client instance."""
    return PgVectorClient(
        engine=_engine,
        session_factory=_session_factory,
    )


@lru_cache(maxsize=1)
def get_session_repo() -> PostgresSessionRepository:
    """Return a cached Postgres session repository instance."""
    return PostgresSessionRepository(session_factory=_session_factory)


@lru_cache(maxsize=1)
def get_retriever() -> RAGRetriever:
    """Return a cached RAG retriever instance."""
    return RAGRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        top_k=settings.retrieval_top_k,
        similarity_threshold=settings.vector_similarity_threshold,
    )


@lru_cache(maxsize=1)
def get_prompt_builder() -> PromptBuilder:
    """Return a cached PromptBuilder instance."""
    return PromptBuilder()


# ── Orchestrator (per-request, not cached — stateless construction) ────────────

def _make_profile_provider(
    base_profile: BaseProfile,
) -> Callable[[ConversationState], BaseProfile]:
    """Create a callable that resolves the effective profile for a session."""
    profile_dir = Path(__file__).parent.parent / "project_profiles" / settings.active_profile
    field_sets_root = profile_dir / "field_sets"

    def provider(state: ConversationState) -> BaseProfile:
        # If we already resolved a template, try loading it
        if state.resolved_vertical and state.resolved_template_key:
            field_set_path = field_sets_root / state.resolved_vertical / f"{state.resolved_template_key}.yaml"
            fields = load_field_set(field_set_path)
            if fields:
                return base_profile.model_copy(update={"required_fields": fields})

        # Find the first user message to extract hints
        first_user_msg = next(
            (t["content"] for t in state.conversation_history if t["role"] == "user"), None
        )
        if not first_user_msg:
            return base_profile

        # Extract vertical hint by looking for available vertical folders
        vertical_hint = None
        msg_lower = first_user_msg.lower()
        if field_sets_root.exists():
            for v_dir in field_sets_root.iterdir():
                if not v_dir.is_dir():
                    continue
                v_name = v_dir.name
                v_clean = v_name.replace("realestate", "real estate").replace("_", " ")
                if v_clean in msg_lower or v_name in msg_lower:
                    vertical_hint = v_name
                    break

        template_hint = first_user_msg

        if vertical_hint and template_hint:
            resolved = resolve(vertical_hint, template_hint, field_sets_root)
            if resolved:
                state.resolved_vertical = resolved.vertical
                state.resolved_template_key = resolved.template_key
                fields = load_field_set(resolved.field_set_path)
                if fields:
                    return base_profile.model_copy(update={"required_fields": fields})

        return base_profile

    return provider


def get_orchestrator(
    profile: Annotated[BaseProfile, Depends(get_profile)],
    session_repo: Annotated[PostgresSessionRepository, Depends(get_session_repo)],
    retriever: Annotated[RAGRetriever, Depends(get_retriever)],
    llm: Annotated[LLMProvider, Depends(get_llm_provider)],
    prompt_builder: Annotated[PromptBuilder, Depends(get_prompt_builder)],
) -> ConversationOrchestrator:
    """Construct a ConversationOrchestrator for the current request.

    All dependencies are singletons — construction is cheap.
    """
    profile_dir = Path(__file__).parent.parent / "project_profiles" / settings.active_profile
    return ConversationOrchestrator(
        session_repo=session_repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=prompt_builder,
        profile_provider=_make_profile_provider(profile),
        structural_resolver=StructuralResolver(
            field_sets_root=profile_dir / "field_sets",
            knowledge_docs_root=profile_dir / "knowledge_docs",
        ),
        field_sets_root=profile_dir / "field_sets",
    )


# ── DB lifecycle helpers (used by main.py lifespan) ───────────────────────────

async def get_pgvector_client_initialised() -> PgVectorClient:
    """Initialise pgvector tables/indexes and return the client.

    Called once during application startup lifespan.
    """
    client = get_vector_store()
    await client.initialise()
    return client


async def dispose_engine() -> None:
    """Dispose the async SQLAlchemy engine pool.

    Called during application shutdown lifespan.
    """
    await _engine.dispose()
    logger.info("Database engine disposed")
