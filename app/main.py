"""
app/main.py
────────────
FastAPI application factory and lifespan.

Creates the FastAPI app, registers routers, configures CORS, and manages
the database connection lifecycle via the async lifespan context manager.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging

# Configure structured logging before any other imports that log
configure_logging(settings.log_level)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — handles startup and shutdown."""
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info(
        "picasso-rag-chat starting",
        extra={
            "profile": settings.active_profile,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
        },
    )

    # Initialise pgvector tables and indexes
    from app.api.deps import get_pgvector_client_initialised
    await get_pgvector_client_initialised()

    # Pre-warm the profile cache (fail fast if profile.yaml is misconfigured)
    from app.api.deps import get_profile
    profile = get_profile()
    logger.info(
        "Profile loaded",
        extra={
            "profile_id": profile.profile_id,
            "required_fields": len(profile.required_fields),
        },
    )

    # ── Neon keepalive ────────────────────────────────────────────────────────
    # Neon serverless suspends its compute unit after ~5 minutes of inactivity.
    # A suspended compute cold-starts cost 5-10s on the first subsequent request.
    # This background task sends a cheap SELECT 1 every 4 minutes to keep the
    # connection warm and prevent suspension during low-traffic periods.
    from app.api.deps import _engine
    from sqlalchemy import text as sa_text

    async def _neon_keepalive() -> None:
        """Background task — ping DB every 4 minutes to prevent Neon cold-starts."""
        while True:
            await asyncio.sleep(240)  # 4 minutes
            try:
                async with _engine.connect() as conn:
                    await conn.execute(sa_text("SELECT 1"))
                logger.debug("Neon keepalive ping sent")
            except Exception as exc:
                logger.warning("Neon keepalive ping failed", exc_info=exc)

    keepalive_task = asyncio.create_task(_neon_keepalive())
    logger.info("Neon keepalive task started (interval: 240s)")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    keepalive_task.cancel()
    try:
        await keepalive_task
    except asyncio.CancelledError:
        pass
    from app.api.deps import dispose_engine
    await dispose_engine()
    logger.info("picasso-rag-chat shutdown complete")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title="Picasso RAG Chat",
        description=(
            "Model-driven, retrieval-augmented conversational AI service "
            "for creative brief capture. Profile-driven — zero domain code."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    from app.api.routes.health import router as health_router
    from app.api.routes.conversation import router as conversation_router

    app.include_router(health_router)
    app.include_router(conversation_router)

    return app


app = create_app()
