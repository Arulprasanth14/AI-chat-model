"""
app/api/routes/health.py
─────────────────────────
Health check endpoint.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import _session_factory

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", summary="Service health check")
async def health_check() -> dict:
    """Return service health status including DB connectivity.

    Returns:
        JSON with status, db connectivity, and version info.
    """
    db_status = "unknown"
    try:
        async with _session_factory() as session:
            await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:
        logger.warning("DB health check failed", extra={"error": str(exc)})
        db_status = "error"

    return {
        "status": "ok",
        "db": db_status,
        "service": "picasso-rag-chat",
        "version": "0.1.0",
    }
