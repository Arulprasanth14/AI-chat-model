"""
scripts/reembed_all.py
───────────────────────
One-off re-embedding script: iterates all rows in knowledge_chunks, generates
new 768-dim embeddings with the active embedder (EmbeddingGemmaEmbedder by
default), and updates each row in-place.

Run this AFTER executing the Alembic migration:
  alembic upgrade head

Usage:
  python scripts/reembed_all.py [--batch-size N] [--dry-run]

Flags:
  --batch-size N   Number of chunks to embed in one call (default: 32).
  --dry-run        Embed but do NOT write back to the DB (useful for testing).

Environment:
  Reads from .env (same as the application).  Set EMBEDDING_PROVIDER and
  LOCAL_EMBEDDING_MODEL to control which embedder is used.

Progress:
  Logs count processed / total to stdout.  Per-row failures are caught and
  logged without aborting the batch — a final summary shows total
  successes and failures.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.core.config import settings  # noqa: E402
from app.domain.conversation.models import KnowledgeChunkORM  # noqa: E402

# Bootstrap pgvector so Vector type is registered with SQLAlchemy
from pgvector.sqlalchemy import Vector  # noqa: E402, F401
from sqlalchemy import Column  # noqa: E402
if not hasattr(KnowledgeChunkORM, "embedding"):
    KnowledgeChunkORM.embedding = Column(Vector(768))  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reembed_all")


async def _get_embedder():
    """Instantiate the embedder chosen by EMBEDDING_PROVIDER."""
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        from app.infrastructure.rag.openai_embedder import OpenAIEmbedder
        logger.info("Using OpenAIEmbedder (1536-dim)")
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    else:
        from app.infrastructure.rag.embedding_gemma_embedder import EmbeddingGemmaEmbedder
        logger.info(
            "Using EmbeddingGemmaEmbedder (%s, 768-dim)",
            settings.local_embedding_model,
        )
        return EmbeddingGemmaEmbedder(model_name=settings.local_embedding_model)


async def reembed_all(batch_size: int = 32, dry_run: bool = False) -> None:
    """Main re-embedding routine.

    Args:
        batch_size: Number of chunks processed per embedding batch call.
        dry_run:    If True, embeddings are computed but NOT written to DB.
    """
    engine = create_async_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=2,
        pool_pre_ping=True,
        echo=False,
    )
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    embedder = await _get_embedder()

    # ── Count total rows ───────────────────────────────────────────────────────
    async with session_factory() as session:
        result = await session.execute(
            select(KnowledgeChunkORM.id, KnowledgeChunkORM.content)
        )
        all_rows: list[tuple[str, str]] = result.all()  # type: ignore[assignment]

    total = len(all_rows)
    if total == 0:
        logger.warning("knowledge_chunks table is empty — nothing to re-embed.")
        await engine.dispose()
        return

    logger.info(
        "Starting re-embed: %d chunks | batch_size=%d | dry_run=%s",
        total, batch_size, dry_run,
    )

    successes = 0
    failures = 0
    offset = 0

    while offset < total:
        batch = all_rows[offset : offset + batch_size]
        chunk_ids = [row[0] for row in batch]
        chunk_texts = [row[1] for row in batch]

        # ── Embed the batch ────────────────────────────────────────────────────
        try:
            embeddings = await embedder.embed_batch(chunk_texts)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Batch embedding FAILED (rows %d–%d): %s",
                offset,
                offset + len(batch) - 1,
                exc,
                exc_info=True,
            )
            failures += len(batch)
            offset += batch_size
            continue

        # ── Write back per-row ─────────────────────────────────────────────────
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            try:
                if not dry_run:
                    async with session_factory() as session:
                        async with session.begin():
                            await session.execute(
                                update(KnowledgeChunkORM)
                                .where(KnowledgeChunkORM.id == chunk_id)
                                .values(embedding=embedding)
                            )
                successes += 1
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Row update FAILED chunk_id=%s: %s",
                    chunk_id,
                    exc,
                    exc_info=True,
                )
                failures += 1

        offset += batch_size
        logger.info(
            "Progress: %d / %d  (ok=%d  fail=%d)",
            min(offset, total),
            total,
            successes,
            failures,
        )

    logger.info(
        "Re-embed complete. total=%d  success=%d  failure=%d  dry_run=%s",
        total,
        successes,
        failures,
        dry_run,
    )

    if failures:
        logger.warning(
            "%d rows failed — check logs above and re-run the script to retry.",
            failures,
        )

    await engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-embed all knowledge_chunks with the current embedder.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks per embedding batch call (default: 32).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compute embeddings but do NOT write to the database.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(reembed_all(batch_size=args.batch_size, dry_run=args.dry_run))
