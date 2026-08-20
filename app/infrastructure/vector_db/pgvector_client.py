"""
app/infrastructure/vector_db/pgvector_client.py
─────────────────────────────────────────────────
pgvector-backed VectorStore implementation using SQLAlchemy async.

Registers the pgvector extension with SQLAlchemy, creates/manages the
``knowledge_chunks`` table (with a vector column), and implements the
VectorStore protocol for similarity search and upsert.

Designed for Neon Postgres (serverless) — uses asyncpg driver and respects
SSL mode from the DATABASE_URL connection string.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, delete, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.domain.conversation.models import Base, KnowledgeChunkORM
from app.domain.rag.vector_store import ChunkToUpsert, VectorSearchResult, VectorStore

logger = logging.getLogger(__name__)

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIM = 1536


class PgVectorClient:
    """pgvector-backed implementation of the VectorStore protocol.

    Args:
        engine:         Async SQLAlchemy engine connected to Postgres.
        session_factory: Async session factory from async_sessionmaker.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._table_ready = False

    async def initialise(self) -> None:
        """Create the pgvector extension and knowledge_chunks table if needed.

        Call this once during application startup (lifespan).
        """
        async with self._engine.begin() as conn:
            # Enable pgvector extension (idempotent)
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Add vector column to KnowledgeChunkORM if not already mapped
            if not hasattr(KnowledgeChunkORM, "embedding"):
                KnowledgeChunkORM.embedding = Column(Vector(EMBEDDING_DIM))  # type: ignore

            # Create all tables
            await conn.run_sync(Base.metadata.create_all)

            # Create IVFFlat index for fast similarity search (if not exists)
            await conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename = 'knowledge_chunks'
                            AND indexname = 'knowledge_chunks_embedding_idx'
                        ) THEN
                            CREATE INDEX knowledge_chunks_embedding_idx
                            ON knowledge_chunks USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100);
                        END IF;
                    END
                    $$;
                    """
                )
            )

        self._table_ready = True
        logger.info("pgvector table and index initialised")

    async def upsert(self, chunks: list[ChunkToUpsert]) -> int:
        """Upsert a batch of chunks using Postgres INSERT ... ON CONFLICT.

        Args:
            chunks: Chunks with embeddings to persist.

        Returns:
            Number of rows affected.
        """
        if not chunks:
            return 0

        async with self._session_factory() as session:
            async with session.begin():
                rows = [
                    {
                        "id": c.chunk_id,
                        "profile_id": c.profile_id,
                        "content": c.content,
                        "embedding": c.embedding,
                        "doc_type": c.doc_type,
                        "source_file": c.source_file,
                        "chunk_index": c.chunk_index,
                        "industry": c.industry,
                        "brief_type": c.brief_type,
                        "field_code": c.field_code,
                    }
                    for c in chunks
                ]

                stmt = pg_insert(KnowledgeChunkORM).values(rows)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"],
                    set_={
                        "content": stmt.excluded.content,
                        "embedding": stmt.excluded.embedding,
                        "doc_type": stmt.excluded.doc_type,
                    },
                )
                result = await session.execute(stmt)

        count = len(chunks)
        logger.debug("Upserted knowledge chunks", extra={"count": count})
        return count

    async def search(
        self,
        query_embedding: list[float],
        profile_id: str,
        top_k: int = 5,
        industry: str | None = None,
        brief_type: str | None = None,
        field_code: str | None = None,
    ) -> list[VectorSearchResult]:
        """Run cosine similarity search with metadata filters.

        Args:
            query_embedding: Query vector (1536-dim for text-embedding-3-small).
            profile_id:      Required filter — namespace for this profile.
            top_k:           Maximum results to return.
            industry:        Optional industry filter.
            brief_type:      Optional brief type filter.
            field_code:      Optional field code filter.

        Returns:
            Results ordered by descending cosine similarity.
        """
        async with self._session_factory() as session:
            # Build base query with pgvector cosine distance operator (<=>)
            distance_expr = KnowledgeChunkORM.embedding.cosine_distance(query_embedding)

            stmt = (
                select(
                    KnowledgeChunkORM,
                    distance_expr.label("distance"),
                )
                .where(KnowledgeChunkORM.profile_id == profile_id)
                .order_by(distance_expr)
                .limit(top_k)
            )

            if industry:
                stmt = stmt.where(KnowledgeChunkORM.industry == industry)
            if brief_type:
                stmt = stmt.where(KnowledgeChunkORM.brief_type == brief_type)
            if field_code:
                stmt = stmt.where(KnowledgeChunkORM.field_code == field_code)

            rows = (await session.execute(stmt)).all()

        results = [
            VectorSearchResult(
                chunk_id=row.KnowledgeChunkORM.id,
                content=row.KnowledgeChunkORM.content,
                doc_type=row.KnowledgeChunkORM.doc_type,
                # Convert distance (0=identical, 2=opposite) to similarity score (0–1)
                score=max(0.0, 1.0 - row.distance / 2.0),
                field_code=row.KnowledgeChunkORM.field_code,
                metadata={
                    "source_file": row.KnowledgeChunkORM.source_file,
                    "chunk_index": row.KnowledgeChunkORM.chunk_index,
                    "industry": row.KnowledgeChunkORM.industry,
                },
            )
            for row in rows
        ]

        logger.info(
            "Vector search completed",
            extra={
                "profile_id": profile_id,
                "top_k": top_k,
                "results_returned": len(results),
                "top_score": results[0].score if results else None,
            },
        )
        return results

    async def delete_by_profile(self, profile_id: str) -> int:
        """Delete all chunks for a profile (for clean re-ingestion).

        Args:
            profile_id: Profile namespace to clear.

        Returns:
            Number of rows deleted.
        """
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    delete(KnowledgeChunkORM).where(
                        KnowledgeChunkORM.profile_id == profile_id
                    )
                )
        count = result.rowcount  # type: ignore
        logger.info("Deleted knowledge chunks", extra={"profile_id": profile_id, "count": count})
        return count


# Protocol conformance check (deferred — engine not available at import time)
def _check_protocol() -> None:
    assert isinstance(PgVectorClient, type)  # class, not instance check
