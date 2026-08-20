"""
app/domain/rag/vector_store.py
───────────────────────────────
Abstract VectorStore interface.

The retriever and ingester depend only on this protocol.
The concrete pgvector implementation lives in app/infrastructure/vector_db/.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class VectorSearchResult:
    """A single result from a vector similarity search."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        doc_type: str,
        score: float,
        field_code: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.doc_type = doc_type
        self.score = score
        self.field_code = field_code
        self.metadata = metadata or {}


class ChunkToUpsert:
    """A knowledge chunk ready to be stored in the vector store."""

    def __init__(
        self,
        chunk_id: str,
        profile_id: str,
        content: str,
        embedding: list[float],
        doc_type: str,
        source_file: str,
        chunk_index: int,
        industry: str | None = None,
        brief_type: str | None = None,
        field_code: str | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.profile_id = profile_id
        self.content = content
        self.embedding = embedding
        self.doc_type = doc_type
        self.source_file = source_file
        self.chunk_index = chunk_index
        self.industry = industry
        self.brief_type = brief_type
        self.field_code = field_code


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector storage backends.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      The retriever and ingester depend only on this protocol.
      Swap from pgvector to any other vector DB by implementing this
      protocol and updating the injection in app/api/deps.py.
    ─────────────────────────────────────────────────────────────────────
    """

    async def upsert(self, chunks: list[ChunkToUpsert]) -> int:
        """Insert or update a batch of knowledge chunks.

        Args:
            chunks: List of chunks with embeddings and metadata.

        Returns:
            Number of chunks successfully upserted.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        profile_id: str,
        top_k: int = 5,
        industry: str | None = None,
        brief_type: str | None = None,
        field_code: str | None = None,
    ) -> list[VectorSearchResult]:
        """Run a vector similarity search with optional metadata filters.

        Args:
            query_embedding: The query vector to search against.
            profile_id:      Required — narrows search to this profile's chunks.
            top_k:           Maximum results to return.
            industry:        Optional industry filter.
            brief_type:      Optional brief-type filter.
            field_code:      Optional field-code filter (find chunks for a specific field).

        Returns:
            List of results ordered by descending similarity score.
        """
        ...

    async def delete_by_profile(self, profile_id: str) -> int:
        """Delete all chunks for a profile (used during re-ingestion).

        Args:
            profile_id: Profile namespace to clear.

        Returns:
            Number of chunks deleted.
        """
        ...
