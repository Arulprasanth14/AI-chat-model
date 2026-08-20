"""
app/domain/rag/retriever.py
────────────────────────────
RAG retriever — converts a conversation context into relevant knowledge chunks.

The retriever is the bridge between the conversation state and the vector store.
It builds a richer query embedding by combining the user's message with
missing field descriptions (so chunks relevant to uncaptured fields rank higher).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.domain.conversation.state import MissingField
from app.domain.rag.embedder import Embedder
from app.domain.rag.vector_store import VectorStore
from app.domain.llm.prompt_builder import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuery:
    """Inputs used to construct the retrieval query embedding."""

    user_message: str
    recent_history: list[dict]  # last N turns [{role, content}]
    missing_fields: list[MissingField]
    profile_id: str
    industry: str | None = None
    brief_type: str | None = None


class RAGRetriever:
    """Retrieves knowledge chunks relevant to the current conversation turn.

    Builds an enriched query from the user message + missing field descriptions,
    embeds it, and runs a pgvector similarity search.

    Args:
        embedder:     Embedder implementation for query vectorisation.
        vector_store: VectorStore backend (pgvector or mock for tests).
        top_k:        Default number of chunks to return.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    async def retrieve(
        self,
        query: RetrievalQuery,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant knowledge chunks for this conversation turn.

        Query construction:
          - Starts with the user's raw message text.
          - Appends descriptions of currently missing fields (so the vector
            search gravitates toward chunks that help address those gaps).
          - Appends the last assistant message for additional context.

        Args:
            query:  Structured retrieval query with conversation context.
            top_k:  Override the default top_k for this call.

        Returns:
            List of RetrievedChunk objects ordered by descending similarity.
        """
        k = top_k or self._top_k
        query_text = self._build_query_text(query)

        logger.debug(
            "Retrieval query constructed",
            extra={
                "profile_id": query.profile_id,
                "query_length": len(query_text),
                "missing_fields": len(query.missing_fields),
            },
        )

        # Embed the enriched query
        embedding = await self._embedder.embed(query_text)

        # Run similarity search
        results = await self._vector_store.search(
            query_embedding=embedding,
            profile_id=query.profile_id,
            top_k=k,
            industry=query.industry,
            brief_type=query.brief_type,
        )

        # Filter out results below the similarity threshold
        filtered_results = [r for r in results if r.score >= self._similarity_threshold]

        # Log retrieval scores for eval pipeline
        logger.info(
            "Retrieval results",
            extra={
                "profile_id": query.profile_id,
                "top_k": k,
                "scores": [round(r.score, 4) for r in filtered_results],
                "doc_types": [r.doc_type for r in filtered_results],
            },
        )

        return [
            RetrievedChunk(
                content=r.content,
                doc_type=r.doc_type,
                score=r.score,
                field_code=r.field_code,
            )
            for r in filtered_results
        ]

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_query_text(self, query: RetrievalQuery) -> str:
        """Construct the rich query text for embedding.

        Combines user message + missing field descriptions + last assistant turn.
        Missing field descriptions push the embedding toward chunks that address
        information gaps in the current brief.
        """
        parts: list[str] = [query.user_message]

        # Add missing field descriptions to pull toward relevant guidance chunks
        if query.missing_fields:
            field_context = " ".join(
                " ".join(mf.description.split()[:12])  # first ~12 words per field
                for mf in query.missing_fields[:5]  # cap at 5 fields
            )
            parts.append(f"Looking for information about: {field_context}")

        # Add last assistant message for conversational continuity
        for turn in reversed(query.recent_history):
            if turn.get("role") == "assistant":
                parts.append(turn["content"][:200])  # first 200 chars
                break

        return " ".join(parts)
