"""
app/domain/rag/embedder.py
───────────────────────────
Abstract Embedder interface.

The RAG pipeline (retriever, ingester) depends only on the Embedder protocol
defined here.  Concrete implementations live in the infrastructure layer:

    app/infrastructure/rag/openai_embedder.py   ← current production impl
    app/infrastructure/rag/<other>_embedder.py  ← future alternative impls

To swap the embedding provider without touching RAG pipeline code:
  1. Create a new implementation in app/infrastructure/rag/.
  2. Update get_embedder() in app/api/deps.py to instantiate it.
  3. Zero changes to retriever, ingestion pipeline, or this file.

Backward-compat note:
  ``OpenAIEmbedder`` is still importable from this module for existing scripts
  that use ``from app.domain.rag.embedder import OpenAIEmbedder``.  New code
  should import it directly from app/infrastructure/rag/openai_embedder.py.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedding providers.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      The retriever and ingester depend only on this protocol.
      Swap the embedding provider without touching RAG pipeline code.
    ─────────────────────────────────────────────────────────────────────
    """

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single API call.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, same order as input.
        """
        ...


# ── Backward-compatibility re-export ──────────────────────────────────────────
# The concrete OpenAIEmbedder implementation has moved to the infrastructure
# layer.  This re-export keeps existing scripts working without modification.
# New code should import directly from:
#   app.infrastructure.rag.openai_embedder
from app.infrastructure.rag.openai_embedder import OpenAIEmbedder  # noqa: E402

__all__ = ["Embedder", "OpenAIEmbedder"]
