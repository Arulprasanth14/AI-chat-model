"""
app/infrastructure/rag/openai_embedder.py
──────────────────────────────────────────
OpenAI implementation of the Embedder protocol.

This module lives in the *infrastructure* layer because it depends on a
third-party SDK (``openai``).  All domain and RAG pipeline code depends
only on the abstract ``Embedder`` protocol defined in
``app/domain/rag/embedder.py``.

To swap embedding providers (e.g., OpenAI → Cohere → a local model):
  1. Create a new implementation file here (e.g., ``cohere_embedder.py``).
  2. Update ``get_embedder()`` in ``app/api/deps.py`` to instantiate it.
  3. Zero changes to retriever, ingestion pipeline, or any domain code.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI text embedding implementation.

    Satisfies the ``Embedder`` protocol from ``app/domain/rag/embedder.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Uses the ``text-embedding-3-small`` model by default (1536 dimensions).
    Batches are sent as a single API request to minimise latency.

    Args:
        api_key: OpenAI API key.
        model:   Embedding model identifier.  Controlled from the
                 provider/configuration layer — never hardcoded in domain code.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This class is the ONLY file that must be modified (or replaced) when
      switching the embedding vendor.  Everything above this layer is agnostic.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one API call.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        # OpenAI recommends replacing newlines for embedding quality
        cleaned = [t.replace("\n", " ").strip() for t in texts]

        logger.debug(
            "Embedding batch",
            extra={"model": self._model, "batch_size": len(cleaned)},
        )

        response = await self._client.embeddings.create(
            input=cleaned,
            model=self._model,
        )

        # Response data is ordered to match input order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def model_name(self) -> str:
        """The embedding model identifier this provider is configured to use."""
        return self._model


# Protocol conformance is verified structurally — OpenAIEmbedder satisfies
# Embedder via duck typing (embed + embed_batch methods match the protocol).
