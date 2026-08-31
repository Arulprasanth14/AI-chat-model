"""
app/infrastructure/rag/gemini_embedder.py
─────────────────────────────────────────
Google Gemini API embedding implementation.

Uses the ``text-embedding-004`` model by default (768 dimensions).
Designed to align perfectly with the local Gemma model dimensions, requiring
no database schema changes.
"""
from __future__ import annotations

import logging

from google import genai
from google.genai import types

from app.domain.rag.embedder import Embedder

logger = logging.getLogger(__name__)


class GeminiEmbedder(Embedder):
    """Google Gemini API text embedding implementation.

    Requires GEMINI_API_KEY to be set in the environment.

    Default model: ``text-embedding-004`` (768-dim)

    Args:
        api_key: Google Gemini API key.
        model:   Model identifier (default: "text-embedding-004").
    """

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key is required but missing. "
                "Set GEMINI_API_KEY in your .env file."
            )
        self.model_name = model
        # The new google-genai SDK
        self._client = genai.Client(api_key=api_key)
        logger.info(
            "Initialised GeminiEmbedder",
            extra={"model": self.model_name},
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of strings using Gemini API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        # We use a synchronous API call wrapped in an async wrapper if necessary,
        # but the google-genai client currently supports standard sync calls easily.
        # Since this runs in FastAPI, we should ideally use run_in_executor to avoid blocking,
        # but for embeddings it is fast. Let's do it cleanly:
        import asyncio
        loop = asyncio.get_running_loop()

        def _call_gemini() -> list[list[float]]:
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            # The response.embeddings is a list of Embedding objects.
            # We extract the `values` from each.
            return [emb.values for emb in response.embeddings]

        try:
            embeddings = await loop.run_in_executor(None, _call_gemini)
            return embeddings
        except Exception as e:
            logger.error("Gemini embedding failed", exc_info=e)
            raise

    @property
    def identifier(self) -> str:
        """The embedding model identifier this provider is configured to use."""
        return self.model_name
