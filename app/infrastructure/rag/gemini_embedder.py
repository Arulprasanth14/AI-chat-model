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

    Default model: ``gemini-embedding-2`` (768-dim)

    Args:
        api_key: Google Gemini API key.
        model:   Model identifier (default: "gemini-embedding-2").
    """

    def __init__(self, api_key: str, model: str = "gemini-embedding-2") -> None:
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

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        batch_result = await self.embed_batch([text])
        return batch_result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of strings using Gemini API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        import asyncio
        loop = asyncio.get_running_loop()

        def _embed_single(text: str) -> list[float]:
            # Gemini API throws an error for completely empty whitespace strings.
            # Handle empty texts gracefully to avoid crashes.
            if not text or not text.strip():
                return [0.0] * 768

            response = self._client.models.embed_content(
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768,
                )
            )
            
            if not response.embeddings:
                logger.warning(f"Gemini API returned empty embeddings for text: {text[:50]}...")
                return [0.0] * 768
                
            values = response.embeddings[0].values
            if not values:
                return [0.0] * 768
            return values

        # Process sequentially with rate limiting to respect 100 req/min quota
        results = []
        for i, text in enumerate(texts):
            if i > 0:
                await asyncio.sleep(0.7)
            
            try:
                res = await loop.run_in_executor(None, _embed_single, text)
                results.append(res)
            except Exception as e:
                logger.error(f"Gemini embedding failed on chunk {i}", exc_info=e)
                raise
                
        return results

    @property
    def identifier(self) -> str:
        """The embedding model identifier this provider is configured to use."""
        return self.model_name
