"""
app/infrastructure/rag/embedding_gemma_embedder.py
───────────────────────────────────────────────────
Local embedding implementation using sentence-transformers to load
``google/embeddinggemma-300m`` (768-dim) directly from HuggingFace.

Model: google/embeddinggemma-300m
  - Architecture: Gemma 3 encoder (300M parameters)
  - Output dimensions: 768
  - Max sequence length: 8192 tokens  ← 32× longer than all-mpnet-base-v2 (384)
  - Training: Instruction-tuned for embedding tasks
  - Quality: Outperforms OpenAI text-embedding-3-small on MTEB benchmarks

Why sentence-transformers and not Ollama?
  ``google/embeddinggemma-300m`` is NOT available as an Ollama-pullable
  model. We load the weights through ``sentence-transformers``, which handles
  pooling and normalisation correctly for this encoder-style model.

SIMILARITY THRESHOLD NOTE:
  Cosine similarity scores from embeddinggemma-300m are distributed differently
  from OpenAI’s models. The typical score range for relevant content is ~0.2–0.65.
  The VECTOR_SIMILARITY_THRESHOLD env var is set to 0.25 to account for this.
  See: .env → VECTOR_SIMILARITY_THRESHOLD
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Output dimension for google/embedding-gemma-3-300m-it-v0 (and the
# google/embeddinggemma-300m HuggingFace model card family).
EMBEDDING_DIM = 768


@lru_cache(maxsize=1)
def _load_model(model_name: str):  # type: ignore[return]
    """Load and cache the SentenceTransformer model (once per process).

    The first call downloads weights from HuggingFace (or loads from local
    cache if already present).  Subsequent calls return the cached instance.

    Args:
        model_name: HuggingFace model identifier string.

    Returns:
        A loaded ``SentenceTransformer`` instance.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for EmbeddingGemmaEmbedder. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info(
        "Loading local embedding model (first call only)",
        extra={"model": model_name},
    )
    model = SentenceTransformer(model_name)
    logger.info(
        "Local embedding model loaded",
        extra={"model": model_name, "embedding_dim": EMBEDDING_DIM},
    )
    return model


class EmbeddingGemmaEmbedder:
    """Local sentence-transformers embedding implementation using Gemma encoder.

    Satisfies the ``Embedder`` protocol from ``app/domain/rag/embedder.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Default model: ``google/embedding-gemma-3-300m-it-v0``
    Output dimensions: 768

    The model is loaded once per process and cached (``_load_model`` is an
    ``lru_cache``), so repeated calls to ``embed`` / ``embed_batch`` incur no
    re-loading overhead.

    Args:
        model_name: HuggingFace model identifier (defaults to the Picasso
                    target model).  Override for testing with a lighter model.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This class is the ONLY file that must be modified (or replaced) when
      switching the embedding vendor.  Everything above this layer is agnostic.
    ─────────────────────────────────────────────────────────────────────
    """

    # Active model: google/embeddinggemma-300m
    # 768-dim | 8192 max tokens | instruction-tuned for retrieval tasks
    DEFAULT_MODEL = "google/embeddinggemma-300m"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        # Eagerly validate the import path so failures surface at startup, not
        # at first request.  The model itself is lazily loaded on first embed
        # call so startup doesn't block on download.
        try:
            import sentence_transformers  # noqa: F401  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            ) from exc

    # ── Public API (matches Embedder protocol) ─────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Delegates to ``embed_batch`` for code-reuse and consistent behaviour.

        Args:
            text: Input text to embed.

        Returns:
            768-dimensional embedding vector as a list of floats.
            Model: google/embeddinggemma-300m (max 8192 tokens).
        """
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, returning one 768-dim vector per input.

        Uses google/embeddinggemma-300m which supports up to 8192 tokens per
        input — no silent truncation for typical knowledge chunk sizes.

        Runs inference in the default executor so the async event loop is not
        blocked during the CPU/GPU inference call.

        Args:
            texts: List of input strings.

        Returns:
            List of 768-dimensional embedding vectors, same order as input.
        """
        if not texts:
            return []

        # Clean text (same convention as OpenAIEmbedder)
        cleaned = [t.replace("\n", " ").strip() for t in texts]

        logger.debug(
            "Embedding batch with local model",
            extra={"model": self._model_name, "batch_size": len(cleaned)},
        )

        # Run the synchronous SentenceTransformer inference off the event loop
        loop = asyncio.get_event_loop()
        embeddings: list[list[float]] = await loop.run_in_executor(
            None, self._encode_sync, cleaned
        )
        return embeddings

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode_sync(self, texts: list[str]) -> list[list[float]]:
        """Synchronous encode call — runs in the default thread executor.

        Args:
            texts: Pre-cleaned list of strings.

        Returns:
            List of embedding vectors as plain Python float lists.
        """
        model = _load_model(self._model_name)
        # encode() returns a numpy ndarray in production; convert to plain
        # Python lists so they are JSON-serialisable and pgvector-compatible.
        # In tests the mock returns a list-of-lists directly, so we handle both.
        raw = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        if hasattr(raw, "tolist"):
            # numpy ndarray path (production)
            return raw.tolist()  # type: ignore[union-attr]
        # Already a list-of-lists (test mock path)
        return [[float(v) for v in row] for row in raw]


    # ── Informational property (mirrors OpenAIEmbedder API) ───────────────────

    @property
    def model_name(self) -> str:
        """The embedding model identifier this provider is configured to use."""
        return self._model_name
