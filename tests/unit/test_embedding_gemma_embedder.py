"""
tests/unit/test_embedding_gemma_embedder.py
────────────────────────────────────────────
Unit tests for EmbeddingGemmaEmbedder.

All tests mock the sentence-transformers model so no model weights need
to be downloaded in CI.  The mocking strategy patches:

  1. ``_load_model`` — the lru_cache'd model loader — so we never touch
     the real SentenceTransformer object.
  2. The ``sentence_transformers`` top-level import — so the ImportError
     guard in __init__ doesn't fire even if the package is absent.

The tests verify:
  • Protocol conformance: embed() and embed_batch() exist and return correct types.
  • Dimensions: output vectors are 768-dim.
  • Empty input handling: embed_batch([]) returns [].
  • Delegation: embed() delegates to embed_batch() (single-element call).
  • Model name property.
  • Batch size: embed_batch returns one vector per input text.
  • Normalisation: the embedder passes normalize_embeddings=True.
  • ImportError path: instantiation fails cleanly when sentence-transformers absent.
"""
from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import MagicMock, patch
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_model(dim: int = 768):
    """Return a mock SentenceTransformer that emits fixed-value embeddings."""
    mock = MagicMock()

    def encode(texts, convert_to_numpy=True, normalize_embeddings=True):
        n = len(texts)
        # Return a list of lists (same as ndarray.tolist() output)
        return [[1.0] * dim for _ in range(n)]

    mock.encode.side_effect = encode
    return mock


def _mock_st_package() -> None:
    """Inject a minimal fake ``sentence_transformers`` into sys.modules."""
    if "sentence_transformers" not in sys.modules:
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = MagicMock  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = fake


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _inject_fake_st():
    """Ensure sentence_transformers is importable (fake) during every test."""
    _mock_st_package()
    yield


@pytest.fixture()
def embedder_with_mock_model():
    """EmbeddingGemmaEmbedder with _load_model patched to return a mock."""
    # Clear the lru_cache so each test gets a fresh mock
    from app.infrastructure.rag import embedding_gemma_embedder as mod

    mod._load_model.cache_clear()
    mock_model = _make_mock_model(dim=768)

    with patch.object(mod, "_load_model", return_value=mock_model):
        from app.infrastructure.rag.embedding_gemma_embedder import EmbeddingGemmaEmbedder
        embedder = EmbeddingGemmaEmbedder()
        yield embedder, mock_model

    mod._load_model.cache_clear()


# ── Protocol conformance ───────────────────────────────────────────────────────

class TestProtocolConformance:
    def test_has_embed_method(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        assert callable(embedder.embed)

    def test_has_embed_batch_method(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        assert callable(embedder.embed_batch)

    def test_satisfies_embedder_protocol(self, embedder_with_mock_model) -> None:
        from app.domain.rag.embedder import Embedder
        embedder, _ = embedder_with_mock_model
        assert isinstance(embedder, Embedder)

    def test_model_name_property(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        assert embedder.model_name == "google/embeddinggemma-300m"


# ── Dimension correctness ──────────────────────────────────────────────────────

class TestEmbeddingDimensions:
    @pytest.mark.asyncio
    async def test_embed_returns_768_dims(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        vec = await embedder.embed("hello world")
        assert len(vec) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_returns_768_dims_each(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        vecs = await embedder.embed_batch(["first", "second", "third"])
        assert all(len(v) == 768 for v in vecs)

    @pytest.mark.asyncio
    async def test_embed_batch_count_matches_input(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        texts = ["a", "b", "c", "d", "e"]
        vecs = await embedder.embed_batch(texts)
        assert len(vecs) == len(texts)

    @pytest.mark.asyncio
    async def test_embed_returns_list_of_float(self, embedder_with_mock_model) -> None:
        embedder, _ = embedder_with_mock_model
        vec = await embedder.embed("test")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)


# ── Empty input handling ───────────────────────────────────────────────────────

class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty_list(self, embedder_with_mock_model) -> None:
        embedder, mock_model = embedder_with_mock_model
        result = await embedder.embed_batch([])
        assert result == []
        # The model should NOT have been called
        mock_model.encode.assert_not_called()


# ── Delegation ─────────────────────────────────────────────────────────────────

class TestDelegation:
    @pytest.mark.asyncio
    async def test_embed_delegates_to_embed_batch(self, embedder_with_mock_model) -> None:
        """embed() must call embed_batch() with a single-element list."""
        embedder, mock_model = embedder_with_mock_model
        # Patch embed_batch to track calls
        original_embed_batch = embedder.embed_batch
        calls: list[list[str]] = []

        async def tracking_embed_batch(texts):
            calls.append(texts)
            return await original_embed_batch(texts)

        embedder.embed_batch = tracking_embed_batch  # type: ignore[method-assign]
        await embedder.embed("single text")
        assert calls == [["single text"]]


# ── Text cleaning ──────────────────────────────────────────────────────────────

class TestTextCleaning:
    @pytest.mark.asyncio
    async def test_newlines_replaced_with_spaces(self, embedder_with_mock_model) -> None:
        """Input newlines should be replaced so the model sees clean text."""
        embedder, mock_model = embedder_with_mock_model
        await embedder.embed("line one\nline two\nline three")
        call_args = mock_model.encode.call_args
        texts_passed: list[str] = call_args[0][0]  # positional arg[0]
        assert "\n" not in texts_passed[0]
        assert "line one line two line three" == texts_passed[0]


# ── Normalisation flag ─────────────────────────────────────────────────────────

class TestNormalisation:
    @pytest.mark.asyncio
    async def test_normalize_embeddings_is_true(self, embedder_with_mock_model) -> None:
        """The embedder must request L2-normalised vectors."""
        embedder, mock_model = embedder_with_mock_model
        await embedder.embed("check normalise")
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True


# ── Custom model name ──────────────────────────────────────────────────────────

class TestCustomModelName:
    def test_custom_model_name_stored(self) -> None:
        from app.infrastructure.rag.embedding_gemma_embedder import EmbeddingGemmaEmbedder
        e = EmbeddingGemmaEmbedder(model_name="BAAI/bge-small-en-v1.5")
        assert e.model_name == "BAAI/bge-small-en-v1.5"


# ── Import error handling ──────────────────────────────────────────────────────

class TestImportError:
    def test_raises_import_error_when_st_absent(self) -> None:
        """EmbeddingGemmaEmbedder.__init__ raises ImportError if sentence-transformers cannot be imported.

        Strategy: patch the built-in __import__ inside the embedder's __init__
        so that 'import sentence_transformers' raises ImportError, regardless
        of whether the package is physically installed in the venv.
        """
        import builtins

        _real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("No module named 'sentence_transformers'")
            return _real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=_blocking_import):
            from app.infrastructure.rag.embedding_gemma_embedder import EmbeddingGemmaEmbedder
            with pytest.raises(ImportError, match="sentence-transformers"):
                EmbeddingGemmaEmbedder()


# ── Retriever compatibility (768-dim mock) ─────────────────────────────────────

class TestRetrieverCompatibility:
    """Verify that the 768-dim output is accepted by RAGRetriever mock infra."""

    @pytest.mark.asyncio
    async def test_retriever_uses_768_dim_embedding(self, embedder_with_mock_model) -> None:
        from app.domain.rag.retriever import RAGRetriever, RetrievalQuery
        from app.domain.rag.vector_store import VectorSearchResult

        captured_embeddings: list[list[float]] = []

        class CapturingVectorStore:
            async def upsert(self, chunks):
                return len(chunks)

            async def search(
                self,
                query_embedding,
                profile_id,
                top_k=5,
                industry=None,
                brief_type=None,
                field_code=None,
            ):
                captured_embeddings.append(query_embedding)
                return []

            async def delete_by_profile(self, profile_id):
                return 0

        embedder, _ = embedder_with_mock_model
        retriever = RAGRetriever(
            embedder=embedder,
            vector_store=CapturingVectorStore(),
            top_k=5,
        )
        await retriever.retrieve(
            RetrievalQuery(
                user_message="test message",
                recent_history=[],
                missing_fields=[],
                profile_id="test",
            )
        )
        assert len(captured_embeddings) == 1
        # The embedding passed to the vector store must be 768-dim
        assert len(captured_embeddings[0]) == 768
