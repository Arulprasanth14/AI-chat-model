"""
tests/evaluation/metrics/test_cosine_similarity.py
────────────────────────────────────────────────────
Metric 1: Cosine Similarity

WHAT IT MEASURES:
  The semantic closeness of a query embedding to retrieved chunk embeddings.
  pgvector returns cosine-based scores — this test validates them directly.

HOW IT IS CALCULATED:
  cosine_similarity(query_vector, chunk_vector) = dot(A,B) / (|A| * |B|)
  Range: -1 (opposite) to 1 (identical). Typical relevant chunks: > 0.30.

VALIDATION APPROACH:
  - DETERMINISTIC (no API key): uses local google/embeddinggemma-300m via
    sentence-transformers.  Runs in --no-live mode and feeds the report.
  - LIVE: Uses real OpenAI embeddings (requires OPENAI_API_KEY).
  - We embed domain-relevant query pairs and assert cosine ≥ 0.75.
  - We assert relevant query scores > THRESHOLDS["min_relevant_similarity"].
  - We verify that the relevant query scores higher than the irrelevant query.
  - We report all actual scores.

MARK: @pytest.mark.live — requires OPENAI_API_KEY for live tests.
      Deterministic tests run always (no API key needed).
"""
from __future__ import annotations

import math

import pytest

from tests.evaluation.conftest import EVAL_METRICS, THRESHOLDS, cosine_similarity


# ── Helper: compute pairwise similarity ────────────────────────────────────────

def _avg_similarity(query_vec: list[float], chunk_vecs: list[list[float]]) -> float:
    """Return average cosine similarity between query and each chunk vector."""
    if not chunk_vecs:
        return 0.0
    scores = [cosine_similarity(query_vec, cv) for cv in chunk_vecs]
    return sum(scores) / len(scores)


# ── Deterministic tests using local embeddinggemma-300m (no API key needed) ───
#
# These use the real sentence-transformers model to produce genuine cosine
# similarity scores in --no-live mode so the evaluation report is never N/A.
#
# Probe pairs — semantically very similar (expected cosine ≥ 0.75)
_PROBE_PAIRS = [
    (
        "What is the budget for this marketing project?",
        "How much money is allocated for the marketing campaign?",
    ),
    (
        "The client wants a logo design for their brand.",
        "We need brand identity work including a new logo.",
    ),
    (
        "Target audience is young professionals aged 25-35.",
        "The demographic focus is millennials and young working adults.",
    ),
    (
        "Project timeline is 6 weeks from kick-off to delivery.",
        "The deadline is 6 weeks after project start.",
    ),
    (
        "Campaign should focus on social media channels like Instagram and LinkedIn.",
        "The marketing strategy targets Instagram and LinkedIn platforms.",
    ),
]

_COSINE_PASS_THRESHOLD = 0.75
_LOCAL_MODEL = "google/embeddinggemma-300m"


def _load_local_model():
    """Load the local sentence-transformers model (cached after first load)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        return SentenceTransformer(_LOCAL_MODEL)
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_cosine_similarity_probe_pairs_local_embedder() -> None:
    """Embed 5 semantically-similar probe pairs with the local embeddinggemma-300m
    model and assert average cosine similarity ≥ 0.75.

    This is a deterministic test — no API key required.  It runs in --no-live
    mode so the evaluation report always shows a real cosine score.
    """
    model = _load_local_model()
    scores: list[float] = []

    for i, (q, doc) in enumerate(_PROBE_PAIRS, 1):
        vecs = model.encode([q, doc], normalize_embeddings=True)
        sim = float(vecs[0] @ vecs[1])   # dot of L2-normalised = cosine
        scores.append(sim)
        EVAL_METRICS["cosine_similarity"].append(sim)
        status = "PASS" if sim >= _COSINE_PASS_THRESHOLD else "FAIL"
        print(
            f"\n  [Cosine Similarity] Pair {i}: {sim:.4f}  [{status}]"
            f"\n    Q:   {q[:70]}"
            f"\n    DOC: {doc[:70]}"
        )

    avg = sum(scores) / len(scores)
    overall = avg >= _COSINE_PASS_THRESHOLD
    print(f"\n  [Cosine Similarity] Average: {avg:.4f}  (threshold ≥ {_COSINE_PASS_THRESHOLD})")
    assert overall, (
        f"Average cosine similarity {avg:.4f} is below the required threshold of "
        f"{_COSINE_PASS_THRESHOLD}. Individual scores: {[f'{s:.4f}' for s in scores]}"
    )


def test_cosine_negative_pair_is_low_local_embedder() -> None:
    """Semantically unrelated texts should have cosine similarity well below 0.70."""
    model = _load_local_model()
    q   = "What is the budget for this marketing project?"
    doc = "The weather in London is rainy and cold today."
    vecs = model.encode([q, doc], normalize_embeddings=True)
    sim  = float(vecs[0] @ vecs[1])
    print(f"\n  [Cosine Similarity] Negative pair: {sim:.4f}  (expected < 0.70)")
    assert sim < 0.70, (
        f"Unrelated texts should have similarity < 0.70, got {sim:.4f}"
    )

# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.live
async def test_relevant_query_produces_nonzero_similarity(live_embedder) -> None:
    """A domain-relevant query should produce a positive cosine similarity.

    We embed both a relevant query and a chunk of related content, then
    verify that cosine similarity is above the minimum threshold.
    This validates that the embedding model correctly represents semantic closeness.
    """
    relevant_query = "how should I ask about the project budget from a client"
    relevant_chunk = (
        "When gathering budget information from clients, use open-ended questions. "
        "Ask 'What is your approximate budget range for this project?' rather than "
        "suggesting figures first. Document the range they provide."
    )

    vecs = await live_embedder.embed_batch([relevant_query, relevant_chunk])
    query_vec, chunk_vec = vecs[0], vecs[1]

    similarity = cosine_similarity(query_vec, chunk_vec)
    EVAL_METRICS["cosine_similarity"].append(similarity)

    print(f"\n  [Cosine Similarity] Relevant pair: {similarity:.4f}")
    print(f"  [Cosine Similarity] Threshold:      {THRESHOLDS['min_relevant_similarity']}")

    assert similarity > THRESHOLDS["min_relevant_similarity"], (
        f"Expected relevant query similarity > {THRESHOLDS['min_relevant_similarity']}, "
        f"got {similarity:.4f}"
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_irrelevant_query_produces_lower_similarity(live_embedder) -> None:
    """An off-topic query should have lower similarity to domain content.

    Validates that the embedding model correctly separates unrelated queries
    from domain-specific knowledge chunks.
    """
    irrelevant_query = "how to bake chocolate cake with ganache frosting"
    domain_chunk = (
        "Creative briefs typically require: client name, project type, "
        "target audience, budget range, and desired timeline."
    )

    vecs = await live_embedder.embed_batch([irrelevant_query, domain_chunk])
    irrel_query_vec, domain_chunk_vec = vecs[0], vecs[1]

    similarity = cosine_similarity(irrel_query_vec, domain_chunk_vec)
    EVAL_METRICS["cosine_similarity"].append(similarity)

    print(f"\n  [Cosine Similarity] Irrelevant pair: {similarity:.4f}")
    print(f"  [Cosine Similarity] Max threshold:   {THRESHOLDS['max_irrelevant_similarity']}")

    assert similarity < THRESHOLDS["max_irrelevant_similarity"], (
        f"Expected irrelevant query similarity < {THRESHOLDS['max_irrelevant_similarity']}, "
        f"got {similarity:.4f}. This may indicate retrieval does not discriminate well."
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_relevant_query_scores_higher_than_irrelevant(live_embedder) -> None:
    """Relevant query must score higher than irrelevant query against domain content.

    This is the key ordering invariant: if both queries are sent to the same
    vector store, the domain-related one MUST rank higher.
    """
    domain_chunk = (
        "Creative briefs typically require: client name, project type, "
        "target audience, budget range, and desired timeline."
    )
    relevant_query = "what information do I need for a creative brief"
    irrelevant_query = "what is the best recipe for carbonara pasta"

    vecs = await live_embedder.embed_batch([relevant_query, irrelevant_query, domain_chunk])
    relevant_vec, irrelevant_vec, chunk_vec = vecs[0], vecs[1], vecs[2]

    relevant_score = cosine_similarity(relevant_vec, chunk_vec)
    irrelevant_score = cosine_similarity(irrelevant_vec, chunk_vec)

    EVAL_METRICS["cosine_similarity"].append(relevant_score)

    print(f"\n  [Cosine Similarity] Relevant query score:   {relevant_score:.4f}")
    print(f"  [Cosine Similarity] Irrelevant query score: {irrelevant_score:.4f}")
    print(f"  [Cosine Similarity] Margin:                 {relevant_score - irrelevant_score:.4f}")

    assert relevant_score > irrelevant_score, (
        f"Relevant query should score higher than irrelevant. "
        f"Relevant: {relevant_score:.4f}, Irrelevant: {irrelevant_score:.4f}"
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_identical_texts_produce_max_similarity(live_embedder) -> None:
    """Identical texts should produce cosine similarity very close to 1.0.

    Validates that the embedding model is self-consistent.
    """
    text = "What is the client's name and the type of project they need?"
    vecs = await live_embedder.embed_batch([text, text])
    similarity = cosine_similarity(vecs[0], vecs[1])

    print(f"\n  [Cosine Similarity] Self-similarity: {similarity:.6f}")

    assert similarity > 0.9999, (
        f"Expected identical texts to have similarity ~1.0, got {similarity:.6f}"
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_multiple_domain_queries_report_scores(live_embedder, eval_dataset) -> None:
    """Report cosine similarity scores for all RAG-query cases in the dataset."""
    rag_cases = [c for c in eval_dataset if "rag_query" in c]

    for case in rag_cases:
        query = case["rag_query"]
        reference_chunk = (
            "Creative brief assistant helps clients capture project requirements "
            "including name, project type, budget, and audience information."
        )
        vecs = await live_embedder.embed_batch([query, reference_chunk])
        score = cosine_similarity(vecs[0], vecs[1])
        EVAL_METRICS["cosine_similarity"].append(score)
        print(f"\n  [Cosine Similarity] Case '{case['id']}': {score:.4f}  (query: '{query}')")

    # This test is informational — always passes, reports scores
    assert len(rag_cases) > 0, "No RAG cases found in dataset"
