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
  - DETERMINISTIC: We use real OpenAI embeddings but compare mathematically.
  - We embed a domain-relevant query and an irrelevant query.
  - We assert relevant query scores > THRESHOLDS["min_relevant_similarity"].
  - We verify that the relevant query scores higher than the irrelevant query.
  - We report all actual scores.

MARK: @pytest.mark.live — requires OPENAI_API_KEY.
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
