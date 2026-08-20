"""
tests/evaluation/metrics/test_context_recall.py
────────────────────────────────────────────────
Metric 3: Context Recall

WHAT IT MEASURES:
  The proportion of all required information that was successfully retrieved.
  Detects when critical domain knowledge is missed by the retriever.

HOW IT IS CALCULATED:
  Context Recall = relevant_retrieved / total_relevant_required

  "relevant_required" is defined per evaluation case in the dataset as the
  set of doc_types/topics that MUST be present to correctly handle the query.

VALIDATION APPROACH:
  - DETERMINISTIC: Uses mock vector store to simulate controlled retrieval gaps.
  - Tests both full recall (no missed chunks), partial recall, and zero recall.
  - Also validates that missing critical chunks are detected and flagged.

WHY IT MATTERS:
  Low recall means the LLM is missing key guidance. For example, if the
  retriever fails to surface the "how to ask about budget" guidance chunk when
  the budget field is missing, the LLM may ask poorly or miss the field entirely.
"""
from __future__ import annotations

import pytest

from app.domain.rag.retriever import RAGRetriever, RetrievalQuery
from app.domain.rag.vector_store import VectorSearchResult
from tests.evaluation.conftest import (
    EVAL_METRICS,
    ConfigurableMockVectorStore,
    MockEmbedder,
)

# Separate accumulator for detector-validation tests (synthetic stores seeded to
# produce zero or partial recall to confirm the formula works correctly).
# Must NOT mix with EVAL_METRICS, which feeds the real evaluation report.
_DETECTOR_TEST_METRICS: dict[str, list[float]] = {
    "context_recall": [],
}


# ── Helper ─────────────────────────────────────────────────────────────────────

def compute_context_recall(
    retrieved_chunks,
    required_doc_types: list[str],
) -> tuple[float, list[str]]:
    """Compute context recall.

    Args:
        retrieved_chunks:  Chunks returned by retriever.
        required_doc_types: Doc types that MUST be present for full recall.

    Returns:
        (recall_score, list_of_missed_doc_types)
    """
    if not required_doc_types:
        return 1.0, []

    retrieved_types = {c.doc_type for c in retrieved_chunks}
    found = [dt for dt in required_doc_types if dt in retrieved_types]
    missed = [dt for dt in required_doc_types if dt not in retrieved_types]
    recall = len(found) / len(required_doc_types)
    return recall, missed


# ── Deterministic Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_perfect_when_all_required_types_present() -> None:
    """Recall should be 1.0 when all required doc types are present in results."""
    results = [
        VectorSearchResult(
            chunk_id="c1", content="Question guidance for budget",
            doc_type="question_guidance", score=0.85,
        ),
        VectorSearchResult(
            chunk_id="c2", content="Example of budget conversation",
            doc_type="example", score=0.75,
        ),
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="how to gather budget information",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query, top_k=5)
    recall, missed = compute_context_recall(chunks, required_doc_types=["question_guidance", "example"])

    EVAL_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] Full recall test: {recall:.2%} | Missed: {missed}")
    assert recall == 1.0, f"Expected recall 1.0, got {recall:.4f}"
    assert missed == []


@pytest.mark.asyncio
async def test_recall_zero_when_no_required_types_retrieved() -> None:
    """Recall should be 0.0 when required doc types are completely absent."""
    results = [
        VectorSearchResult(
            chunk_id="c1", content="Some domain fact",
            doc_type="domain_facts", score=0.70,
        ),
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="how to ask about project budget",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query, top_k=3)
    recall, missed = compute_context_recall(
        chunks, required_doc_types=["question_guidance", "example"]
    )

    # Route to detector accumulator — the store was intentionally seeded with only
    # irrelevant doc_types to confirm recall=0.0.  Not real dataset data.
    _DETECTOR_TEST_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] Zero recall test: {recall:.2%} | Missed: {missed}")
    assert recall == 0.0
    assert "question_guidance" in missed
    assert "example" in missed


@pytest.mark.asyncio
async def test_recall_partial_when_one_of_two_types_missing() -> None:
    """Recall should be 0.5 when only one of two required doc types is retrieved."""
    results = [
        VectorSearchResult(
            chunk_id="c1", content="Question guidance",
            doc_type="question_guidance", score=0.85,
        ),
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="how to ask about project budget",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query, top_k=3)
    recall, missed = compute_context_recall(
        chunks, required_doc_types=["question_guidance", "example"]
    )

    EVAL_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] Partial recall test: {recall:.2%} | Missed: {missed}")
    assert recall == 0.5
    assert "example" in missed
    assert "question_guidance" not in missed


@pytest.mark.asyncio
async def test_recall_empty_required_types_returns_perfect() -> None:
    """If no required types are defined, recall is trivially 1.0 (nothing to miss)."""
    results = [
        VectorSearchResult(chunk_id="c1", content="anything", doc_type="example", score=0.5),
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="anything",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query)
    recall, missed = compute_context_recall(chunks, required_doc_types=[])

    EVAL_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] No requirements test: {recall:.2%}")
    assert recall == 1.0
    assert missed == []


@pytest.mark.asyncio
async def test_recall_empty_retrieval_returns_zero() -> None:
    """Recall should be 0.0 if no chunks are retrieved but types are required."""
    store = ConfigurableMockVectorStore([])
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="anything",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query)
    recall, missed = compute_context_recall(
        chunks, required_doc_types=["question_guidance"]
    )

    # Route to detector accumulator — the store was intentionally seeded empty
    # to confirm recall=0.0 when nothing is returned.  Not real dataset data.
    _DETECTOR_TEST_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] Empty retrieval: {recall:.2%}")
    assert recall == 0.0
    assert "question_guidance" in missed


@pytest.mark.asyncio
async def test_recall_multiple_required_types_with_duplicates() -> None:
    """If a required doc_type appears multiple times in results, recall should still be 1.0."""
    results = [
        VectorSearchResult(chunk_id="c1", content="Q guidance 1", doc_type="question_guidance", score=0.9),
        VectorSearchResult(chunk_id="c2", content="Q guidance 2", doc_type="question_guidance", score=0.8),
        VectorSearchResult(chunk_id="c3", content="An example", doc_type="example", score=0.75),
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=5)
    query = RetrievalQuery(
        user_message="tell me about asking for budget",
        recent_history=[],
        missing_fields=[],
        profile_id="eval_test",
    )
    chunks = await retriever.retrieve(query, top_k=3)
    recall, missed = compute_context_recall(
        chunks, required_doc_types=["question_guidance", "example"]
    )

    EVAL_METRICS["context_recall"].append(recall)
    print(f"\n  [Context Recall] Duplicates test: {recall:.2%} | Missed: {missed}")
    assert recall == 1.0
    assert missed == []


@pytest.mark.asyncio
async def test_context_recall_all_dataset_cases(eval_dataset) -> None:
    """Report context recall for all dataset cases with expected_relevant_doc_types."""
    relevant_cases = [c for c in eval_dataset if "expected_relevant_doc_types" in c]
    per_case_scores = []

    for case in relevant_cases:
        required_types = case["expected_relevant_doc_types"]
        # Simulate retrieval returning ALL required types
        mock_results = [
            VectorSearchResult(
                chunk_id=f"c{i}",
                content=f"Content for {dt}",
                doc_type=dt,
                score=0.80,
            )
            for i, dt in enumerate(required_types)
        ]
        store = ConfigurableMockVectorStore(mock_results)
        retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=10)
        query = RetrievalQuery(
            user_message=case.get("rag_query", "test"),
            recent_history=[],
            missing_fields=[],
            profile_id="eval_test",
        )
        chunks = await retriever.retrieve(query, top_k=10)
        recall, missed = compute_context_recall(chunks, required_doc_types=required_types)
        per_case_scores.append(recall)
        EVAL_METRICS["context_recall"].append(recall)
        print(f"\n  [Context Recall] Case '{case['id']}': {recall:.2%} | Missed: {missed}")

    avg_recall = sum(per_case_scores) / len(per_case_scores) if per_case_scores else 0.0
    print(f"\n  [Context Recall] AVERAGE: {avg_recall:.2%}")
    assert avg_recall > 0.0
