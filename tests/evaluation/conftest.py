"""
tests/evaluation/conftest.py
─────────────────────────────
Shared fixtures and helpers for the Picasso AI evaluation suite.

This module provides:
  - A reusable eval_profile (BaseProfile with 3 required fields).
  - A reusable StubLLMProvider that can be configured to return specific
    tool-call responses for evaluation scenarios.
  - A reusable InMemorySessionRepository.
  - A reusable MockVectorStore and MockEmbedder for deterministic tests.
  - A LiveEmbedder fixture that uses the real OpenAI embedding API (requires .env).
  - A LiveVectorStore fixture for real pgvector searches (requires .env + DB).
  - An evaluate_faithfulness() helper for LLM-as-evaluator checks.
  - An evaluate_answer_relevance() helper.
  - Metric accumulation via the EVAL_METRICS global dict.
  - The evaluation dataset (evaluation_cases.json) loaded as a fixture.

ARCHITECTURE NOTE:
  - Tests marked @pytest.mark.live require OPENAI_API_KEY and DATABASE_URL.
  - All other tests are deterministic and use stubs/mocks only.
  - Do NOT import production code that touches FastAPI lifespan or DB connections
    at module level — use lazy imports inside fixtures.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

# Make sure the app root is on sys.path for all eval tests
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.domain.conversation.state import ConversationState, ExtractedAnswer
from app.domain.llm.prompt_builder import RetrievedChunk
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from app.project_profiles.base_profile import BaseProfile, FieldDefinition

# ── Global metric accumulator ──────────────────────────────────────────────────
# Each metric test appends its score here. run_evaluation.py reads this dict
# after pytest to print the summary table.
EVAL_METRICS: dict[str, list[float]] = {
    "cosine_similarity": [],
    "context_precision": [],
    "context_recall": [],
    "extraction_precision": [],
    "extraction_recall": [],
    "faithfulness": [],
    "answer_relevance": [],
    "state_transition_correct": [],
    "turns_to_completion": [],
    "goal_completion_success": [],
}

# ── Thresholds (single source of truth) ───────────────────────────────────────
THRESHOLDS = {
    # RAG: minimum acceptable cosine similarity for a domain-relevant query
    "min_relevant_similarity": 0.30,
    # RAG: maximum acceptable cosine similarity for an off-topic query
    "max_irrelevant_similarity": 0.65,
    # LLM: minimum confidence threshold to count a field as captured
    "extraction_confidence": 0.70,
    # LLM: minimum faithfulness score to pass the faithfulness test
    "faithfulness_pass": 0.60,
    # LLM: minimum answer relevance score to pass
    "answer_relevance_pass": 0.60,
    # End-to-end: maximum allowed turns before flagging as inefficient
    "max_efficient_turns": 8,
}


# ── Dataset loader ─────────────────────────────────────────────────────────────

def load_dataset() -> list[dict[str, Any]]:
    """Load the evaluation_cases.json dataset."""
    dataset_path = Path(__file__).parent / "datasets" / "evaluation_cases.json"
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def eval_dataset() -> list[dict[str, Any]]:
    return load_dataset()


# ── Standard eval profile (3 required fields) ─────────────────────────────────

@pytest.fixture
def eval_profile() -> BaseProfile:
    """A lightweight evaluation profile with 3 required fields."""
    return BaseProfile(
        profile_id="eval_test",
        persona_prompt=(
            "You are a creative brief assistant. Your goal is to gather information "
            "from clients to produce a complete creative brief. Ask questions naturally "
            "and warmly. Extract information as the client provides it."
        ),
        knowledge_namespace="eval_test",
        required_fields=[
            FieldDefinition(
                code="client_name",
                description="Full legal name of the client or company commissioning the project.",
                required=True,
            ),
            FieldDefinition(
                code="project_type",
                description="The type of creative project, e.g. brand identity, social campaign, pitch deck.",
                required=True,
            ),
            FieldDefinition(
                code="budget_range",
                description="Approximate budget or budget range available for this project.",
                required=True,
            ),
        ],
    )


# ── Stub / Mock infrastructure ─────────────────────────────────────────────────

class StubLLMProvider:
    """Returns a configurable tool-call JSON response without calling OpenAI.

    NOTE: This is a test stub, NOT a production component.
    """

    def __init__(self, tool_result: dict[str, Any]) -> None:
        self._tool_result = tool_result

    async def stream_tool_call(
        self,
        messages: list[dict],
        tool_schema: dict,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        yield json.dumps(self._tool_result)

    async def complete(self, messages: list[dict], temperature: float = 0.7) -> str:
        return self._tool_result.get("message", "")

    async def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        tool_calls = []
        for i, ans in enumerate(self._tool_result.get("extracted_answers", [])):
            tool_calls.append({
                "id": f"call_{i}",
                "name": "save_text_field",
                "arguments": json.dumps({
                    "field_code": ans["field_hint"].replace(" ", "_"),
                    "value": ans["value"],
                    "confidence": ans.get("confidence", 0.9),
                }),
            })
        return tool_calls


class MockEmbedder:
    """Returns deterministic embeddings for testing.

    NOTE: This is a test double. Do NOT use in production.
    """

    async def embed(self, text: str) -> list[float]:
        """Returns a fixed 768-dim vector seeded by text length for determinism."""
        seed = sum(ord(c) for c in text[:50]) / 10000.0
        return [seed] * 768

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class ConfigurableMockVectorStore:
    """Mock vector store with configurable results per query.

    NOTE: This is a test double.
    """

    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self._results = results or []

    async def upsert(self, chunks: list) -> int:
        return len(chunks)

    async def search(
        self,
        query_embedding: list[float],
        profile_id: str,
        top_k: int = 5,
        industry: str | None = None,
        brief_type: str | None = None,
        field_code: str | None = None,
    ) -> list[VectorSearchResult]:
        return self._results[:top_k]

    async def delete_by_profile(self, profile_id: str) -> int:
        return 0


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def in_memory_repo() -> InMemorySessionRepository:
    return InMemorySessionRepository()


# ── Live API fixtures (require real env vars) ──────────────────────────────────

@pytest.fixture(scope="function")
def openai_api_key() -> str:
    """Returns the OpenAI API key from the environment."""
    # Load from .env if not already set
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — skipping live API test")
    return key


@pytest.fixture(scope="function")
def live_embedder(openai_api_key: str):
    """Real OpenAI embedder. Used for live RAG metric tests."""
    from app.domain.rag.embedder import OpenAIEmbedder
    return OpenAIEmbedder(api_key=openai_api_key, model="text-embedding-3-small")


@pytest.fixture(scope="function")
def live_llm_provider(openai_api_key: str):
    """Real OpenAI LLM provider. Used for live LLM metric tests."""
    from app.domain.llm.openai_provider import OpenAIProvider
    return OpenAIProvider(api_key=openai_api_key, model="gpt-4.1")



# ── Cosine similarity helper ───────────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Higher = more similar.

    DETERMINISTIC — no model calls required.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── LLM-as-evaluator helpers (semantic checks) ────────────────────────────────

async def evaluate_faithfulness(
    llm_provider,
    assistant_message: str,
    retrieved_context: list[str],
) -> float:
    """Use an LLM to evaluate whether an assistant message is grounded in context.

    IMPORTANT: This is an EVALUATOR function, not part of the production system.
    It calls the LLM separately to score faithfulness.

    Returns:
        Float between 0.0 (hallucinated) and 1.0 (fully grounded).
    """
    if not retrieved_context:
        return 0.5  # Cannot evaluate without context

    context_block = "\n".join(f"- {c}" for c in retrieved_context)
    prompt = f"""You are an evaluation assistant. Your task is to assess whether an AI assistant's response is grounded in the provided context.

RETRIEVED CONTEXT:
{context_block}

ASSISTANT RESPONSE:
"{assistant_message}"

Evaluate: Does the assistant's response make any claims that are NOT supported by the retrieved context above?
- Score 1.0 if the response is fully grounded OR only adds harmless conversational warmth/empathy without inventing concrete domain specifics (like prices, timelines, guarantees, or specific project examples).
- Score 0.5 if the response is mostly neutral.
- Score 0.0 if the response invents NEW concrete domain specifics, examples, or guarantees not present in the context.

Respond with ONLY a single decimal number between 0.0 and 1.0."""

    messages = [
        {"role": "system", "content": "You are a precise evaluation assistant. Respond with only a decimal number."},
        {"role": "user", "content": prompt},
    ]
    try:
        response_text = await llm_provider.complete(messages, temperature=0.0)
        score = float(response_text.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.5  # Conservative default on parse error


async def evaluate_answer_relevance(
    llm_provider,
    user_message: str,
    assistant_message: str,
    expected_next_topic: str | None = None,
) -> float:
    """Use an LLM to evaluate whether the assistant's response is relevant to the user's input.

    IMPORTANT: This is an EVALUATOR function, not part of the production system.

    Returns:
        Float between 0.0 (irrelevant) and 1.0 (highly relevant).
    """
    next_topic_hint = (
        f"\nThe expected next conversation topic is: '{expected_next_topic}'"
        if expected_next_topic else ""
    )
    prompt = f"""You are an evaluation assistant. Assess whether an AI assistant's response is relevant to the user's message.

USER MESSAGE:
"{user_message}"

ASSISTANT RESPONSE:
"{assistant_message}"
{next_topic_hint}

Evaluate:
1. Does the assistant acknowledge or address the user's message?
2. Does the response naturally move the conversation forward?
3. (If expected_next_topic is given) Does the assistant move toward that topic?

Score:
- 1.0: Fully relevant, addresses user message, transitions naturally.
- 0.7: Mostly relevant, minor issues.
- 0.5: Partially relevant — assistant continues but somewhat ignores user input.
- 0.2: Mostly irrelevant or ignores user entirely.
- 0.0: Completely off-topic or broken response.

Respond with ONLY a single decimal number between 0.0 and 1.0."""

    messages = [
        {"role": "system", "content": "You are a precise evaluation assistant. Respond with only a decimal number."},
        {"role": "user", "content": prompt},
    ]
    try:
        response_text = await llm_provider.complete(messages, temperature=0.0)
        score = float(response_text.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.5


# ── Extraction accuracy helpers ────────────────────────────────────────────────

def compute_extraction_precision_recall(
    extracted: dict[str, str],
    expected: dict[str, str],
) -> tuple[float, float]:
    """Compute precision and recall for extracted field values.

    DETERMINISTIC — uses exact string matching (case-insensitive).

    Precision = correctly extracted fields / total extracted fields
    Recall    = correctly extracted fields / total expected fields

    Returns:
        (precision, recall) as floats in [0, 1].
    """
    if not extracted and not expected:
        return 1.0, 1.0
    if not extracted:
        return 0.0, 0.0
    if not expected:
        # Extracted when nothing expected = low precision
        return 0.0, 1.0

    true_positives = 0
    for key, expected_value in expected.items():
        actual_value = extracted.get(key, "")
        if actual_value and expected_value.lower() in actual_value.lower():
            true_positives += 1

    precision = true_positives / len(extracted) if extracted else 0.0
    recall = true_positives / len(expected) if expected else 0.0
    return precision, recall


# ── Auto-generated by run_evaluation.py ───────────────────────────────────────
import json as _json
from pathlib import Path as _Path

def pytest_sessionfinish(session, exitstatus):
    """Serialize EVAL_METRICS to JSON so run_evaluation.py can read the scores."""
    output_path = _Path(__file__).parent / "_metrics_report.json"
    try:
        output_path.write_text(_json.dumps(EVAL_METRICS, indent=2))
    except Exception:
        pass
