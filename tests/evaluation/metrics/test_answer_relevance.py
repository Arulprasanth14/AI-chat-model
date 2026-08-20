"""
tests/evaluation/metrics/test_answer_relevance.py
──────────────────────────────────────────────────
Metric 6: Answer Relevance

WHAT IT MEASURES:
  Whether the generated assistant message actually addresses the user's latest
  message and moves the conversation toward the expected next topic.

HOW IT IS CALCULATED:
  1. DETERMINISTIC checks: verify that the response does not completely ignore
     the user's message (e.g., after user says "Acme Corp", the reply should
     acknowledge it). Uses substring / keyword cross-checking.
  2. SEMANTIC (LLM-as-evaluator): Score from 0.0 to 1.0 measuring relevance
     and topic progression.

VALIDATION APPROACH:
  - Positive tests: response that correctly acknowledges user input.
  - Negative tests: response that ignores user input and asks random question.
  - Tests use both deterministic logic and live LLM evaluation.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.domain.rag.retriever import RAGRetriever
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from tests.evaluation.conftest import (
    EVAL_METRICS,
    THRESHOLDS,
    ConfigurableMockVectorStore,
    MockEmbedder,
    evaluate_answer_relevance,
)


# ── Deterministic Relevance Check ─────────────────────────────────────────────

def check_response_acknowledges_input(
    user_message: str,
    assistant_response: str,
    key_terms: list[str] | None = None,
) -> bool:
    """Deterministically check if the response acknowledges the user's input.

    A response is considered relevant if:
    - It contains at least one key term from the user's message, OR
    - It contains one of the expected key terms provided.

    Returns True if the response appears to acknowledge the user's message.
    """
    user_lower = user_message.lower()
    response_lower = assistant_response.lower()

    # Check provided key terms (e.g., "Acme Corp", "budget")
    if key_terms:
        for term in key_terms:
            if term.lower() in response_lower:
                return True

    # Fallback: check if any significant word from user message appears in response
    user_words = [w for w in user_lower.split() if len(w) > 4]
    for word in user_words:
        if word in response_lower:
            return True

    return False


# ── Deterministic Tests ────────────────────────────────────────────────────────

def test_response_acknowledges_company_name() -> None:
    """Response should acknowledge the user's company name."""
    user_message = "My company is called Acme Corp."
    assistant_response = "Great to meet you, Acme Corp! What type of project are you working on?"

    result = check_response_acknowledges_input(
        user_message, assistant_response, key_terms=["Acme Corp"]
    )
    score = 1.0 if result else 0.0
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Acknowledge company name: {'PASS' if result else 'FAIL'}")
    assert result, "Response should acknowledge the company name 'Acme Corp'"


def test_response_ignoring_user_input_detected() -> None:
    """A response that ignores the user's input should score low relevance."""
    user_message = "My company is called XYZ Studios."
    # Response ignores the name and asks about something completely different
    irrelevant_response = "What industry does your project target?"

    result = check_response_acknowledges_input(
        user_message, irrelevant_response, key_terms=["XYZ Studios"]
    )
    score = 1.0 if result else 0.0
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Ignored input detected: {'FAIL (detected)' if not result else 'PASS (not detected)'}")
    # The response does NOT mention "XYZ Studios" so result should be False
    assert not result, (
        "Response ignoring user's company name should fail acknowledgement check. "
        "This indicates the assistant is not tracking user input."
    )


def test_relevant_follow_up_on_budget_question() -> None:
    """After user mentions budget, response should acknowledge or reference it."""
    user_message = "Our budget is around $15,000 for this project."
    assistant_response = "A $15,000 budget sounds reasonable. What type of creative project do you have in mind?"

    result = check_response_acknowledges_input(
        user_message, assistant_response, key_terms=["$15,000", "budget"]
    )
    score = 1.0 if result else 0.0
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Budget acknowledgement: {'PASS' if result else 'FAIL'}")
    assert result


def test_positive_transition_toward_next_topic() -> None:
    """After extracting client_name, response should transition to asking about project_type."""
    user_message = "My name is BrightPath Inc."
    assistant_response = "Wonderful, BrightPath Inc.! What type of creative project do you need help with — brand identity, a social media campaign, or something else?"

    result = check_response_acknowledges_input(
        user_message, assistant_response, key_terms=["BrightPath", "project"]
    )
    score = 1.0 if result else 0.0
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Topic transition: {'PASS' if result else 'FAIL'}")
    assert result


def test_negative_abrupt_topic_change() -> None:
    """After user gives their name, response that randomly asks about something unrelated should be detected."""
    user_message = "Hi, I'm from Acme Corp."
    # Abruptly asking about timeline without acknowledging the name or even context
    off_topic_response = "Please describe your ideal ROI targets for this campaign."

    result = check_response_acknowledges_input(
        user_message, off_topic_response, key_terms=["Acme", "welcome", "hello", "name"]
    )
    score = 1.0 if result else 0.0
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Abrupt topic change: {'FAIL (detected)' if not result else 'PASS'}")
    # off_topic_response doesn't acknowledge "Acme Corp" or the greeting
    assert not result


def test_batch_relevance_deterministic() -> None:
    """Batch check multiple response pairs and report average relevance."""
    test_pairs = [
        # (user_message, assistant_response, key_terms, expect_relevant)
        ("We are TechFlow Inc.", "Welcome, TechFlow Inc.!", ["TechFlow"], True),
        ("Our budget is $5,000", "Got it, $5,000 budget!", ["5,000", "budget"], True),
        ("Hello!", "What is your project timeline?", ["hello", "hi", "welcome", "great"], False),
        ("Brand identity project", "Brand identity is exciting!", ["brand", "identity"], True),
        ("$20,000", "Tell me about your ROI.", ["20,000", "budget"], False),
    ]

    scores = []
    for user_msg, assistant_msg, key_terms, expect_relevant in test_pairs:
        result = check_response_acknowledges_input(user_msg, assistant_msg, key_terms)
        score = 1.0 if (result == expect_relevant) else 0.0
        scores.append(score)
        EVAL_METRICS["answer_relevance"].append(1.0 if result else 0.0)
        print(
            f"\n  [Answer Relevance] '{user_msg[:30]}' -> "
            f"{'relevant' if result else 'irrelevant'} | Expected: {'relevant' if expect_relevant else 'irrelevant'} | {'OK' if result == expect_relevant else 'FAIL'}"
        )

    avg = sum(scores) / len(scores)
    print(f"\n  [Answer Relevance] Batch accuracy: {avg:.2%}")
    assert avg >= 0.6, f"Answer relevance batch accuracy {avg:.2%} is too low"


# ── LLM-based Tests (Live API) ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.live
async def test_answer_relevance_relevant_response_scores_high(live_llm_provider) -> None:
    """LLM evaluator should give high score to a relevant response."""
    user_message = "My company is Stellar Studios."
    assistant_response = (
        "Great to meet you, Stellar Studios! To continue, could you tell me "
        "what type of creative project you need help with?"
    )

    score = await evaluate_answer_relevance(
        live_llm_provider, user_message, assistant_response, expected_next_topic="project_type"
    )
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Relevant response (LLM eval): {score:.2f}")
    assert score >= THRESHOLDS["answer_relevance_pass"], (
        f"Relevant response should score >= {THRESHOLDS['answer_relevance_pass']}, got {score:.2f}"
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_answer_relevance_irrelevant_response_scores_low(live_llm_provider) -> None:
    """LLM evaluator should give low score to a response that ignores user input."""
    user_message = "My company is called BrightPath Inc."
    # Response completely ignores the company name and asks something unrelated
    irrelevant_response = "Please describe your experience with marketing ROI metrics in detail."

    score = await evaluate_answer_relevance(
        live_llm_provider, user_message, irrelevant_response
    )
    EVAL_METRICS["answer_relevance"].append(score)
    print(f"\n  [Answer Relevance] Irrelevant response (LLM eval): {score:.2f}")
    assert score < THRESHOLDS["answer_relevance_pass"], (
        f"Irrelevant response should score < {THRESHOLDS['answer_relevance_pass']}, got {score:.2f}"
    )
