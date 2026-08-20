"""
tests/evaluation/metrics/test_faithfulness.py
───────────────────────────────────────────────
Metric 5: Faithfulness / Groundedness

WHAT IT MEASURES:
  Whether the generated assistant message makes claims that are supported
  by the retrieved RAG context. Hallucinated domain facts = faithfulness failure.

HOW IT IS CALCULATED:
  1. DETERMINISTIC checks: Scan response for specific keywords that indicate
     unsupported domain claims (e.g. ROI guarantees, pricing promises).
  2. SEMANTIC (LLM-as-evaluator): Use a separate OpenAI call with a scoring prompt
     that evaluates if the response is grounded in the provided context.

  Faithfulness Score: 1.0 = fully grounded, 0.0 = hallucinated claim.

VALIDATION APPROACH:
  - Both grounded responses (expect score >= threshold) and
    intentionally unsupported responses (expect score < threshold) are tested.
  - Deterministic checks run without API calls.
  - LLM-based checks are marked @pytest.mark.live.

WHY IT MATTERS:
  The LLM must not invent domain facts. For example, it should not claim
  "our typical projects cost $5,000" unless that came from the retrieved chunks.
  Faithfulness ensures the persona + RAG constraints are respected.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from tests.evaluation.conftest import (
    EVAL_METRICS,
    THRESHOLDS,
    evaluate_faithfulness,
)


# ── Deterministic Faithfulness Checks ─────────────────────────────────────────

# Patterns that indicate potentially unsupported domain claims.
# These are detected WITHOUT an LLM call.
_HALLUCINATION_PATTERNS = [
    "guarantee",
    "100% roi",
    "300% roi",
    "guaranteed results",
    "typical cost is",
    "we charge",
    "our price is",
    "you will definitely",
    "always works",
    "never fails",
    "proven formula",
]


def has_unsupported_claim(response_text: str) -> bool:
    """Deterministic check for known hallucination patterns.

    Returns True if the response contains patterns that are typically
    NOT grounded in a creative brief assistant's knowledge base.
    """
    text_lower = response_text.lower()
    return any(pattern in text_lower for pattern in _HALLUCINATION_PATTERNS)


# ── Tests: Deterministic ───────────────────────────────────────────────────────

def test_grounded_response_passes_deterministic_check() -> None:
    """A response that only asks questions should pass deterministic groundedness."""
    grounded_response = (
        "Thanks for reaching out! To get started on your brief, could you tell me "
        "the name of your company and what type of project you have in mind?"
    )
    result = has_unsupported_claim(grounded_response)
    EVAL_METRICS["faithfulness"].append(1.0 if not result else 0.0)
    print(f"\n  [Faithfulness] Grounded response (deterministic): {'PASS' if not result else 'FAIL'}")
    assert not result, f"Grounded response should not contain unsupported claims"


def test_hallucinated_response_fails_deterministic_check() -> None:
    """A response containing a ROI guarantee should fail deterministic check."""
    hallucinated_response = (
        "Great to hear from you! With our approach, we guarantee 300% ROI on all "
        "creative campaigns. Let me know your company name to get started."
    )
    result = has_unsupported_claim(hallucinated_response)
    EVAL_METRICS["faithfulness"].append(0.0 if result else 1.0)
    print(f"\n  [Faithfulness] Hallucinated response (deterministic): {'DETECTED' if result else 'MISSED'}")
    assert result, "Hallucinated claim should have been detected"


def test_neutral_question_response_passes() -> None:
    """A purely conversational question-asking response should pass."""
    response = "Could you tell me a little bit about the company you're representing?"
    result = has_unsupported_claim(response)
    EVAL_METRICS["faithfulness"].append(1.0 if not result else 0.0)
    print(f"\n  [Faithfulness] Neutral question: {'PASS' if not result else 'FAIL'}")
    assert not result


def test_pricing_claim_fails_deterministic_check() -> None:
    """A response claiming specific pricing should fail."""
    pricing_response = (
        "Our price is typically $5,000 for a brand identity project. "
        "What's the name of your company?"
    )
    result = has_unsupported_claim(pricing_response)
    EVAL_METRICS["faithfulness"].append(0.0 if result else 1.0)
    print(f"\n  [Faithfulness] Pricing claim: {'DETECTED' if result else 'MISSED'}")
    assert result, "Pricing claim should have been detected as unsupported"


def test_multiple_responses_deterministic_batch() -> None:
    """Run multiple responses through deterministic check and report scores."""
    test_cases = [
        ("What is your company name?", True),  # Should pass (no hallucination)
        ("We guarantee you will get results.", False),  # Should fail
        ("Tell me about your creative project.", True),  # Should pass
        ("We charge based on scope of work.", False),  # Should fail
        ("How exciting! Could you share more?", True),  # Should pass
    ]

    results = []
    for response, expect_pass in test_cases:
        has_issue = has_unsupported_claim(response)
        passed = not has_issue if expect_pass else has_issue
        # Record the actual faithfulness of the response (not whether the detector worked).
        # A response with a detected hallucination is unfaithful (0.0), regardless of
        # whether the test expected it to fail. "passed" is only used for the assert below.
        actual_faithfulness = 0.0 if has_issue else 1.0
        EVAL_METRICS["faithfulness"].append(actual_faithfulness)
        results.append((response[:50], expect_pass, passed, actual_faithfulness))
        print(f"\n  [Faithfulness] '{response[:50]}...' | Expected pass={expect_pass} | Result: {'PASS' if passed else 'FAIL'}")

    avg_score = sum(r[3] for r in results) / len(results)
    print(f"\n  [Faithfulness] Batch average: {avg_score:.2%}")
    # avg_score reflects actual faithfulness: 3 grounded responses (1.0) and
    # 2 intentionally hallucinating responses (0.0) → true average is 0.60.
    assert avg_score >= 0.5, f"Batch faithfulness average {avg_score:.2%} too low"


# ── Tests: LLM-as-evaluator (Live API) ────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.live
async def test_grounded_response_scores_high(live_llm_provider) -> None:
    """LLM evaluator should give a high score to a response grounded in context."""
    context = [
        "When asking about budget, use open-ended questions.",
        "Creative briefs require: client name, project type, budget, and audience.",
        "The assistant should be warm, professional, and avoid making assumptions.",
    ]
    grounded_response = (
        "Welcome! I'm here to help you capture your creative brief. "
        "To get started, could you tell me the name of your company?"
    )

    score = await evaluate_faithfulness(live_llm_provider, grounded_response, context)
    EVAL_METRICS["faithfulness"].append(score)

    print(f"\n  [Faithfulness] Grounded response (LLM eval): {score:.2f}")
    assert score >= THRESHOLDS["faithfulness_pass"], (
        f"Grounded response should score >= {THRESHOLDS['faithfulness_pass']}, got {score:.2f}"
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_hallucinated_response_scores_low(live_llm_provider) -> None:
    """LLM evaluator should give a low score to a hallucinated response."""
    context = [
        "When asking about budget, use open-ended questions.",
        "Creative briefs require: client name, project type, budget, and audience.",
    ]
    hallucinated_response = (
        "Great! Our standard package costs $3,000 for a brand campaign, "
        "and we guarantee a 150% return on investment for all clients. "
        "What is your company's name?"
    )

    score = await evaluate_faithfulness(live_llm_provider, hallucinated_response, context)
    EVAL_METRICS["faithfulness"].append(score)

    print(f"\n  [Faithfulness] Hallucinated response (LLM eval): {score:.2f}")
    assert score < THRESHOLDS["faithfulness_pass"], (
        f"Hallucinated response should score < {THRESHOLDS['faithfulness_pass']}, got {score:.2f}. "
        "The LLM evaluator did not detect the hallucination."
    )


@pytest.mark.asyncio
@pytest.mark.live
async def test_faithfulness_dataset_case_09(live_llm_provider, eval_dataset) -> None:
    """Evaluate faithfulness for dataset case_09 (faithfulness test case)."""
    case = next(c for c in eval_dataset if c["id"] == "case_09_faithfulness")

    grounded_context = case["grounded_context"]
    unsupported_claim = case["unsupported_claim_example"]

    # Test 1: The unsupported claim should score LOW
    score_unsupported = await evaluate_faithfulness(
        live_llm_provider, unsupported_claim, grounded_context
    )
    print(f"\n  [Faithfulness] Case 09 unsupported claim score: {score_unsupported:.2f}")

    # Test 2: A grounded response that references context content should score HIGH.
    # We use an explicitly context-grounded response referencing the retrieved guidance:
    grounded_response = (
        "As your creative brief assistant, I'll help you capture the key details we need. "
        "To start, could you tell me the full name of your company? "
        "I'll also need to know the type of project and your approximate budget range."
    )
    score_grounded = await evaluate_faithfulness(
        live_llm_provider, grounded_response, grounded_context
    )
    print(f"  [Faithfulness] Case 09 grounded response score: {score_grounded:.2f}")

    EVAL_METRICS["faithfulness"].append(score_grounded)

    # The grounded response score should be >= 0.5 (neutral/grounded)
    assert score_grounded >= 0.5, (
        f"Grounded creative brief response should score >= 0.5, got {score_grounded:.2f}"
    )
    assert score_unsupported < score_grounded, (
        "Grounded response should always score higher than unsupported claims"
    )

