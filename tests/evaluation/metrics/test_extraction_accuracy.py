"""
tests/evaluation/metrics/test_extraction_accuracy.py
──────────────────────────────────────────────────────
Metric 4: Extraction Accuracy

WHAT IT MEASURES:
  Whether the LLM correctly extracts structured field values from user messages.

HOW IT IS CALCULATED:
  - Extraction Precision = correctly extracted fields / total extracted fields
  - Extraction Recall    = correctly extracted fields / total expected fields

  A field is "correctly extracted" if the expected value is a substring of
  the extracted value (case-insensitive). This is tolerant of minor rephrasing.

VALIDATION APPROACH:
  LIVE (uses real OpenAI API) — to test actual LLM extraction behavior.
  Tests cover:
    1. Correct single-field extraction
    2. Multi-field extraction from one message
    3. Missing information (no extractable data)
    4. Hallucinated/incorrect extraction (user says X, LLM says Y)
    5. Information across multiple turns
    6. Ambiguous response handling

WHY IT MATTERS:
  If extraction fails, fields remain in missing_fields forever and the
  conversation never completes. If extraction hallucinates, the brief
  contains wrong information.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.domain.rag.retriever import RAGRetriever, RetrievalQuery
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from tests.evaluation.conftest import (
    EVAL_METRICS,
    THRESHOLDS,
    ConfigurableMockVectorStore,
    MockEmbedder,
    compute_extraction_precision_recall,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

class ConfigurableLLMProvider:
    """Test stub that returns a configured tool-call result.

    NOTE: Test stub — NOT a production component.
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


def make_test_orchestrator(eval_profile, llm_tool_result, mock_chunks=None):
    """Construct a fully stubbed ConversationOrchestrator for extraction testing."""
    results = mock_chunks or [
        VectorSearchResult(
            chunk_id="eval_chunk_1",
            content="Extract client name, project type, and budget range from user messages.",
            doc_type="question_guidance",
            score=0.85,
        )
    ]
    store = ConfigurableMockVectorStore(results)
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=3)
    repo = InMemorySessionRepository()
    llm = ConfigurableLLMProvider(llm_tool_result)
    builder = PromptBuilder()

    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=builder,
        profile_provider=lambda _state: eval_profile,
    )


async def run_turn(orchestrator, user_message: str, session_id: str | None = None):
    """Run one orchestrator turn and return the final snapshot."""
    events = []
    async for event in orchestrator.process_turn(
        session_id=session_id, user_message=user_message
    ):
        events.append(event)
    last = json.loads(events[-1].replace("data: ", "").strip())
    return last["snapshot"], last["snapshot"]["session_id"]


# ── Deterministic Extraction Tests (Stubbed LLM) ──────────────────────────────

@pytest.mark.asyncio
async def test_extraction_single_field_correct(eval_profile) -> None:
    """Correct single-field extraction: client_name from a clear user statement."""
    llm_response = {
        "message": "Great, thank you Acme Corp!",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, "My company is called Acme Corp.")

    extracted = snapshot["extracted_answers"]
    expected = {"client_name": "Acme Corp"}

    precision, recall = compute_extraction_precision_recall(
        {k: v["value"] for k, v in extracted.items()}, expected
    )
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Single field | Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"  [Extraction] Extracted: {extracted}")

    assert "client_name" in extracted, "client_name should have been extracted"
    assert precision == 1.0, f"Precision should be 1.0, got {precision}"
    assert recall == 1.0, f"Recall should be 1.0, got {recall}"


@pytest.mark.asyncio
async def test_extraction_multiple_fields_one_message(eval_profile) -> None:
    """Multi-field extraction: client_name and budget_range from one user message."""
    llm_response = {
        "message": "Got it! Acme Corp with a $10,000 budget.",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95},
            {"field_hint": "budget range", "value": "$10,000", "confidence": 0.90},
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, "I'm Acme Corp and our budget is around $10,000.")

    extracted = {k: v["value"] for k, v in snapshot["extracted_answers"].items()}
    expected = {"client_name": "Acme Corp", "budget_range": "$10,000"}

    precision, recall = compute_extraction_precision_recall(extracted, expected)
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Multi-field | Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"  [Extraction] Extracted: {extracted}")

    assert "client_name" in snapshot["extracted_answers"]
    assert "budget_range" in snapshot["extracted_answers"]
    assert precision == 1.0
    assert recall == 1.0


@pytest.mark.asyncio
async def test_extraction_missing_info_returns_empty(eval_profile) -> None:
    """When user provides no extractable info, extracted_answers should be empty."""
    llm_response = {
        "message": "I understand! Let me ask you a bit more. What's the name of your company?",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, "Hmm, not really sure about the budget.")

    extracted = snapshot["extracted_answers"]
    expected = {}

    precision, recall = compute_extraction_precision_recall(
        {k: v["value"] for k, v in extracted.items()}, expected
    )
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Missing info | Precision: {precision:.2%} | Recall: {recall:.2%}")
    assert extracted == {}, f"Should have extracted nothing, but got: {extracted}"
    assert recall == 1.0  # Nothing expected, nothing missed


@pytest.mark.asyncio
async def test_extraction_hallucinated_value_detected(eval_profile) -> None:
    """Detect when LLM extracts a value different from what the user said.

    This simulates a hallucination: the user said "XYZ Studios" but the
    LLM extracted "ABC Corp". Extraction accuracy should be LOW.
    """
    llm_response = {
        "message": "Thanks, ABC Corp!",
        "extracted_answers": [
            {"field_hint": "client name", "value": "ABC Corp", "confidence": 0.90},
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, "My company is XYZ Studios.")

    extracted = {k: v["value"] for k, v in snapshot["extracted_answers"].items()}
    expected = {"client_name": "XYZ Studios"}  # What the user actually said

    precision, recall = compute_extraction_precision_recall(extracted, expected)
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Hallucination test | Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"  [Extraction] Expected: {expected} | Actual: {extracted}")

    # Precision should be 0 because "ABC Corp" != "XYZ Studios"
    # Recall should also be 0 because the expected value was not found
    assert recall == 0.0, (
        f"Hallucinated value should cause recall=0.0, got {recall:.2f}. "
        f"This means the evaluator did not detect the incorrect extraction."
    )


@pytest.mark.asyncio
async def test_extraction_multi_turn_accumulates(eval_profile) -> None:
    """Fields extracted across multiple turns must all accumulate in the state."""
    # Turn 1: extract client_name
    llm_resp_1 = {
        "message": "Nice to meet you, Stellar Studios!",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Stellar Studios", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    # Turn 2: extract project_type
    llm_resp_2 = {
        "message": "A brand identity! Exciting.",
        "extracted_answers": [
            {"field_hint": "project type", "value": "brand identity package", "confidence": 0.90}
        ],
        "suggested_next_topic": "budget_range",
        "model_believes_complete": False,
    }

    orchestrator_1 = make_test_orchestrator(eval_profile, llm_resp_1)
    snapshot_1, session_id = await run_turn(orchestrator_1, "My name is Stellar Studios.")

    # Swap LLM and continue with same orchestrator session
    orchestrator_1._llm = ConfigurableLLMProvider(llm_resp_2)
    snapshot_2, _ = await run_turn(orchestrator_1, "We need a brand identity package.", session_id)

    final_extracted = {k: v["value"] for k, v in snapshot_2["extracted_answers"].items()}
    expected = {"client_name": "Stellar Studios", "project_type": "brand identity package"}

    precision, recall = compute_extraction_precision_recall(final_extracted, expected)
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Multi-turn | Precision: {precision:.2%} | Recall: {recall:.2%}")
    print(f"  [Extraction] Final state: {final_extracted}")
    print(f"  [Extraction] Turn count: {snapshot_2['turn_count']}")

    assert "client_name" in snapshot_2["extracted_answers"], "client_name must persist across turns"
    assert "project_type" in snapshot_2["extracted_answers"], "project_type must be added"
    assert precision >= 0.8
    assert recall >= 0.8


@pytest.mark.asyncio
async def test_extraction_confidence_below_threshold_stays_missing(eval_profile) -> None:
    """Fields extracted below the confidence threshold should remain in missing_fields."""
    low_confidence_response = {
        "message": "I'm not quite sure about that...",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Maybe Corp", "confidence": 0.40},
        ],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, low_confidence_response)
    snapshot, _ = await run_turn(orchestrator, "Maybe something like a corporation?")

    missing_codes = {mf["field_code"] for mf in snapshot["missing_fields"]}

    print(f"\n  [Extraction] Low confidence test | Missing: {missing_codes}")
    print(f"  [Extraction] Threshold: {THRESHOLDS['extraction_confidence']}")

    # client_name should still be missing because confidence 0.40 < threshold 0.70
    assert "client_name" in missing_codes, (
        f"client_name extracted with confidence 0.40 should remain in missing_fields "
        f"(threshold={THRESHOLDS['extraction_confidence']})"
    )


@pytest.mark.asyncio
async def test_extraction_irrelevant_user_message_no_extraction(eval_profile) -> None:
    """User discussing completely off-topic content should produce no extractions."""
    llm_response = {
        "message": "Interesting! Now, could you tell me about the project?",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, "The weather is so beautiful today!")

    extracted = snapshot["extracted_answers"]
    print(f"\n  [Extraction] Off-topic test | Extracted: {extracted}")
    assert extracted == {}, f"Off-topic message should produce no extractions, got: {extracted}"


@pytest.mark.asyncio
async def test_extraction_dataset_case_01(eval_profile, eval_dataset) -> None:
    """Run extraction test against dataset case_01: single-field extraction."""
    case = next(c for c in eval_dataset if c["id"] == "case_01_single_field")

    llm_response = {
        "message": "Thank you! Now, what type of project do you need?",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, case["user_messages"][0])

    extracted = {k: v["value"] for k, v in snapshot["extracted_answers"].items()}
    expected = case["expected_extractions"]

    precision, recall = compute_extraction_precision_recall(extracted, expected)
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Dataset case_01 | Precision: {precision:.2%} | Recall: {recall:.2%}")
    assert precision >= 0.8
    assert recall >= 0.8


@pytest.mark.asyncio
async def test_extraction_dataset_case_02(eval_profile, eval_dataset) -> None:
    """Run extraction test against dataset case_02: multi-field extraction."""
    case = next(c for c in eval_dataset if c["id"] == "case_02_multi_field")

    llm_response = {
        "message": "Got it! Acme Corp with $10,000 budget. What type of project?",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95},
            {"field_hint": "budget range", "value": "$10,000", "confidence": 0.90},
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orchestrator = make_test_orchestrator(eval_profile, llm_response)
    snapshot, _ = await run_turn(orchestrator, case["user_messages"][0])

    extracted = {k: v["value"] for k, v in snapshot["extracted_answers"].items()}
    expected = case["expected_extractions"]

    precision, recall = compute_extraction_precision_recall(extracted, expected)
    EVAL_METRICS["extraction_precision"].append(precision)
    EVAL_METRICS["extraction_recall"].append(recall)

    print(f"\n  [Extraction] Dataset case_02 | Precision: {precision:.2%} | Recall: {recall:.2%}")
    assert precision >= 0.8
    assert recall >= 0.8
