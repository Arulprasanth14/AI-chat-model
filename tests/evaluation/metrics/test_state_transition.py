"""
tests/evaluation/metrics/test_state_transition.py
───────────────────────────────────────────────────
Metric 7: State Transition Correctness

WHAT IT MEASURES:
  Whether the ConversationState transitions correctly after every orchestrator turn.
  This is the most critical deterministic metric — the Python ledger, not the LLM,
  is the source of truth for what has been captured.

HOW IT IS CALCULATED:
  Binary per-check: each assertion is either correct (1) or incorrect (0).
  State Transition Correctness % = correct assertions / total assertions.

WHAT IS CHECKED PER TURN:
  1. extracted_answers dict is correctly populated.
  2. Captured fields are removed from missing_fields.
  3. Uncaptured fields remain in missing_fields.
  4. turn_count is correct (2 per full turn: user + assistant).
  5. is_complete reflects actual field coverage.
  6. model_believes_complete is stored correctly.
  7. session_id persists across turns.
  8. status changes to "complete" when all fields are captured.

VALIDATION APPROACH:
  - FULLY DETERMINISTIC — uses stubbed LLM and in-memory repositories.
  - No OpenAI API calls required.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.llm.prompt_builder import PromptBuilder
from app.domain.rag.retriever import RAGRetriever
from app.domain.rag.vector_store import VectorSearchResult
from app.infrastructure.persistence.session_repository import InMemorySessionRepository
from tests.evaluation.conftest import (
    EVAL_METRICS,
    ConfigurableMockVectorStore,
    MockEmbedder,
)


# ── Helper infrastructure ──────────────────────────────────────────────────────

class ConfigurableLLMStub:
    """Simple stub returning preconfigured tool-call JSON."""

    def __init__(self, tool_result: dict[str, Any]) -> None:
        self._tool_result = tool_result

    async def call_with_tools(self, messages, tools, temperature=0.7):
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

    async def stream_tool_call(self, messages, tool_schema, temperature=0.7) -> AsyncIterator[str]:
        yield json.dumps(self._tool_result)

    async def complete(self, messages, temperature=0.7) -> str:
        return self._tool_result.get("message", "")


def build_orchestrator(eval_profile, llm_result: dict[str, Any]):
    """Build a fully stubbed ConversationOrchestrator."""
    store = ConfigurableMockVectorStore([
        VectorSearchResult(
            chunk_id="c1",
            content="Guide for creative brief gathering.",
            doc_type="question_guidance",
            score=0.85,
        )
    ])
    repo = InMemorySessionRepository()
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=3)
    llm = ConfigurableLLMStub(llm_result)
    builder = PromptBuilder()

    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=builder,
        profile_provider=lambda _state: eval_profile,
    )


async def do_turn(orchestrator, user_message, session_id=None) -> tuple[dict, str]:
    """Execute one turn and return (snapshot, session_id)."""
    events = []
    async for event in orchestrator.process_turn(session_id=session_id, user_message=user_message):
        events.append(event)
    last = json.loads(events[-1].replace("data: ", "").strip())
    snap = last["snapshot"]
    return snap, snap["session_id"]


# ── State Transition Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_turn_count_correct_after_one_turn(eval_profile) -> None:
    """After one user message, turn_count should be 2 (1 user + 1 assistant)."""
    llm_result = {
        "message": "Hi! Tell me about your project.",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(orch, "Hello!")

    EVAL_METRICS["state_transition_correct"].append(1.0 if snap["turn_count"] == 2 else 0.0)
    print(f"\n  [State Transition] Turn count after 1 turn: {snap['turn_count']} (expected: 2)")
    assert snap["turn_count"] == 2, f"Expected turn_count=2, got {snap['turn_count']}"


@pytest.mark.asyncio
async def test_extracted_field_removed_from_missing(eval_profile) -> None:
    """After extracting client_name, it must not appear in missing_fields."""
    llm_result = {
        "message": "Thanks, Acme Corp!",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(orch, "My company is Acme Corp.")

    missing_codes = {mf["field_code"] for mf in snap["missing_fields"]}
    captured_codes = set(snap["extracted_answers"].keys())

    EVAL_METRICS["state_transition_correct"].append(1.0 if "client_name" not in missing_codes else 0.0)
    print(f"\n  [State Transition] Missing after extraction: {missing_codes}")
    print(f"  [State Transition] Captured: {captured_codes}")

    assert "client_name" not in missing_codes, (
        "client_name was extracted but still appears in missing_fields"
    )
    assert "client_name" in captured_codes


@pytest.mark.asyncio
async def test_uncaptured_fields_remain_in_missing(eval_profile) -> None:
    """Fields not mentioned by the user should remain in missing_fields."""
    llm_result = {
        "message": "Noted! What type of project are you working on?",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(orch, "My name is Acme Corp.")

    missing_codes = {mf["field_code"] for mf in snap["missing_fields"]}

    EVAL_METRICS["state_transition_correct"].append(
        1.0 if "project_type" in missing_codes and "budget_range" in missing_codes else 0.0
    )
    print(f"\n  [State Transition] Remaining missing: {missing_codes}")

    assert "project_type" in missing_codes, "project_type should still be missing"
    assert "budget_range" in missing_codes, "budget_range should still be missing"


@pytest.mark.asyncio
async def test_is_complete_false_when_fields_missing(eval_profile) -> None:
    """is_complete should be False when required fields are still missing."""
    llm_result = {
        "message": "Good start! What's the project type?",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(orch, "I'm Acme Corp.")

    EVAL_METRICS["state_transition_correct"].append(1.0 if not snap["is_complete"] else 0.0)
    print(f"\n  [State Transition] is_complete (should be False): {snap['is_complete']}")
    assert not snap["is_complete"]


@pytest.mark.asyncio
async def test_is_complete_true_when_all_fields_captured(eval_profile) -> None:
    """is_complete should be True only when ALL required fields are captured."""
    # Extract all 3 fields in one turn (simulating completion)
    llm_result = {
        "message": "We have everything! Let me summarize your brief.",
        "extracted_answers": [
            {"field_hint": "client name", "value": "BrightPath Inc", "confidence": 0.95},
            {"field_hint": "project type", "value": "brand identity", "confidence": 0.90},
            {"field_hint": "budget range", "value": "$50,000", "confidence": 0.92},
        ],
        "suggested_next_topic": "",
        "model_believes_complete": True,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(
        orch,
        "BrightPath Inc needs a brand identity for $50,000.",
    )

    EVAL_METRICS["state_transition_correct"].append(1.0 if snap["is_complete"] else 0.0)
    print(f"\n  [State Transition] is_complete (should be True): {snap['is_complete']}")
    print(f"  [State Transition] Captured: {list(snap['extracted_answers'].keys())}")
    assert snap["is_complete"], (
        f"All fields extracted but is_complete=False. "
        f"Extracted: {list(snap['extracted_answers'].keys())}"
    )


@pytest.mark.asyncio
async def test_model_believes_complete_stored_correctly(eval_profile) -> None:
    """model_believes_complete from LLM response must be stored in state."""
    # Test with model_believes_complete = True
    llm_result_true = {
        "message": "I think we have everything!",
        "extracted_answers": [],
        "suggested_next_topic": "",
        "model_believes_complete": True,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result_true)
    snap, _ = await do_turn(orch, "Yes, that should be everything.")

    score = 1.0 if snap["model_believes_complete"] is True else 0.0
    EVAL_METRICS["state_transition_correct"].append(score)
    print(f"\n  [State Transition] model_believes_complete=True stored: {snap['model_believes_complete']}")
    assert snap["model_believes_complete"] is True

    # Test with model_believes_complete = False
    llm_result_false = {
        "message": "Let me ask a few more questions.",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch2 = build_orchestrator(eval_profile, llm_result_false)
    snap2, _ = await do_turn(orch2, "Hello!")

    score2 = 1.0 if snap2["model_believes_complete"] is False else 0.0
    EVAL_METRICS["state_transition_correct"].append(score2)
    print(f"  [State Transition] model_believes_complete=False stored: {snap2['model_believes_complete']}")
    assert snap2["model_believes_complete"] is False


@pytest.mark.asyncio
async def test_session_id_persists_across_turns(eval_profile) -> None:
    """The same session_id must be returned for follow-up turns."""
    llm_result = {
        "message": "Got it!",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)

    snap1, session_id_1 = await do_turn(orch, "Hello!")
    snap2, session_id_2 = await do_turn(orch, "Follow up message.", session_id=session_id_1)

    EVAL_METRICS["state_transition_correct"].append(1.0 if session_id_1 == session_id_2 else 0.0)
    print(f"\n  [State Transition] Session continuity: {session_id_1[:8]}... == {session_id_2[:8]}...")
    assert session_id_1 == session_id_2, "Session ID must be the same across turns"


@pytest.mark.asyncio
async def test_status_becomes_complete_when_all_fields_captured(eval_profile) -> None:
    """Session status must change to 'complete' when all fields are captured."""
    llm_result = {
        "message": "All done!",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95},
            {"field_hint": "project type", "value": "brand identity", "confidence": 0.92},
            {"field_hint": "budget range", "value": "$10,000", "confidence": 0.88},
        ],
        "suggested_next_topic": "",
        "model_believes_complete": True,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(
        orch,
        "Acme Corp, brand identity project, $10,000 budget.",
    )

    EVAL_METRICS["state_transition_correct"].append(1.0 if snap["status"] == "complete" else 0.0)
    print(f"\n  [State Transition] Session status: '{snap['status']}' (expected: 'complete')")
    assert snap["status"] == "complete", (
        f"Expected status 'complete', got '{snap['status']}'. "
        "All required fields were captured."
    )


@pytest.mark.asyncio
async def test_turn_count_increments_correctly_across_multiple_turns(eval_profile) -> None:
    """turn_count must increment by 2 for each full conversation turn."""
    llm_result = {
        "message": "Tell me more.",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)

    snap1, sid = await do_turn(orch, "Turn 1")
    snap2, _ = await do_turn(orch, "Turn 2", session_id=sid)
    snap3, _ = await do_turn(orch, "Turn 3", session_id=sid)

    expected_counts = [2, 4, 6]
    actual_counts = [snap1["turn_count"], snap2["turn_count"], snap3["turn_count"]]

    EVAL_METRICS["state_transition_correct"].append(1.0 if actual_counts == expected_counts else 0.0)
    print(f"\n  [State Transition] Turn counts: {actual_counts} (expected: {expected_counts})")
    assert actual_counts == expected_counts, (
        f"Turn counts incorrect. Got {actual_counts}, expected {expected_counts}"
    )


@pytest.mark.asyncio
async def test_suggested_next_topic_stored_in_snapshot(eval_profile) -> None:
    """The suggested_next_topic from LLM should appear in the snapshot."""
    llm_result = {
        "message": "Great, I'll ask about your project type next.",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
        "intent": "ask_question",
    }
    orch = build_orchestrator(eval_profile, llm_result)
    snap, _ = await do_turn(orch, "We are Acme Corp.")

    EVAL_METRICS["state_transition_correct"].append(
        1.0 if snap["suggested_next_topic"] == "project_type" else 0.0
    )
    print(f"\n  [State Transition] suggested_next_topic: '{snap['suggested_next_topic']}' (expected: 'project_type')")
    assert snap["suggested_next_topic"] == "project_type"


@pytest.mark.asyncio
async def test_full_state_snapshot_matches_expected_after_extraction(eval_profile, eval_dataset) -> None:
    """Compare actual snapshot against dataset expected_final_state for case_10."""
    case = next(c for c in eval_dataset if c["id"] == "case_10_complete_conversation")

    expected_extractions = case["expected_final_extractions"]

    messages = [turn["message"] for turn in case["conversation"]]

    # Build LLM stubs for each turn — one per message
    llm_results = [
        {
            "message": "Hello! Tell me about your project.",
            "extracted_answers": [],
            "suggested_next_topic": "client_name",
            "model_believes_complete": False,
        },
        {
            "message": "Nice to meet you, BrightPath Inc.!",
            "extracted_answers": [
                {"field_hint": "client name", "value": "BrightPath Inc.", "confidence": 0.95}
            ],
            "suggested_next_topic": "project_type",
            "model_believes_complete": False,
        },
        {
            "message": "A brand identity redesign — great!",
            "extracted_answers": [
                {"field_hint": "project type", "value": "brand identity redesign", "confidence": 0.92}
            ],
            "suggested_next_topic": "budget_range",
            "model_believes_complete": False,
        },
        {
            "message": "Perfect, we have everything we need!",
            "extracted_answers": [
                {"field_hint": "budget range", "value": "$50,000 to $75,000", "confidence": 0.92}
            ],
            "suggested_next_topic": "",
            "model_believes_complete": True,
        },
    ]

    # Start with the first LLM stub
    orch = build_orchestrator(eval_profile, llm_results[0])
    session_id = None
    snap = None

    for msg, llm_resp in zip(messages, llm_results):
        # Swap the LLM stub for each turn using the pre-defined stub class
        orch._llm = ConfigurableLLMStub(llm_resp)
        snap, session_id = await do_turn(orch, msg, session_id=session_id)

    assert snap is not None
    actual_extracted = {k: v["value"] for k, v in snap["extracted_answers"].items()}

    correct = 0
    total = len(expected_extractions)
    for field, expected_val in expected_extractions.items():
        actual_val = actual_extracted.get(field, "")
        match = expected_val.lower() in actual_val.lower() or actual_val.lower() in expected_val.lower()
        if match:
            correct += 1
        print(
            f"\n  [State Transition] Case 10 | {field}: "
            f"expected='{expected_val}' actual='{actual_val}' {'OK' if match else 'FAIL'}"
        )

    score = correct / total if total > 0 else 0.0
    EVAL_METRICS["state_transition_correct"].append(score)
    print(f"\n  [State Transition] Case 10 final score: {score:.2%}")
    print(f"  [State Transition] is_complete: {snap['is_complete']} (expected: {case['expected_completion']})")
    assert score >= 0.6, f"State transition case 10 score {score:.2%} too low"
    assert snap["is_complete"] == case["expected_completion"]
