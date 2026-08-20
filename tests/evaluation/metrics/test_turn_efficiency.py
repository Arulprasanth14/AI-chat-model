"""
tests/evaluation/metrics/test_turn_efficiency.py
──────────────────────────────────────────────────
Metric 8: Goal Completion / Turn Efficiency

WHAT IT MEASURES:
  How many conversation turns it takes to capture all required fields,
  and whether the conversation eventually reaches completion.

HOW IT IS CALCULATED:
  turns_to_completion  = total conversation messages / 2 (user + assistant)
  completion_success   = is_complete == True at end
  efficiency_flag      = turns_to_completion <= max_efficient_turns threshold

WHAT IS NOT MEASURED:
  The LOWEST possible turns. A perfectly efficient system that asks 3 questions
  for 3 fields is not inherently better if the conversation feels robotic.
  We validate that completion is achieved within an ACCEPTABLE RANGE.

VALIDATION APPROACH:
  - FULLY DETERMINISTIC — multi-turn simulations with stubbed LLM responses.
  - Tests cover: happy path, inefficient path, and incomplete conversation detection.
  - Reports turns_to_completion and whether it falls within acceptable_turn_range.
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
    THRESHOLDS,
    ConfigurableMockVectorStore,
    MockEmbedder,
)

# Separate accumulator for tests that validate detector logic, not real performance.
# These tests deliberately produce synthetic results (e.g. an orchestrator that
# never extracts anything) to confirm the evaluation framework itself works.
# They must NOT pollute EVAL_METRICS, which feeds the real evaluation report.
_DETECTOR_TEST_METRICS: dict[str, list[float]] = {
    "turns_to_completion": [],
    "goal_completion_success": [],
}


# ── Helpers ────────────────────────────────────────────────────────────────────

class SequentialLLMStub:
    """Returns tool-call results from a predefined sequence.

    NOTE: Test stub — NOT a production component.
    """

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self._index = 0

    async def call_with_tools(self, messages, tools, temperature=0.7):
        result = self._results[min(self._index, len(self._results) - 1)]
        tool_calls = []
        for i, ans in enumerate(result.get("extracted_answers", [])):
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
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        yield json.dumps(result)

    async def complete(self, messages, temperature=0.7) -> str:
        return self._results[0].get("message", "")


def build_sequential_orchestrator(eval_profile, llm_results: list[dict[str, Any]]):
    """Build an orchestrator with a sequential LLM stub."""
    store = ConfigurableMockVectorStore([
        VectorSearchResult(
            chunk_id="c1",
            content="Guidance on gathering brief information.",
            doc_type="question_guidance",
            score=0.85,
        )
    ])
    repo = InMemorySessionRepository()
    retriever = RAGRetriever(embedder=MockEmbedder(), vector_store=store, similarity_threshold=0.0, top_k=3)
    llm = SequentialLLMStub(llm_results)
    builder = PromptBuilder()

    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=builder,
        profile_provider=lambda _state: eval_profile,
    )


async def simulate_multi_turn(
    orchestrator, messages: list[str]
) -> tuple[dict, int]:
    """Run a multi-turn simulation and return (final_snapshot, actual_turns)."""
    session_id = None
    snap = {}

    for msg in messages:
        events = []
        async for event in orchestrator.process_turn(session_id=session_id, user_message=msg):
            events.append(event)
        last = json.loads(events[-1].replace("data: ", "").strip())
        snap = last["snapshot"]
        session_id = snap["session_id"]

    actual_turns = snap.get("turn_count", 0) // 2  # user+assistant pairs
    return snap, actual_turns


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_completes_within_acceptable_turns(eval_profile, eval_dataset) -> None:
    """Case 10 (complete conversation) should complete within 4–10 turns."""
    case = next(c for c in eval_dataset if c["id"] == "case_10_complete_conversation")
    min_turns, max_turns = case["acceptable_turn_range"]

    llm_results = [
        {
            "message": "Hello! Tell me about your project.",
            "extracted_answers": [],
            "suggested_next_topic": "client_name",
            "model_believes_complete": False,
        },
        {
            "message": "Nice to meet you, BrightPath Inc.!",
            "extracted_answers": [{"field_hint": "client name", "value": "BrightPath Inc.", "confidence": 0.95}],
            "suggested_next_topic": "project_type",
            "model_believes_complete": False,
        },
        {
            "message": "Brand identity redesign — wonderful!",
            "extracted_answers": [{"field_hint": "project type", "value": "brand identity redesign", "confidence": 0.92}],
            "suggested_next_topic": "budget_range",
            "model_believes_complete": False,
        },
        {
            "message": "Perfect, I have all the details!",
            "extracted_answers": [{"field_hint": "budget range", "value": "$50,000 to $75,000", "confidence": 0.92}],
            "suggested_next_topic": "",
            "model_believes_complete": True,
        },
    ]

    orch = build_sequential_orchestrator(eval_profile, llm_results)
    user_messages = [turn["message"] for turn in case["conversation"]]
    snap, actual_turns = await simulate_multi_turn(orch, user_messages)

    EVAL_METRICS["turns_to_completion"].append(float(actual_turns))
    EVAL_METRICS["goal_completion_success"].append(1.0 if snap["is_complete"] else 0.0)

    print(f"\n  [Turn Efficiency] Case 10 | Turns: {actual_turns} | Acceptable: {min_turns}–{max_turns}")
    print(f"  [Turn Efficiency] is_complete: {snap['is_complete']}")

    assert snap["is_complete"], "Case 10 should result in a complete brief"
    assert min_turns <= actual_turns <= max_turns, (
        f"Completion took {actual_turns} turns, outside acceptable range [{min_turns}, {max_turns}]"
    )


@pytest.mark.asyncio
async def test_single_field_case_completes_within_range(eval_profile, eval_dataset) -> None:
    """Case 01 (single field) — completion should require minimal turns."""
    case = next(c for c in eval_dataset if c["id"] == "case_01_single_field")
    min_turns, max_turns = case["acceptable_turn_range"]

    llm_result = {
        "message": "Noted, Acme Corp! What type of project do you need?",
        "extracted_answers": [{"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }
    orch = build_sequential_orchestrator(eval_profile, [llm_result])
    snap, actual_turns = await simulate_multi_turn(orch, case["user_messages"])

    EVAL_METRICS["turns_to_completion"].append(float(actual_turns))

    print(f"\n  [Turn Efficiency] Case 01 | Turns: {actual_turns} | Acceptable: {min_turns}–{max_turns}")
    assert actual_turns <= max_turns, (
        f"Single-field case took {actual_turns} turns, exceeds max={max_turns}"
    )


@pytest.mark.asyncio
async def test_excessive_turns_are_flagged(eval_profile) -> None:
    """Conversations taking more than max_efficient_turns should be flagged."""
    max_turns = THRESHOLDS["max_efficient_turns"]

    # 10 turns with no extraction (inefficient conversation)
    no_extraction_result = {
        "message": "I see. Could you please tell me your company name?",
        "extracted_answers": [],
        "suggested_next_topic": "client_name",
        "model_believes_complete": False,
    }
    messages = [f"Message {i}" for i in range(max_turns + 1)]  # More than max

    orch = build_sequential_orchestrator(eval_profile, [no_extraction_result])
    snap, actual_turns = await simulate_multi_turn(orch, messages)

    is_efficient = actual_turns <= max_turns
    # Route to detector accumulator — this is a synthetic failure, not a real dataset run.
    _DETECTOR_TEST_METRICS["turns_to_completion"].append(float(actual_turns))
    _DETECTOR_TEST_METRICS["goal_completion_success"].append(0.0)  # expected: not complete

    print(f"\n  [Turn Efficiency] Excessive turns test | Turns: {actual_turns} | Max efficient: {max_turns}")
    print(f"  [Turn Efficiency] Efficiency flag: {'PASS' if is_efficient else 'FLAGGED (excessive)'}")

    # We assert that the conversation is NOT flagged as complete (the goal was not reached)
    assert not snap["is_complete"], "Conversation with no extractions should not be complete"
    # And we can report that it exceeded the threshold
    assert actual_turns > max_turns, (
        f"Expected >={max_turns + 1} turns for inefficient test, got {actual_turns}"
    )


@pytest.mark.asyncio
async def test_multi_turn_accumulation_reaches_completion(eval_profile, eval_dataset) -> None:
    """Case 03 (multi-turn) should complete with fields spread across 4 turns."""
    case = next(c for c in eval_dataset if c["id"] == "case_03_multi_turn")
    min_turns, max_turns = case["acceptable_turn_range"]

    llm_results = [
        {
            "message": "Hi! Could you tell me your company name?",
            "extracted_answers": [],
            "suggested_next_topic": "client_name",
            "model_believes_complete": False,
        },
        {
            "message": "Stellar Studios — great!",
            "extracted_answers": [{"field_hint": "client name", "value": "Stellar Studios", "confidence": 0.95}],
            "suggested_next_topic": "project_type",
            "model_believes_complete": False,
        },
        {
            "message": "Brand identity package — excellent choice!",
            "extracted_answers": [{"field_hint": "project type", "value": "brand identity package", "confidence": 0.92}],
            "suggested_next_topic": "budget_range",
            "model_believes_complete": False,
        },
        {
            "message": "A $25,000 budget — perfect! Let me summarize.",
            "extracted_answers": [{"field_hint": "budget range", "value": "$25,000", "confidence": 0.92}],
            "suggested_next_topic": "",
            "model_believes_complete": True,
        },
    ]

    orch = build_sequential_orchestrator(eval_profile, llm_results)
    snap, actual_turns = await simulate_multi_turn(orch, case["user_messages"])

    EVAL_METRICS["turns_to_completion"].append(float(actual_turns))
    EVAL_METRICS["goal_completion_success"].append(1.0 if snap["is_complete"] else 0.0)

    print(f"\n  [Turn Efficiency] Case 03 | Turns: {actual_turns} | Acceptable: {min_turns}–{max_turns}")
    print(f"  [Turn Efficiency] is_complete: {snap['is_complete']}")

    assert snap["is_complete"], "Case 03 should complete with all 3 fields captured"
    assert min_turns <= actual_turns <= max_turns, (
        f"Multi-turn case took {actual_turns} turns, outside [{min_turns}, {max_turns}]"
    )


@pytest.mark.asyncio
async def test_required_fields_count_in_report(eval_profile) -> None:
    """Verify that required_fields_count matches profile configuration."""
    required_count = len(eval_profile.get_required_field_codes())

    print(f"\n  [Turn Efficiency] Required fields count: {required_count}")
    assert required_count == 3, f"Eval profile should have 3 required fields, got {required_count}"


@pytest.mark.asyncio
async def test_completion_failure_detected(eval_profile) -> None:
    """A conversation that ends before all fields are captured should be marked as failure."""
    only_one_field = {
        "message": "I see you're Acme Corp. What's the project type?",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.95}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }

    # Only send 1 turn — not enough to collect all 3 fields
    orch = build_sequential_orchestrator(eval_profile, [only_one_field])
    snap, actual_turns = await simulate_multi_turn(orch, ["We are Acme Corp."])

    # Route to detector accumulator — this is a synthetic 1-turn truncation, not a real dataset case.
    _DETECTOR_TEST_METRICS["goal_completion_success"].append(0.0 if not snap["is_complete"] else 1.0)
    print(f"\n  [Turn Efficiency] Incomplete conversation | is_complete: {snap['is_complete']} | Turns: {actual_turns}")
    assert not snap["is_complete"], "Conversation with only 1 of 3 fields should not be complete"


@pytest.mark.asyncio
async def test_summary_report_for_all_efficiency_cases(eval_profile, eval_dataset) -> None:
    """Generate a turn efficiency summary report for all multi-turn cases."""
    multi_turn_cases = [
        c for c in eval_dataset
        if "acceptable_turn_range" in c and "user_messages" in c
    ]

    print("\n\n  [Turn Efficiency] === EFFICIENCY SUMMARY ===")
    for case in multi_turn_cases:
        min_t, max_t = case["acceptable_turn_range"]
        actual_msgs = len(case["user_messages"])
        within_range = min_t <= actual_msgs <= max_t
        print(
            f"  Case '{case['id']}': {actual_msgs} message(s) | "
            f"Acceptable: {min_t}-{max_t} | {'OK' if within_range else 'OUTSIDE RANGE'}"
        )


    assert len(multi_turn_cases) > 0
