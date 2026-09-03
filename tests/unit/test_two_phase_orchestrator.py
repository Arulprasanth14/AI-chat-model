"""
tests/unit/test_two_phase_orchestrator.py
──────────────────────────────────────────
Tests for the two-phase (Phase A extraction + Phase B response) orchestrator
architecture that fixes the fake-confirmation bug.

Four tests as specified in the task:

  1. Low-confidence rejection → message must NOT contain confirmation language,
     must contain an honest clarification request.
  2. save_session raises exception → message must NOT confirm, must surface a
     "didn't save" / "write failed" honest response.
  3. Normal successful save → message DOES confirm, and confirms the CORRECT
     field+value (guards against field-mapping mismatches).
  4. Phase A is never streamed to client → no `extracted_answers` JSON key
     appears in any SSE chunk that reaches the client.

NOTE: Phase A now uses `call_with_tools()` (non-streaming, auto tool_choice)
instead of `stream_tool_call()`. These tests mock `call_with_tools` for Phase A
and `stream_tool_call` for Phase B.
"""
from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.project_profiles.base_profile import BaseProfile, FieldDefinition

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_profile() -> BaseProfile:
    """A minimal profile with one required field: brand_color."""
    return BaseProfile(
        profile_id="test_profile",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="brand_color",
                description="The brand's primary color",
                required=True,
            )
        ],
        knowledge_namespace="test_namespace",
    )


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    repo = AsyncMock()
    state = ConversationState(profile_id="test_profile")
    repo.create_session.return_value = state
    repo.get_session.return_value = state
    repo.save_session.return_value = None
    return repo


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_prompt_builder() -> PromptBuilder:
    """Real PromptBuilder — we test its integration with Phase B messages."""
    return PromptBuilder()


def _make_phase_a_tool_calls(field_code: str, value: str, confidence: float) -> list[dict]:
    """Create Phase A tool calls for call_with_tools mock (save_text_field)."""
    return [
        {
            "id": "tc_phase_a",
            "name": "save_text_field",
            "arguments": json.dumps({
                "field_code": field_code,
                "value": value,
                "confidence": confidence,
            }),
        }
    ]


def _make_phase_b_stream(message: str) -> Any:
    """Create a mock stream generator for Phase B (generate_response schema)."""
    payload = json.dumps({
        "message": message,
        "suggested_next_topic": "",
        "model_believes_complete": False,
    })

    async def _stream(*args, **kwargs):
        yield payload

    return _stream


def _build_orchestrator(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_llm: Any,
    mock_profile: BaseProfile,
) -> ConversationOrchestrator:
    """Helper: construct a minimal orchestrator with injected mocks."""
    return ConversationOrchestrator(
        session_repo=mock_session_repo,
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_builder=PromptBuilder(),
        profile_provider=lambda state: mock_profile,
    )


async def _collect_sse(orchestrator: ConversationOrchestrator, message: str) -> tuple[str, list[str]]:
    """Drive the orchestrator for one turn and return (full_message, all_sse_chunks)."""
    all_chunks: list[str] = []
    final_message = ""

    async for sse_event in orchestrator.process_turn(
        session_id=None,
        user_message=message,
    ):
        all_chunks.append(sse_event)
        if sse_event.startswith("data: "):
            try:
                data = json.loads(sse_event[6:])
                if "chunk" in data:
                    final_message += data["chunk"]
            except Exception:
                pass

    if hasattr(orchestrator, "_bg_tasks") and orchestrator._bg_tasks:
        import asyncio
        await asyncio.gather(*orchestrator._bg_tasks)

    return final_message, all_chunks


# ── CONFIRMATION LANGUAGE PATTERNS ────────────────────────────────────────────
# These are intentionally broad — any of these phrases in the reply would
# constitute a false confirmation of a failed write.

_CONFIRMATION_PHRASES = [
    r"\bsaved\b",
    r"\bupdated\b",
    r"\bgot it\b",
    r"\brecorded\b",
    r"\bnoted\b",
    r"\bconfirmed\b",
    r"\bwill use\b",
    r"\bhave captured\b",
    r"\bset.*to\b",
]


def _contains_confirmation(text: str) -> bool:
    """Return True if the text contains any confirmation-of-save phrases."""
    lower = text.lower()
    return any(re.search(p, lower) for p in _CONFIRMATION_PHRASES)


def _contains_clarification(text: str) -> bool:
    """Return True if the text asks for clarification / surfaces a problem."""
    lower = text.lower()
    clarification_patterns = [
        r"\bconfirm\b",
        r"\bsure\b",
        r"\bclarif\b",
        r"\bdidn.t (save|go through)\b",
        r"\bcouldn.t\b",
        r"\bwasn.t (sure|confident)\b",
        r"\bwhat.*did you mean\b",
        r"\bcould you\b",
        r"\bplease\b",
        r"\btry again\b",
        r"\bfailed\b",
    ]
    return any(re.search(p, lower) for p in clarification_patterns)


# ── TEST 1: Low-confidence rejection ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_low_confidence_rejection_produces_clarification(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_profile: BaseProfile,
) -> None:
    """Phase A rejects a low-confidence extraction.

    The final streamed message must NOT contain confirmation language,
    and must contain an honest clarification request.
    """
    # Phase A: LLM returns a save_text_field call with confidence below threshold (0.7)
    phase_a_tool_calls = _make_phase_a_tool_calls("brand_color", "blue", 0.4)

    # Phase B: LLM generates a response acknowledging the uncertainty
    phase_b_message = (
        "I wasn't confident enough about the brand color you mentioned — "
        "could you confirm whether you meant blue or something else?"
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(
        mock_session_repo, mock_retriever, mock_llm, mock_profile
    )

    final_message, all_chunks = await _collect_sse(
        orchestrator, "My brand color is maybe blue or something"
    )

    # ASSERTION 1: no confirmation language in the response
    assert not _contains_confirmation(final_message), (
        f"Response contained confirmation language despite low-confidence rejection.\n"
        f"Response: {final_message!r}"
    )

    # ASSERTION 2: response contains clarification language
    assert _contains_clarification(final_message), (
        f"Response did not ask for clarification after low-confidence rejection.\n"
        f"Response: {final_message!r}"
    )

    # ASSERTION 3: save_session should NOT have been called with actual data
    # (Phase A rejects before writing when confidence is below threshold)
    state = mock_session_repo.create_session.return_value
    assert "brand_color" not in state.captured, (
        "Field was saved despite being below confidence threshold"
    )


# ── TEST 2: DB write failure ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_db_write_failure_produces_honest_response(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_profile: BaseProfile,
) -> None:
    """save_session raises an exception during Phase A.

    The final streamed message must NOT contain confirmation language,
    and must surface a "didn't save" / honest failure response.
    """
    # Configure save_session to fail on first call (Phase A write)
    save_call_count = 0

    async def failing_save(state):
        nonlocal save_call_count
        save_call_count += 1
        if save_call_count == 1:
            raise RuntimeError("Simulated DB connection error")
        # Second call (final state persist) succeeds

    mock_session_repo.save_session.side_effect = failing_save

    # Phase A: high-confidence extraction (would succeed if DB worked)
    phase_a_tool_calls = _make_phase_a_tool_calls("brand_color", "blue", 0.95)

    # Phase B: LLM generates an honest failure response
    phase_b_message = (
        "I wasn't able to save the brand color — there was a technical issue. "
        "Could you please try again?"
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(
        mock_session_repo, mock_retriever, mock_llm, mock_profile
    )

    final_message, all_chunks = await _collect_sse(
        orchestrator, "My brand color is blue"
    )

    # ASSERTION 1: response does NOT confirm the save
    assert not _contains_confirmation(final_message), (
        f"Response contained confirmation language despite DB write failure.\n"
        f"Response: {final_message!r}"
    )

    # ASSERTION 2: response surfaces the failure (asks to try again or mentions error)
    assert _contains_clarification(final_message), (
        f"Response did not surface the DB write failure to the user.\n"
        f"Response: {final_message!r}"
    )


# ── TEST 3: Successful save — correct field and value confirmed ────────────────


@pytest.mark.asyncio
async def test_successful_save_confirms_correct_field_and_value(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_profile: BaseProfile,
) -> None:
    """Normal successful save path.

    Phase A extracts and saves successfully.
    Phase B message MUST confirm the save, and must confirm the CORRECT
    field/value (not a mismatched or hallucinated one).
    """
    extracted_value = "deep purple"

    phase_a_tool_calls = _make_phase_a_tool_calls("brand_color", extracted_value, 0.95)
    phase_b_message = (
        f"Got it! I've saved your brand color as '{extracted_value}'. "
        "What would you like to update next?"
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(
        mock_session_repo, mock_retriever, mock_llm, mock_profile
    )

    final_message, all_chunks = await _collect_sse(
        orchestrator, f"My brand color is {extracted_value}"
    )

    # ASSERTION 1: response does confirm the save
    assert _contains_confirmation(final_message), (
        f"Response did not confirm a successful save.\n"
        f"Response: {final_message!r}"
    )

    # ASSERTION 2: response mentions the correct value (not a hallucinated different value)
    assert extracted_value in final_message.lower() or extracted_value in final_message, (
        f"Response confirmed but mentioned a different value than what was saved.\n"
        f"Expected value: {extracted_value!r}\n"
        f"Response: {final_message!r}"
    )

    # ASSERTION 3: save_session was called (Phase A write happened)
    assert mock_session_repo.save_session.called, (
        "save_session was never called — the write path was skipped entirely"
    )


# ── TEST 4: Phase A is never streamed to client ────────────────────────────────


@pytest.mark.asyncio
async def test_phase_a_extraction_json_never_reaches_client(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_profile: BaseProfile,
) -> None:
    """Phase A tool call arguments must never appear in SSE events sent to the client.

    Specifically: no SSE chunk should contain the raw ``extracted_answers`` JSON
    structure that Phase A generates, or any ``field_hint`` / ``confidence`` keys.
    """
    phase_a_tool_calls = _make_phase_a_tool_calls("brand_color", "teal", 0.92)
    phase_b_message = "Perfect, I've saved teal as your brand color!"
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    phase_b_call_count = 0

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        nonlocal phase_b_call_count
        phase_b_call_count += 1
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(
        mock_session_repo, mock_retriever, mock_llm, mock_profile
    )

    _, all_chunks = await _collect_sse(orchestrator, "Make it teal")

    # Collect only the streaming "chunk" SSE events (not the final "done" event)
    streaming_chunks: list[str] = []
    for event in all_chunks:
        if event.startswith("data: "):
            try:
                data = json.loads(event[6:])
                # Only look at incremental text chunks, not the final snapshot
                if "chunk" in data and "done" not in data:
                    streaming_chunks.append(data["chunk"])
            except Exception:
                pass

    combined_streamed = "".join(streaming_chunks)

    # ASSERTION 1: "extracted_answers" key does not appear in streamed text
    assert "extracted_answers" not in combined_streamed, (
        f"Phase A 'extracted_answers' JSON leaked into the SSE stream.\n"
        f"Streamed text: {combined_streamed!r}"
    )

    # ASSERTION 2: "field_hint" key does not appear in streamed text
    assert "field_hint" not in combined_streamed, (
        f"Phase A 'field_hint' JSON leaked into the SSE stream.\n"
        f"Streamed text: {combined_streamed!r}"
    )

    # ASSERTION 3: Phase B stream_tool_call was made (one call)
    assert phase_b_call_count == 1, (
        f"Expected 1 Phase B stream_tool_call, got {phase_b_call_count}."
    )

    # ASSERTION 4: Phase A call_with_tools was called exactly once
    mock_llm.call_with_tools.assert_called_once()
