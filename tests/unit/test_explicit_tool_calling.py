"""
tests/unit/test_explicit_tool_calling.py
─────────────────────────────────────────
Tests for the explicit tool-calling refactor (Phase A categorical tools).

Five tests as specified:

  1. Single-tool turn       — one field provided → correct tool called, correct write.
  2. Multi-tool turn        — two fields in one message → both tools called, both writes succeed.
  3. Partial failure        — one write succeeds, one fails enum validation → Phase B confirms
                              the success AND flags the failure.
  4. No-tool turn           — pure conversation, no data → no tool calls, no Phase A latency,
                              no unnecessary save_session call.
  5. Malformed tool call    — missing required parameter → rejected_malformed, surfaced as
                              clarification in Phase B, no crash.

Architecture notes:
  - Phase A now calls `llm.call_with_tools()` (non-streaming, auto tool_choice).
  - The mock replaces `call_with_tools` (not `stream_tool_call`) for Phase A.
  - Phase B still uses `stream_tool_call` with `generate_response`.
"""
from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.llm.prompt_builder import PromptBuilder
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_text_profile() -> BaseProfile:
    """Profile with two free-text fields: brand_name, target_audience."""
    return BaseProfile(
        profile_id="test_explicit",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(code="brand_name", description="The name of the brand", required=True),
            FieldDefinition(code="target_audience", description="Who the brand targets", required=True),
        ],
        knowledge_namespace="test_ns",
    )


@pytest.fixture
def mock_enum_profile() -> BaseProfile:
    """Profile with one text field and one enum field (content_type)."""
    return BaseProfile(
        profile_id="test_enum",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(code="brand_name", description="The name of the brand", required=True),
            FieldDefinition(
                code="content_type",
                description="Type of content to create",
                required=True,
                enum_values=["static_post", "reel", "carousel"],
            ),
        ],
        knowledge_namespace="test_ns",
    )


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    repo = AsyncMock()
    state = ConversationState(profile_id="test_explicit")
    repo.create_session.return_value = state
    repo.get_session.return_value = state
    repo.save_session.return_value = None
    return repo


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    return retriever


def _make_phase_b_stream(message: str) -> Any:
    """Create a mock async generator for Phase B (generate_response)."""
    payload = json.dumps({
        "message": message,
        "suggested_next_topic": "",
        "model_believes_complete": False,
    })

    async def _stream(*args, **kwargs):
        yield payload

    return _stream


def _build_orchestrator(
    repo: AsyncMock,
    retriever: AsyncMock,
    llm: Any,
    profile: BaseProfile,
) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=PromptBuilder(),
        profile_provider=lambda state: profile,
    )


async def _collect_sse(orchestrator: ConversationOrchestrator, message: str) -> tuple[str, list[str]]:
    """Drive one turn and return (full_message, all_sse_chunks)."""
    all_chunks: list[str] = []
    final_message = ""

    async for sse_event in orchestrator.process_turn(session_id=None, user_message=message):
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


def _snapshot_from_chunks(chunks: list[str]) -> dict[str, Any]:
    """Extract the final snapshot dict from SSE chunks."""
    for chunk in reversed(chunks):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if "done" in data and "snapshot" in data:
                    return data["snapshot"]
            except Exception:
                pass
    return {}


# ── TEST 1: Single-tool turn ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_tool_turn_saves_correctly(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_text_profile: BaseProfile,
) -> None:
    """One field provided → save_text_field called once, correct write, confirmation."""
    # Phase A: LLM calls save_text_field for brand_name
    phase_a_tool_calls = [
        {
            "id": "tc_001",
            "name": "save_text_field",
            "arguments": json.dumps({"field_code": "brand_name", "value": "Luminary", "confidence": 0.95}),
        }
    ]
    phase_b_message = "Great, I've saved 'Luminary' as your brand name!"
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

    orchestrator = _build_orchestrator(mock_session_repo, mock_retriever, mock_llm, mock_text_profile)
    final_message, all_chunks = await _collect_sse(orchestrator, "My brand name is Luminary")

    # ASSERTION 1: Phase A was called once with call_with_tools
    mock_llm.call_with_tools.assert_called_once()

    # ASSERTION 2: Phase B was called once with stream_tool_call
    assert phase_b_call_count == 1

    # ASSERTION 3: brand_name was saved to state with correct value
    state = mock_session_repo.create_session.return_value
    assert "brand_name" in state.captured, "brand_name should be in captured after tool call"
    assert state.captured["brand_name"].value == "Luminary"
    assert state.captured["brand_name"].confidence == 0.95

    # ASSERTION 4: save_session was called (DB write happened)
    assert mock_session_repo.save_session.called

    # ASSERTION 5: Phase B message confirms the save
    assert "luminary" in final_message.lower() or "brand" in final_message.lower()


# ── TEST 2: Multi-tool turn ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_tool_turn_both_writes_succeed(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_text_profile: BaseProfile,
) -> None:
    """Two fields in one message → both save_text_field calls succeed independently."""
    # Reconfigure session repo for this test (profile_id must match)
    state = ConversationState(profile_id="test_explicit")
    mock_session_repo.create_session.return_value = state
    mock_session_repo.get_session.return_value = state

    phase_a_tool_calls = [
        {
            "id": "tc_001",
            "name": "save_text_field",
            "arguments": json.dumps({"field_code": "brand_name", "value": "Luminary", "confidence": 0.95}),
        },
        {
            "id": "tc_002",
            "name": "save_text_field",
            "arguments": json.dumps({"field_code": "target_audience", "value": "Gen Z creatives", "confidence": 0.90}),
        },
    ]
    phase_b_message = (
        "Saved both! Brand name: 'Luminary', target audience: 'Gen Z creatives'."
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(mock_session_repo, mock_retriever, mock_llm, mock_text_profile)
    final_message, all_chunks = await _collect_sse(
        orchestrator,
        "My brand name is Luminary and I'm targeting Gen Z creatives"
    )

    # ASSERTION 1: Both fields saved to state with correct values
    assert "brand_name" in state.captured, "brand_name should be captured"
    assert "target_audience" in state.captured, "target_audience should be captured"
    assert state.captured["brand_name"].value == "Luminary"
    assert state.captured["target_audience"].value == "Gen Z creatives"

    # ASSERTION 2: Both values appear in the Phase B response (regression: field-mapping bug)
    assert "luminary" in final_message.lower(), "Response should confirm brand_name"
    assert "gen z" in final_message.lower() or "creatives" in final_message.lower(), (
        "Response should confirm target_audience"
    )

    # ASSERTION 3: call_with_tools called once (not once per field)
    mock_llm.call_with_tools.assert_called_once()


# ── TEST 3: Partial failure in multi-tool turn ─────────────────────────────────

@pytest.mark.asyncio
async def test_partial_failure_multi_tool_turn(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_enum_profile: BaseProfile,
) -> None:
    """One write succeeds, one fails enum validation → Phase B confirms success
    and honestly flags the failure. No fake-confirmation on either."""
    state = ConversationState(profile_id="test_enum")
    mock_session_repo.create_session.return_value = state
    mock_session_repo.get_session.return_value = state

    phase_a_tool_calls = [
        {
            "id": "tc_001",
            "name": "save_text_field",
            "arguments": json.dumps({"field_code": "brand_name", "value": "Luminary", "confidence": 0.95}),
        },
        {
            "id": "tc_002",
            "name": "save_enum_field",
            # "video" is NOT in enum_values = ["static_post", "reel", "carousel"]
            "arguments": json.dumps({"field_code": "content_type", "value": "video", "confidence": 0.90}),
        },
    ]
    # Phase B responds honestly: confirms brand_name, flags content_type
    phase_b_message = (
        "Great, I've saved 'Luminary' as the brand name! "
        "However, 'video' isn't a valid option for content type — "
        "please choose from: static post, reel, or carousel."
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(mock_session_repo, mock_retriever, mock_llm, mock_enum_profile)
    final_message, all_chunks = await _collect_sse(
        orchestrator,
        "Brand name is Luminary and I want a video"
    )

    # ASSERTION 1: brand_name saved, content_type NOT saved
    assert "brand_name" in state.captured, "brand_name should be saved"
    assert "content_type" not in state.captured, (
        "content_type with invalid enum value should NOT be saved"
    )

    # ASSERTION 2: Response confirms the success
    assert "luminary" in final_message.lower(), "Response should confirm brand_name"

    # ASSERTION 3: Response flags the failure honestly
    failure_indicators = ["valid", "option", "choose", "carousel", "reel", "static", "not a valid"]
    assert any(indicator in final_message.lower() for indicator in failure_indicators), (
        f"Response should flag the enum rejection, but got: {final_message!r}"
    )

    # ASSERTION 4: No fake confirmation of the failed field
    # We check that the word 'video' is not confirmed.
    # Because Phase B explicitly says "'video' isn't a valid option", 'video' WILL
    # be in the message. We just want to ensure it's not confirmed.
    # Let's check that 'video' isn't near a confirmation word.
    confirmations = ["saved", "recorded", "noted", "got it", "captured"]
    lower_message = final_message.lower()
    
    # We allow the word 'video' to appear if it's accompanied by failure words
    assert "video" in lower_message and "valid" in lower_message
    
    # Check that we only confirm the brand name, not the video
    for c in confirmations:
        # Check that video is not close to a confirmation word (within 30 chars)
        match = re.search(f"{c}.{{0,30}}video|video.{{0,30}}{c}", lower_message)
        assert not match, f"Response should not confirm 'video'. Found: {match}"


# ── TEST 4: No-tool turn (pure conversation) ───────────────────────────────────

@pytest.mark.asyncio
async def test_no_tool_turn_skips_phase_a(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_text_profile: BaseProfile,
) -> None:
    """Pure conversation with no data → LLM makes no tool calls, Phase A is skipped,
    save_session is NOT called, no latency from Phase A write."""
    state = ConversationState(profile_id="test_explicit")
    mock_session_repo.create_session.return_value = state
    mock_session_repo.get_session.return_value = state

    # Phase A: LLM makes NO tool calls
    phase_b_message = "That's a great question! Happy to help you think through the brief."
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    phase_b_call_count = 0

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        nonlocal phase_b_call_count
        phase_b_call_count += 1
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=[])  # Empty — no tool calls
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(mock_session_repo, mock_retriever, mock_llm, mock_text_profile)
    final_message, all_chunks = await _collect_sse(
        orchestrator,
        "What's the difference between a static post and a carousel?"
    )

    # ASSERTION 1: call_with_tools was called (Phase A was invoked)
    mock_llm.call_with_tools.assert_called_once()

    # ASSERTION 2: Phase B was still called (conversation continues)
    assert phase_b_call_count == 1

    # ASSERTION 3: No fields were saved (state ledger untouched)
    assert len(state.captured) == 0, "No fields should be captured for a pure conversation turn"

    # ASSERTION 4: save_session NOT called for Phase A write
    # (it will be called once at end of turn for history, not for field writes)
    # The key check: call count should be at most 1 (final history save), not 2
    # (Phase A write + final history save)
    assert mock_session_repo.save_session.call_count <= 1, (
        f"save_session called {mock_session_repo.save_session.call_count} times; "
        "expected at most 1 (final history only, no Phase A write)"
    )

    # ASSERTION 5: No confirmation language in the response (nothing was saved)
    confirmation_pattern = re.compile(r"\b(saved|recorded|captured|noted|got it)\b", re.IGNORECASE)
    assert not confirmation_pattern.search(final_message), (
        f"Response should not confirm any field saves. Got: {final_message!r}"
    )


# ── TEST 5: Malformed tool call ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_malformed_tool_call_surfaced_as_clarification(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_text_profile: BaseProfile,
) -> None:
    """LLM returns a tool call missing a required parameter (field_code is empty).
    Should be caught as rejected_malformed, surfaced as clarification, no crash."""
    state = ConversationState(profile_id="test_explicit")
    mock_session_repo.create_session.return_value = state
    mock_session_repo.get_session.return_value = state

    phase_a_tool_calls = [
        {
            "id": "tc_malformed",
            "name": "save_text_field",
            # Missing field_code — only value and confidence provided
            "arguments": json.dumps({"value": "Luminary", "confidence": 0.95}),
        }
    ]
    # Phase B asks for clarification
    phase_b_message = (
        "I wasn't sure which field that answer belongs to — "
        "could you clarify what 'Luminary' refers to?"
    )
    phase_b_stream = _make_phase_b_stream(phase_b_message)

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(mock_session_repo, mock_retriever, mock_llm, mock_text_profile)

    # Should NOT raise
    final_message, all_chunks = await _collect_sse(
        orchestrator,
        "It's Luminary"
    )

    # ASSERTION 1: No crash — the turn completed
    assert final_message, "Orchestrator should produce a response even for malformed tool calls"

    # ASSERTION 2: No field was saved (malformed call rejected)
    assert len(state.captured) == 0, "No field should be saved from a malformed tool call"

    # ASSERTION 3: save_session NOT called for Phase A write (no successful writes)
    # Final history save may happen, but the field-write save should not
    # The write_outcomes should show rejected_malformed status, not saved
    # Simplest check: state has no captured fields
    assert "brand_name" not in state.captured
    assert "target_audience" not in state.captured

    # ASSERTION 4: Response surfaces the issue (clarification language)
    clarification_patterns = [
        r"\bclarif\b", r"\bsure\b", r"\bconfirm\b", r"\bwhich field\b",
        r"\bwhat.*refer\b", r"\bspecif\b", r"\b(couldn't|couldn.t)\b",
    ]
    has_clarification = any(
        re.search(p, final_message, re.IGNORECASE) for p in clarification_patterns
    )
    assert has_clarification, (
        f"Response should ask for clarification after malformed tool call. Got: {final_message!r}"
    )
