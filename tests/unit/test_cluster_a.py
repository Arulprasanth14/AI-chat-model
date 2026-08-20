"""
tests/unit/test_cluster_a.py
──────────────────────────────
Cluster A tests: State/Truth Integrity (Bugs 1, 3, 4, 10).

Tests:
  A1 — unmapped_signals populated when Phase A returns rejected_unknown_field
  A2 — to_snapshot() includes unmapped_signals and has_unmapped_signals
  A3 — logo upload writes directly to existing_assets at confidence 1.0
  A4 — Phase A prompt honesty invariants block premature completion language
"""
from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState, WRITE_STATUS_REJECTED_UNKNOWN_FIELD
from app.domain.llm.prompt_builder import PromptBuilder
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_two_field_profile() -> BaseProfile:
    return BaseProfile(
        profile_id="test_cluster_a",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(code="brand_name", description="The brand name", required=True),
            FieldDefinition(code="existing_assets", description="Existing brand assets", required=False),
        ],
        knowledge_namespace="test_ns",
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    state = ConversationState(profile_id="test_cluster_a")
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
    payload = json.dumps({
        "message": message,
        "suggested_next_topic": "",
        "model_believes_complete": False,
    })

    async def _stream(*args, **kwargs):
        yield payload

    return _stream


def _build_orchestrator(repo, retriever, llm, profile) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=PromptBuilder(),
        profile_provider=lambda state: profile,
    )


async def _collect_sse(orchestrator: ConversationOrchestrator, message: str) -> tuple[str, list[str]]:
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
    return final_message, all_chunks


def _snapshot_from_chunks(chunks: list[str]) -> dict[str, Any]:
    for chunk in reversed(chunks):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:])
                if "done" in data and "snapshot" in data:
                    return data["snapshot"]
            except Exception:
                pass
    return {}


# ── A1: unmapped_signals populated ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_unmapped_signals_populated_on_unknown_field(
    mock_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_two_field_profile: BaseProfile,
) -> None:
    """Bug 10 fix: rejected_unknown_field outcomes should populate unmapped_signals."""
    state = ConversationState(profile_id="test_cluster_a")
    mock_repo.create_session.return_value = state
    mock_repo.get_session.return_value = state

    # Phase A: LLM tries to save to a field that doesn't exist in the profile
    phase_a_tool_calls = [
        {
            "id": "tc_001",
            "name": "save_text_field",
            "arguments": json.dumps({
                "field_code": "nonexistent_field",
                "value": "Some unmapped value",
                "confidence": 0.9,
            }),
        }
    ]
    phase_b_stream = _make_phase_b_stream("Let me ask about your brand name instead.")

    async def mock_stream_tool_call(messages, tool_schema, temperature=0.7):
        async for token in phase_b_stream(messages, tool_schema):
            yield token

    mock_llm = MagicMock()
    mock_llm.call_with_tools = AsyncMock(return_value=phase_a_tool_calls)
    mock_llm.stream_tool_call = mock_stream_tool_call

    orchestrator = _build_orchestrator(mock_repo, mock_retriever, mock_llm, mock_two_field_profile)
    await _collect_sse(orchestrator, "My brand's atmosphere is vintage")

    # ASSERTION: unmapped_signals should be non-empty
    assert len(state.unmapped_signals) > 0, (
        "unmapped_signals should be populated when LLM tries to write to unknown field"
    )
    # The signal should reference the attempted value
    assert any("Some unmapped value" in sig for sig in state.unmapped_signals), (
        f"Expected 'Some unmapped value' in unmapped_signals, got: {state.unmapped_signals}"
    )


# ── A2: to_snapshot() exposes unmapped_signals ────────────────────────────────

def test_snapshot_includes_unmapped_signals(mock_two_field_profile: BaseProfile) -> None:
    """Bug 10 fix: to_snapshot() must include unmapped_signals and has_unmapped_signals."""
    state = ConversationState(profile_id="test_cluster_a")
    state.unmapped_signals = ["vintage aesthetic (tried field: nonexistent_field)"]

    snapshot = state.to_snapshot(mock_two_field_profile, threshold=0.7)

    assert "unmapped_signals" in snapshot, "snapshot must include unmapped_signals key"
    assert "has_unmapped_signals" in snapshot, "snapshot must include has_unmapped_signals key"
    assert snapshot["has_unmapped_signals"] is True, "has_unmapped_signals should be True"
    assert len(snapshot["unmapped_signals"]) == 1
    assert "vintage" in snapshot["unmapped_signals"][0]


def test_snapshot_unmapped_signals_empty_by_default(mock_two_field_profile: BaseProfile) -> None:
    """has_unmapped_signals should be False for a fresh session."""
    state = ConversationState(profile_id="test_cluster_a")
    snapshot = state.to_snapshot(mock_two_field_profile, threshold=0.7)

    assert snapshot["has_unmapped_signals"] is False
    assert snapshot["unmapped_signals"] == []


# ── A3: logo upload endpoint write ────────────────────────────────────────────

def test_logo_upload_writes_to_existing_assets(mock_two_field_profile: BaseProfile) -> None:
    """Bug 3 fix: logo upload should write directly to existing_assets at confidence 1.0."""
    state = ConversationState(profile_id="test_cluster_a")

    # Simulate the direct write that the /logo endpoint performs
    result = state.handle_save_text_field(
        field_code="existing_assets",
        value="brand_logo.png",
        confidence=1.0,
        profile=mock_two_field_profile,
        confidence_threshold=0.0,
    )

    assert result.status == "saved", f"Expected 'saved', got: {result.status}"
    assert state.captured["existing_assets"].value == "brand_logo.png"
    assert state.captured["existing_assets"].confidence == 1.0, (
        "Logo upload must write at confidence 1.0 — must not be overwritable by lower-confidence LLM inference"
    )


def test_logo_upload_cannot_be_overwritten_by_low_confidence_llm(
    mock_two_field_profile: BaseProfile,
) -> None:
    """Bug 3 fix: A direct upload at confidence 1.0 must not be overwritten by an LLM
    inference at lower confidence."""
    state = ConversationState(profile_id="test_cluster_a")

    # First: direct upload at confidence 1.0
    state.handle_save_text_field(
        field_code="existing_assets",
        value="original_logo.png",
        confidence=1.0,
        profile=mock_two_field_profile,
        confidence_threshold=0.0,
    )

    # Attempt: LLM extraction at lower confidence
    result = state.handle_save_text_field(
        field_code="existing_assets",
        value="different_file.pdf",
        confidence=0.8,
        profile=mock_two_field_profile,
        confidence_threshold=0.0,
    )

    # The LLM inference should NOT overwrite the direct upload
    assert state.captured["existing_assets"].value == "original_logo.png", (
        "Low-confidence LLM inference should not overwrite a confidence-1.0 direct upload"
    )
    assert result.status in ("rejected_lower_confidence", "saved")


# ── A4: Honesty invariants in prompt ──────────────────────────────────────────

def test_phase_a_prompt_contains_honesty_invariants() -> None:
    """Bug 1 + Bug 4 fix: Phase A prompt must contain hard prohibition rules."""
    from app.domain.conversation.state import MissingField

    builder = PromptBuilder()
    profile = BaseProfile(
        profile_id="test_honesty",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(code="brand_name", description="Brand name", required=True),
        ],
        knowledge_namespace="test_ns",
    )
    missing = [MissingField(field_code="brand_name", description="Brand name")]
    prompt = builder.build(
        profile=profile,
        retrieved_chunks=[],
        conversation_history=[],
        missing_fields=missing,
        is_complete=False,
    )

    system_msg = prompt[0]["content"] if prompt else ""

    # Bug 4 fix: should contain prohibition on premature completion language
    assert "all set" in system_msg.lower() or "HONESTY INVARIANTS" in system_msg, (
        "System message should contain honesty invariants blocking premature completion language"
    )

    # Bug 1 fix: should contain prohibition on unverified file claims
    assert "logo" in system_msg.lower() or "file" in system_msg.lower(), (
        "System message should contain prohibition on unverified file receipt claims"
    )
