"""
tests/unit/test_prompt_builder.py
───────────────────────────────────
Unit tests for PromptBuilder.

No LLM calls, no DB, no external IO.
"""
from __future__ import annotations

import pytest

from app.domain.conversation.state import MissingField
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


@pytest.fixture
def profile() -> BaseProfile:
    return BaseProfile(
        profile_id="test",
        persona_prompt="You are a test creative strategist. Be helpful and professional.",
        knowledge_namespace="test",
        required_fields=[
            FieldDefinition(code="client_name", description="Name of the client", required=True),
            FieldDefinition(code="budget", description="Budget range for the project", required=True),
        ],
    )


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder()


class TestBuildSystemMessage:
    def test_persona_appears_in_system_message(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=[],
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "test creative strategist" in system["content"]

    def test_missing_fields_summary_in_system_message(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        missing = [
            MissingField(field_code="client_name", description="Name of the client"),
            MissingField(field_code="budget", description="Budget range for the project"),
        ]
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=missing,
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "client_name" in system["content"]
        assert "budget" in system["content"]

    def test_all_captured_message_when_no_missing_fields(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=[],
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "All required fields" in system["content"]

    def test_retrieved_chunks_appear_in_system_message(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        chunks = [
            RetrievedChunk(content="Ask about budget diplomatically.", doc_type="question_guidance", score=0.9),
            RetrievedChunk(content="Budget tier examples: starter, growth.", doc_type="domain_fact", score=0.8),
        ]
        phase_a = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=[],
        )
        messages = builder.build_response_phase(
            phase_a_messages=phase_a,
            phase_a_tool_calls=[],
            write_outcomes=[],
            profile=profile,
            retrieved_chunks=chunks,
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "Ask about budget diplomatically" in system["content"]
        assert "Budget tier examples" in system["content"]

    def test_chunk_count_budget_truncation(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        """Chunks that exceed MAX_TOTAL_CHUNK_CHARS should be truncated."""
        # Create chunks that exceed the budget when combined
        large_chunks = [
            RetrievedChunk(
                content="X" * 1000,
                doc_type="domain_fact",
                score=0.9 - i * 0.1,
            )
            for i in range(10)
        ]
        phase_a = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=[],
        )
        messages = builder.build_response_phase(
            phase_a_messages=phase_a,
            phase_a_tool_calls=[],
            write_outcomes=[],
            profile=profile,
            retrieved_chunks=large_chunks,
        )
        system = next(m for m in messages if m["role"] == "system")
        # Should not include all 10 chunks (each 1000 chars)
        chunk_markers = system["content"].count("[Chunk")
        assert chunk_markers < 10


class TestHistoryInclusion:
    def test_history_included_in_messages(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Welcome!"},
        ]
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=history,
            missing_fields=[],
        )
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_history_window_respected(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        history = [{"role": "user", "content": f"message {i}"} for i in range(50)]
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=history,
            missing_fields=[],
            history_window=10,
        )
        # 1 system message + 10 history messages
        assert len(messages) == 11

    def test_system_messages_in_history_filtered_out(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        history = [
            {"role": "system", "content": "old system message"},
            {"role": "user", "content": "actual user message"},
        ]
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=history,
            missing_fields=[],
        )
        # The old system message should be filtered; only the new one + user msg
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert user_messages[0]["content"] == "actual user message"

    def test_tool_call_instructions_in_system(
        self, builder: PromptBuilder, profile: BaseProfile
    ) -> None:
        messages = builder.build(
            profile=profile,
            retrieved_chunks=[],
            conversation_history=[],
            missing_fields=[],
        )
        system = next(m for m in messages if m["role"] == "system")
        assert "save_text_field" in system["content"]
