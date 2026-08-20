"""
tests/unit/test_llm_provider.py
────────────────────────────────
Unit tests for the LLM Provider abstraction layer.

Three test groups:

  1. Protocol conformance — verifies that OpenAIProvider structurally satisfies
     the LLMProvider protocol (no SDK call needed).

  2. Provider independence — verifies that ConversationOrchestrator works
     identically with a FakeLLMProvider that has zero OpenAI dependency.
     If the orchestrator is truly provider-agnostic, the FakeLLMProvider must
     be fully interchangeable with OpenAIProvider.

  3. Swap simulation — verifies that a "different vendor" fake provider can be
     injected and produces a valid turn result, proving the backend logic
     doesn't care about the concrete provider.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock

import pytest

from app.domain.llm.provider import LLMProvider
from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.llm.prompt_builder import PromptBuilder
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fake providers (zero third-party SDK dependency) ──────────────────────────

class FakeLLMProvider:
    """A fully self-contained fake LLM provider with no external dependencies.

    This is the primary test vehicle for verifying provider independence.
    It satisfies the LLMProvider protocol without importing the openai SDK.

    - stream_tool_call yields a pre-configured JSON payload.
    - call_with_tools returns a configurable list of tool call dicts.
    - complete returns a fixed string.
    """

    def __init__(
        self,
        stream_payload: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        complete_text: str = "Fake response",
    ) -> None:
        self._stream_payload = stream_payload or {
            "message": "Fake streamed response",
            "suggested_next_topic": "",
            "model_believes_complete": False,
        }
        self._tool_calls = tool_calls or []
        self._complete_text = complete_text

    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        yield json.dumps(self._stream_payload)

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        return list(self._tool_calls)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> str:
        return self._complete_text


class FakeAnthropicStyleProvider:
    """Mimics a hypothetical Anthropic-style provider interface shape.

    Same method signatures as LLMProvider — satisfies the protocol via
    duck typing without any Anthropic SDK dependency.
    """

    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        payload = json.dumps({
            "message": "Response from Anthropic-style provider",
            "suggested_next_topic": "",
            "model_believes_complete": False,
        })
        yield payload

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        return []  # No extractions in this test scenario

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> str:
        return "Anthropic-style complete response"


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_profile() -> BaseProfile:
    """A profile with a single required field for minimal test setup."""
    return BaseProfile(
        profile_id="provider_test",
        persona_prompt="You are a test assistant.",
        knowledge_namespace="provider_test",
        required_fields=[
            FieldDefinition(
                code="client_name",
                description="Name of the client",
                required=True,
            )
        ],
    )


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    repo = AsyncMock()
    state = ConversationState(profile_id="provider_test")
    repo.create_session.return_value = state
    repo.get_session.return_value = state
    repo.save_session.return_value = None
    return repo


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    return retriever


def _build_orchestrator(
    llm: Any,
    session_repo: AsyncMock,
    retriever: AsyncMock,
    profile: BaseProfile,
) -> ConversationOrchestrator:
    return ConversationOrchestrator(
        session_repo=session_repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=PromptBuilder(),
        profile_provider=lambda _: profile,
    )


# ── Group 1: Protocol conformance ─────────────────────────────────────────────

class TestProtocolConformance:
    """Verifies structural conformance with the LLMProvider protocol."""

    def test_openai_provider_satisfies_llm_provider_protocol(self) -> None:
        """OpenAIProvider must satisfy LLMProvider without importing openai SDK here."""
        from app.infrastructure.llm.openai_provider import OpenAIProvider

        # isinstance check works because LLMProvider is @runtime_checkable
        # We create a dummy instance with fake credentials — no real API call
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4.1")
        assert isinstance(provider, LLMProvider), (
            "OpenAIProvider does not satisfy the LLMProvider protocol. "
            "Check that stream_tool_call, call_with_tools, and complete "
            "have matching signatures."
        )

    def test_fake_llm_provider_satisfies_llm_provider_protocol(self) -> None:
        """FakeLLMProvider must also satisfy the protocol (validates the protocol itself)."""
        provider = FakeLLMProvider()
        assert isinstance(provider, LLMProvider), (
            "FakeLLMProvider does not satisfy the LLMProvider protocol."
        )

    def test_anthropic_style_provider_satisfies_llm_provider_protocol(self) -> None:
        """Any class with the right method signatures satisfies the protocol."""
        provider = FakeAnthropicStyleProvider()
        assert isinstance(provider, LLMProvider), (
            "FakeAnthropicStyleProvider does not satisfy the LLMProvider protocol."
        )

    def test_llm_provider_protocol_has_required_methods(self) -> None:
        """The protocol must expose exactly the three required methods."""
        required_methods = {"stream_tool_call", "call_with_tools", "complete"}
        # Protocol methods are accessible as annotations on the class
        protocol_members = {
            name for name in dir(LLMProvider)
            if not name.startswith("_")
        }
        for method in required_methods:
            assert method in protocol_members, (
                f"LLMProvider protocol is missing required method: {method!r}"
            )

    def test_domain_orchestrator_has_no_openai_import(self) -> None:
        """The orchestrator module must not import any OpenAI-specific symbol."""
        import importlib
        import sys

        # Force a fresh module load to inspect its imports accurately
        mod_name = "app.domain.conversation.orchestrator"
        mod = sys.modules.get(mod_name) or importlib.import_module(mod_name)

        # The orchestrator's globals must not contain any openai-SDK symbol
        openai_leaked = [
            name for name in vars(mod)
            if "openai" in name.lower() and not name.startswith("_")
        ]
        assert not openai_leaked, (
            f"Orchestrator module has OpenAI symbols in its namespace: {openai_leaked}. "
            "The orchestrator must not depend directly on the OpenAI SDK."
        )


# ── Group 2: Provider independence ────────────────────────────────────────────

class TestProviderIndependence:
    """Verifies that the orchestrator works identically with any LLMProvider."""

    @pytest.mark.asyncio
    async def test_orchestrator_processes_turn_with_fake_provider(
        self,
        mock_session_repo: AsyncMock,
        mock_retriever: AsyncMock,
        minimal_profile: BaseProfile,
    ) -> None:
        """Full turn must succeed with FakeLLMProvider (no OpenAI SDK involved)."""
        fake_llm = FakeLLMProvider()
        orchestrator = _build_orchestrator(
            fake_llm, mock_session_repo, mock_retriever, minimal_profile
        )

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hello"):
            events.append(event)

        assert len(events) >= 1, "No SSE events were emitted"
        last = json.loads(events[-1].replace("data: ", "").strip())
        assert last.get("done") is True, "Final SSE event must be the done/snapshot event"
        assert "snapshot" in last, "Final event must contain a session snapshot"

    @pytest.mark.asyncio
    async def test_snapshot_contains_session_id_with_fake_provider(
        self,
        mock_session_repo: AsyncMock,
        mock_retriever: AsyncMock,
        minimal_profile: BaseProfile,
    ) -> None:
        """Session ID must be present in the snapshot regardless of provider."""
        fake_llm = FakeLLMProvider()
        orchestrator = _build_orchestrator(
            fake_llm, mock_session_repo, mock_retriever, minimal_profile
        )

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hi"):
            events.append(event)

        last = json.loads(events[-1].replace("data: ", "").strip())
        snapshot = last["snapshot"]
        assert "session_id" in snapshot
        assert snapshot["session_id"] is not None

    @pytest.mark.asyncio
    async def test_message_streamed_from_fake_provider(
        self,
        mock_session_repo: AsyncMock,
        mock_retriever: AsyncMock,
        minimal_profile: BaseProfile,
    ) -> None:
        """Message tokens yielded by FakeLLMProvider must reach the client."""
        expected_message = "This is the fake streamed reply"
        fake_llm = FakeLLMProvider(
            stream_payload={
                "message": expected_message,
                "suggested_next_topic": "",
                "model_believes_complete": False,
            }
        )
        orchestrator = _build_orchestrator(
            fake_llm, mock_session_repo, mock_retriever, minimal_profile
        )

        chunks = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hi"):
            data = json.loads(event.replace("data: ", "").strip())
            if "chunk" in data:
                chunks.append(data["chunk"])

        combined = "".join(chunks)
        assert expected_message in combined, (
            f"Expected message not found in streamed output.\n"
            f"Expected: {expected_message!r}\n"
            f"Got: {combined!r}"
        )


# ── Group 3: Swap simulation ───────────────────────────────────────────────────

class TestProviderSwap:
    """Verifies that swapping providers produces equivalent orchestrator behavior."""

    @pytest.mark.asyncio
    async def test_anthropic_style_provider_produces_valid_turn(
        self,
        mock_session_repo: AsyncMock,
        mock_retriever: AsyncMock,
        minimal_profile: BaseProfile,
    ) -> None:
        """Replacing FakeLLMProvider with FakeAnthropicStyleProvider must work identically."""
        anthropic_llm = FakeAnthropicStyleProvider()
        orchestrator = _build_orchestrator(
            anthropic_llm, mock_session_repo, mock_retriever, minimal_profile
        )

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hello"):
            events.append(event)

        last = json.loads(events[-1].replace("data: ", "").strip())
        assert last.get("done") is True
        assert "snapshot" in last

    @pytest.mark.asyncio
    async def test_provider_swap_does_not_affect_orchestrator_state_logic(
        self,
        mock_session_repo: AsyncMock,
        mock_retriever: AsyncMock,
        minimal_profile: BaseProfile,
    ) -> None:
        """Orchestrator state logic (missing fields, snapshot structure) must be
        identical regardless of which provider is injected.
        """
        # Run with FakeLLMProvider
        fake_events = []
        fake_llm = FakeLLMProvider()
        orch1 = _build_orchestrator(fake_llm, mock_session_repo, mock_retriever, minimal_profile)
        async for event in orch1.process_turn(session_id=None, user_message="Hi"):
            fake_events.append(event)

        # Run with FakeAnthropicStyleProvider (reset repo state between runs)
        state2 = ConversationState(profile_id="provider_test")
        mock_session_repo.create_session.return_value = state2

        anthropic_events = []
        anthropic_llm = FakeAnthropicStyleProvider()
        orch2 = _build_orchestrator(
            anthropic_llm, mock_session_repo, mock_retriever, minimal_profile
        )
        async for event in orch2.process_turn(session_id=None, user_message="Hi"):
            anthropic_events.append(event)

        # Both must emit a valid done event with equivalent snapshot structure
        fake_snap = json.loads(fake_events[-1].replace("data: ", "").strip())["snapshot"]
        anthropic_snap = json.loads(anthropic_events[-1].replace("data: ", "").strip())["snapshot"]

        # Structural keys must be identical
        assert set(fake_snap.keys()) == set(anthropic_snap.keys()), (
            "Snapshot structure differs between providers — orchestrator is not provider-agnostic."
        )
        # Both must have the same missing fields (since neither extracted anything)
        fake_missing = {mf["field_code"] for mf in fake_snap.get("missing_fields", [])}
        anthropic_missing = {mf["field_code"] for mf in anthropic_snap.get("missing_fields", [])}
        assert fake_missing == anthropic_missing, (
            "Missing fields differ between providers — state logic is provider-dependent."
        )
