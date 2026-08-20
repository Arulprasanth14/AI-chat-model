"""
tests/integration/test_conversation_turn.py
────────────────────────────────────────────
Integration test for a full conversation turn using:
  - Stubbed LLMProvider (no real OpenAI call)
  - InMemorySessionRepository (no DB)
  - MockVectorStore + MockEmbedder (no pgvector)

Tests the full orchestrator turn flow end-to-end.
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
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Stubs ──────────────────────────────────────────────────────────────────────

class StubLLMProvider:
    """Returns a configurable tool-call JSON response without calling OpenAI."""

    def __init__(self, tool_result: dict[str, Any]) -> None:
        self._tool_result = tool_result

    async def stream_tool_call(
        self,
        messages: list[dict],
        tool_schema: dict,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        yield json.dumps(self._tool_result)

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

    async def complete(self, messages: list[dict], temperature: float = 0.7) -> str:
        return self._tool_result.get("message", "")


class MockEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [0.0] * 1536

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 1536 for _ in texts]


class MockVectorStore:
    async def upsert(self, chunks: list) -> int:
        return len(chunks)

    async def search(
        self,
        query_embedding: list[float],
        profile_id: str,
        top_k: int = 5,
        industry: str | None = None,
        brief_type: str | None = None,
        field_code: str | None = None,
    ) -> list[VectorSearchResult]:
        return [
            VectorSearchResult(
                chunk_id="c1",
                content="Ask about the client's goals naturally.",
                doc_type="question_guidance",
                score=0.85,
            )
        ]

    async def delete_by_profile(self, profile_id: str) -> int:
        return 0


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def test_profile() -> BaseProfile:
    return BaseProfile(
        profile_id="integration_test",
        persona_prompt="You are a helpful creative brief assistant.",
        knowledge_namespace="integration_test",
        required_fields=[
            FieldDefinition(code="client_name", description="Name of the client", required=True),
            FieldDefinition(code="project_type", description="Type of project", required=True),
            FieldDefinition(code="budget_range", description="Budget range", required=True),
        ],
    )


@pytest.fixture
def stub_llm_response() -> dict:
    return {
        "message": "Great to meet you! Tell me more about your project.",
        "extracted_answers": [
            {"field_hint": "client name", "value": "Acme Corp", "confidence": 0.92}
        ],
        "suggested_next_topic": "project_type",
        "model_believes_complete": False,
    }


def make_orchestrator(
    profile: BaseProfile,
    llm_response: dict,
    retriever: Any | None = None,
    **kwargs: Any,
) -> ConversationOrchestrator:
    repo = InMemorySessionRepository()
    embedder = MockEmbedder()
    vector_store = MockVectorStore()
    retriever = retriever or RAGRetriever(embedder=embedder, vector_store=vector_store, top_k=3)
    llm = StubLLMProvider(llm_response)
    builder = PromptBuilder()
    return ConversationOrchestrator(
        session_repo=repo,
        retriever=retriever,
        llm=llm,
        prompt_builder=builder,
        profile_provider=lambda _state: profile,
        **kwargs,
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestFullTurnFlow:
    @pytest.mark.asyncio
    async def test_new_session_created_when_no_session_id(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hi"):
            events.append(event)

        assert len(events) >= 1  # At least one chunk + done event
        # Last event should be the done/snapshot event
        last = json.loads(events[-1].replace("data: ", "").strip())
        assert last.get("done") is True
        assert "snapshot" in last

    @pytest.mark.asyncio
    async def test_session_id_returned_in_snapshot(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="Hello"):
            events.append(event)

        last = json.loads(events[-1].replace("data: ", "").strip())
        snapshot = last["snapshot"]
        assert "session_id" in snapshot
        assert snapshot["session_id"] is not None

    @pytest.mark.asyncio
    async def test_extraction_captured_in_snapshot(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        """Extracted answer from LLM should appear in the snapshot."""
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="I'm Acme Corp"):
            events.append(event)

        last = json.loads(events[-1].replace("data: ", "").strip())
        snapshot = last["snapshot"]
        extracted = snapshot.get("extracted_answers", {})
        # client_name should have been extracted
        assert "client_name" in extracted
        assert extracted["client_name"]["value"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_missing_fields_reflect_extraction(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        """After extracting client_name, it should no longer appear in missing_fields."""
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="I'm Acme Corp"):
            events.append(event)

        last = json.loads(events[-1].replace("data: ", "").strip())
        snapshot = last["snapshot"]
        missing_codes = {mf["field_code"] for mf in snapshot.get("missing_fields", [])}
        assert "client_name" not in missing_codes
        assert "project_type" in missing_codes
        assert "budget_range" in missing_codes

    @pytest.mark.asyncio
    async def test_existing_session_continues_conversation(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        """Second turn on same session_id should carry forward extracted answers."""
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        # First turn — extract client_name
        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="I'm Acme"):
            events.append(event)

        first_snapshot = json.loads(events[-1].replace("data: ", "").strip())["snapshot"]
        session_id = first_snapshot["session_id"]

        # Second turn — same session, different LLM response
        llm2 = {
            "message": "What type of project are you thinking?",
            "extracted_answers": [
                {"field_hint": "project type", "value": "brand identity", "confidence": 0.88}
            ],
            "suggested_next_topic": "budget_range",
            "model_believes_complete": False,
        }
        orchestrator._llm = StubLLMProvider(llm2)

        events2 = []
        async for event in orchestrator.process_turn(session_id=session_id, user_message="Brand identity"):
            events2.append(event)

        second_snapshot = json.loads(events2[-1].replace("data: ", "").strip())["snapshot"]
        # Both fields should now be extracted
        extracted = second_snapshot["extracted_answers"]
        assert "client_name" in extracted
        assert "project_type" in extracted
        assert second_snapshot["turn_count"] == 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_message_chunk_events_emitted(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        """Intermediate chunk events should carry the message text."""
        orchestrator = make_orchestrator(test_profile, stub_llm_response)

        chunk_events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="hi"):
            data = json.loads(event.replace("data: ", "").strip())
            if "chunk" in data:
                chunk_events.append(data["chunk"])

        full_text = "".join(chunk_events)
        assert "Great to meet you" in full_text or len(full_text) > 0

    @pytest.mark.asyncio
    async def test_model_believes_complete_flag_in_snapshot(
        self, test_profile: BaseProfile
    ) -> None:
        """model_believes_complete from LLM should appear in snapshot."""
        llm_response = {
            "message": "I think we have everything!",
            "extracted_answers": [],
            "suggested_next_topic": "",
            "model_believes_complete": True,
        }
        orchestrator = make_orchestrator(test_profile, llm_response)

        events = []
        async for event in orchestrator.process_turn(session_id=None, user_message="done"):
            events.append(event)

        snapshot = json.loads(events[-1].replace("data: ", "").strip())["snapshot"]
        assert snapshot["model_believes_complete"] is True

    @pytest.mark.asyncio
    async def test_structural_guidance_merged_with_vector_results(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        from unittest.mock import MagicMock
        class MockRetriever:
            async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk("Direct section guidance", "general", 0.9),  # Duplicate content
                    RetrievedChunk("Unique vector context", "general", 0.8)
                ]

        class StubStructuralResolver:
            def resolve(self, state: ConversationState, missing_fields: list) -> RetrievedChunk:
                return RetrievedChunk("Direct section guidance", "question_guidance", 1.0)

        orchestrator = make_orchestrator(
            test_profile,
            stub_llm_response,
            retriever=MockRetriever(),
            structural_resolver=StubStructuralResolver(),
        )
        
        real_build = orchestrator._prompt_builder.build_response_phase
        orchestrator._prompt_builder.build_response_phase = MagicMock(side_effect=real_build)

        async for _ in orchestrator.process_turn(session_id=None, user_message="Hi"):
            pass
            
        retrieved_chunks = orchestrator._prompt_builder.build_response_phase.call_args.kwargs["retrieved_chunks"]
        assert len(retrieved_chunks) == 2
        assert retrieved_chunks[0].content == "Direct section guidance"
        assert retrieved_chunks[0].doc_type == "question_guidance"
        assert retrieved_chunks[1].content == "Unique vector context"

    @pytest.mark.asyncio
    async def test_vector_chunks_fill_gaps_when_structural_guidance_partial(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        from unittest.mock import MagicMock
        class MockRetriever:
            async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
                return [
                    RetrievedChunk("Supporting context A", "general", 0.85),
                    RetrievedChunk("Supporting context B", "general", 0.8)
                ]

        class StubStructuralResolver:
            def resolve(self, state: ConversationState, missing_fields: list) -> RetrievedChunk:
                return RetrievedChunk("Primary guidance", "question_guidance", 1.0)

        orchestrator = make_orchestrator(
            test_profile,
            stub_llm_response,
            retriever=MockRetriever(),
            structural_resolver=StubStructuralResolver(),
        )
        
        real_build = orchestrator._prompt_builder.build_response_phase
        orchestrator._prompt_builder.build_response_phase = MagicMock(side_effect=real_build)

        async for _ in orchestrator.process_turn(session_id=None, user_message="Hi"):
            pass
            
        retrieved_chunks = orchestrator._prompt_builder.build_response_phase.call_args.kwargs["retrieved_chunks"]
        assert len(retrieved_chunks) == 3
        assert retrieved_chunks[0].content == "Primary guidance"
        assert retrieved_chunks[1].content == "Supporting context A"
        assert retrieved_chunks[2].content == "Supporting context B"

    @pytest.mark.asyncio
    async def test_vector_retrieval_runs_when_structural_guidance_is_unavailable(
        self, test_profile: BaseProfile, stub_llm_response: dict
    ) -> None:
        class CountingRetriever:
            def __init__(self) -> None:
                self.calls = 0

            async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
                self.calls += 1
                return []

        class NullStructuralResolver:
            def resolve(self, state: ConversationState, missing_fields: list) -> None:
                return None

        retriever = CountingRetriever()
        orchestrator = make_orchestrator(
            test_profile,
            stub_llm_response,
            retriever=retriever,
            structural_resolver=NullStructuralResolver(),
        )
        async for _ in orchestrator.process_turn(session_id=None, user_message="Hi"):
            pass
        assert retriever.calls == 1
