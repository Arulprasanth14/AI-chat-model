"""Unit tests for the orchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.conversation.orchestrator import ConversationOrchestrator
from app.domain.conversation.state import ConversationState
from app.domain.llm.prompt_builder import PromptBuilder, RetrievedChunk
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


@pytest.fixture
def mock_profile() -> BaseProfile:
    # Must have at least one required field so the session starts as
    # incomplete — otherwise is_complete=True short-circuits the retriever
    # and structural resolver before they are called.
    return BaseProfile(
        profile_id="test_profile",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="test_field",
                description="A required test field",
                required=True,
            )
        ],
        knowledge_namespace="test_namespace",
    )


@pytest.fixture
def mock_session_repo() -> AsyncMock:
    repo = AsyncMock()
    # Ensure create_session and get_session return a valid state
    state = ConversationState(profile_id="test_profile")
    repo.create_session.return_value = state
    repo.get_session.return_value = state
    return repo


@pytest.fixture
def mock_retriever() -> AsyncMock:
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    return retriever


@pytest.fixture
def mock_llm() -> AsyncMock:
    llm = AsyncMock()

    # Phase A uses call_with_tools (non-streaming, returns list of tool call dicts).
    # Return an empty list to simulate a no-tool turn (pure conversation).
    llm.call_with_tools = AsyncMock(return_value=[])

    # Phase B uses stream_tool_call (streaming, forced generate_response).
    async def mock_stream(*args, **kwargs):
        yield '{"message": "Hello", "suggested_next_topic": "", "model_believes_complete": false}'

    llm.stream_tool_call = mock_stream
    return llm


@pytest.fixture
def mock_prompt_builder() -> PromptBuilder:
    """Real PromptBuilder — the orchestrator appends tool-call/tool-result
    messages to its output, so a MagicMock return value won't work."""
    return PromptBuilder()


@pytest.mark.asyncio
async def test_orchestrator_calls_retriever_even_when_structural_guidance_found(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_llm: AsyncMock,
    mock_prompt_builder: PromptBuilder,
    mock_profile: BaseProfile,
) -> None:
    """Retriever SHOULD be called even if structural_resolver returns guidance.
    The structural guidance should be merged with vector results and appear first.
    """
    
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = RetrievedChunk(
        content="Structural guidance",
        doc_type="question_guidance",
        score=1.0,
        field_code=None,
    )
    
    # Setup mock retriever to return a vector chunk
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(content="Vector context", doc_type="general", score=0.8)
    ]
    
    # Spy on build_response_phase to capture retrieved_chunks
    real_build = mock_prompt_builder.build_response_phase
    mock_prompt_builder.build_response_phase = MagicMock(side_effect=real_build)
    
    orchestrator = ConversationOrchestrator(
        session_repo=mock_session_repo,
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_builder=mock_prompt_builder,
        profile_provider=lambda state: mock_profile,
        structural_resolver=mock_resolver,
    )
    
    # Consume the generator
    chunks = [chunk async for chunk in orchestrator.process_turn(session_id=None, user_message="test")]
    
    # Assert retriever was called exactly once
    mock_retriever.retrieve.assert_called_once()
    mock_resolver.resolve.assert_called_once()
    
    # Assert structural guidance appears first in the merged list
    retrieved_chunks = mock_prompt_builder.build_response_phase.call_args.kwargs["retrieved_chunks"]
    assert len(retrieved_chunks) == 2
    assert retrieved_chunks[0].content == "Structural guidance"
    assert retrieved_chunks[1].content == "Vector context"


@pytest.mark.asyncio
async def test_orchestrator_calls_retriever_when_structural_guidance_is_none(
    mock_session_repo: AsyncMock,
    mock_retriever: AsyncMock,
    mock_llm: AsyncMock,
    mock_prompt_builder: MagicMock,
    mock_profile: BaseProfile,
) -> None:
    """Retriever SHOULD be called if structural_resolver returns None."""
    
    mock_resolver = MagicMock()
    mock_resolver.resolve.return_value = None
    
    orchestrator = ConversationOrchestrator(
        session_repo=mock_session_repo,
        retriever=mock_retriever,
        llm=mock_llm,
        prompt_builder=mock_prompt_builder,
        profile_provider=lambda state: mock_profile,
        structural_resolver=mock_resolver,
    )
    
    # Consume the generator
    chunks = [chunk async for chunk in orchestrator.process_turn(session_id=None, user_message="test")]
    
    # Assert retriever was called
    mock_retriever.retrieve.assert_called_once()
    mock_resolver.resolve.assert_called_once()

