"""
tests/unit/test_ollama_provider.py
────────────────────────────────────
Unit tests for OllamaProvider.

All tests mock ``httpx.AsyncClient`` so no live Ollama instance is needed.
The key invariant being tested: OllamaProvider produces exactly the same
output *shape* that the orchestrator expects — i.e. the same as OpenAIProvider.

Shape contract verified:
  call_with_tools  → list[{id: str, name: str, arguments: str (JSON)}]
  stream_tool_call → AsyncIterator[str]  (yields valid JSON string)
  complete         → str
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.llm.provider import LLMProvider
from app.infrastructure.llm.ollama_provider import OllamaProvider


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ollama_tool_response(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mock Ollama /api/chat response with tool_calls."""
    return {
        "model": "qwen2.5:7b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        },
        "done": True,
    }


def _make_ollama_content_response(content: str) -> dict[str, Any]:
    """Build a mock Ollama /api/chat response with a plain content reply."""
    return {
        "model": "qwen2.5:7b",
        "message": {
            "role": "assistant",
            "content": content,
        },
        "done": True,
    }


def _mock_httpx_post(response_json: dict[str, Any]) -> MagicMock:
    """Return a context-manager mock that simulates httpx.AsyncClient.post."""
    mock_response = MagicMock()
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ── Protocol conformance ───────────────────────────────────────────────────────

class TestOllamaProviderProtocol:
    """OllamaProvider must satisfy the LLMProvider protocol."""

    def test_ollama_provider_satisfies_llm_provider_protocol(self) -> None:
        """OllamaProvider must satisfy LLMProvider via structural duck-typing."""
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        assert isinstance(provider, LLMProvider), (
            "OllamaProvider does not satisfy the LLMProvider protocol. "
            "Check that stream_tool_call, call_with_tools, and complete "
            "have matching signatures."
        )

    def test_model_name_property(self) -> None:
        provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
        assert provider.model_name == "qwen2.5:7b"


# ── call_with_tools ────────────────────────────────────────────────────────────

class TestCallWithTools:
    """Verify call_with_tools produces the exact shape the orchestrator expects."""

    @pytest.mark.asyncio
    async def test_returns_normalized_shape_for_single_tool_call(self) -> None:
        """Single tool call must be normalised to {id, name, arguments (str)}."""
        ollama_response = _make_ollama_tool_response([
            {
                "function": {
                    "name": "save_text_field",
                    "arguments": {
                        "field_code": "client_name",
                        "value": "Acme Corp",
                        "confidence": 0.95,
                    },
                }
            }
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "My name is Acme Corp"}],
                tools=[{"type": "function", "function": {"name": "save_text_field", "parameters": {}}}],
                temperature=0.3,
            )

        assert len(result) == 1
        tc = result[0]

        # Verify all three required keys are present
        assert "id" in tc, "Tool call must have an 'id' key"
        assert "name" in tc, "Tool call must have a 'name' key"
        assert "arguments" in tc, "Tool call must have an 'arguments' key"

        # Verify types match what the orchestrator expects
        assert isinstance(tc["id"], str), f"id must be str, got {type(tc['id'])}"
        assert isinstance(tc["name"], str), f"name must be str, got {type(tc['name'])}"
        assert isinstance(tc["arguments"], str), (
            f"arguments must be a JSON str, got {type(tc['arguments'])}. "
            "Ollama returns a dict — OllamaProvider must json.dumps() it."
        )

        # Verify name is correct
        assert tc["name"] == "save_text_field"

        # Verify arguments is valid JSON with expected keys
        args = json.loads(tc["arguments"])
        assert args["field_code"] == "client_name"
        assert args["value"] == "Acme Corp"
        assert args["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_returns_normalized_shape_for_multiple_tool_calls(self) -> None:
        """Multiple tool calls must all be normalised and returned."""
        ollama_response = _make_ollama_tool_response([
            {
                "function": {
                    "name": "save_text_field",
                    "arguments": {"field_code": "client_name", "value": "Acme", "confidence": 0.9},
                }
            },
            {
                "function": {
                    "name": "save_enum_field",
                    "arguments": {"field_code": "project_type", "value": "branding", "confidence": 0.85},
                }
            },
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Acme Corp, branding project"}],
                tools=[],
                temperature=0.3,
            )

        assert len(result) == 2
        for tc in result:
            assert "id" in tc and "name" in tc and "arguments" in tc
            assert isinstance(tc["arguments"], str)
            json.loads(tc["arguments"])  # Must be valid JSON

        assert result[0]["name"] == "save_text_field"
        assert result[1]["name"] == "save_enum_field"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_tool_calls(self) -> None:
        """When the model makes no tool calls, must return an empty list — not crash."""
        ollama_response = _make_ollama_content_response("Tell me more about your project.")
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Hi"}],
                tools=[],
                temperature=0.3,
            )

        assert result == [], (
            "call_with_tools must return [] when the model makes no tool calls. "
            "The orchestrator treats [] as a fast-path skip for Phase A."
        )

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_connection_error(self) -> None:
        """Connection refused must be caught gracefully — return [] instead of crashing."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[],
                temperature=0.3,
            )

        assert result == [], "Connection error must return [], not raise an exception"

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_http_error(self) -> None:
        """Non-2xx HTTP response must be caught and return []."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=mock_response,
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=http_error)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Hello"}],
                tools=[],
                temperature=0.3,
            )

        assert result == [], "HTTP error must return [], not raise an exception"

    @pytest.mark.asyncio
    async def test_handles_string_arguments_passthrough(self) -> None:
        """Some Ollama builds may return arguments as a JSON string — pass through as-is."""
        ollama_response = _make_ollama_tool_response([
            {
                "function": {
                    "name": "save_text_field",
                    "arguments": '{"field_code": "client_name", "value": "Acme", "confidence": 0.9}',
                }
            }
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Acme"}],
                tools=[],
                temperature=0.3,
            )

        assert len(result) == 1
        assert isinstance(result[0]["arguments"], str)
        args = json.loads(result[0]["arguments"])
        assert args["field_code"] == "client_name"

    @pytest.mark.asyncio
    async def test_synthetic_ids_are_unique(self) -> None:
        """Each tool call must receive a unique synthetic ID."""
        ollama_response = _make_ollama_tool_response([
            {"function": {"name": "save_text_field", "arguments": {"field_code": "a", "value": "x", "confidence": 0.9}}},
            {"function": {"name": "save_text_field", "arguments": {"field_code": "b", "value": "y", "confidence": 0.9}}},
            {"function": {"name": "save_text_field", "arguments": {"field_code": "c", "value": "z", "confidence": 0.9}}},
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "xyz"}],
                tools=[],
                temperature=0.3,
            )

        ids = [tc["id"] for tc in result]
        assert len(ids) == len(set(ids)), "All tool call IDs must be unique"


# ── stream_tool_call ───────────────────────────────────────────────────────────

class TestStreamToolCall:
    """Verify stream_tool_call yields a valid JSON string matching Phase B schema."""

    @pytest.mark.asyncio
    async def test_yields_json_string_from_tool_call(self) -> None:
        """stream_tool_call must yield a single valid JSON string."""
        expected_payload = {
            "intent": "PROVIDE_INFO",
            "message": "Great to meet you! Tell me more.",
            "suggested_next_topic": "project_type",
            "model_believes_complete": False,
        }
        ollama_response = _make_ollama_tool_response([
            {
                "function": {
                    "name": "generate_response",
                    "arguments": expected_payload,
                }
            }
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            tool_schema = {
                "type": "function",
                "function": {
                    "name": "generate_response",
                    "parameters": {},
                },
            }

            chunks = []
            async for chunk in provider.stream_tool_call(
                messages=[{"role": "user", "content": "Hello"}],
                tool_schema=tool_schema,
                temperature=0.3,
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1, "stream_tool_call must yield at least one chunk"
        accumulated = "".join(chunks)

        # Must be valid JSON
        parsed = json.loads(accumulated)
        assert parsed["message"] == "Great to meet you! Tell me more."
        assert parsed["model_believes_complete"] is False

    @pytest.mark.asyncio
    async def test_yields_fallback_json_on_no_tool_call(self) -> None:
        """When model ignores the tool, a fallback valid JSON must still be yielded."""
        ollama_response = _make_ollama_content_response("Let me help you with that.")
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            tool_schema = {"type": "function", "function": {"name": "generate_response", "parameters": {}}}

            chunks = []
            async for chunk in provider.stream_tool_call(
                messages=[{"role": "user", "content": "Hello"}],
                tool_schema=tool_schema,
                temperature=0.3,
            ):
                chunks.append(chunk)

        accumulated = "".join(chunks)
        parsed = json.loads(accumulated)
        assert "message" in parsed, "Fallback JSON must contain 'message' key"
        assert isinstance(parsed["message"], str)

    @pytest.mark.asyncio
    async def test_yields_error_fallback_on_connection_error(self) -> None:
        """Connection error must yield a valid JSON fallback — not raise."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            tool_schema = {"type": "function", "function": {"name": "generate_response", "parameters": {}}}

            chunks = []
            async for chunk in provider.stream_tool_call(
                messages=[{"role": "user", "content": "Hello"}],
                tool_schema=tool_schema,
                temperature=0.3,
            ):
                chunks.append(chunk)

        accumulated = "".join(chunks)
        parsed = json.loads(accumulated)
        assert "message" in parsed
        assert isinstance(parsed["message"], str)
        assert len(parsed["message"]) > 0


# ── complete ───────────────────────────────────────────────────────────────────

class TestComplete:
    """Verify complete() returns a plain string."""

    @pytest.mark.asyncio
    async def test_returns_content_string(self) -> None:
        ollama_response = _make_ollama_content_response("This is the assistant reply.")
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.complete(
                messages=[{"role": "user", "content": "What is your name?"}],
                temperature=0.3,
            )

        assert result == "This is the assistant reply."

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_error(self) -> None:
        """Errors in complete() must return '' — not raise."""
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.complete(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.3,
            )

        assert result == ""


# ── Response shape equivalence to orchestrator expectations ────────────────────

class TestResponseShapeEquivalence:
    """Prove that OllamaProvider returns exactly the same shape as OpenAIProvider.

    The orchestrator in _run_phase_a() accesses:
        tc["name"]      → str
        tc["id"]        → str  (used as tool_call_id)
        tc["arguments"] → str  (raw JSON, parsed by parse_phase_a_tool_calls)

    These tests verify that shape without calling OpenAI.
    """

    @pytest.mark.asyncio
    async def test_call_with_tools_keys_match_orchestrator_expectations(self) -> None:
        """Each item in the return list must have exactly the keys the orchestrator reads."""
        ollama_response = _make_ollama_tool_response([
            {
                "function": {
                    "name": "save_text_field",
                    "arguments": {"field_code": "client_name", "value": "Test", "confidence": 0.9},
                }
            }
        ])
        mock_client = _mock_httpx_post(ollama_response)

        with patch("app.infrastructure.llm.ollama_provider.httpx.AsyncClient", return_value=mock_client):
            provider = OllamaProvider(base_url="http://localhost:11434", model="qwen2.5:7b")
            result = await provider.call_with_tools(
                messages=[{"role": "user", "content": "Test Corp"}],
                tools=[],
                temperature=0.3,
            )

        assert len(result) == 1
        tc = result[0]

        # These are the EXACT keys the orchestrator reads in _run_phase_a and
        # parse_phase_a_tool_calls — any missing key would crash the orchestrator
        required_keys = {"id", "name", "arguments"}
        assert required_keys.issubset(set(tc.keys())), (
            f"Tool call dict missing required keys. "
            f"Required: {required_keys}, Got: {set(tc.keys())}"
        )
        # arguments must be a str (parse_phase_a_tool_calls calls json.loads on it)
        assert isinstance(tc["arguments"], str)
        # Confirm it round-trips through json.loads without error
        parsed_args = json.loads(tc["arguments"])
        assert isinstance(parsed_args, dict)
