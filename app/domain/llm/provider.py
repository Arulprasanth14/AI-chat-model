"""
app/domain/llm/provider.py
───────────────────────────
Abstract LLM provider interface.

All orchestrator code depends only on this protocol — never on a concrete
provider class. To swap from OpenAI to another vendor, implement a new class
satisfying this protocol and update the dependency injection in app/api/deps.py.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for all LLM backends.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      The orchestrator and all domain code depend ONLY on this protocol.
      Concrete implementations (OpenAIProvider, etc.) live in the
      infrastructure layer and are injected via app/api/deps.py.
    ─────────────────────────────────────────────────────────────────────
    """

    def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a chat completion with a forced single tool/function call.

        Used for Phase B (generate_response). The model MUST invoke the tool
        on every response. The caller accumulates yielded tokens and then
        parses the complete tool call result as JSON.

        Args:
            messages:    Full prompt as a list of {role, content} dicts.
            tool_schema: JSON Schema of the tool the model must call.
            temperature: Sampling temperature.

        Yields:
            Raw string tokens from the streaming response. The complete
            accumulated string is valid JSON matching the tool schema.
        """
        ...

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Non-streaming call with auto tool_choice, returning all tool calls made.

        Used for Phase A explicit tool calling. The LLM may call zero, one,
        or multiple tools in a single response. Returns an empty list if the
        LLM made no tool calls (pure conversation turn — Phase A is skipped).

        Args:
            messages:    Prompt as {role, content} dicts.
            tools:       List of tool schemas the LLM may call.
            temperature: Sampling temperature.

        Returns:
            List of tool call dicts, each with keys:
              id (str), name (str), arguments (str — raw JSON).
            Empty list if the LLM chose not to call any tools.
        """
        ...

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming completion for non-conversational uses (e.g. eval).

        Args:
            messages:    Prompt as {role, content} dicts.
            temperature: Sampling temperature.

        Returns:
            Complete response text.
        """
        ...
