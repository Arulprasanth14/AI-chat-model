"""
app/infrastructure/llm/openai_provider.py
──────────────────────────────────────────
OpenAI implementation of the LLMProvider protocol.

This module lives in the *infrastructure* layer because it depends on a
third-party SDK (``openai``).  All domain and orchestration code depends
only on the abstract ``LLMProvider`` protocol defined in
``app/domain/llm/provider.py``.

To swap providers entirely (e.g., OpenAI → Anthropic → Gemini):
  1. Create a new implementation file here (e.g., ``anthropic_provider.py``).
  2. Update ``get_llm_provider()`` in ``app/api/deps.py`` to instantiate it.
  3. Zero changes to orchestrator, prompt builder, or any domain code.

Two calling modes:
  stream_tool_call  — forced single-tool call (used in Phase B for generate_response)
  call_with_tools   — auto tool_choice supporting zero or multiple tool calls (Phase A)
  complete          — non-streaming completion for evals / scripts
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

logger = logging.getLogger(__name__)

# Maximum number of rate-limit retries
_MAX_RETRIES = 3


class OpenAIProvider:
    """Concrete LLMProvider backed by OpenAI's chat completions API.

    Satisfies the ``LLMProvider`` protocol from ``app/domain/llm/provider.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Args:
        api_key:    OpenAI API key.
        model:      Chat model identifier (e.g. ``"gpt-4.1"``).
                    Controlled entirely from the provider/configuration layer —
                    never hardcoded in business logic.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This class is the ONLY file that must be modified (or replaced) when
      switching the LLM vendor.  Everything above this layer is agnostic.
    ─────────────────────────────────────────────────────────────────────
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, max_retries=_MAX_RETRIES)
        self._model = model

    # ── LLMProvider protocol methods ───────────────────────────────────────────

    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a forced tool call and yield tool argument tokens.

        The tool arguments accumulate into a JSON string that matches the
        provided tool schema.  The caller is responsible for buffering tokens
        and parsing the final result.

        Used for Phase B (generate_response) where a single forced tool call
        is always required.

        Yields:
            Raw JSON token fragments from the tool call arguments stream.
        """
        tool_name = tool_schema["function"]["name"]

        logger.debug(
            "OpenAI stream_tool_call started",
            extra={"model": self._model, "tool": tool_name, "messages": len(messages)},
        )

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=[cast(ChatCompletionToolParam, tool_schema)],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            temperature=temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Tool call argument fragments arrive in delta.tool_calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.arguments:
                        yield tc.function.arguments

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Non-streaming call with tool_choice="auto", returning all tool calls made.

        Used for Phase A explicit tool calling, where the LLM may call zero,
        one, or multiple tools in a single response.

        Each returned dict has:
            id:        Tool call ID string.
            name:      Tool function name.
            arguments: Raw JSON arguments string.

        Args:
            messages:    Full prompt as {role, content} dicts.
            tools:       List of tool schemas the LLM may call.
            temperature: Sampling temperature.

        Returns:
            List of tool call dicts (may be empty if the LLM made no tool calls).
        """
        logger.debug(
            "OpenAI call_with_tools started",
            extra={
                "model": self._model,
                "tools": [t["function"]["name"] for t in tools],
                "messages": len(messages),
            },
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools),
            tool_choice="auto",
            temperature=temperature,
            stream=False,
        )

        message = response.choices[0].message
        if not message.tool_calls:
            logger.debug("call_with_tools: LLM made no tool calls (pure conversation turn)")
            return []

        result: list[dict[str, Any]] = []
        for tc in message.tool_calls:
            if tc.type == "function":
                result.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                })

        logger.debug(
            "call_with_tools: LLM made tool calls",
            extra={"tool_call_count": len(result), "tools_called": [r["name"] for r in result]},
        )

        return result

    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming completion (used for evals / scripts).

        Args:
            messages:    Prompt as {role, content} dicts.
            temperature: Sampling temperature.

        Returns:
            Full response text.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model


# Protocol conformance is verified structurally — OpenAIProvider satisfies
# LLMProvider via duck typing (stream_tool_call + call_with_tools + complete
# methods all match the protocol signatures).
