"""
app/infrastructure/llm/groq_provider.py
──────────────────────────────────────────
Groq implementation of the LLMProvider protocol.

This module lives in the *infrastructure* layer because it depends on a
third-party SDK (``groq``). All domain and orchestration code depends
only on the abstract ``LLMProvider`` protocol defined in
``app/domain/llm/provider.py``.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, cast

from groq import AsyncGroq
from groq.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

logger = logging.getLogger(__name__)

# Maximum number of rate-limit retries
_MAX_RETRIES = 3


class GroqProvider:
    """Concrete LLMProvider backed by Groq's chat completions API.

    Satisfies the ``LLMProvider`` protocol from ``app/domain/llm/provider.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Args:
        api_key:    Groq API key.
        model:      Chat model identifier (e.g. ``"llama-3.1-8b-instant"``).
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncGroq(api_key=api_key, max_retries=_MAX_RETRIES)
        self._model = model

    # ── LLMProvider protocol methods ───────────────────────────────────────────

    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a forced tool call and yield tool argument tokens."""
        tool_name = tool_schema["function"]["name"]

        logger.debug(
            "Groq stream_tool_call started",
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
        """Non-streaming call with tool_choice="auto", returning all tool calls made."""
        logger.debug(
            "Groq call_with_tools started",
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
            logger.debug("call_with_tools: LLM made no tool calls")
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
        """Non-streaming completion."""
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
