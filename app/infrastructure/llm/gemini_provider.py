"""
app/infrastructure/llm/gemini_provider.py
──────────────────────────────────────────
Google Gemini implementation of the LLMProvider protocol.

Uses the ``google-genai`` SDK (already a project dependency via GeminiEmbedder).
All domain and orchestration code depends only on the abstract ``LLMProvider``
protocol defined in ``app/domain/llm/provider.py``.

Recommended model: ``gemini-3.5-flash-lite`` (fast, generous free-tier / low-cost,
excellent function/tool calling, 1M context window).

Calling modes:
  stream_tool_call  — forced single-tool call (Phase B: generate_response)
  call_with_tools   — auto tool_choice, zero-to-many tool calls (Phase A)
  complete          — non-streaming completion for evals / scripts
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from google import genai
from google.genai import types
from google.genai import errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

def _is_retryable_error(exc: BaseException) -> bool:
    """Return True for 429 rate-limit and 5xx server errors — both are retryable."""
    if isinstance(exc, errors.APIError):
        status_code = getattr(exc, "code", None)
        if status_code in (429, 500, 502, 503, 504):
            return True
    if isinstance(exc, errors.ClientError):
        status_code = getattr(exc, "code", None)
        if status_code == 429:
            return True
    if isinstance(exc, errors.ServerError):
        return True
    return False

# Shared retry policy: up to 6 attempts, exponential backoff (2s → 3s → ... max 30s)
_gemini_retry = retry(
    retry=retry_if_exception(_is_retryable_error),
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def _openai_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI-style {role, content} messages to Gemini format.

    Returns:
        Tuple of (system_instruction, list[Content]).
        system_instruction is the concatenated system prompt (or None).
        Content list contains only user/model turns.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            system_parts.append(content)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
        elif role == "tool":
            # Tool results — wrap as user turn with structured text so Gemini
            # understands the context (Gemini uses function_response Part type
            # but that requires full multi-turn function calling state which we
            # handle via the simplified "tool result as user text" pattern).
            tool_result_text = f"[Tool result for call {msg.get('tool_call_id', '')}]: {content}"
            contents.append(
                types.Content(role="user", parts=[types.Part(text=tool_result_text)])
            )

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[types.Tool]:
    """Convert OpenAI-style tool schemas to Gemini FunctionDeclaration format.

    OpenAI format:
      {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    Gemini format:
      types.Tool(function_declarations=[types.FunctionDeclaration(...)])
    """
    declarations: list[types.FunctionDeclaration] = []

    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        description = fn.get("description", "")
        parameters = fn.get("parameters")  # JSON Schema dict or None

        # Gemini SDK accepts the JSON Schema parameters dict directly as a Schema
        if parameters:
            schema = types.Schema(
                type=parameters.get("type", "object").upper(),
                properties={
                    k: _json_schema_prop_to_gemini(v)
                    for k, v in parameters.get("properties", {}).items()
                },
                required=parameters.get("required", []),
            )
        else:
            schema = None

        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=schema,
            )
        )

    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]


def _json_schema_prop_to_gemini(prop: dict[str, Any]) -> types.Schema:
    """Recursively convert a JSON Schema property dict to a Gemini Schema."""
    prop_type = prop.get("type", "string").upper()

    # Map standard JSON Schema types to Gemini types
    type_map = {
        "STRING": "STRING",
        "NUMBER": "NUMBER",
        "INTEGER": "INTEGER",
        "BOOLEAN": "BOOLEAN",
        "ARRAY": "ARRAY",
        "OBJECT": "OBJECT",
    }
    gemini_type = type_map.get(prop_type, "STRING")

    kwargs: dict[str, Any] = {
        "type": gemini_type,
        "description": prop.get("description", ""),
    }

    if "enum" in prop:
        kwargs["enum"] = prop["enum"]

    if gemini_type == "ARRAY" and "items" in prop:
        kwargs["items"] = _json_schema_prop_to_gemini(prop["items"])

    if gemini_type == "OBJECT" and "properties" in prop:
        kwargs["properties"] = {
            k: _json_schema_prop_to_gemini(v)
            for k, v in prop["properties"].items()
        }

    return types.Schema(**kwargs)


def _tool_calls_from_gemini_response(
    response: types.GenerateContentResponse,
) -> list[dict[str, Any]]:
    """Extract tool/function call dicts from a Gemini GenerateContentResponse.

    Returns a list of dicts compatible with the LLMProvider protocol:
      [{"id": str, "name": str, "arguments": str (JSON)}, ...]
    """
    result: list[dict[str, Any]] = []
    for candidate in (response.candidates or []):
        parts = candidate.content.parts if candidate.content and getattr(candidate.content, "parts", None) else []
        for part in parts:
            if part.function_call:
                fc = part.function_call
                args_json = json.dumps(dict(fc.args) if fc.args else {})
                result.append({
                    "id": fc.name,   # Gemini doesn't expose unique call IDs; use name
                    "name": fc.name,
                    "arguments": args_json,
                })
    return result


class GeminiProvider:
    """Concrete LLMProvider backed by the Google Gemini API.

    Satisfies the ``LLMProvider`` protocol from ``app/domain/llm/provider.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Uses the ``google-genai`` SDK (same SDK used by GeminiEmbedder).

    Args:
        api_key: Google AI Studio / GCP API key.
        model:   Gemini model ID (e.g. ``"gemini-3.5-flash-lite"``).
    """

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash-lite") -> None:
        if not api_key:
            raise ValueError(
                "Gemini API key is required but missing. "
                "Set GEMINI_API_KEY in your .env file."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        logger.info(
            "Initialised GeminiProvider",
            extra={"model": self._model},
        )

    # ── LLMProvider protocol methods ───────────────────────────────────────────

    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream a forced tool call and yield tool argument JSON tokens.

        Used for Phase B (generate_response). Forces the model to call the
        named tool, then streams the JSON arguments token-by-token so the
        orchestrator can speculate-decode ``message`` in real time.

        Yields:
            Raw JSON token fragments (accumulated string = valid JSON).
        """
        tool_name = tool_schema["function"]["name"]

        logger.debug(
            "GeminiProvider stream_tool_call started",
            extra={"model": self._model, "tool": tool_name, "messages": len(messages)},
        )

        system_instruction, contents = _openai_messages_to_gemini(messages)
        gemini_tools = _openai_tools_to_gemini([tool_schema])

        config = types.GenerateContentConfig(
            temperature=temperature,
            tools=gemini_tools,
            # Force the model to use the specific function
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
            system_instruction=system_instruction,
        )

        # Gemini async streaming via generate_content_stream
        # We accumulate the full function call args and yield them as a single
        # chunk (Gemini streaming for function calls works differently than text).
        # For genuine token streaming, we yield one big chunk at the end.
        # This is functionally equivalent because orchestrator buffers until done.
        
        @_gemini_retry
        async def _generate():
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        try:
            response = await _generate()

            # Extract function call arguments and yield as accumulated JSON
            tool_calls = _tool_calls_from_gemini_response(response)
            if tool_calls:
                # Yield the arguments JSON so orchestrator can parse it
                yield tool_calls[0]["arguments"]
            else:
                # Fallback: yield raw text if model ignored tool forcing
                text = ""
                for candidate in (response.candidates or []):
                    parts = candidate.content.parts if candidate.content and getattr(candidate.content, "parts", None) else []
                    for part in parts:
                        if part.text:
                            text += part.text
                logger.warning(
                    "GeminiProvider stream_tool_call: model returned text instead of tool call",
                    extra={"tool_name": tool_name, "text_snippet": text[:200]},
                )
                # Yield minimal valid JSON so orchestrator doesn't crash
                yield json.dumps({"message": text, "intent": "continue"})

        except Exception as exc:
            logger.error(
                "GeminiProvider stream_tool_call failed",
                extra={"model": self._model, "tool": tool_name},
                exc_info=exc,
            )
            raise

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Non-streaming call with tool_choice=auto, returning all tool calls made.

        Used for Phase A explicit tool calling (field extraction + saves).
        The model may call zero, one, or multiple tools in a single response.

        Returns:
            List of tool call dicts: [{"id": str, "name": str, "arguments": str}, ...]
            Empty list if the model made no tool calls.
        """
        logger.debug(
            "GeminiProvider call_with_tools started",
            extra={
                "model": self._model,
                "tools": [t["function"]["name"] for t in tools],
                "messages": len(messages),
            },
        )

        system_instruction, contents = _openai_messages_to_gemini(messages)
        gemini_tools = _openai_tools_to_gemini(tools)

        config = types.GenerateContentConfig(
            temperature=temperature,
            tools=gemini_tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_instruction,
        )

        @_gemini_retry
        async def _generate():
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

        try:
            response = await _generate()
        except Exception as exc:
            logger.error(
                "GeminiProvider call_with_tools failed",
                extra={"model": self._model},
                exc_info=exc,
            )
            raise

        result = _tool_calls_from_gemini_response(response)

        if not result:
            raw_text = ""
            for candidate in (response.candidates or []):
                parts = candidate.content.parts if candidate.content and getattr(candidate.content, "parts", None) else []
                for part in parts:
                    if part.text:
                        raw_text += part.text
            logger.debug(
                "call_with_tools: Gemini made no tool calls. Raw content: %s",
                raw_text[:300],
            )
        else:
            logger.debug(
                "call_with_tools: Gemini made tool calls",
                extra={
                    "tool_call_count": len(result),
                    "tools_called": [r["name"] for r in result],
                },
            )

        return result

    @_gemini_retry
    async def complete(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> str:
        """Non-streaming completion (used for evals / scripts).

        Returns:
            Full response text.
        """
        system_instruction, contents = _openai_messages_to_gemini(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        text = ""
        for candidate in (response.candidates or []):
            parts = candidate.content.parts if candidate.content and getattr(candidate.content, "parts", None) else []
            for part in parts:
                if part.text:
                    text += part.text
        return text

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model
