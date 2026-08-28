"""
app/infrastructure/llm/ollama_provider.py
─────────────────────────────────────────
Ollama implementation of the LLMProvider protocol.

Uses Ollama's **native** /api/chat endpoint (NOT the OpenAI-compat shim at
/v1/chat/completions).  The native endpoint gives cleaner tool_calls parsing
because arguments arrive as already-parsed dicts, not raw JSON strings.

Two calling modes (identical signatures to OpenAIProvider):
  call_with_tools   — non-streaming, auto tool_choice, Phase A
  stream_tool_call  — forced single-tool call (Phase B generate_response)
  complete          — non-streaming completion for evals / scripts

Constructor: OllamaProvider(base_url: str, model: str)

─────────────────────────────────────────────────────────────────────
REUSABILITY BOUNDARY:
  This class is the ONLY file that must be modified (or replaced) when
  switching the LLM vendor.  Everything above this layer is agnostic.

  To revert to OpenAI: in app/api/deps.py, replace
      return OllamaProvider(...)
  with
      return OpenAIProvider(...)
  That single line change is all that is required.
─────────────────────────────────────────────────────────────────────

Error handling policy:
  - If Ollama returns zero tool calls → return []  (fast path, not an error)
  - If arguments JSON is malformed → log warning, treat call as having {} args
    (parse_phase_a_tool_calls handles {} and produces a rejected_malformed outcome)
  - If connection is refused / server unreachable → log error, return []
    (orchestrator's _run_phase_a fallback already handles empty returns)
  - Never raise to the orchestrator from these error paths.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# Timeout for non-streaming Ollama calls (seconds).
# Kept generous because local inference can be slow on CPU.
_TIMEOUT_SECONDS = 600.0

# Timeout for streaming calls — same generous allowance.
_STREAM_TIMEOUT_SECONDS = 600.0


class OllamaProvider:
    """Concrete LLMProvider backed by Ollama's native /api/chat endpoint.

    Satisfies the ``LLMProvider`` protocol from ``app/domain/llm/provider.py``
    via structural (duck-typed) conformance — no explicit subclassing required.

    Key difference from OpenAIProvider:
      Ollama's native endpoint returns ``message.tool_calls[].function.arguments``
      as an **already-parsed dict**, not a JSON string.  This class serialises
      it back to a JSON string before returning, so the orchestrator sees the
      same shape it expects from OpenAIProvider.

    Args:
        base_url: Ollama server base URL (e.g. ``"http://localhost:11434"``).
        model:    Model identifier (e.g. ``"qwen2.5:7b"``).
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    # ── LLMProvider protocol methods ───────────────────────────────────────────

    async def call_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Non-streaming call with tool_choice="auto", returning all tool calls made.

        Used for Phase A explicit tool calling, where the LLM may call zero,
        one, or multiple tools in a single response.

        OLLAMA-SPECIFIC STRATEGY (differs from OpenAIProvider):
        ────────────────────────────────────────────────────────
        ``qwen2.5:7b`` does not reliably call tools when a ``tools`` array is
        provided — it ignores tool schemas and replies in plain text. This was
        already discovered for Phase B and solved with JSON content mode.

        We apply the SAME strategy here for Phase A:
          1. Do NOT send a ``tools`` array in the payload.
          2. Use Ollama's ``format: json`` to get valid JSON output.
          3. Inject a tightly structured JSON schema instruction into a final
             system message, telling the model exactly what to return.
          4. Parse the returned JSON and convert it back into the standard
             tool call dict format the orchestrator expects.

        Each returned dict has:
            id:        Synthetic tool call ID string.
            name:      Tool function name (save_text_field / save_enum_field / etc.).
            arguments: Raw JSON arguments string.

        Args:
            messages:    Full prompt as {role, content} dicts.
            tools:       List of tool schemas (used only to build the JSON instruction).
            temperature: Sampling temperature.

        Returns:
            List of tool call dicts (may be empty if the LLM made no tool calls).
        """
        logger.debug(
            "OllamaProvider call_with_tools started (JSON content mode)",
            extra={
                "model": self._model,
                "tools": [t["function"]["name"] for t in tools],
                "messages": len(messages),
            },
        )

        # ── Build the JSON extraction instruction ──────────────────────────────
        # Describe the available field-saving tools in a compact, model-friendly way
        field_saving_tools = {t["function"]["name"]: t for t in tools
                              if t["function"]["name"] in (
                                  "save_text_field", "save_enum_field",
                                  "save_quantitative_field"
                              )}

        tool_description = (
            "You MUST respond with ONLY a valid JSON object. No markdown, no prose, no code fences.\n\n"
            "JSON format:\n"
            "{\n"
            '  "tool_calls": [\n'
            "    {\n"
            '      "name": "<tool_name>",\n'
            '      "field_code": "<exact field code>",\n'
            '      "value": "<extracted value>",\n'
            '      "confidence": <0.0 to 1.0>\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Available tools:\n"
            '- "save_text_field": for free-text fields (client_name, primary_objective, '
            "target_audience, timeline, deliverables, key_messages)\n"
            '- "save_enum_field": for fields that must use a value from a fixed list '
            "(project_type, brand_personality, distribution_channels, brand_identity_preference)\n"
            '- "save_quantitative_field": for numeric/KPI fields (success_metrics, budget_range)\n\n'
            "INSTRUCTIONS:\n"
            "1. Read the user's last message carefully.\n"
            "2. For EACH piece of information the user gave that matches a required field, "
            "add one entry to the tool_calls array.\n"
            "3. If the user gave NO field data (just chatting), return: {\"tool_calls\": []}\n"
            "4. Use confidence=0.95 when the user stated something very clearly, "
            "0.80 for clear but with minor interpretation, 0.65 for uncertain.\n"
            "5. For save_enum_field, value MUST exactly match one of the allowed options.\n\n"
            "EXAMPLE — if the user said 'my company is Acme and I need a logo':\n"
            '{"tool_calls": [{"name": "save_text_field", "field_code": "client_name", "value": "Acme", "confidence": 0.95}]}\n'
        )

        augmented_messages = list(messages) + [
            {"role": "system", "content": tool_description}
        ]

        payload = {
            "model": self._model,
            "messages": augmented_messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        # NOTE: No "tools" key — we skip native tool-calling mode intentionally

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(
                "OllamaProvider call_with_tools: connection refused — is Ollama running?",
                extra={"base_url": self._base_url, "error": str(exc)},
            )
            return []
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OllamaProvider call_with_tools: HTTP error from Ollama",
                extra={"status_code": exc.response.status_code, "error": str(exc)},
            )
            return []
        except Exception as exc:
            logger.error(
                "OllamaProvider call_with_tools: unexpected error",
                exc_info=exc,
            )
            return []

        message = data.get("message", {})
        content = message.get("content", "")

        # ── Parse the JSON response into tool call dicts ───────────────────────
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "OllamaProvider call_with_tools: could not parse JSON content",
                extra={"error": str(exc), "content": content[:500]},
            )
            return []

        raw_calls = parsed.get("tool_calls", [])
        if not raw_calls:
            logger.debug(
                "OllamaProvider call_with_tools: LLM made no tool calls (pure conversation turn)"
            )
            return []

        result: list[dict[str, Any]] = []
        for i, tc in enumerate(raw_calls):
            name = tc.get("name", "")
            if not name:
                continue
            # Build arguments dict from the flat JSON fields
            arguments_dict = {
                "field_code": tc.get("field_code", ""),
                "value": tc.get("value", ""),
                "confidence": tc.get("confidence", 0.5),
            }
            try:
                arguments_str = json.dumps(arguments_dict)
            except (TypeError, ValueError):
                arguments_str = "{}"

            result.append({
                "id": f"call_{i}_{uuid.uuid4().hex[:8]}",
                "name": name,
                "arguments": arguments_str,
            })

        logger.debug(
            "OllamaProvider call_with_tools: LLM made tool calls",
            extra={
                "tool_call_count": len(result),
                "tools_called": [r["name"] for r in result],
            },
        )

        return result



    async def stream_tool_call(
        self,
        messages: list[dict[str, Any]],
        tool_schema: dict[str, Any],
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Generate the Phase B response as structured JSON content.

        OLLAMA-SPECIFIC STRATEGY (differs from OpenAIProvider):
        ────────────────────────────────────────────────────────
        ``qwen2.5:7b`` does not reliably honour forced single-tool-call
        instructions — it ignores the ``tools`` array and replies in plain
        content mode.  Rather than fighting the model, we lean into its
        natural strength:

          1. Do NOT send a ``tools`` array in the payload.
          2. Use Ollama's ``"format": "json"`` to guarantee valid JSON output.
          3. Inject a system instruction describing the expected JSON schema
             (intent, message, suggested_next_topic, model_believes_complete).
          4. Read the model's ``message.content`` and yield it directly.

        The orchestrator sees the same JSON string it would get from
        OpenAI's tool-call arguments — no downstream changes required.

        Safety nets (kept, should rarely fire now):
          - If the model somehow still returns ``tool_calls``, we extract from
            those (bonus path).
          - If the content is not valid JSON with a ``message`` key, we wrap it
            in the expected schema (content-fallback path).

        Args:
            messages:    Full prompt as {role, content} dicts.
            tool_schema: JSON Schema of the single tool (used only to extract
                         the schema description for the JSON instruction).
            temperature: Sampling temperature.

        Yields:
            The full response as a single JSON string chunk.
        """
        tool_name = tool_schema["function"]["name"]

        logger.debug(
            "OllamaProvider stream_tool_call started (JSON content mode)",
            extra={"model": self._model, "tool": tool_name, "messages": len(messages)},
        )

        # ── Build JSON-output instruction from the tool schema ──────────────
        # Extract the expected properties from the tool schema so the model
        # knows what JSON shape to produce.
        schema_props = tool_schema.get("function", {}).get("parameters", {}).get("properties", {})
        required_keys = tool_schema.get("function", {}).get("parameters", {}).get("required", [])

        schema_description = "Respond with a JSON object containing these fields:\n"
        for key, prop in schema_props.items():
            prop_type = prop.get("type", "string")
            prop_desc = prop.get("description", "")
            req_marker = " (REQUIRED)" if key in required_keys else ""
            if "enum" in prop:
                schema_description += f'  - "{key}": {prop_type}{req_marker} — {prop_desc} One of: {prop["enum"]}\n'
            else:
                schema_description += f'  - "{key}": {prop_type}{req_marker} — {prop_desc}\n'

        json_instruction = (
            "CRITICAL: You must respond with ONLY a valid JSON object — no markdown, "
            "no code fences, no extra text.\n\n"
            + schema_description
        )

        # Inject the JSON schema instruction as a final system message so it
        # takes highest priority.  We copy the messages list to avoid mutating
        # the caller's data.
        augmented_messages = list(messages) + [
            {"role": "system", "content": json_instruction}
        ]

        # Ollama's native /api/chat expects `arguments` in tool_calls history to be a dict,
        # unlike OpenAI which requires a JSON string. We parse it back to a dict here.
        for msg in augmented_messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except json.JSONDecodeError:
                            func["arguments"] = {}

        payload = {
            "model": self._model,
            "messages": augmented_messages,
            "format": "json",
            "stream": False,
            "options": {"temperature": temperature},
        }
        # NOTE: No "tools" key — we intentionally skip tool-calling mode
        # for Ollama so qwen2.5:7b responds in plain JSON content mode.

        try:
            async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error(
                "OllamaProvider stream_tool_call: connection refused",
                extra={"base_url": self._base_url, "error": str(exc)},
            )
            # Yield a minimal valid JSON so the orchestrator's Phase B parse
            # fails gracefully with its own error handler rather than crashing
            yield json.dumps({
                "intent": "PROVIDE_INFO",
                "message": "I'm sorry, I'm having trouble connecting right now. Please try again.",
                "suggested_next_topic": "",
                "model_believes_complete": False,
            })
            return
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OllamaProvider stream_tool_call: HTTP error from Ollama",
                extra={"status_code": exc.response.status_code, "error": str(exc)},
            )
            yield json.dumps({
                "intent": "PROVIDE_INFO",
                "message": "I encountered an error generating a response. Please try again.",
                "suggested_next_topic": "",
                "model_believes_complete": False,
            })
            return
        except Exception as exc:
            logger.error(
                "OllamaProvider stream_tool_call: unexpected error",
                exc_info=exc,
            )
            yield json.dumps({
                "intent": "PROVIDE_INFO",
                "message": "Sorry, something went wrong. Please try again.",
                "suggested_next_topic": "",
                "model_believes_complete": False,
            })
            return

        message = data.get("message", {})

        # ── Bonus path: if the model still returned tool_calls, extract ─────
        # This shouldn't happen (we didn't send a tools array), but handle it
        # defensively in case a future Ollama version auto-detects tools.
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls:
            logger.debug(
                "OllamaProvider stream_tool_call: model returned tool_calls despite no tools in payload",
                extra={"tool_name": tool_name},
            )
            func = raw_tool_calls[0].get("function", {})
            raw_arguments = func.get("arguments", {})
            if isinstance(raw_arguments, dict):
                try:
                    yield json.dumps(raw_arguments)
                except (TypeError, ValueError):
                    yield json.dumps({
                        "intent": "PROVIDE_INFO",
                        "message": "I had trouble formatting my response. Please try again.",
                        "suggested_next_topic": "",
                        "model_believes_complete": False,
                    })
            elif isinstance(raw_arguments, str):
                yield raw_arguments
            else:
                yield json.dumps({
                    "intent": "PROVIDE_INFO",
                    "message": "I had trouble formatting my response. Please try again.",
                    "suggested_next_topic": "",
                    "model_believes_complete": False,
                })
            return

        # ── Primary path: extract from message.content (JSON content mode) ──
        content = message.get("content", "")

        # With format:"json", content should be valid JSON.  Parse and validate.
        try:
            parsed = json.loads(content)
            if "message" in parsed:
                logger.debug(
                    "OllamaProvider stream_tool_call: valid JSON content with 'message' key",
                    extra={"tool_name": tool_name, "content_len": len(content)},
                )
                yield json.dumps(parsed)
                return
            else:
                # Valid JSON but missing the 'message' key — wrap it
                logger.warning(
                    "OllamaProvider stream_tool_call: JSON content missing 'message' key, wrapping",
                    extra={"tool_name": tool_name, "keys": list(parsed.keys())},
                )
                # Try to salvage: maybe the model used a different key
                fallback_message = (
                    parsed.get("response", "")
                    or parsed.get("text", "")
                    or parsed.get("content", "")
                    or str(parsed)
                )
                yield json.dumps({
                    "intent": parsed.get("intent", "PROVIDE_INFO"),
                    "message": fallback_message,
                    "suggested_next_topic": parsed.get("suggested_next_topic", ""),
                    "model_believes_complete": parsed.get("model_believes_complete", False),
                })
                return
        except (json.JSONDecodeError, TypeError):
            pass

        # ── Content-fallback: wrap raw text in the expected schema ───────────
        # This should be very rare with format:"json" but kept as safety net.
        logger.warning(
            "OllamaProvider stream_tool_call: content not valid JSON, using content fallback",
            extra={"tool_name": tool_name, "content_len": len(content)},
        )
        yield json.dumps({
            "intent": "PROVIDE_INFO",
            "message": content or "I'm here to help. What would you like to know?",
            "suggested_next_topic": "",
            "model_believes_complete": False,
        })

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
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.error(
                "OllamaProvider complete: error calling Ollama",
                exc_info=exc,
            )
            return ""

        return data.get("message", {}).get("content", "")

    # ── Internal helpers ───────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        """The model identifier this provider is configured to use."""
        return self._model


# Protocol conformance is verified structurally — OllamaProvider satisfies
# LLMProvider via duck typing (stream_tool_call + call_with_tools + complete
# methods all match the protocol signatures).
