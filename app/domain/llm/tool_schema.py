"""
app/domain/llm/tool_schema.py
──────────────────────────────
Tool-call schemas for all conversation turn phases.

TWO-PHASE ARCHITECTURE
─────────────────────
Phase A — Explicit Tool Calling (silent, not streamed):
  The LLM chooses from a set of NAMED tools to explicitly act on what the
  user said.  Each tool corresponds to a distinct backend action with its own
  typed schema.  The orchestrator dispatches each returned tool call to a
  named handler — it never pattern-matches on a generic payload.

  Tools exposed to the LLM in Phase A:
    save_text_field         — save a free-text field value
    save_enum_field         — save a field whose value must match a fixed list
    save_quantitative_field — save a field that must contain a numeric/KPI signal
    mark_session_complete   — signal the LLM believes all fields are captured
    set_next_topic          — advisory: which field to ask about next

  If the LLM emits no tool calls (pure conversation, no data to capture),
  Phase A is skipped entirely and only Phase B runs.

Phase B — Response (streamed to client):
  Schema:  ``generate_response``
  Returns: message, suggested_next_topic, model_believes_complete
  Purpose: Generate the conversational reply AFTER the write outcome is known.
           The write outcomes from Phase A are injected into context as
           tool-result messages so the LLM can only confirm what actually saved.

Legacy schemas ``conversation_turn_result`` and ``extract_answers`` are retained
for import compatibility with existing test harnesses and scripts.
"""
from __future__ import annotations

import json
from typing import Any

from app.project_profiles.base_profile import BaseProfile


# ── Phase A: Categorized tool schemas ─────────────────────────────────────────

def get_phase_a_tools(profile: BaseProfile) -> list[dict[str, Any]]:
    """Return the list of Phase A action tools for the given profile.

    The tool list is built dynamically:
      - ``save_text_field`` is always included.
      - ``save_enum_field`` is included only when the profile has enum fields.
      - ``save_quantitative_field`` is included only when the profile has
        fields tagged with quantitative-signal validation (currently identified
        by containing the word "metric" or "kpi" in their description).
      - ``mark_session_complete`` and ``set_next_topic`` are always included.

    Keeping the tool list minimal reduces prompt token usage and avoids
    confusing the LLM with near-duplicate choices.

    Args:
        profile: Active project profile (used to introspect field types).

    Returns:
        List of tool dicts in the OpenAI function/tool calling format.
    """
    # Collect all field codes with descriptions for the save tools
    all_field_codes = [f.code for f in profile.required_fields]
    enum_field_codes = [f.code for f in profile.required_fields if f.enum_values]
    quantitative_field_codes = [
        f.code for f in profile.required_fields
        if f.enum_values is None and any(
            kw in (f.description or "").lower()
            for kw in ("metric", "kpi", "measur", "target", "rate", "roi", "revenue",
                       "growth", "uplift", "conversion", "impressions", "clicks",
                       "reach", "signups", "leads", "sales", "percent")
        )
    ]

    tools: list[dict[str, Any]] = []

    # ── save_text_field ───────────────────────────────────────────────────────
    # Always included — handles any free-text field.
    tools.append({
        "type": "function",
        "function": {
            "name": "save_text_field",
            "description": (
                "Save a free-text field value extracted from the user's message. "
                "Call this once per field you can identify. "
                "Use the EXACT field code from the list of required fields. "
                "Do NOT call this for fields that require a value from a fixed list "
                "(use save_enum_field for those) or that must be a numeric/KPI value "
                "(use save_quantitative_field for those)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field_code": {
                        "type": "string",
                        "description": (
                            "The exact machine-readable field code. "
                            f"Must be one of: {all_field_codes!r}."
                        ),
                        "enum": all_field_codes,
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "The extracted value as a concise string. "
                            "Preserve the client's own words where possible."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": _CONFIDENCE_DESC,
                    },
                },
                "required": ["field_code", "value", "confidence"],
            },
        },
    })

    # ── save_enum_field ───────────────────────────────────────────────────────
    # Only included when there are enum-constrained fields.
    if enum_field_codes:
        # Bug 8 fix: Build enum_map with BOTH machine values AND labels so the LLM
        # can match whatever form the user types (e.g. "Percentage Discount" OR "percentage_discount").
        enum_map: dict[str, list[str]] = {}
        for f in profile.required_fields:
            if f.enum_options:
                # Include both label and value so either form is recognizable
                combined = []
                for o in f.enum_options:
                    label = o.get("label", "")
                    value = o.get("value", "")
                    if label and label not in combined:
                        combined.append(label)
                    if value and value not in combined and value != label:
                        combined.append(value)
                enum_map[f.code] = combined
            elif f.enum_values:
                enum_map[f.code] = f.enum_values
                
        tools.append({
            "type": "function",
            "function": {
                "name": "save_enum_field",
                "description": (
                    "Save a field whose value must be chosen from a fixed list of allowed options. "
                    "ONLY use this tool for fields with a defined options list. "
                    "The value you provide should match one of the allowed options as closely as possible. "
                    f"Enum options available: {json.dumps(enum_map, indent=None)}. "
                    "If the user's answer does not clearly map to an option, do NOT call this tool — "
                    "instead, respond in Phase B asking them to choose from the options."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field_code": {
                            "type": "string",
                            "description": "The exact field code for an enum-constrained field.",
                            "enum": enum_field_codes,
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "The user's chosen value. Must match one of the allowed options "
                                "for this field."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": _CONFIDENCE_DESC,
                        },
                    },
                    "required": ["field_code", "value", "confidence"],
                },
            },
        })

    # ── save_quantitative_field ───────────────────────────────────────────────
    # Only included when there are quantitative metric fields.
    if quantitative_field_codes:
        tools.append({
            "type": "function",
            "function": {
                "name": "save_quantitative_field",
                "description": (
                    "Save a field that must contain a measurable, numeric, or KPI-style value. "
                    "Examples: '20% increase in conversions', 'reach 10k impressions', "
                    "'grow revenue by 2x in Q3'. "
                    "Only call this if the user provided a value with a quantitative signal "
                    "(a number, percentage, rate, or measurable target). "
                    "If the user gave a vague qualitative answer (e.g. 'grow the brand'), "
                    "do NOT call this tool — use Phase B to ask them for a specific metric. "
                    f"Quantitative fields: {quantitative_field_codes!r}."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field_code": {
                            "type": "string",
                            "description": "The exact field code for the quantitative field.",
                            "enum": quantitative_field_codes,
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "The extracted value. Must include a numeric or measurable "
                                "signal (number, percentage, specific target, etc.)."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": _CONFIDENCE_DESC,
                        },
                    },
                    "required": ["field_code", "value", "confidence"],
                },
            },
        })

    # ── mark_session_complete ─────────────────────────────────────────────────
    tools.append({
        "type": "function",
        "function": {
            "name": "mark_session_complete",
            "description": (
                "Call this when you believe ALL required fields have now been captured. "
                "This is advisory — the system validates independently against the "
                "required fields list. Only call this once per turn, and only when "
                "all fields appear to be filled."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    })

    # ── set_next_topic ────────────────────────────────────────────────────────
    tools.append({
        "type": "function",
        "function": {
            "name": "set_next_topic",
            "description": (
                "Advisory: indicate which topic or field you plan to ask about next. "
                "This helps the system track conversation flow. "
                "Call this at most once per turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": (
                            "Natural-language description of the next field/topic to ask about "
                            "(e.g. 'target audience', 'campaign timeline')."
                        ),
                    },
                },
                "required": ["topic"],
            },
        },
    })
    # ── save_color_suggestion ─────────────────────────────────────────────────
    # Bug 5 fix: Structured output forces the LLM to include a valid HEX code.
    # Only include when the profile has brand_personality or custom_colors_and_fonts
    # fields (i.e., creative colour context is relevant).
    color_field_codes = {f.code for f in profile.required_fields if f.required is False or f.required}
    if "brand_personality" in color_field_codes or "custom_colors_and_fonts" in color_field_codes or "hybrid_notes" in color_field_codes:
        tools.append({
            "type": "function",
            "function": {
                "name": "save_color_suggestion",
                "description": (
                    "Save a structured color suggestion for the brand palette. "
                    "Call this ONLY when you are suggesting a specific color as part of "
                    "a color palette recommendation. You MUST provide a valid HEX code — "
                    "a 6-digit code starting with '#' (e.g. '#FF5733'). "
                    "3-digit shorthand (#RGB) is NOT accepted. "
                    "If you cannot determine a specific HEX code, do NOT call this tool — "
                    "instead ask the user for more information about their color preferences."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Human-readable color name (e.g. 'Deep Navy', 'Warm Ivory').",
                        },
                        "hex": {
                            "type": "string",
                            "description": (
                                "Exact 6-digit HEX code with leading '#'. "
                                "Format: #RRGGBB (e.g. '#1A2B3C'). "
                                "REQUIRED — do not omit or approximate."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": "1-2 sentences explaining why this color suits the brand.",
                        },
                    },
                    "required": ["name", "hex", "rationale"],
                },
            },
        })

    return tools


_CONFIDENCE_DESC = (
    "Your confidence that this extracted value correctly represents the field. "
    "Calibrate carefully — do NOT default to 0.9 for everything:\n"
    "• 0.95–1.0: User selected an option directly or stated the exact value verbatim, "
    "or is explicitly correcting/updating a previously captured field.\n"
    "• 0.80–0.94: Clear and unambiguous but required minor interpretation.\n"
    "• 0.60–0.79: Answer present but paraphrased or partial.\n"
    "• 0.40–0.59: Vague or hedged — guessing intent.\n"
    "• 0.10–0.39: Highly uncertain — inferring from indirect context."
)


# ── Phase A: Parse multi-tool-call response ────────────────────────────────────

def parse_phase_a_tool_calls(response_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse the list of tool call dicts from a Phase A LLM response.

    Each item in ``response_data`` is one tool call returned by the LLM.
    Returns a normalised list of dicts:
        {
            "tool_call_id": str,
            "tool_name":    str,
            "arguments":    dict,   # parsed JSON arguments
            "parse_error":  str | None,
        }

    Args:
        response_data: List of raw tool call dicts from the LLM response,
                       each with keys: id, name, arguments (str JSON).

    Returns:
        List of normalised tool call dicts, one per tool call.
    """
    parsed: list[dict[str, Any]] = []
    for raw in response_data:
        tool_call_id = raw.get("id", "")
        tool_name = raw.get("name", "")
        arguments_str = raw.get("arguments", "{}")
        parse_error: str | None = None
        arguments: dict[str, Any] = {}

        try:
            arguments = json.loads(arguments_str) if arguments_str else {}
        except json.JSONDecodeError as exc:
            parse_error = f"JSON parse error: {exc}"

        parsed.append({
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "parse_error": parse_error,
        })

    return parsed


# ── Phase B: Response schema ───────────────────────────────────────────────────

def get_response_tool_schema() -> dict[str, Any]:
    """Return the JSON schema for Phase B: streamed response generation.

    The LLM generates its conversational reply here. By the time this call is
    made, the write outcomes from Phase A are already in the message context as
    tool-result messages, so the LLM cannot generate false confirmations —
    it can only reference what the system reported as actually saved.

    Returns:
        A dict conforming to the OpenAI function/tool calling spec.
    """
    return {
        "type": "function",
        "function": {
            "name": "generate_response",
            "description": (
                "Generate the conversational response to the user. "
                "You have just received a structured write-outcome report in your context "
                "(from the Phase A tool calls). "
                "RULES:\n"
                "- Only confirm fields that appear with status 'saved' in the write outcome.\n"
                "- For fields with status 'rejected_low_confidence': tell the user you weren't "
                "confident about their answer and ask for clarification.\n"
                "- For fields with status 'rejected_enum': tell the user the value wasn't valid "
                "and list the valid options.\n"
                "- For fields with status 'rejected_unasked': do not mention this to the user, "
                "simply ask for that information naturally.\n"
                "- For fields with status 'rejected_malformed': tell the user you couldn't "
                "understand what they meant for that field and ask them to clarify.\n"
                "- For fields with status 'write_failed': tell the user the save failed and "
                "that you'll try again, or ask them to repeat the information.\n"
                "- If write_outcomes is empty (no tool calls were made in Phase A), respond "
                "naturally to the user's message without confirming any saves."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "PROVIDE_INFO",
                            "CORRECT_PREVIOUS",
                            "ASK_QUESTION",
                            "REQUEST_CLARIFICATION",
                            "CONFIRM",
                            "REJECT",
                            "OFF_TOPIC"
                        ],
                        "description": (
                            "The primary intent of the user's latest message.\n"
                            "- CORRECT_PREVIOUS: User is explicitly correcting or changing a previously provided value.\n"
                            "- PROVIDE_INFO: User is providing new information.\n"
                            "- ASK_QUESTION: User is asking you a question.\n"
                            "- REQUEST_CLARIFICATION: User doesn't understand your question or needs more details.\n"
                            "- CONFIRM: User is confirming or agreeing.\n"
                            "- REJECT: User is rejecting or disagreeing.\n"
                            "- OFF_TOPIC: User is talking about something unrelated.\n"
                            "If ambiguous, default to 'PROVIDE_INFO' or 'ASK_QUESTION'."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Your conversational response to the user. This is what the user "
                            "will see. Be warm, natural, and on-brand. Only confirm fields "
                            "that were successfully saved per the write outcome report."
                        ),
                    },
                    "suggested_next_topic": {
                        "type": "string",
                        "description": "Advisory: best topic to explore next.",
                    },
                    "model_believes_complete": {
                        "type": "boolean",
                        "description": (
                            "Set to true if you believe all required information has been "
                            "gathered. Advisory only — the system validates independently."
                        ),
                    },
                },
                "required": ["intent", "message", "suggested_next_topic", "model_believes_complete"],
            },
        },
    }


def parse_response_result(tool_call_json: str) -> dict[str, Any]:
    """Parse the Phase B tool call JSON.

    Args:
        tool_call_json: Raw JSON string from the streamed generate_response call.

    Returns:
        Parsed dict with keys: message, suggested_next_topic, model_believes_complete.

    Raises:
        ValueError: If JSON is malformed or missing required keys.
    """
    try:
        result = json.loads(tool_call_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Phase B tool call result is not valid JSON: {exc}") from exc

    required_keys = {"message", "suggested_next_topic", "model_believes_complete"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Phase B tool call result missing required keys: {missing}")

    return result


# ── Legacy schemas (kept for import compatibility) ─────────────────────────────

def get_extract_tool_schema() -> dict[str, Any]:
    """Legacy single-tool extraction schema. No longer used by the orchestrator.

    Retained so external test harnesses and scripts that import this function
    do not break. The orchestrator now uses get_phase_a_tools() (Phase A)
    and get_response_tool_schema() (Phase B) instead.
    """
    return {
        "type": "function",
        "function": {
            "name": "extract_answers",
            "description": (
                "Extract structured field values from the user's most recent message. "
                "[LEGACY — use get_phase_a_tools() instead]"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "extracted_answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field_hint": {"type": "string"},
                                "value": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["field_hint", "value", "confidence"],
                        },
                    },
                    "suggested_next_topic": {"type": "string"},
                    "model_believes_complete": {"type": "boolean"},
                },
                "required": ["extracted_answers", "suggested_next_topic", "model_believes_complete"],
            },
        },
    }


def parse_extract_result(tool_call_json: str) -> dict[str, Any]:
    """Legacy parse function for the old extract_answers schema.

    Retained for import compatibility with existing test files.
    """
    try:
        result = json.loads(tool_call_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Phase A tool call result is not valid JSON: {exc}") from exc

    required_keys = {"extracted_answers", "suggested_next_topic", "model_believes_complete"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Phase A tool call result missing required keys: {missing}")

    return result


def get_conversation_tool_schema() -> dict[str, Any]:
    """Legacy single-pass schema. No longer used by the orchestrator.

    Retained so external test harnesses and scripts that import this function
    do not break.
    """
    return {
        "type": "function",
        "function": {
            "name": "conversation_turn_result",
            "description": (
                "Record the result of one conversation turn. Always call this function "
                "with every response. The 'message' field contains what you say to the "
                "user. The 'extracted_answers' field contains any field values you "
                "extracted from the user's most recent message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "extracted_answers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field_hint": {"type": "string"},
                                "value": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["field_hint", "value", "confidence"],
                        },
                    },
                    "suggested_next_topic": {"type": "string"},
                    "model_believes_complete": {"type": "boolean"},
                },
                "required": [
                    "message",
                    "extracted_answers",
                    "suggested_next_topic",
                    "model_believes_complete",
                ],
            },
        },
    }


def parse_tool_call_result(tool_call_json: str) -> dict[str, Any]:
    """Legacy parse function. No longer used by the orchestrator.

    Retained for import compatibility with existing test files.
    """
    try:
        result = json.loads(tool_call_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool call result is not valid JSON: {exc}") from exc

    required_keys = {"message", "extracted_answers", "suggested_next_topic", "model_believes_complete"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Tool call result missing required keys: {missing}")

    return result
