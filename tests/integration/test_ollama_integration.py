"""
tests/integration/test_ollama_integration.py
─────────────────────────────────────────────
Live integration test for OllamaProvider against a real local Ollama instance.

REQUIREMENTS:
  - Ollama running at http://localhost:11434
  - Model pulled: ollama pull qwen2.5:7b

Run with:
  python -m pytest tests/integration/test_ollama_integration.py -v -m live

These tests are deliberately excluded from the default CI run (deselect with
  -m "not live"  or simply don't pass -m live).

The test runs a 5-turn creative brief conversation and reports per-turn:
  - How many Phase A tool calls the model made
  - How many were valid (parseable + had required fields)
  - How many were malformed/empty
  - Raw model output for any malformed turn

This provides the evidence needed to judge whether qwen2.5:7b is accurate
enough for production use or should be swapped for a larger model.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from app.infrastructure.llm.ollama_provider import OllamaProvider
from app.domain.llm.tool_schema import get_phase_a_tools, get_response_tool_schema
from app.project_profiles.base_profile import BaseProfile, FieldDefinition

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Minimal creative-brief profile matching the real Picasso Fusion domain shape
_TEST_PROFILE = BaseProfile(
    profile_id="ollama_integration_test",
    persona_prompt=(
        "You are a friendly creative brief assistant helping gather information "
        "for a design project. Ask natural follow-up questions to collect the required fields."
    ),
    knowledge_namespace="ollama_integration_test",
    required_fields=[
        FieldDefinition(code="client_name", description="Name of the client or company", required=True),
        FieldDefinition(code="project_type", description="Type of creative project (e.g. logo, website, campaign)", required=True),
        FieldDefinition(code="target_audience", description="Who is the intended audience for this project", required=True),
        FieldDefinition(code="project_goals", description="Primary business or creative goals for the project", required=True),
        FieldDefinition(code="timeline", description="Desired timeline or deadline for the project", required=True),
    ],
)

# 5-turn scripted conversation providing progressively more information
_CONVERSATION_TURNS = [
    "Hi! I'm from Acme Corp and we need a new logo design.",
    "We sell B2B software to enterprise IT teams — mostly CTOs and IT directors.",
    "The goal is to modernize our brand image to look more premium and trustworthy. We're rebranding after a merger.",
    "We have about 6 weeks — the new brand needs to launch at our annual conference on September 15th.",
    "I think that covers everything we discussed. We're ready to proceed with the brief.",
]


def _is_ollama_reachable() -> bool:
    """Check if Ollama is running and reachable before running live tests."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def _is_model_available() -> bool:
    """Check if the required model is pulled in Ollama."""
    try:
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        return any(OLLAMA_MODEL in m for m in models)
    except Exception:
        return False


# ── Live integration tests ─────────────────────────────────────────────────────

@pytest.mark.live
class TestOllamaLiveIntegration:
    """Live tests that require a running Ollama instance with qwen2.5:7b."""

    @pytest.fixture(autouse=True)
    def check_ollama_available(self) -> None:
        if not _is_ollama_reachable():
            pytest.skip(
                "Ollama is not running at http://localhost:11434. "
                "Start Ollama and re-run with -m live."
            )
        if not _is_model_available():
            pytest.skip(
                f"Model {OLLAMA_MODEL!r} is not pulled. "
                f"Run: ollama pull {OLLAMA_MODEL}"
            )

    @pytest.mark.asyncio
    async def test_provider_satisfies_protocol_live(self) -> None:
        """Basic sanity: OllamaProvider works with real Ollama, complete() returns text."""
        from app.domain.llm.provider import LLMProvider

        provider = OllamaProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        assert isinstance(provider, LLMProvider)

        result = await provider.complete(
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            temperature=0.3,
        )
        assert isinstance(result, str)
        assert len(result) > 0, "complete() returned an empty string from live Ollama"
        print(f"\n[complete() response]: {result!r}")

    @pytest.mark.asyncio
    async def test_five_turn_conversation_tool_call_accuracy(self) -> None:
        """Run a 5-turn conversation and report per-turn Phase A tool call accuracy.

        This is the primary evidence test for qwen2.5:7b fitness.
        Output format (printed to stdout for --capture=no or -s flag):

            Turn 1: user_message
              Phase A tool calls: 2
              Valid:    2  [save_text_field(client_name=...), save_text_field(project_type=...)]
              Malformed: 0

            Turn 2: ...

            ═══ SUMMARY ═══
            Total turns: 5
            Total tool calls: 8
            Valid: 7  |  Malformed: 1
            Malformed rate: 12.5%
        """
        provider = OllamaProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        tools = get_phase_a_tools(_TEST_PROFILE)

        conversation_history: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _TEST_PROFILE.persona_prompt,
            }
        ]

        # Per-turn stats
        turn_stats: list[dict[str, Any]] = []
        total_tool_calls = 0
        total_valid = 0
        total_malformed = 0

        print(f"\n\n{'='*70}")
        print(f"LIVE OLLAMA INTEGRATION TEST -- {OLLAMA_MODEL}")
        print(f"{'='*70}\n")

        for turn_idx, user_message in enumerate(_CONVERSATION_TURNS, start=1):
            print(f"Turn {turn_idx}: {user_message!r}")

            # Add user message to history
            conversation_history.append({"role": "user", "content": user_message})

            # Run Phase A
            raw_tool_calls = await provider.call_with_tools(
                messages=conversation_history,
                tools=tools,
                temperature=0.3,
            )

            # Validate each tool call
            valid_calls: list[dict[str, Any]] = []
            malformed_calls: list[dict[str, Any]] = []

            for tc in raw_tool_calls:
                name = tc.get("name", "")
                arguments_str = tc.get("arguments", "{}")

                # Try parsing arguments
                try:
                    args = json.loads(arguments_str)
                except json.JSONDecodeError as exc:
                    malformed_calls.append({
                        "name": name,
                        "error": f"JSON parse error: {exc}",
                        "raw_arguments": arguments_str,
                    })
                    continue

                # Check required fields for field-saving tools
                field_saving_tools = {"save_text_field", "save_enum_field", "save_quantitative_field"}
                if name in field_saving_tools:
                    missing = []
                    for required_key in ("field_code", "value", "confidence"):
                        if not args.get(required_key) and args.get(required_key) != 0:
                            missing.append(required_key)
                    if missing:
                        malformed_calls.append({
                            "name": name,
                            "error": f"Missing required keys: {missing}",
                            "args": args,
                        })
                        continue

                valid_calls.append({"name": name, "args": args})

            if raw_tool_calls:
                # Replay the assistant tool-call message in Ollama's native format.
                # IMPORTANT: Ollama's /api/chat requires arguments as a DICT in the
                # message thread (not a JSON string). Our call_with_tools normalises
                # arguments to a JSON string for orchestrator compatibility, so we
                # parse it back here when building the conversation history.
                history_tool_calls = []
                for tc in raw_tool_calls:
                    try:
                        args_dict = json.loads(tc["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args_dict = {}
                    history_tool_calls.append({
                        "function": {
                            "name": tc["name"],
                            "arguments": args_dict,  # dict, not string
                        }
                    })
                conversation_history.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": history_tool_calls,
                })
                # Add tool results (one per tool call)
                for tc in raw_tool_calls:
                    conversation_history.append({
                        "role": "tool",
                        "content": json.dumps({"status": "saved"}),
                    })

            # Get Phase B response
            response_schema = get_response_tool_schema()
            phase_b_chunks = []
            async for chunk in provider.stream_tool_call(
                messages=conversation_history,
                tool_schema=response_schema,
                temperature=0.3,
            ):
                phase_b_chunks.append(chunk)

            phase_b_json = "".join(phase_b_chunks)
            try:
                phase_b_result = json.loads(phase_b_json)
                assistant_message = phase_b_result.get("message", "")
            except json.JSONDecodeError:
                assistant_message = phase_b_json  # raw text fallback

            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            # Record stats
            turn_count = len(raw_tool_calls)
            valid_count = len(valid_calls)
            malformed_count = len(malformed_calls)
            total_tool_calls += turn_count
            total_valid += valid_count
            total_malformed += malformed_count

            turn_stat = {
                "turn": turn_idx,
                "user_message": user_message,
                "tool_call_count": turn_count,
                "valid_count": valid_count,
                "malformed_count": malformed_count,
                "valid_calls": valid_calls,
                "malformed_calls": malformed_calls,
                "assistant_response": assistant_message,
            }
            turn_stats.append(turn_stat)

            # Print per-turn report
            print(f"  Phase A tool calls: {turn_count}")
            if valid_calls:
                print(f"  Valid ({valid_count}):")
                for vc in valid_calls:
                    if vc["name"] in {"save_text_field", "save_enum_field", "save_quantitative_field"}:
                        print(f"    [OK] {vc['name']}({vc['args'].get('field_code')}={vc['args'].get('value')!r}, conf={vc['args'].get('confidence')})")
                    else:
                        print(f"    [OK] {vc['name']}({vc.get('args', {})})")  # noqa: E501
            if malformed_calls:
                print(f"  [MALFORMED] ({malformed_count}):")
                for mc in malformed_calls:
                    print(f"    [FAIL] {mc['name']}: {mc['error']}")
                    if "raw_arguments" in mc:
                        print(f"      Raw arguments: {mc['raw_arguments']!r}")
                    elif "args" in mc:
                        print(f"      Parsed args: {json.dumps(mc['args'])}")
            print(f"  Assistant: {assistant_message[:120]!r}{'...' if len(assistant_message) > 120 else ''}")
            print()

        # Print summary
        print(f"{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Total turns:       {len(_CONVERSATION_TURNS)}")
        print(f"Total tool calls:  {total_tool_calls}")
        print(f"Valid:             {total_valid}")
        print(f"Malformed:         {total_malformed}")
        malformed_rate = (total_malformed / total_tool_calls * 100) if total_tool_calls else 0
        print(f"Malformed rate:    {malformed_rate:.1f}%")
        print(f"{'='*70}\n")

        # Show raw output for any malformed turns
        has_malformed = any(ts["malformed_count"] > 0 for ts in turn_stats)
        if has_malformed:
            print("RAW OUTPUT FOR MALFORMED TURNS:")
            for ts in turn_stats:
                if ts["malformed_count"] > 0:
                    print(f"\nTurn {ts['turn']}: {ts['user_message']!r}")
                    for mc in ts["malformed_calls"]:
                        print(f"  Tool: {mc['name']}")
                        print(f"  Error: {mc['error']}")
                        raw = mc.get("raw_arguments", json.dumps(mc.get("args", {})))
                        print(f"  Raw model output for arguments: {raw!r}")

        # The test passes regardless — this is a reporting/evidence test.
        # The data determines whether qwen2.5:7b is adequate.
        assert total_tool_calls >= 0  # Always passes — remove to make it a strict gate

    @pytest.mark.asyncio
    async def test_stream_tool_call_with_real_ollama(self) -> None:
        """Verify stream_tool_call works end-to-end with real Ollama."""
        provider = OllamaProvider(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
        schema = get_response_tool_schema()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Always respond using the generate_response tool.",
            },
            {"role": "user", "content": "Hello! What's your name?"},
        ]

        chunks = []
        async for chunk in provider.stream_tool_call(
            messages=messages,
            tool_schema=schema,
            temperature=0.3,
        ):
            chunks.append(chunk)

        accumulated = "".join(chunks)
        print(f"\n[stream_tool_call raw output]: {accumulated!r}")

        # Must be parseable JSON
        try:
            parsed = json.loads(accumulated)
        except json.JSONDecodeError as exc:
            pytest.fail(f"stream_tool_call did not yield valid JSON: {exc}\nRaw: {accumulated!r}")

        assert "message" in parsed, f"Parsed JSON missing 'message' key: {parsed}"
        assert isinstance(parsed["message"], str)
        assert len(parsed["message"]) > 0, "message field is empty"

        print(f"[stream_tool_call message]: {parsed['message']!r}")
