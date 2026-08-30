"""
app/domain/llm/prompt_builder.py
──────────────────────────────────
Assembles the complete LLM prompt for each conversation turn.

The prompt builder is profile-agnostic: it receives a BaseProfile, retrieved
chunks, conversation history, and missing fields — and constructs a list of
{role, content} messages. No domain strings are hardcoded here.

TWO-PHASE PROMPT ARCHITECTURE
──────────────────────────────
Phase A — Explicit Tool Calling (``build``):
  The LLM receives a set of named action tools and chooses which ones to call
  based on the user's message.  Zero, one, or multiple tool calls may be made.
  No conversational reply is generated here.

  Tools available: save_text_field, save_enum_field, save_quantitative_field,
                   mark_session_complete, set_next_topic.

Phase B — Response (``build_response_phase``):
  Reuses Phase A messages, then appends one tool-result message per Phase A
  tool call so the LLM knows exactly what was saved vs rejected/failed before
  it generates text.  Tool: ``generate_response``.

Phase A prompt structure:
  [system]   — Profile persona + tool instructions + missing fields summary
  [user/assistant] × N — Trimmed conversation history (last ``window`` messages)
  [user]     — Current user message (always last)

Phase B prompt structure (extends Phase A list):
  ... all Phase A messages ...
  [assistant tool_calls] — The tool calls the LLM made during Phase A (may be
                           zero if no data was extracted — Phase A is skipped)
  [tool] × N            — One write-outcome result per Phase A tool call
"""
from __future__ import annotations

import json

import logging
from typing import Any

from app.domain.conversation.state import MissingField
from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)


class RetrievedChunk:
    """A chunk returned from the vector store with its metadata."""

    def __init__(
        self,
        content: str,
        doc_type: str,
        score: float,
        field_code: str | None = None,
    ) -> None:
        self.content = content
        self.doc_type = doc_type
        self.score = score
        self.field_code = field_code


class PromptBuilder:
    """Builds the full LLM message list for a conversation turn.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY BOUNDARY:
      This class contains zero domain-specific strings. All persona text,
      field descriptions, and conversation guidance come from:
        1. profile.persona_prompt       (system message base)
        2. retrieved chunks             (knowledge retrieval)
        3. missing_fields list          (from state.py completion ledger)
    ─────────────────────────────────────────────────────────────────────
    """

    # Max characters per retrieved chunk in the prompt
    MAX_CHUNK_CHARS: int = 800
    # Max total characters for all retrieved chunks combined
    MAX_TOTAL_CHUNK_CHARS: int = 4000
    # Separator used between sections in the system message
    SECTION_SEP: str = "\n\n---\n\n"

    def build(
        self,
        profile: BaseProfile,
        retrieved_chunks: list[RetrievedChunk],
        conversation_history: list[dict[str, Any]],
        missing_fields: list[MissingField],
        history_window: int = 20,
        is_complete: bool = False,
        brief_summary: str | None = None,
        suggestion_gate_rule: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the full message list for the LLM API call.

        Args:
            profile:              Active project profile (persona, field defs).
            retrieved_chunks:     Top-k chunks from vector similarity search.
            conversation_history: Full conversation history from state.
            missing_fields:       Fields not yet captured (from ledger).
            history_window:       Max number of history messages to include.
            is_complete:          True if all required fields are captured.
            brief_summary:        Deterministic brief summary string.
            suggestion_gate_rule: If not None, a hard prohibition string from
                                  suggestion_gate.format_gate_rule() to inject
                                  into the system message (Cluster C, Bug 2 fix).

        Returns:
            List of {role: str, content: str} dicts ready for the LLM API.
        """
        system_content = self._build_system_message(
            profile, retrieved_chunks, missing_fields,
            is_complete=is_complete, brief_summary=brief_summary,
            suggestion_gate_rule=suggestion_gate_rule,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

        # Add trimmed conversation history (excluding any system messages)
        history = [
            m for m in conversation_history
            if m.get("role") in ("user", "assistant")
        ]
        trimmed = history[-history_window:] if history_window > 0 else history
        messages.extend({"role": m["role"], "content": m["content"]} for m in trimmed)

        logger.debug(
            "Prompt built",
            extra={
                "system_chars": len(system_content),
                "history_turns": len(trimmed),
                "chunks_included": len(retrieved_chunks),
                "missing_field_count": len(missing_fields),
                "is_complete": is_complete,
            },
        )

        return messages

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_system_message(
        self,
        profile: BaseProfile,
        chunks: list[RetrievedChunk],
        missing_fields: list[MissingField],
        is_complete: bool = False,
        brief_summary: str | None = None,
        suggestion_gate_rule: str | None = None,
    ) -> str:
        """Assemble the system message from its three components."""
        parts: list[str] = []

        # 1. Profile persona (tone, role, high-level instructions)
        parts.append(profile.persona_prompt.strip())

        # 2. Retrieved knowledge context (moved to Phase B)
        # We no longer inject retrieved chunks during Phase A to allow parallel execution
        # of RAG and Phase A. Chunks will be injected in build_response_phase.

        # 3. Suggestion gate rule (Bug 2 fix) — injected before the brief status block
        #    so it appears early enough in the system message to take effect.
        if suggestion_gate_rule:
            parts.append(
                "## Creative Suggestion Gate\n\n"
                + suggestion_gate_rule
            )

        # 3. Brief status block — switches mode based on completion state
        status_block = self._format_brief_status(
            missing_fields, is_complete=is_complete, brief_summary=brief_summary
        )

        # Build a simple, direct extraction instruction that smaller models can follow
        extraction_examples = self._format_extraction_examples(missing_fields)
        parts.append(
            "## STEP 1 — EXTRACT AND SAVE FIELDS (do this first, before any reply)\n\n"
            "Read the user's latest message carefully. For every piece of information they provided "
            "that matches a required field below, call the appropriate save tool immediately.\n\n"
            "## Required Fields Still Missing\n\n"
            + status_block
            + "\n\n"
            "## How to Call the Save Tools\n\n"
            "You are equipped with a set of named action tools. "
            "Use them NOW to explicitly save any field values the user just provided. "
            "Call save_text_field, save_enum_field, or save_quantitative_field once per field. "
            "You may call multiple tools in one turn if the user provided multiple fields. "
            "If the user provided NO field data (pure conversation), call no tools — leave this turn tool-free. "
            "Do NOT generate a conversational reply here — that happens in a separate step.\n\n"
            + extraction_examples
            + "\n\nRULES:\n"
            "- Use the EXACT field_code from the list above — never invent new codes.\n"
            "- Call one tool per field. You may call several tools in a single turn.\n"
            "- If the user volunteered a field without being asked, still save it.\n"
            "- If the user is correcting a field they already gave, save it with confidence=1.0.\n"
            "- Do NOT call a tool if you are not sure. Only save what is clearly stated.\n"
            "HONESTY INVARIANTS (enforced — never override):\n"
            "- NEVER use language like 'all set', 'you\'re good to go', 'we\'re done', or 'brief is complete' "
            "unless the system explicitly tells you STATUS: BRIEF COMPLETE.\n"
            "- NEVER claim a file, logo, or asset was received unless it appears as a SAVED field "
            "in the write outcome report. The user uploading a file in chat is not confirmation."
        )

        return self.SECTION_SEP.join(parts)

    def build_response_phase(
        self,
        phase_a_messages: list[dict[str, Any]],
        phase_a_tool_calls: list[dict[str, Any]],
        write_outcomes: list[dict[str, Any]],
        profile: BaseProfile,
        is_complete: bool = False,
        brief_summary: str | None = None,
        retrieved_chunks: list[RetrievedChunk] | None = None,
    ) -> list[dict[str, Any]]:
        """Build the Phase B message list for the response-generation LLM call.

        Extends the Phase A message list with:
          1. A synthetic ``assistant`` message with a ``tool_calls`` list
             containing one entry per Phase A tool call (required by OpenAI API
             to form a valid tool-result thread).
          2. One ``tool`` message per Phase A tool call, each containing the
             write outcome for that specific tool call.

        If Phase A made no tool calls (pure conversation turn), no tool-call
        or tool-result messages are appended — Phase B sees a clean conversation
        thread with an empty outcomes note.

        The Phase B system message is also patched to instruct the LLM to
        use the ``generate_response`` tool and to follow honesty rules.

        Args:
            phase_a_messages:   The exact messages list used for Phase A.
            phase_a_tool_calls: List of tool call dicts from Phase A, each with
                                keys: tool_call_id, tool_name, arguments (dict).
                                Empty list if Phase A made no tool calls.
            write_outcomes:     Structured list of per-field write results.
                                Each item: {field_code, value, status, reason,
                                tool_name, tool_call_id}.
            profile:            Active project profile.
            is_complete:        Whether the brief is now complete.
            brief_summary:      Optional deterministic brief summary string.

        Returns:
            Message list ready for the Phase B LLM call.
        """
        # Start from a copy of Phase A messages
        messages: list[dict[str, Any]] = list(phase_a_messages)

        # Patch the system message for Phase B instructions
        if messages and messages[0]["role"] == "system":
            original_system = messages[0]["content"]
            patched = original_system.replace(
                "You are equipped with a set of named action tools. "
                "Use them NOW to explicitly save any field values the user just provided. "
                "Call save_text_field, save_enum_field, or save_quantitative_field once per field. "
                "You may call multiple tools in one turn if the user provided multiple fields. "
                "If the user provided NO field data (pure conversation), call no tools — leave this turn tool-free. "
                "Do NOT generate a conversational reply here — that happens in a separate step.",
                "You MUST call `generate_response` with every response. "
                "Generate your conversational reply in the `message` field. "
                "You have already saved fields in a prior step; do NOT re-save here.\n\n"
                "## RESPONSE STYLE RULES (MANDATORY)\n"
                "- Keep replies SHORT and conversational — 1 to 3 sentences max.\n"
                "- Do NOT echo, praise, or over-validate what the user said. A brief 'Got it' or 'Perfect' is enough.\n"
                "- You may ask UP TO 2–3 closely related questions in a single turn if they belong to the same topic cluster "
                "(e.g., asking for dish name + price together, or timeline + go-live date together). "
                "But do NOT dump all remaining questions at once — prioritise the most important ones first.\n"
                "- NEVER repeat a question that already appears answered in the conversation history.\n"
                "- Write like a human creative strategist, not a customer-service bot.",
            )
            
            if retrieved_chunks and not is_complete:
                context_block = self._format_retrieved_chunks(retrieved_chunks)
                if context_block:
                    patched += (
                        "\n\n## Retrieved Knowledge\n\n"
                        "The following context has been retrieved from the knowledge base "
                        "to help guide this conversation turn. Use it to inform your "
                        "questions and extraction, but don't quote it verbatim.\n"
                        "CRITICAL: You may use a warm, empathetic tone, but DO NOT invent new unlisted examples, options, pricing, or specific guarantees that are not in the retrieved context (e.g., do not suggest 'brand identity revamp or social media campaign' unless those exact phrases exist in the chunks).\n"
                        "CRITICAL: Always acknowledge the user's message directly before moving to a new topic. If retrieved context is irrelevant to the user's input, ignore it.\n\n"
                        + context_block
                    )
            
            messages[0] = {"role": "system", "content": patched}

        if phase_a_tool_calls:
            # Build a synthetic assistant message with all Phase A tool calls.
            # OpenAI requires the tool_calls list to precede any tool results.
            assistant_tool_calls = [
                {
                    "id": tc["tool_call_id"],
                    "type": "function",
                    "function": {
                        "name": tc["tool_name"],
                        "arguments": json.dumps(tc.get("arguments", {})),
                    },
                }
                for tc in phase_a_tool_calls
            ]
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })

            # Build a per-tool-call outcome index keyed by tool_call_id
            outcome_by_id: dict[str, dict[str, Any]] = {
                o["tool_call_id"]: o
                for o in write_outcomes
                if o.get("tool_call_id")
            }

            # Append one tool-result message per Phase A tool call
            for tc in phase_a_tool_calls:
                tc_id = tc["tool_call_id"]
                outcome = outcome_by_id.get(tc_id)
                if outcome:
                    result_text = self._format_single_outcome(outcome)
                else:
                    result_text = "No write outcome recorded for this tool call."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_text,
                })

        # Always append a summary outcomes message (even if Phase A was skipped)
        # so Phase B has the full picture in one place.
        outcomes_summary = self._format_write_outcomes(write_outcomes)
        # Only inject the summary as a system-role note to avoid OpenAI
        # API errors (tool results must immediately follow a tool_calls message)
        if not phase_a_tool_calls:
            # No tool calls were made — inject as a user-visible note
            messages.append({
                "role": "user",
                "content": (
                    "[SYSTEM NOTE — not from the real user]: "
                    + outcomes_summary
                ),
            })

        logger.debug(
            "Phase B prompt built",
            extra={
                "total_messages": len(messages),
                "phase_a_tool_call_count": len(phase_a_tool_calls),
                "write_outcomes_count": len(write_outcomes),
                "is_complete": is_complete,
            },
        )

        return messages

    @staticmethod
    def _format_single_outcome(outcome: dict[str, Any]) -> str:
        """Format a single tool call write outcome as the tool-result content.

        This is injected as the ``content`` of a ``tool`` message, one per
        Phase A tool call. The LLM sees this in its message history before
        generating the Phase B response.

        Args:
            outcome: Single write outcome dict with keys: field_code, value,
                     status, reason, tool_name.

        Returns:
            Human-readable status string for the tool-result message.
        """
        status = outcome.get("status", "unknown")
        field_code = outcome.get("field_code", "?")
        value = outcome.get("value", "?")
        reason = outcome.get("reason", "")

        if status == "saved":
            return f"✅ SAVED: field_code={field_code!r}, value={value!r}"
        else:
            return (
                f"❌ NOT SAVED: field_code={field_code!r}, value={value!r}, "
                f"status={status!r}, reason={reason!r}"
            )

    @staticmethod
    def _format_write_outcomes(write_outcomes: list[dict[str, Any]]) -> str:
        """Format all write outcomes as a consolidated summary for Phase B context.

        This string is injected as a summary note (into a user-role message when
        Phase A made no tool calls, or appended as the last tool-result block).
        It is both human readable (for LLM honesty rules) and machine readable
        (raw JSON appended for future tooling).

        Args:
            write_outcomes: List of per-field outcome dicts from Phase A.

        Returns:
            Formatted string for injection into the Phase B context.
        """
        if not write_outcomes:
            return (
                "WRITE OUTCOMES: No tool calls were made in this turn. "
                "No saves were attempted. Respond naturally to the user's message "
                "without confirming any field saves."
            )

        saved = [o for o in write_outcomes if o["status"] == "saved"]
        not_saved = [o for o in write_outcomes if o["status"] != "saved"]

        lines = ["WRITE OUTCOMES (ground truth — you MUST base your reply on these results):\n"]

        if saved:
            lines.append("✅ SAVED (confirm these to the user):")
            for o in saved:
                lines.append(f"  - field_code={o.get('field_code')!r} → value={o.get('value')!r}")

        if not_saved:
            lines.append("\n❌ NOT SAVED (handle these honestly per the rules below):")
            for o in not_saved:
                reason = f" ({o['reason']})" if o.get("reason") else ""
                lines.append(f"  - field_code={o.get('field_code')!r}: status={o['status']}{reason}")

        lines.append(
            "\nHONESTY RULES:\n"
            "- Only say 'saved', 'updated', 'got it', etc. for ✅ SAVED fields.\n"
            "- For 'rejected_low_confidence': say you weren't sure what they meant and ask to confirm.\n"
            "- For 'rejected_enum': say the value wasn't valid and offer the valid options.\n"
            "- For 'rejected_qualitative': say you need a specific number or measurable target.\n"
            "- For 'rejected_lower_confidence': note that the existing capture was kept (higher confidence).\n"
            "- For 'rejected_unknown_field': do not mention — field mismatch is a system issue.\n"
            "- For 'rejected_malformed': say you couldn't understand the value and ask to clarify.\n"
            "- For 'write_failed': tell the user the save didn't go through and you'll try again.\n"
            # Bug 1 fix: hard prohibition on file/logo claims not backed by a write outcome
            "- NEVER say 'I can see your logo', 'I received your file', or 'your asset was uploaded' "
            "unless that file's field_code appears with status='saved' in the ✅ SAVED list above.\n"
            # Bug 4 fix: prohibit premature completion language
            "- NEVER use 'all set', 'you\'re good to go', 'we\'re done', or any completion language "
            "unless the system explicitly states STATUS: BRIEF COMPLETE in the context above."
        )

        # Also append raw JSON for precise machine parsing by future tooling
        lines.append("\nRAW JSON: " + json.dumps(write_outcomes))

        return "\n".join(lines)

    def _format_extraction_examples(self, missing_fields: list[MissingField]) -> str:
        """Generate concrete few-shot examples for the missing fields.

        Shows the model exactly which tool to call and what arguments to use,
        making it easier for smaller LLMs to reliably call tools correctly.

        Args:
            missing_fields: Fields not yet captured (from ledger).

        Returns:
            Formatted string with extraction examples, or empty string if no fields.
        """
        if not missing_fields:
            return ""

        lines = ["## Examples — How to Save Fields\n"]

        # Categorise fields into text vs enum vs quantitative for examples
        enum_fields = [mf for mf in missing_fields if getattr(mf, "enum_values", None)]
        quant_fields = [mf for mf in missing_fields if mf.field_code in (
            "success_metrics", "budget_range", "key_messages",
            "industry_vertical", "competitors_or_references", "existing_assets",
        )]
        text_fields = [mf for mf in missing_fields
                       if mf not in enum_fields and mf not in quant_fields]

        example_count = 0

        # Show up to 2 text-field examples
        for mf in text_fields[:2]:
            lines.append(
                f'  If the user gives you their {mf.field_code.replace("_", " ")}, call:\n'
                f'    save_text_field(field_code="{mf.field_code}", value="<what they said>", confidence=0.9)'
            )
            example_count += 1

        # Show 1 enum example
        for mf in enum_fields[:1]:
            opts = getattr(mf, "enum_values", [])
            example_val = opts[0] if opts else "value_from_list"
            lines.append(
                f'  If the user names their {mf.field_code.replace("_", " ")}, call:\n'
                f'    save_enum_field(field_code="{mf.field_code}", value="{example_val}", confidence=0.9)\n'
                f'    (value MUST be one of: {", ".join(str(v) for v in opts[:5])}{"..." if len(opts) > 5 else ""})'
            )
            example_count += 1

        # Show 1 quantitative example
        for mf in quant_fields[:1]:
            lines.append(
                f'  If the user gives a number or KPI for {mf.field_code.replace("_", " ")}, call:\n'
                f'    save_quantitative_field(field_code="{mf.field_code}", value="<the number/target>", confidence=0.9)'
            )
            example_count += 1

        if example_count == 0:
            return ""

        return "\n".join(lines)

    def _format_retrieved_chunks(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a labeled context block."""
        if not chunks:
            return ""

        lines: list[str] = []
        total_chars = 0

        for i, chunk in enumerate(chunks, start=1):
            truncated = chunk.content[: self.MAX_CHUNK_CHARS]
            if len(chunk.content) > self.MAX_CHUNK_CHARS:
                truncated += "…"

            chunk_text = (
                f"[Chunk {i} | type={chunk.doc_type}"
                + (f" | field={chunk.field_code}" if chunk.field_code else "")
                + f" | score={chunk.score:.2f}]\n{truncated}"
            )

            if total_chars + len(chunk_text) > self.MAX_TOTAL_CHUNK_CHARS:
                logger.debug("Truncating retrieved chunks at char budget", extra={"stopped_at": i})
                break

            lines.append(chunk_text)
            total_chars += len(chunk_text)

        return "\n\n".join(lines)

    def _format_brief_status(
        self,
        missing_fields: list[MissingField],
        is_complete: bool = False,
        brief_summary: str | None = None,
    ) -> str:
        """Format the brief status block for the LLM.

        POST-COMPLETION MODE (is_complete=True):
          The LLM must NOT default to re-summarising the brief.
          It must read the user's actual message and respond directly to it.
          The captured brief is provided as a reference so the LLM can quote
          it verbatim if and only if the user explicitly asks to see it.

        IN-PROGRESS MODE (is_complete=False):
          Lists remaining required fields so the LLM knows what to collect.
        """
        if is_complete:
            summary_ref = ""
            if brief_summary:
                # Provide the brief as a reference — LLM must not paraphrase it
                summary_ref = (
                    "\n\n**Captured Brief (for reference only — do not repeat unless asked):**\n\n"
                    + brief_summary
                )

            return (
                "**STATUS: BRIEF COMPLETE — All required fields have been captured.**\n\n"
                "You are now in post-completion Q&A mode. Rules for this mode:\n"
                "1. READ the user's message carefully. Respond DIRECTLY and SPECIFICALLY to what they asked.\n"
                "2. Do NOT recite or summarise the full brief again unless the user explicitly asks "
                "   (e.g. 'show me the brief', 'what did we capture', 'summarise everything').\n"
                "3. If the user asks to change or update a field, acknowledge it warmly and note the change.\n"
                "4. If the user asks a question about the project, answer it from the captured context.\n"
                "5. If the user says they are done, confirm readiness to proceed.\n"
                "6. NEVER give the same response twice for different questions — your response must "
                "   be specifically tailored to what was just asked."
                + summary_ref
            )

        # In-progress: list remaining fields
        if not missing_fields:
            return (
                "**All required fields have been captured.** "
                "Consider confirming the brief summary with the client."
            )

        lines = [
            f"**{len(missing_fields)} required field(s) not yet captured:**\n"
        ]
        for mf in missing_fields:
            # Use description (from profile) — no hardcoded wording
            lines.append(f"- `{mf.field_code}`: {mf.description.strip()}")

        return "\n".join(lines)
