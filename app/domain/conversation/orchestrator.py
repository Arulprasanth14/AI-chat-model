"""
app/domain/conversation/orchestrator.py
─────────────────────────────────────────
Conversation turn orchestrator.

This module is the central engine for processing a single conversation turn.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ARCHITECTURE INVARIANT — READ BEFORE EDITING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  This file MUST remain free of:
    ✗ Hardcoded question strings or scripts
    ✗ Domain-specific field names or brief type names
    ✗ Branching logic based on field names or conversation state values
    ✗ Any string that references the Picasso Fusion domain specifically

  The orchestrator operates on abstract types only:
    ✓ BaseProfile  (loaded from YAML, injected by deps.py)
    ✓ ConversationState  (from state.py — profile-agnostic)
    ✓ MissingField  (computed by the state ledger)
    ✓ LLMProvider, RAGRetriever, SessionRepository  (all injected interfaces)

  To add a new project integration, create a new profile folder.
  Zero changes to this file are required.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TWO-PHASE TURN ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per turn with extraction attempts, the orchestrator runs two LLM calls:

  Phase A — Extraction (silent, not streamed):
    1. Call LLM with ``extract_answers`` tool schema.
    2. Validate confidence thresholds + apply to state ledger.
    3. Attempt ``save_session``. Capture the outcome (saved / rejected / failed).
    4. Log every outcome at INFO level for audit.

  Phase B — Response (streamed to client):
    1. Build Phase B prompt: Phase A messages + tool-result (write outcomes).
    2. Call LLM with ``generate_response`` tool schema.
    3. Stream ``message`` tokens to client via SSE.
    4. The LLM can only confirm what appears as "saved" in the outcome context.

For turns where Phase A extracts nothing (empty extracted_answers), Phase A
is skipped and only Phase B runs — preserving single-call latency for Q&A
turns that contain no new field data.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.domain.conversation.brief_renderer import render_brief
from app.domain.conversation.state import (
    ConversationState,
    ExtractedAnswer,
    FieldWriteResult,
    WRITE_STATUS_SAVED,
    WRITE_STATUS_REJECTED_LOW_CONFIDENCE,
    WRITE_STATUS_REJECTED_ENUM,
    WRITE_STATUS_REJECTED_QUALITATIVE,
    WRITE_STATUS_REJECTED_UNKNOWN_FIELD,
    WRITE_STATUS_REJECTED_LOWER_CONFIDENCE,
    WRITE_STATUS_REJECTED_MALFORMED,
)
from app.domain.conversation.structural_resolver import StructuralResolver
from app.domain.conversation.suggestion_gate import can_generate_suggestions, format_gate_rule
from app.domain.conversation.color_suggestion import parse_color_suggestion
from app.domain.llm.prompt_builder import PromptBuilder
from app.domain.llm.provider import LLMProvider
from app.domain.llm.tool_schema import (
    get_phase_a_tools,
    parse_phase_a_tool_calls,
    get_response_tool_schema,
    parse_response_result,
    # Legacy — kept so external test harnesses that import these names still work
    get_extract_tool_schema,
    parse_extract_result,
)
from app.domain.rag.retriever import RAGRetriever, RetrievalQuery
from app.infrastructure.persistence.session_repository import SessionRepository
from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)


# Write outcome status constants are now imported from state.py.
# Kept as module-level aliases here so any code that references them via
# `orchestrator._STATUS_*` continues to work without changes.
_STATUS_SAVED = WRITE_STATUS_SAVED
_STATUS_REJECTED_LOW_CONFIDENCE = WRITE_STATUS_REJECTED_LOW_CONFIDENCE
_STATUS_REJECTED_ENUM = WRITE_STATUS_REJECTED_ENUM
_STATUS_REJECTED_UNASKED = "rejected_unasked"  # legacy alias
_STATUS_WRITE_FAILED = "write_failed"


class ConversationOrchestrator:
    """Processes a single conversation turn end-to-end.

    Injected dependencies (all behind interfaces — no concrete types here):
        session_repo:    Persists and loads conversation state.
        retriever:       Retrieves relevant knowledge chunks.
        llm:             Calls the language model with forced tool schema.
        prompt_builder:  Assembles the full LLM message list.
        profile:         Active project profile (loaded from YAML).
        field_sets_root: Path to field_sets directory, used by the brief renderer
                         to locate the resolved YAML template for summary output.

    Args:
        session_repo:      SessionRepository implementation.
        retriever:         RAGRetriever instance.
        llm:               LLMProvider implementation.
        prompt_builder:    PromptBuilder instance.
        profile_provider:  Callable that provides the active BaseProfile for a given state.
        structural_resolver: Optional resolver for deterministic section guidance.
        field_sets_root:   Root path for field-set YAMLs (for the brief renderer).
    """
    def __init__(
        self,
        session_repo: SessionRepository,
        retriever: RAGRetriever,
        llm: LLMProvider,
        prompt_builder: PromptBuilder,
        profile_provider: Callable[[ConversationState], BaseProfile],
        structural_resolver: StructuralResolver | None = None,
        field_sets_root: Path | None = None,
    ) -> None:
        self._repo = session_repo
        self._retriever = retriever
        self._llm = llm
        self._prompt_builder = prompt_builder
        self._profile_provider = profile_provider
        self._structural_resolver = structural_resolver
        self._field_sets_root = field_sets_root

    async def process_turn(
        self,
        session_id: str | None,
        user_message: str,
        vertical: str | None = None,
        template_key: str | None = None,
    ) -> AsyncIterator[str]:
        import time
        t_start = time.perf_counter()
        timings = {}
        
        # ── Step 1: Load or create session state ────────────────────────────

        """Process one user message and stream the response.

        Turn flow (all steps operate on abstract profile/state types):
          1. Load or create session state.
          2. Add user message to history.
          3. Compute missing fields from the completion ledger.
          4. Build retrieval query from message + history + missing fields.
          5. Retrieve top-k knowledge chunks.
          6. Build Phase A prompt (persona + chunks + history + missing fields).
          7. [Phase A] Call LLM silently — extract field values only.
          8. Validate + apply extracted answers; attempt save_session.
          9. Build per-field write_outcomes list; log all outcomes.
          10. [Phase B] Build response prompt (Phase A messages + write outcomes).
          11. Stream LLM response — yield message tokens to caller.
          12. Update state with assistant message + completion status.
          13. Persist final state.
          14. Yield final SSE event with session snapshot.

        Phase A is skipped when extracted_answers is empty (no write attempted).
        This preserves single-call latency for pure Q&A turns.

        Args:
            session_id:   Existing session ID or None to start a new session.
            user_message: The user's latest message text.

        Yields:
            SSE-formatted strings. Regular chunks contain the streaming
            assistant message. The final chunk is a JSON snapshot event.
        """
        # ── Step 1: Load or create session state ────────────────────────────
        base_profile = self._profile_provider(ConversationState(profile_id="placeholder"))

        if session_id:
            state = await self._repo.get_session(session_id)
            if state is None:
                logger.warning("Session not found, creating new", extra={"session_id": session_id})
                state = await self._repo.create_session(base_profile.profile_id)
        else:
            state = await self._repo.create_session(base_profile.profile_id)
            if vertical:
                state.resolved_vertical = vertical
                logger.info(
                    "Pre-selected vertical applied",
                    extra={"vertical": vertical, "session_id": state.session_id},
                )
            if template_key:
                state.resolved_template_key = template_key
                logger.info(
                    "Pre-selected template_key applied",
                    extra={"template_key": template_key, "session_id": state.session_id},
                )

        logger.info(
            "Turn started",
            extra={
                "session_id": state.session_id,
                "turn": len(state.conversation_history) // 2,
                "base_profile_id": base_profile.profile_id,
            },
        )

        # ── Step 2: Add user turn to history ────────────────────────────────
        # __start__ is a hidden UI trigger to generate the opening greeting.
        # We do NOT add it to conversation history so the LLM sees an empty
        # history and produces a natural first-turn opener for the selected vertical.
        is_start_trigger = user_message.strip() == "__start__"

        # Bug 4+5 fix: __field_saved__:field_code:value is a hidden trigger from chip clicks.
        # The field was already written via direct_field_write at confidence=1.0.
        # We skip Phase A extraction (nothing to extract) and let Phase B acknowledge the save.
        is_field_saved_trigger = user_message.strip().startswith("__field_saved__:")
        field_saved_info: str | None = None
        if is_field_saved_trigger:
            # Extract "field_code:value" for Phase B context injection
            parts = user_message.strip().split(":", 2)
            field_saved_info = f"{parts[1]}={parts[2]}" if len(parts) >= 3 else None

        if not is_start_trigger and not is_field_saved_trigger:
            state.add_turn("user", user_message)

        # Obtain the effective profile for this turn (applies field sets if resolved)
        active_profile = self._profile_provider(state)

        # ── Step 3: Compute missing fields ──────────────────────────────────
        missing_fields = state.compute_missing_fields(
            active_profile,
            settings.extraction_confidence_threshold,
        )

        is_complete = state.is_complete(active_profile, settings.extraction_confidence_threshold)

        # Build brief summary when complete (ready for the post-completion block)
        brief_summary: str | None = None
        if is_complete and self._field_sets_root:
            field_set_yaml_path: Path | None = None
            if state.resolved_vertical and state.resolved_template_key:
                field_set_yaml_path = (
                    self._field_sets_root
                    / state.resolved_vertical
                    / f"{state.resolved_template_key}.yaml"
                )
            try:
                brief_summary = render_brief(
                    captured=state.captured,
                    field_set_yaml_path=field_set_yaml_path,
                    profile=active_profile,
                    include_confidence=False,
                )
            except Exception as exc:
                logger.warning(
                    "Brief renderer failed, proceeding without summary",
                    exc_info=exc,
                )

        # ── Step 6: Build Phase A prompt (Without RAG chunks) ───────────────
        t_prompt_start = time.perf_counter()
        # Cluster C: compute suggestion gate before building the prompt so the
        # gate rule can be injected as a hard constraint in the system message.
        can_suggest, gate_missing = can_generate_suggestions(
            state, active_profile, settings.extraction_confidence_threshold
        )
        suggestion_gate_rule: str | None = None if can_suggest else format_gate_rule(gate_missing)

        # Bug 2 fix: Identify which field the AI most recently asked about so we can
        # inject a "CURRENT CONTEXT" annotation into Phase A. This tells the LLM exactly
        # which field the user is answering, eliminating guessing for short/Tanglish replies.
        current_field_hint: str | None = None
        if not is_start_trigger and not is_complete and missing_fields:
            current_field_hint = self._build_current_context_hint(
                state=state,
                missing_fields=missing_fields,
            )

        phase_a_messages = self._prompt_builder.build(
            profile=active_profile,
            retrieved_chunks=[],  # Omitted for Phase A to allow concurrency
            conversation_history=state.get_recent_history(settings.history_window),
            missing_fields=missing_fields,
            history_window=settings.history_window,
            is_complete=is_complete,
            brief_summary=brief_summary,
            suggestion_gate_rule=suggestion_gate_rule,
            current_field_hint=current_field_hint,
        )

        # For __start__ trigger: inject a directive so the LLM generates a
        # context-aware opening greeting instead of asking about a user message.
        if is_start_trigger:
            vertical_label = state.resolved_vertical or "creative"
            template_label = (state.resolved_template_key or "project").replace("_", " ")
            start_directive = (
                f"\n\n## OPENING TURN DIRECTIVE\n"
                f"This is the very first turn of a new brief session. The client has selected:\n"
                f"- **Vertical:** {vertical_label}\n"
                f"- **Content type:** {template_label}\n\n"
                f"Generate a warm, SHORT (1-2 sentences) opening greeting that:\n"
                f"1. Acknowledges the specific vertical and content type they selected.\n"
                f"2. Asks for the client/brand name as the very first question.\n"
                f"Do NOT ask multiple questions. Do NOT say 'I'm excited' or similar filler phrases.\n"
                f"Write like a confident creative consultant meeting a client for the first time."
            )
            if phase_a_messages and phase_a_messages[0]["role"] == "system":
                phase_a_messages[0]["content"] += start_directive
        timings["prompt_assembly"] = time.perf_counter() - t_prompt_start

        # ── Concurrent Execution: RAG Retrieval + Phase A ───────────────────
        t_parallel_start = time.perf_counter()

        async def fetch_retrieval() -> list[Any]:
            if is_complete:
                return []
            
            structural_guidance = (
                self._structural_resolver.resolve(state, missing_fields)
                if self._structural_resolver
                else None
            )

            retrieval_query = RetrievalQuery(
                user_message=user_message,
                recent_history=state.get_recent_history(window=6),
                missing_fields=missing_fields,
                profile_id=active_profile.knowledge_namespace,
                industry=state.resolved_vertical or None,
                brief_type=state.resolved_template_key or None,
            )
            vector_chunks = await self._retriever.retrieve(retrieval_query)

            chunks = []
            seen = set()
            if structural_guidance:
                chunks.append(structural_guidance)
                seen.add(structural_guidance.content)
            for chunk in vector_chunks:
                if chunk.content not in seen:
                    chunks.append(chunk)
                    seen.add(chunk.content)
            return chunks

        import asyncio
        retrieval_task = asyncio.create_task(fetch_retrieval())
        
        # Skip Phase A entirely for the __start__ trigger (no user message to extract from)
        # Also skip for __field_saved__ trigger (field already saved via direct_field_write — nothing to extract)
        if is_start_trigger or is_field_saved_trigger:
            async def _dummy_phase_a():
                return [], [], {"model_believes_complete": False, "suggested_next_topic": None}
            phase_a_task = asyncio.create_task(_dummy_phase_a())
        else:
            phase_a_task = asyncio.create_task(self._run_phase_a(
                state=state,
                active_profile=active_profile,
                phase_a_messages=phase_a_messages,
            ))

        try:
            retrieved_chunks, phase_a_result = await asyncio.gather(retrieval_task, phase_a_task)
        except Exception as exc:
            logger.error(
                "Phase A or RAG retrieval failed — aborting turn to prevent data loss",
                extra={"session_id": state.session_id},
                exc_info=exc,
            )
            # Cancel the sibling task to avoid leaking it
            retrieval_task.cancel()
            phase_a_task.cancel()
            yield self._sse_chunk(
                "I'm experiencing high load right now and couldn't save your message. "
                "Please send it again in a moment!"
            )
            yield self._sse_done(state.to_snapshot(active_profile, settings.extraction_confidence_threshold))
            return

        write_outcomes, phase_a_dispatched_tool_calls, phase_a_advisory = phase_a_result
        
        timings["rag_and_phase_a_parallel"] = time.perf_counter() - t_parallel_start

        # Propagate advisory flags from Phase A
        state.model_believes_complete = phase_a_advisory.get("model_believes_complete", False)
        state.suggested_next_topic = phase_a_advisory.get("suggested_next_topic") or None

        # Recompute completion after Phase A write
        is_complete = state.is_complete(active_profile, settings.extraction_confidence_threshold)
        if is_complete:
            state.status = "complete"

        # ── Step 10: Build Phase B prompt ───────────────────────────────────
        # Recompute missing_fields AFTER Phase A so Phase B has an accurate
        # view of what still needs to be collected (prevents re-asking saved fields).
        missing_fields_post_a = state.compute_missing_fields(
            active_profile, settings.extraction_confidence_threshold
        )

        phase_b_messages = self._prompt_builder.build_response_phase(
            phase_a_messages=phase_a_messages,
            phase_a_tool_calls=phase_a_dispatched_tool_calls,
            write_outcomes=write_outcomes,
            profile=active_profile,
            is_complete=is_complete,
            brief_summary=brief_summary,
            retrieved_chunks=retrieved_chunks,
            missing_fields=missing_fields_post_a,
            field_saved_note=field_saved_info,  # Bug 5 fix: inject chip-save context for Phase B
        )

        # ── Step 11: Phase B — Stream response to client ────────────────────
        t_phase_b_start = time.perf_counter()
        response_schema = get_response_tool_schema()
        accumulated_json = ""
        streamed_message = ""

        async for token in self._llm.stream_tool_call(
            messages=phase_b_messages,
            tool_schema=response_schema,
            temperature=active_profile.llm_temperature,
        ):
            if "ttft" not in timings:
                timings["ttft"] = time.perf_counter() - t_phase_b_start
            accumulated_json += token
            streamed_chunk = self._extract_message_delta(accumulated_json, len(streamed_message))
            if streamed_chunk:
                streamed_message += streamed_chunk
                yield self._sse_chunk(streamed_chunk)

        # Parse complete Phase B result
        try:
            phase_b_result = parse_response_result(accumulated_json)
        except ValueError as exc:
            logger.error("Failed to parse Phase B response result", exc_info=exc)
            phase_b_result: dict[str, Any] = {
                "message": "Sorry, I encountered an error generating a response. Please try again.",
                "suggested_next_topic": "",
                "model_believes_complete": False,
            }

        final_message = phase_b_result.get("message", "")
        remaining = final_message[len(streamed_message):]
        if remaining:
            yield self._sse_chunk(remaining)
            
        timings["llm_phase_b_total"] = time.perf_counter() - t_phase_b_start

        # Update advisory flags from Phase B
        state.model_believes_complete = bool(
            phase_b_result.get("model_believes_complete", state.model_believes_complete)
        )
        state.suggested_next_topic = (
            phase_b_result.get("suggested_next_topic") or state.suggested_next_topic
        )
        state.last_intent = phase_b_result.get("intent")

        # ── Step 12: Add assistant turn to history ──────────────────────────
        state.add_turn("assistant", final_message)

        # ── Step 13: Persist final state (Background Write) ─────────────────
        # Phase A already persisted the extraction changes. We persist again
        # here to capture the updated conversation history (assistant message).
        # Optimization: We fire-and-forget this write to return the final 
        # SSE chunk to the user immediately, eliminating the DB wait time.
        t_write_start = time.perf_counter()

        async def _bg_save(state_to_save: ConversationState) -> None:
            try:
                await self._repo.save_session(state_to_save)
            except Exception as exc:
                logger.error(
                    "Final state persist failed (conversation history not saved)",
                    extra={"session_id": state_to_save.session_id},
                    exc_info=exc,
                )

        import asyncio
        bg_task = asyncio.create_task(_bg_save(state))
        
        # Strong reference to prevent GC during fire-and-forget
        if not hasattr(self, "_bg_tasks"):
            self._bg_tasks = set()
        self._bg_tasks.add(bg_task)
        bg_task.add_done_callback(self._bg_tasks.discard)

        timings["backend_write_b"] = time.perf_counter() - t_write_start
        timings["total_turn"] = time.perf_counter() - t_start

        with open("perf_logs_after.txt", "a") as f:
            f.write(
                f"PERF_AFTER|{state.session_id}|"
                f"PARALLEL_RAG_A:{timings.get('rag_and_phase_a_parallel', 0):.3f}s|"
                f"PROMPT:{timings.get('prompt_assembly', 0):.3f}s|"
                f"PHASE_B_TTFT:{timings.get('ttft', 0):.3f}s|"
                f"PHASE_B_TOT:{timings.get('llm_phase_b_total', 0):.3f}s|"
                f"WRITE_B:{timings.get('backend_write_b', 0):.3f}s|"
                f"TOTAL:{timings.get('total_turn', 0):.3f}s\n"
            )

        logger.info(
            "Turn complete",
            extra={
                "session_id": state.session_id,
                "phase_a_write_outcomes": write_outcomes,
                "missing_remaining": len(
                    state.compute_missing_fields(active_profile, settings.extraction_confidence_threshold)
                ),
                "model_believes_complete": state.model_believes_complete,
                "intent": phase_b_result.get("intent"),
            },
        )

        # ── Step 14: Yield final snapshot event ─────────────────────────────
        snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
        yield self._sse_done(snapshot)

    # ── Phase A implementation ─────────────────────────────────────────────────

    async def _run_phase_a(
        self,
        state: ConversationState,
        active_profile: BaseProfile,
        phase_a_messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Run Phase A: explicit tool calling + backend writes.

        Calls the LLM with the full set of Phase A action tools (tool_choice="auto").
        The LLM may call zero, one, or multiple tools in a single response.

        - If no tool calls are made: Phase A is skipped entirely (fast path).
        - Each tool call is dispatched to the corresponding handler on state.
        - All write results are collected before Phase B runs.
        - Advisory tools (mark_session_complete, set_next_topic) update state
          flags but are not reported in write_outcomes.
        - Malformed tool calls (missing required parameters) are caught and
          reported as rejected_malformed outcomes without crashing.

        Args:
            state:            Current conversation state (mutated in-place).
            active_profile:   Active project profile.
            phase_a_messages: Prompt messages for the Phase A call.

        Returns:
            Tuple of (write_outcomes, dispatched_tool_calls, advisory_flags).
            write_outcomes:         List of per-field outcome dicts.
            dispatched_tool_calls:  List of {tool_call_id, tool_name, arguments}
                                    dicts for the Phase B message thread.
            advisory_flags:         Dict with model_believes_complete and
                                    suggested_next_topic keys.
        """
        tools = get_phase_a_tools(active_profile)
        advisory_flags: dict[str, Any] = {
            "model_believes_complete": False,
            "suggested_next_topic": None,
        }

        # Call the LLM with auto tool_choice (zero, one, or many tool calls)
        # Bug 6 fix: Use temperature=0.0 for Phase A (extraction) to make tool calling
        # fully deterministic. High temperature was causing probabilistic non-calls.
        # Phase B still uses the profile's configured temperature for natural conversation.
        try:
            raw_tool_calls = await self._llm.call_with_tools(
                messages=phase_a_messages,
                tools=tools,
                temperature=0.0,  # Bug 6 fix: deterministic extraction
            )
        except Exception as exc:
            logger.error("Phase A LLM request failed", extra={"session_id": state.session_id}, exc_info=exc)
            failure_outcome = FieldWriteResult(
                field_code=None,
                value=None,
                status=WRITE_STATUS_REJECTED_MALFORMED,
                reason=f"Phase A LLM request failed: {exc}",
                tool_name="phase_a_execution",
                tool_call_id="error",
            ).to_outcome_dict()
            return [failure_outcome], [], advisory_flags

        # Fast path: no tool calls — pure conversation, skip Phase A
        if not raw_tool_calls:
            logger.debug(
                "Phase A: LLM made no tool calls — skipping write",
                extra={"session_id": state.session_id},
            )
            return [], [], advisory_flags

        logger.info(
            "Phase A: LLM made tool calls",
            extra={
                "session_id": state.session_id,
                "tool_calls": [tc["name"] for tc in raw_tool_calls],
            },
        )

        # Parse all tool calls
        parsed_calls = parse_phase_a_tool_calls(raw_tool_calls)

        write_outcomes: list[dict[str, Any]] = []
        dispatched_tool_calls: list[dict[str, Any]] = []  # for Phase B thread

        for call in parsed_calls:
            tool_name = call["tool_name"]
            tool_call_id = call["tool_call_id"]
            arguments = call["arguments"]
            parse_error = call["parse_error"]

            # Track for Phase B message thread
            dispatched_tool_calls.append({
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            })

            # ── Handle parse errors ────────────────────────────────────────
            if parse_error:
                write_outcomes.append(FieldWriteResult(
                    field_code=None,
                    value=None,
                    status=WRITE_STATUS_REJECTED_MALFORMED,
                    reason=f"Tool call JSON parse error: {parse_error}",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                ).to_outcome_dict())
                logger.warning(
                    "Phase A: malformed tool call arguments",
                    extra={"session_id": state.session_id, "tool_name": tool_name, "error": parse_error},
                )
                continue

            # ── Advisory tools (no write outcome) ─────────────────────────
            if tool_name == "mark_session_complete":
                advisory_flags["model_believes_complete"] = True
                logger.debug("Phase A: model marked session complete", extra={"session_id": state.session_id})
                continue

            if tool_name == "set_next_topic":
                topic = arguments.get("topic", "")
                advisory_flags["suggested_next_topic"] = topic
                logger.debug("Phase A: model set next topic", extra={"session_id": state.session_id, "topic": topic})
                continue

            # ── Color suggestion tool (Bug 5 fix) ─────────────────────────
            if tool_name == "save_color_suggestion":
                try:
                    color = parse_color_suggestion(arguments)
                    # Store as formatted string in hybrid_notes (or custom_colors_and_fonts fallback)
                    store_field = "hybrid_notes" if active_profile.get_field_by_code("hybrid_notes") else "existing_assets"
                    # Use a neutral free-text field to accumulate color suggestions
                    # The stored value is appended to any existing value
                    existing_colors = state.captured.get(store_field)
                    color_str = color.to_stored_string()
                    new_value = (
                        f"{existing_colors.value}; {color_str}" if existing_colors else color_str
                    )
                    color_result = state.handle_save_text_field(
                        field_code=store_field,
                        value=new_value,
                        confidence=1.0,
                        profile=active_profile,
                        confidence_threshold=0.0,
                        tool_call_id=tool_call_id,
                    )
                    write_outcomes.append(color_result.to_outcome_dict())
                    logger.info(
                        "Phase A: color suggestion saved",
                        extra={"session_id": state.session_id, "color": color_str, "field": store_field},
                    )
                except Exception as exc:
                    write_outcomes.append(FieldWriteResult(
                        field_code="color_suggestion",
                        value=str(arguments.get("hex", "?")),
                        status=WRITE_STATUS_REJECTED_MALFORMED,
                        reason=f"Color suggestion validation failed: {exc}",
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                    ).to_outcome_dict())
                    logger.warning(
                        "Phase A: color suggestion rejected — validation failed",
                        extra={"session_id": state.session_id, "error": str(exc)},
                    )
                continue

            # ── Field-saving tools — extract and validate arguments ────────
            field_code = arguments.get("field_code")
            value = arguments.get("value")
            confidence_raw = arguments.get("confidence")

            # Validate required parameters are present
            if not field_code or value is None or confidence_raw is None:
                missing_params = [
                    p for p, v in [("field_code", field_code), ("value", value), ("confidence", confidence_raw)]
                    if v is None or v == ""
                ]
                write_outcomes.append(FieldWriteResult(
                    field_code=field_code,
                    value=value,
                    status=WRITE_STATUS_REJECTED_MALFORMED,
                    reason=f"Missing required parameters: {missing_params}",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                ).to_outcome_dict())
                logger.warning(
                    "Phase A: tool call missing required params",
                    extra={
                        "session_id": state.session_id,
                        "tool_name": tool_name,
                        "missing": missing_params,
                    },
                )
                continue

            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                write_outcomes.append(FieldWriteResult(
                    field_code=field_code,
                    value=str(value),
                    status=WRITE_STATUS_REJECTED_MALFORMED,
                    reason=f"confidence is not a valid number: {confidence_raw!r}",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                ).to_outcome_dict())
                continue

            # ── Dispatch to the appropriate handler ────────────────────────
            if tool_name == "save_text_field":
                result = state.handle_save_text_field(
                    field_code=field_code,
                    value=str(value),
                    confidence=confidence,
                    profile=active_profile,
                    confidence_threshold=settings.extraction_confidence_threshold,
                    tool_call_id=tool_call_id,
                )
            elif tool_name == "save_enum_field":
                result = state.handle_save_enum_field(
                    field_code=field_code,
                    value=str(value),
                    confidence=confidence,
                    profile=active_profile,
                    confidence_threshold=settings.extraction_confidence_threshold,
                    tool_call_id=tool_call_id,
                )
            elif tool_name == "save_quantitative_field":
                result = state.handle_save_quantitative_field(
                    field_code=field_code,
                    value=str(value),
                    confidence=confidence,
                    profile=active_profile,
                    confidence_threshold=settings.extraction_confidence_threshold,
                    tool_call_id=tool_call_id,
                )
            else:
                # Unknown tool name — log and skip
                logger.warning(
                    "Phase A: unknown tool name received from LLM",
                    extra={"session_id": state.session_id, "tool_name": tool_name},
                )
                write_outcomes.append(FieldWriteResult(
                    field_code=field_code,
                    value=str(value),
                    status=WRITE_STATUS_REJECTED_MALFORMED,
                    reason=f"Unknown tool name: {tool_name!r}",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                ).to_outcome_dict())
                continue

            write_outcomes.append(result.to_outcome_dict())
            logger.info(
                "Phase A: tool dispatched",
                extra={
                    "session_id": state.session_id,
                    "tool_name": tool_name,
                    "field_code": field_code,
                    "status": result.status,
                },
            )

        # Bug 10 fix: collect unmapped signals from rejected_unknown_field outcomes.
        # These are values the LLM found meaningful but couldn't map to any required field.
        # Appending (not replacing) preserves signals across turns.
        for outcome in write_outcomes:
            if outcome.get("status") == WRITE_STATUS_REJECTED_UNKNOWN_FIELD and outcome.get("value"):
                signal = f"{outcome['value']} (tried field: {outcome.get('field_code', '?')})"
                if signal not in state.unmapped_signals:
                    state.unmapped_signals.append(signal)

        # Attempt DB write — We NO LONGER save session in Phase A! 
        # This saves ~1.5s of blocking DB write time. The state changes are
        # held in memory and will be persisted at the end of the turn (Step 13).
        # We only log that Phase A completed.
        if any(o["status"] == WRITE_STATUS_SAVED for o in write_outcomes):
            logger.info(
                "Phase A: extraction applied to memory (DB write deferred to end of turn)",
                extra={
                    "session_id": state.session_id,
                    "write_outcomes": write_outcomes,
                },
            )

        return write_outcomes, dispatched_tool_calls, advisory_flags

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_current_context_hint(
        state: "ConversationState",
        missing_fields: "list",
    ) -> str | None:
        """Build a CURRENT CONTEXT annotation for Phase A.

        Identifies which field the AI most recently asked about by checking the
        last assistant message in conversation history. Injects an explicit
        annotation so Phase A never has to guess what the user is answering.

        This is the primary fix for Bug 2: empty phase_a_write_outcomes caused
        by short/Tanglish answers the LLM couldn't map to a field without context.

        Args:
            state:          Current conversation state.
            missing_fields: Fields not yet captured (first entry = current question).

        Returns:
            A formatted CURRENT CONTEXT annotation string, or None if not applicable.
        """
        if not missing_fields:
            return None

        # The first missing field is the one Phase B most recently asked about
        # (Phase B is always instructed to ask only about missing_fields[0]).
        current_field = missing_fields[0]

        # Build a rich description of the current field including enum options if present
        field_info_lines = [
            f"[CURRENT CONTEXT — FOR EXTRACTION ONLY]",
            f"The AI's last question was about the field: '{current_field.field_code}'",
            f"Field description: {current_field.description}",
        ]

        if current_field.enum_values:
            from app.project_profiles.base_profile import FieldDefinition
            opts = current_field.enum_options
            if opts:
                pairs = ", ".join(
                    f'"{o.get("label", o.get("value"))}" → "{o.get("value", o.get("label"))}"'
                    for o in opts[:8]
                )
                field_info_lines.append(
                    f"This is an ENUM field. The user's answer must map to one of these options: {pairs}"
                )
                field_info_lines.append(
                    f"Accept EITHER the label (human-readable) OR the machine value — both are valid inputs."
                )
            else:
                vals = ", ".join(f'"{v}"' for v in current_field.enum_values[:8])
                field_info_lines.append(
                    f"This is an ENUM field. Allowed values: {vals}"
                )

        field_info_lines.append(
            "The user's NEXT message is their answer to this question. "
            "Even if the answer is short (a single word, a name, or casual language), "
            "you MUST extract it and call the appropriate save tool."
        )

        return "\n".join(field_info_lines)

    @staticmethod
    def _sse_chunk(text: str) -> str:
        """Format a streaming text chunk as an SSE data event."""
        payload = json.dumps({"chunk": text})
        return f"data: {payload}\n\n"

    @staticmethod
    def _sse_done(snapshot: dict[str, Any]) -> str:
        """Format the final snapshot as an SSE done event."""
        payload = json.dumps({"done": True, "snapshot": snapshot})
        return f"data: {payload}\n\n"

    @staticmethod
    def _extract_message_delta(accumulated_json: str, already_streamed: int) -> str:
        """Speculatively extract new message characters from partial JSON.

        As the LLM streams token fragments, the accumulated JSON gradually
        builds up. We attempt to parse the ``message`` field value as it
        appears in the partial JSON string — enabling low-latency token
        streaming to the client without waiting for the full JSON.

        Returns:
            Newly available message characters not yet streamed (may be "").
        """
        import re
        pattern = r'"message"\s*:\s*"((?:[^"\\]|\\.)*)'
        match = re.search(pattern, accumulated_json)
        if not match:
            return ""

        raw_value = match.group(1)
        try:
            unescaped = json.loads(f'"{raw_value}"')
        except json.JSONDecodeError:
            unescaped = raw_value

        return unescaped[already_streamed:]
