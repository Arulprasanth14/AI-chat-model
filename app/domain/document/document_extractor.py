"""
app/domain/document/document_extractor.py
──────────────────────────────────────────
Core orchestrator for LLM-based field extraction from uploaded documents.

DESIGN CONTRACT
───────────────
This module does NOT call process_turn(). It has a dedicated extraction
pipeline that:
  1. Uses the same Phase A tools (save_text_field, save_enum_field,
     save_quantitative_field, save_custom_field) so the write path is identical.
  2. Uses the same state handlers on ConversationState — the ledger is the
     single source of truth regardless of whether values came from chat or doc.
  3. Processes document chunks sequentially so earlier extractions already
     reside in the ledger when later chunks are processed (prevents duplicates).
  4. Never overwrites a field that was captured at a confidence >= the incoming
     extraction — the existing _write_field logic enforces this automatically.

ADDITIONAL FIELDS
─────────────────
The LLM is explicitly instructed to:
  a) Identify required fields first (primary extraction target).
  b) After required fields, scan the document for meaningful non-required
     information that could be useful to a designer or content creator.
  c) Save each such piece of information via save_custom_field with a
     descriptive snake_case field name (prefixed with custom_).
  d) Skip irrelevant, generic, or noise information entirely.

ANTI-HALLUCINATION
──────────────────
  • The extraction prompt requires each extracted value to be directly
    supported by text in the document (source_excerpt).
  • Confidence calibration in the prompt means inferred/paraphrased values
    receive confidence < 0.65, below the system threshold (0.7) — they are
    automatically rejected by the state handlers.
  • source_excerpt is logged with each write outcome for auditability
    but is NOT persisted to the database (Q3 answer).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.domain.conversation.state import (
    ConversationState,
    MissingField,
    WRITE_STATUS_SAVED,
    WRITE_STATUS_REJECTED_MALFORMED,
    WRITE_STATUS_REJECTED_UNKNOWN_FIELD,
    FieldWriteResult,
)
from app.domain.llm.provider import LLMProvider
from app.domain.llm.tool_schema import (
    get_document_extraction_tools,
    parse_phase_a_tool_calls,
)
from app.domain.llm.prompt_builder import PromptBuilder
from app.infrastructure.document_parser import ParsedDocument, DocumentChunk
from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)

# Characters-per-token estimate for chunk budget calculation
_CHARS_PER_TOKEN = 4

# Overhead tokens for the system prompt (fields list, instructions, etc.)
_SYSTEM_PROMPT_OVERHEAD_TOKENS = 2_000


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DocumentExtractionResult:
    """Full result of a document extraction run.

    Attributes:
        write_outcomes:            All write attempt dicts (one per tool call).
        fields_saved:              Required field codes successfully written.
        fields_failed:             Required field codes attempted but rejected.
        additional_fields_saved:   custom_ field codes successfully written.
        missing_required_fields:   Required fields still outstanding after extraction.
        extraction_summary:        Human-readable summary for the API response message.
        warnings:                  Non-fatal issues (low confidence, skipped, etc.).
        chunks_processed:          How many document chunks were sent to the LLM.
    """
    write_outcomes: list[dict[str, Any]]
    fields_saved: list[str]
    fields_failed: list[str]
    additional_fields_saved: list[str]
    missing_required_fields: list[MissingField]
    extraction_summary: str
    warnings: list[str]
    chunks_processed: int


# ── Main extractor class ───────────────────────────────────────────────────────

class DocumentExtractor:
    """Extracts field values from a ParsedDocument using LLM tool calls.

    This is a stateless service — all state mutations happen on the
    ConversationState object passed in by the caller.

    Args:
        llm:            LLMProvider to call for extraction.
        prompt_builder: PromptBuilder (used for the document extraction prompt).
    """

    def __init__(self, llm: LLMProvider, prompt_builder: PromptBuilder) -> None:
        self._llm = llm
        self._prompt_builder = prompt_builder

    async def extract(
        self,
        state: ConversationState,
        active_profile: BaseProfile,
        parsed_doc: ParsedDocument,
        confidence_threshold: float | None = None,
    ) -> DocumentExtractionResult:
        """Extract fields from a parsed document into the session state.

        Processing flow:
          1. Determine missing required fields (pre-extraction snapshot).
          2. Calculate token budget per chunk.
          3. For each chunk (sequentially):
               a. Build extraction prompt.
               b. Call LLM → get tool calls.
               c. Dispatch each tool call to the appropriate state handler.
               d. Collect write outcomes.
          4. Recompute missing fields post-extraction.
          5. Build human-readable summary.
          6. Return DocumentExtractionResult.

        Args:
            state:               Mutable ConversationState (mutated in place).
            active_profile:      Active profile for field validation.
            parsed_doc:          ParsedDocument from DocumentParser.
            confidence_threshold: Override for the extraction confidence threshold.
                                  Defaults to settings.extraction_confidence_threshold.

        Returns:
            DocumentExtractionResult with a full accounting of what happened.
        """
        threshold = confidence_threshold if confidence_threshold is not None \
            else settings.extraction_confidence_threshold

        # Pre-extraction state snapshot
        missing_before = state.compute_missing_fields(active_profile, threshold)
        already_captured = set(state.captured.keys())

        # Determine which chunks to process within the token budget
        chunks_to_process = self._select_chunks(parsed_doc)

        logger.info(
            "Document extraction started",
            extra={
                "session_id": state.session_id,
                "file_name": parsed_doc.filename,
                "total_chunks": len(parsed_doc.chunks),
                "chunks_to_process": len(chunks_to_process),
                "missing_fields_count": len(missing_before),
            },
        )

        # ── Sequential chunk processing ────────────────────────────────────────
        all_write_outcomes: list[dict[str, Any]] = []
        warnings: list[str] = list(parsed_doc.warnings)

        for chunk_idx, chunk in enumerate(chunks_to_process):
            logger.debug(
                "Processing document chunk",
                extra={
                    "session_id": state.session_id,
                    "chunk_index": chunk.chunk_index,
                    "chunk_chars": len(chunk.text),
                    "chunk_type": chunk.chunk_type,
                },
            )

            # Recompute missing fields so each chunk benefits from earlier saves
            current_missing = state.compute_missing_fields(active_profile, threshold)

            chunk_outcomes, chunk_warnings = await self._process_chunk(
                state=state,
                active_profile=active_profile,
                chunk=chunk,
                chunk_position=f"{chunk_idx + 1}/{len(chunks_to_process)}",
                missing_fields=current_missing,
                threshold=threshold,
            )

            all_write_outcomes.extend(chunk_outcomes)
            warnings.extend(chunk_warnings)

        # ── Post-extraction accounting ─────────────────────────────────────────
        missing_after = state.compute_missing_fields(active_profile, threshold)
        newly_captured = set(state.captured.keys()) - already_captured

        required_codes = {f.code for f in active_profile.required_fields if f.required}

        fields_saved = [
            o["field_code"] for o in all_write_outcomes
            if o.get("status") == WRITE_STATUS_SAVED
            and o.get("field_code") in required_codes
        ]
        # Deduplicate (same field may be saved from multiple chunks)
        fields_saved = list(dict.fromkeys(fields_saved))

        additional_fields_saved = [
            o["field_code"] for o in all_write_outcomes
            if o.get("status") == WRITE_STATUS_SAVED
            and o.get("field_code", "").startswith("custom_")
        ]
        additional_fields_saved = list(dict.fromkeys(additional_fields_saved))

        # Fields that were attempted but rejected (only required fields)
        attempted_required = {
            o["field_code"] for o in all_write_outcomes
            if o.get("field_code") in required_codes
        }
        fields_failed = [
            code for code in attempted_required
            if code not in fields_saved
        ]

        # Build the user-facing summary message
        extraction_summary = self._build_summary(
            fields_saved=fields_saved,
            additional_fields_saved=additional_fields_saved,
            missing_after=missing_after,
            warnings=warnings,
            filename=parsed_doc.filename,
        )

        logger.info(
            "Document extraction complete",
            extra={
                "session_id": state.session_id,
                "file_name": parsed_doc.filename,
                "fields_saved": fields_saved,
                "additional_fields_saved": additional_fields_saved,
                "fields_failed": fields_failed,
                "missing_remaining": len(missing_after),
                "chunks_processed": len(chunks_to_process),
            },
        )

        return DocumentExtractionResult(
            write_outcomes=all_write_outcomes,
            fields_saved=fields_saved,
            fields_failed=fields_failed,
            additional_fields_saved=additional_fields_saved,
            missing_required_fields=missing_after,
            extraction_summary=extraction_summary,
            warnings=warnings,
            chunks_processed=len(chunks_to_process),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _process_chunk(
        self,
        state: ConversationState,
        active_profile: BaseProfile,
        chunk: DocumentChunk,
        chunk_position: str,
        missing_fields: list[MissingField],
        threshold: float,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Run LLM extraction on a single document chunk.

        Returns:
            (write_outcomes, warnings)
        """
        warnings: list[str] = []

        # Build the document extraction prompt for this chunk
        messages = self._prompt_builder.build_document_extraction_phase(
            profile=active_profile,
            missing_fields=missing_fields,
            chunk_text=chunk.text,
            chunk_position=chunk_position,
            chunk_type=chunk.chunk_type,
            captured_fields={k: v.value for k, v in state.captured.items()},
        )

        # Get the appropriate tools (document mode — no advisory tools)
        tools = get_document_extraction_tools(active_profile)

        # Call the LLM (temperature=0.0 for deterministic extraction)
        try:
            raw_tool_calls = await self._llm.call_with_tools(
                messages=messages,
                tools=tools,
                temperature=0.0,
            )
        except Exception as exc:
            logger.error(
                "Document extraction: LLM call failed for chunk",
                extra={"session_id": state.session_id, "chunk_position": chunk_position},
                exc_info=exc,
            )
            warnings.append(
                f"LLM extraction failed for document chunk {chunk_position}: {exc}"
            )
            return [], warnings

        if not raw_tool_calls:
            logger.debug(
                "Document extraction: no tool calls for chunk (no extractable fields found)",
                extra={"session_id": state.session_id, "chunk_position": chunk_position},
            )
            return [], warnings

        # Parse and dispatch each tool call
        parsed_calls = parse_phase_a_tool_calls(raw_tool_calls)
        write_outcomes: list[dict[str, Any]] = []

        for call in parsed_calls:
            outcome, call_warnings = self._dispatch_tool_call(
                call=call,
                state=state,
                active_profile=active_profile,
                threshold=threshold,
                missing_fields=missing_fields,
            )
            if outcome:
                write_outcomes.append(outcome)
            warnings.extend(call_warnings)

        return write_outcomes, warnings

    def _dispatch_tool_call(
        self,
        call: dict[str, Any],
        state: ConversationState,
        active_profile: BaseProfile,
        threshold: float,
        missing_fields: list[MissingField],
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Dispatch a single parsed tool call to the appropriate state handler.

        Returns:
            (outcome_dict | None, warnings)
        """
        warnings: list[str] = []
        tool_name = call["tool_name"]
        tool_call_id = call["tool_call_id"]
        arguments = call["arguments"]
        parse_error = call.get("parse_error")

        if parse_error:
            outcome = FieldWriteResult(
                field_code=None,
                value=None,
                status=WRITE_STATUS_REJECTED_MALFORMED,
                reason=f"Tool call JSON parse error: {parse_error}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ).to_outcome_dict()
            warnings.append(f"Malformed tool call ({tool_name}): {parse_error}")
            return outcome, warnings

        # Extract source_excerpt for logging (not persisted — Q3 answer)
        source_excerpt: str = arguments.pop("source_excerpt", "") or ""

        is_custom = (tool_name == "save_custom_field")
        field_code = arguments.get("field_code")
        field_name = arguments.get("field_name")
        value = arguments.get("value")
        confidence_raw = arguments.get("confidence")

        req_field = field_name if is_custom else field_code
        req_field_key = "field_name" if is_custom else "field_code"

        # Validate required parameters
        if not req_field or value is None or confidence_raw is None:
            missing_params = [
                p for p, v in [(req_field_key, req_field), ("value", value), ("confidence", confidence_raw)]
                if v is None or v == ""
            ]
            outcome = FieldWriteResult(
                field_code=req_field or "unknown",
                value=str(value) if value else "",
                status=WRITE_STATUS_REJECTED_MALFORMED,
                reason=f"Missing required parameters: {missing_params}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ).to_outcome_dict()
            warnings.append(f"Skipped malformed tool call ({tool_name}) — missing: {missing_params}")
            return outcome, warnings

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            outcome = FieldWriteResult(
                field_code=req_field,
                value=str(value),
                status=WRITE_STATUS_REJECTED_MALFORMED,
                reason=f"confidence is not a valid number: {confidence_raw!r}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ).to_outcome_dict()
            return outcome, warnings

        # Fuzzy field code resolution for non-custom tools
        if not is_custom and field_code:
            field_code = self._resolve_field_code(
                field_code=field_code,
                profile=active_profile,
                missing_fields=missing_fields,
            )

        # Dispatch to the appropriate state handler
        if tool_name == "save_text_field":
            result = state.handle_save_text_field(
                field_code=field_code,
                value=str(value),
                confidence=confidence,
                profile=active_profile,
                confidence_threshold=threshold,
                tool_call_id=tool_call_id,
            )
        elif tool_name == "save_enum_field":
            result = state.handle_save_enum_field(
                field_code=field_code,
                value=str(value),
                confidence=confidence,
                profile=active_profile,
                confidence_threshold=threshold,
                tool_call_id=tool_call_id,
            )
        elif tool_name == "save_quantitative_field":
            result = state.handle_save_quantitative_field(
                field_code=field_code,
                value=str(value),
                confidence=confidence,
                profile=active_profile,
                confidence_threshold=threshold,
                tool_call_id=tool_call_id,
            )
        elif tool_name == "save_custom_field":
            result = state.handle_save_custom_field(
                field_name=field_name,
                value=str(value),
                confidence=confidence,
                profile=active_profile,
                confidence_threshold=threshold,
                tool_call_id=tool_call_id,
            )
        else:
            logger.warning(
                "Document extraction: unknown tool name received",
                extra={"tool_name": tool_name},
            )
            outcome = FieldWriteResult(
                field_code=field_code,
                value=str(value),
                status=WRITE_STATUS_REJECTED_MALFORMED,
                reason=f"Unknown tool name: {tool_name!r}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            ).to_outcome_dict()
            return outcome, warnings

        outcome = result.to_outcome_dict()

        # Log with source_excerpt for audit/traceability (Q3: logs only)
        logger.info(
            "Document extraction: tool dispatched",
            extra={
                "session_id": state.session_id,
                "tool_name": tool_name,
                "field_code": result.field_code,
                "status": result.status,
                "confidence": confidence,
                "source_excerpt": source_excerpt[:200] if source_excerpt else None,
            },
        )

        # Collect low-confidence warning for the extraction report
        if result.status != WRITE_STATUS_SAVED and result.status != "rejected_lower_confidence":
            warnings.append(
                f"Field '{result.field_code}' extraction rejected: {result.status}"
                + (f" — {result.reason}" if result.reason else "")
            )

        return outcome, warnings

    def _select_chunks(self, parsed_doc: ParsedDocument) -> list[DocumentChunk]:
        """Select which chunks to process within the configured token budget.

        Respects settings.document_extraction_max_tokens by estimating token
        count per chunk (4 chars ≈ 1 token) and stopping before the budget
        for document text is exhausted.

        The budget only applies to the document text portion — the system prompt
        overhead (_SYSTEM_PROMPT_OVERHEAD_TOKENS) is subtracted first.
        """
        doc_budget_tokens = settings.document_extraction_max_tokens - _SYSTEM_PROMPT_OVERHEAD_TOKENS
        doc_budget_chars = doc_budget_tokens * _CHARS_PER_TOKEN

        selected: list[DocumentChunk] = []
        used_chars = 0

        for chunk in parsed_doc.chunks:
            chunk_len = len(chunk.text)
            if used_chars + chunk_len > doc_budget_chars and selected:
                logger.info(
                    "Document extraction: stopping chunk selection at token budget",
                    extra={
                        "stopped_at_chunk": chunk.chunk_index,
                        "total_chunks": len(parsed_doc.chunks),
                        "chars_used": used_chars,
                        "budget_chars": doc_budget_chars,
                    },
                )
                break
            selected.append(chunk)
            used_chars += chunk_len

        return selected or parsed_doc.chunks[:1]  # Always process at least 1 chunk

    def _build_summary(
        self,
        fields_saved: list[str],
        additional_fields_saved: list[str],
        missing_after: list[MissingField],
        warnings: list[str],
        filename: str,
    ) -> str:
        """Build a human-readable extraction summary message for the API response."""
        parts: list[str] = []

        if fields_saved:
            field_names = ", ".join(f"`{c}`" for c in fields_saved[:6])
            suffix = f" and {len(fields_saved) - 6} more" if len(fields_saved) > 6 else ""
            parts.append(
                f"✅ I extracted **{len(fields_saved)} required field(s)** from `{filename}`: "
                f"{field_names}{suffix}."
            )
        else:
            parts.append(
                f"I reviewed `{filename}` but could not confidently extract any required fields."
            )

        if additional_fields_saved:
            add_names = ", ".join(c.replace("custom_", "").replace("_", " ") for c in additional_fields_saved[:4])
            suffix = f" and {len(additional_fields_saved) - 4} more" if len(additional_fields_saved) > 4 else ""
            parts.append(
                f"📎 I also captured **{len(additional_fields_saved)} additional detail(s)** "
                f"that may be useful for your brief: {add_names}{suffix}."
            )

        if missing_after:
            missing_names = ", ".join(f"`{mf.field_code}`" for mf in missing_after[:5])
            suffix = f" and {len(missing_after) - 5} more" if len(missing_after) > 5 else ""
            parts.append(
                f"⚠️ **{len(missing_after)} required field(s)** still need to be filled in: "
                f"{missing_names}{suffix}. Please answer these in the chat."
            )
        else:
            parts.append(
                "🎉 All required fields are now captured! Please review and confirm."
            )

        # Include non-trivial warnings (filter out parser-level noise)
        meaningful_warnings = [w for w in warnings if "LLM extraction failed" in w or "rejected:" in w]
        if meaningful_warnings:
            parts.append(
                f"ℹ️ Note: {meaningful_warnings[0]}"
                + (f" (+{len(meaningful_warnings)-1} more)" if len(meaningful_warnings) > 1 else "")
            )

        return "\n\n".join(parts)

    @staticmethod
    def _resolve_field_code(
        field_code: str,
        profile: BaseProfile,
        missing_fields: list[MissingField],
    ) -> str:
        """Fuzzy-resolve a descriptive field code to a valid profile code.

        Mirrors the logic in ConversationOrchestrator._resolve_field_code.
        """
        valid_codes = {f.code for f in profile.required_fields}
        if field_code in valid_codes:
            return field_code

        probe = field_code.lower().replace("_", " ").replace("-", " ")
        probe_words = set(probe.split())

        best_code: str | None = None
        best_score: int = 0

        missing_codes = [mf.field_code for mf in missing_fields]
        priority_codes = missing_codes + [f.code for f in profile.required_fields if f.code not in missing_codes]

        for code in priority_codes:
            field_def = profile.get_field_by_code(code)
            if field_def is None:
                continue
            search_text = " ".join(filter(None, [
                code.lower().replace("_", " "),
                (field_def.description or "").lower(),
            ]))
            search_words = set(search_text.split())
            overlap = len(probe_words & search_words)
            bonus = 2 if code in missing_codes else 0
            score = overlap + bonus
            if score > best_score:
                best_score = score
                best_code = code

        if best_code and best_score > 0:
            if best_code != field_code:
                logger.info(
                    "Document extraction: fuzzy field code resolved",
                    extra={"original": field_code, "resolved": best_code, "score": best_score},
                )
            return best_code

        logger.warning(
            "Document extraction: field code not resolvable",
            extra={"field_code": field_code},
        )
        return field_code
