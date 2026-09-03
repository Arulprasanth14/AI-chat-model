"""
app/domain/conversation/state.py
──────────────────────────────────
Conversation state and completion ledger.

This module is the ONLY deterministic gate in the system:
  - It tracks which fields have been saved (above confidence threshold).
  - It computes which required fields are still missing.
  - It does NOT dictate question wording, order, or conversation flow.
  - It does NOT contain any domain-specific strings (no field names, no brief types).
  - It operates on any BaseProfile — 100% profile-agnostic.

The orchestrator passes ``missing_fields`` to the LLM as context. The LLM
decides how to ask. The ledger just keeps score.

EXPLICIT TOOL CALL HANDLERS
────────────────────────────
The primary write path is now via named handler methods that correspond
directly to the LLM's named tool calls:

  handle_save_text_field(field_code, value, confidence, profile)
  handle_save_enum_field(field_code, value, confidence, profile)
  handle_save_quantitative_field(field_code, value, confidence, profile)

Each handler returns a FieldWriteResult with a status and optional reason.
The orchestrator collects all results before Phase B runs — no state
is guessed from text, and no confirmation language is emitted until
the write result is known.

apply_extraction() is retained for backward compatibility with tests and
scripts that still use the old extraction pattern.
"""
from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)


# ── Write status constants ─────────────────────────────────────────────────────

WRITE_STATUS_SAVED = "saved"
WRITE_STATUS_REJECTED_LOW_CONFIDENCE = "rejected_low_confidence"
WRITE_STATUS_REJECTED_ENUM = "rejected_enum"
WRITE_STATUS_REJECTED_QUALITATIVE = "rejected_qualitative"  # quantitative field got non-numeric value
WRITE_STATUS_REJECTED_UNKNOWN_FIELD = "rejected_unknown_field"
WRITE_STATUS_REJECTED_LOWER_CONFIDENCE = "rejected_lower_confidence"
WRITE_STATUS_REJECTED_MALFORMED = "rejected_malformed"  # missing required tool arguments


# ── Data transfer objects ──────────────────────────────────────────────────────

@dataclass
class FieldWriteResult:
    """Result of a single field write operation, returned by each tool handler.

    Attributes:
        field_code:  Profile field code that was targeted (may be None if unknown).
        value:       The value that was attempted (or None for advisory tools).
        status:      One of the WRITE_STATUS_* constants.
        reason:      Human-readable explanation for non-saved statuses.
        tool_name:   Name of the tool that was called.
        tool_call_id: Tool call ID from the LLM, used to wire up tool-result messages.
    """
    field_code: str | None
    value: str | None
    status: str
    reason: str | None
    tool_name: str
    tool_call_id: str = ""

    def to_outcome_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the existing write_outcomes format."""
        return {
            "field_code": self.field_code,
            "field_hint": self.field_code,  # tool-calling path uses field_code directly
            "value": self.value,
            "status": self.status,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
        }


class ExtractedAnswer(BaseModel):
    """A single field value extracted by the LLM from user message(s).

    Used by the legacy apply_extraction() path.

    Attributes:
        field_hint:  Natural-language hint from the LLM about which field
                     this answer relates to (e.g. "target audience", "budget").
                     The state ledger maps this to a field_code via the profile.
        value:       The extracted value as a string.
        confidence:  0.0–1.0 confidence score assigned by the LLM.
        field_code:  Resolved field code (set by ``apply_extraction`` after
                     matching field_hint against profile field descriptions).
    """

    field_hint: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    field_code: str | None = None  # resolved after matching


class CapturedField(BaseModel):
    """A confirmed captured field in the completion ledger.

    Attributes:
        field_code:    Matching key from profile.required_fields[].code.
        value:         The best extracted value so far.
        confidence:    Confidence of the best extraction.
        turn_index:    Which conversation turn this was extracted from.
        extracted_at:  ISO timestamp of extraction.
    """

    field_code: str
    value: str
    confidence: float
    turn_index: int
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class MissingField(BaseModel):
    """A required field that has not yet been captured above the threshold."""

    field_code: str
    description: str  # from profile — fed to LLM as context
    enum_values: list[str] | None = None
    enum_options: list[dict[str, str]] | None = None
    input_type: str = "text"


# ── Core state object ──────────────────────────────────────────────────────────

class ConversationState(BaseModel):
    """Mutable state for a single conversation session.

    This is the domain object that flows through the system. The repository
    layer serialises/deserialises it to/from the ORM model.

    Usage:
        state = ConversationState(session_id=..., profile_id=...)
        state.add_turn("user", "I need a brand identity for my startup")
        result = state.handle_save_text_field("client_name", "Acme", 0.95, profile, threshold)
        missing = state.compute_missing_fields(profile, threshold)
    """

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    profile_id: str
    status: str = "active"

    # Ordered list of conversation turns: [{role, content, timestamp}]
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)

    # Completion ledger: field_code → CapturedField
    captured: dict[str, CapturedField] = Field(default_factory=dict)

    # Advisory flags from last LLM turn
    model_believes_complete: bool = False
    suggested_next_topic: str | None = None
    last_intent: str | None = None

    # Multi-template resolution state
    resolved_vertical: str | None = None
    resolved_template_key: str | None = None

    # Unmapped signals: meaningful content that was mentioned but has no field home.
    # Populated when Phase A returns rejected_unknown_field for a value the LLM tried
    # to save. Non-empty means the session has valuable signal not tracked in the ledger
    # — the UI should show a caveat state rather than a pure green complete badge.
    unmapped_signals: list[str] = Field(default_factory=list)

    # Turn counter (len(conversation_history) // 2 ≈ turn count)
    _turn_index: int = 0

    # ── History management ─────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role:    "user" or "assistant"
            content: Message text
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._turn_index += 1

    def get_recent_history(self, window: int) -> list[dict[str, Any]]:
        """Return the most recent ``window`` messages (for prompt construction).

        Args:
            window: Maximum number of messages to return.

        Returns:
            Slice of conversation_history, newest last.
        """
        return self.conversation_history[-window:] if window > 0 else []

    # ── Explicit tool call handlers ────────────────────────────────────────────

    def handle_save_text_field(
        self,
        field_code: str,
        value: str,
        confidence: float,
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
        tool_call_id: str = "",
    ) -> FieldWriteResult:
        """Handle a save_text_field tool call from the LLM.

        Validates that:
          - field_code exists in the profile.
          - confidence meets the threshold.
          - value is not empty.
          - New value has higher or equal confidence than the existing capture
            (allows explicit corrections at confidence == 1.0).

        Args:
            field_code:           Exact field code from the LLM tool call.
            value:                Extracted value string.
            confidence:           LLM-assigned confidence (0.0–1.0).
            profile:              Active profile for field validation.
            confidence_threshold: Minimum confidence to accept.
            tool_call_id:         LLM tool call ID for tool-result wiring.

        Returns:
            FieldWriteResult with status and reason.
        """
        tool_name = "save_text_field"

        # Validate field exists
        field_def = profile.get_field_by_code(field_code)
        if field_def is None:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_UNKNOWN_FIELD,
                reason=f"field_code {field_code!r} not found in profile",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Validate confidence
        if confidence < confidence_threshold:
            logger.info(
                "save_text_field: rejected — low confidence",
                extra={"field_code": field_code, "confidence": confidence, "threshold": confidence_threshold},
            )
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_LOW_CONFIDENCE,
                reason=f"confidence {confidence:.2f} < threshold {confidence_threshold:.2f}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Look up input_type from profile to decide merge semantics
        is_list_field = (field_def.input_type == "list")

        return self._write_field(field_code, value, confidence, tool_name, tool_call_id, is_list_field=is_list_field)

    def handle_save_enum_field(
        self,
        field_code: str,
        value: str,
        confidence: float,
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
        tool_call_id: str = "",
    ) -> FieldWriteResult:
        """Handle a save_enum_field tool call from the LLM.

        Validates that:
          - field_code exists and has enum_values.
          - value matches one of the allowed enum values (case-insensitive,
            underscore/hyphen normalised).
          - confidence meets the threshold.

        Args:
            field_code:           Exact field code.
            value:                Extracted value (must be a valid enum option).
            confidence:           LLM-assigned confidence.
            profile:              Active profile.
            confidence_threshold: Minimum confidence to accept.
            tool_call_id:         LLM tool call ID.

        Returns:
            FieldWriteResult with status and reason.
        """
        tool_name = "save_enum_field"

        field_def = profile.get_field_by_code(field_code)
        if field_def is None:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_UNKNOWN_FIELD,
                reason=f"field_code {field_code!r} not found in profile",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        if not field_def.enum_values:
            logger.warning(
                "save_enum_field called for field with no enum_values — routing to text handler",
                extra={"field_code": field_code},
            )
            return self.handle_save_text_field(
                field_code, value, confidence, profile, confidence_threshold, tool_call_id
            )

        # Confidence check
        if confidence < confidence_threshold:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_LOW_CONFIDENCE,
                reason=f"confidence {confidence:.2f} < threshold {confidence_threshold:.2f}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Look up input_type from profile to decide merge semantics
        is_list_field = (field_def.input_type == "list")

        # Enum validation
        raw_values = [v.strip() for v in value.split(",")] if is_list_field else [value.strip()]
        allowed_values = [v.lower() for v in field_def.enum_values]
        
        allowed_labels = []
        if field_def.enum_options:
            allowed_labels = [o.get("label", "").lower() for o in field_def.enum_options]
            
        canonical_values = []

        for raw_val in raw_values:
            val_lower = raw_val.lower()
            normalised = val_lower.replace(" ", "_").replace("-", "_")
            if val_lower in allowed_values:
                canonical_values.append(field_def.enum_values[allowed_values.index(val_lower)])
            elif normalised in allowed_values:
                canonical_values.append(field_def.enum_values[allowed_values.index(normalised)])
            elif val_lower in allowed_labels:
                canonical_values.append(field_def.enum_values[allowed_labels.index(val_lower)])
            else:
                logger.info(
                    "save_enum_field: rejected — value not in enum list",
                    extra={"field_code": field_code, "value": raw_val, "allowed": field_def.enum_values[:5]},
                )
                return FieldWriteResult(
                    field_code=field_code,
                    value=value,
                    status=WRITE_STATUS_REJECTED_ENUM,
                    reason=(
                        f"value {raw_val!r} not in allowed list "
                        f"{field_def.enum_values[:5]!r}"
                        + (" …" if len(field_def.enum_values) > 5 else "")
                    ),
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )

        canonical = ", ".join(canonical_values)

        return self._write_field(field_code, canonical, confidence, tool_name, tool_call_id, is_list_field=is_list_field)

    def handle_save_quantitative_field(
        self,
        field_code: str,
        value: str,
        confidence: float,
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
        tool_call_id: str = "",
    ) -> FieldWriteResult:
        """Handle a save_quantitative_field tool call from the LLM.

        Validates that:
          - field_code exists in the profile.
          - value contains a quantitative signal (digit, percentage, or KPI keyword).
          - confidence meets the threshold.

        If the value has no quantitative signal, it is rejected with
        WRITE_STATUS_REJECTED_QUALITATIVE so Phase B can ask the user
        for a specific number or measurable target.

        Args:
            field_code:           Exact field code.
            value:                Extracted value (must contain numeric/KPI signal).
            confidence:           LLM-assigned confidence.
            profile:              Active profile.
            confidence_threshold: Minimum confidence to accept.
            tool_call_id:         LLM tool call ID.

        Returns:
            FieldWriteResult with status and reason.
        """
        tool_name = "save_quantitative_field"

        field_def = profile.get_field_by_code(field_code)
        if field_def is None:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_UNKNOWN_FIELD,
                reason=f"field_code {field_code!r} not found in profile",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Confidence check
        if confidence < confidence_threshold:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_LOW_CONFIDENCE,
                reason=f"confidence {confidence:.2f} < threshold {confidence_threshold:.2f}",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Quantitative signal check
        quantitative_signals = (
            re.search(r"\d", value) or
            re.search(
                r"\b(percent|increase|decrease|roi|kpi|conversion|rate|target|"
                r"uplift|reach|impressions|clicks|revenue|sales|growth|"
                r"signups?|leads?)\b",
                value.lower(),
            )
        )
        if not quantitative_signals:
            logger.warning(
                "save_quantitative_field: rejected — no quantitative signal",
                extra={"field_code": field_code, "value": value},
            )
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_QUALITATIVE,
                reason=(
                    "value has no quantitative signal (no number, percentage, or measurable "
                    "target). Please provide a specific metric, e.g. '20% increase in conversions'."
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # Look up input_type from profile to decide merge semantics
        is_list_field = (field_def.input_type == "list")

        return self._write_field(field_code, value, confidence, tool_name, tool_call_id, is_list_field=is_list_field)

    # ── Completion ledger ──────────────────────────────────────────────────────

    def compute_missing_fields(
        self,
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
    ) -> list[MissingField]:
        """Compute the list of required fields not yet captured above threshold.

        This is the ONLY gate that determines whether a brief is considered
        complete. It is purely deterministic — no LLM involvement.

        Args:
            profile:              Active profile (provides required_fields list).
            confidence_threshold: Fields captured below this are treated as missing.

        Returns:
            List of MissingField objects, each with code + description for LLM context.
        """
        missing: list[MissingField] = []
        for field_def in profile.required_fields:
            if not field_def.required:
                if not field_def.show_if:
                    continue
                # Evaluate show_if dynamically
                dependency = self.captured.get(field_def.show_if.field_code)
                if not dependency or dependency.confidence < confidence_threshold:
                    continue # Dependent field not yet captured, so this conditional field isn't required yet
                
                dep_vals = [v.strip().lower() for v in dependency.value.split(",")]
                condition_met = False
                
                if field_def.show_if.in_:
                    target_vals = [v.lower() for v in field_def.show_if.in_]
                    condition_met = any(v in target_vals for v in dep_vals)
                elif field_def.show_if.not_in:
                    target_vals = [v.lower() for v in field_def.show_if.not_in]
                    condition_met = not any(v in target_vals for v in dep_vals)
                
                if not condition_met:
                    continue # Condition not met, field is not required

            captured = self.captured.get(field_def.code)
            if captured is None or captured.confidence < confidence_threshold:
                missing.append(
                    MissingField(
                        field_code=field_def.code,
                        description=field_def.description,
                        enum_values=field_def.enum_values,
                        enum_options=field_def.enum_options,
                        input_type=field_def.input_type,
                    )
                )
        return missing

    def is_complete(
        self,
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
    ) -> bool:
        """Return True if all required fields are captured above the threshold.

        Args:
            profile:              Active profile.
            confidence_threshold: Minimum confidence to count as captured.
        """
        return len(self.compute_missing_fields(profile, confidence_threshold)) == 0

    # ── Serialisation helpers ──────────────────────────────────────────────────

    def to_snapshot(self, profile: BaseProfile, threshold: float = 0.7) -> dict[str, Any]:
        """Produce a JSON-serialisable snapshot for the API response.

        This is what the SSE final event sends to the frontend, and what
        populates the state panel in the testing UI.

        The ``has_unmapped_signals`` flag is True when content was mentioned
        in conversation that had no matching required field — the frontend should
        show a caveat state on the completeness badge instead of a pure green check.
        """
        missing = self.compute_missing_fields(profile, threshold)
        is_complete = self.is_complete(profile, threshold)
        return {
            "session_id": self.session_id,
            "profile_id": self.profile_id,
            "status": self.status,
            "extracted_answers": {
                code: {
                    "value": cf.value,
                    "confidence": cf.confidence,
                    "turn_index": cf.turn_index,
                }
                for code, cf in self.captured.items()
            },
            "missing_fields": [
                {
                    "field_code": mf.field_code,
                    "description": mf.description,
                    "enum_values": mf.enum_values,
                    "enum_options": mf.enum_options,  # Bug 7 fix: include {label, value} pairs for UI chip rendering
                    "input_type": mf.input_type,
                }
                for mf in missing
            ],
            "resolved_vertical": self.resolved_vertical,
            "resolved_template_key": self.resolved_template_key,
            "model_believes_complete": self.model_believes_complete,
            "suggested_next_topic": self.suggested_next_topic,
            "last_intent": self.last_intent,
            "is_complete": is_complete,
            "turn_count": len(self.conversation_history),
            # Bug 10 fix: expose unmapped signals so the UI can show caveat badge
            "unmapped_signals": self.unmapped_signals,
            "has_unmapped_signals": len(self.unmapped_signals) > 0,
        }

    # ── Private write helper ───────────────────────────────────────────────────

    def _write_field(
        self,
        field_code: str,
        value: str,
        confidence: float,
        tool_name: str,
        tool_call_id: str,
        is_list_field: bool = False,
    ) -> FieldWriteResult:
        """Attempt to write a validated field value to the captured ledger.

        For regular fields (is_list_field=False):
          Accept the write if:
            1. No existing capture for this field.
            2. New confidence >= existing confidence (including explicit corrections).
            3. New confidence >= 0.8 and is from a newer turn (overrides older captures).

        For list fields (is_list_field=True) — Bugs 6, 7, 8 fix:
          Values are MERGED (deduplicated union) rather than replaced. Each new value
          extracted by the LLM is appended to the existing comma-separated value so that
          earlier-mentioned items (e.g. 'premium' from turn 1) are never silently dropped
          when the LLM re-extracts a fresh list on turn 2.

        Args:
            field_code:    Resolved field code.
            value:         Validated value to write.
            confidence:    Confidence score.
            tool_name:     Originating tool name (for result reporting).
            tool_call_id:  LLM tool call ID.
            is_list_field: True if the field accumulates multiple values additively.

        Returns:
            FieldWriteResult — saved or rejected_lower_confidence.
        """
        turn_index = len(self.conversation_history)
        existing = self.captured.get(field_code)

        # ── List-field additive merge path ─────────────────────────────────────
        if is_list_field and existing is not None:
            # Split both existing and new values, merge deduplicated, preserve order
            existing_items = [v.strip() for v in existing.value.split(",") if v.strip()]
            new_items = [v.strip() for v in value.split(",") if v.strip()]
            # Deduplicated union (preserve insertion order)
            merged_items = list(dict.fromkeys(existing_items + new_items))
            merged_value = ", ".join(merged_items)

            # Use the higher confidence of old vs new
            merged_confidence = max(existing.confidence, confidence)

            self.captured[field_code] = CapturedField(
                field_code=field_code,
                value=merged_value,
                confidence=merged_confidence,
                turn_index=turn_index,
            )

            logger.info(
                "List field merged",
                extra={
                    "field_code": field_code,
                    "merged_value": merged_value,
                    "added_items": [i for i in new_items if i not in existing_items],
                    "tool_name": tool_name,
                },
            )

            return FieldWriteResult(
                field_code=field_code,
                value=merged_value,
                status=WRITE_STATUS_SAVED,
                reason=None,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        # ── Standard single-value replace path ─────────────────────────────────
        should_update = False
        if existing is None:
            should_update = True
        elif confidence >= existing.confidence:
            should_update = True
        elif confidence >= 0.8 and turn_index > existing.turn_index:
            should_update = True

        if existing is not None and not should_update:
            return FieldWriteResult(
                field_code=field_code,
                value=value,
                status=WRITE_STATUS_REJECTED_LOWER_CONFIDENCE,
                reason=(
                    f"existing capture has higher confidence "
                    f"({existing.confidence:.2f} vs {confidence:.2f})"
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        self.captured[field_code] = CapturedField(
            field_code=field_code,
            value=value,
            confidence=confidence,
            turn_index=turn_index,
        )

        logger.info(
            "Field written to ledger",
            extra={
                "field_code": field_code,
                "value": value,
                "confidence": confidence,
                "tool_name": tool_name,
                "was_update": existing is not None,
            },
        )

        return FieldWriteResult(
            field_code=field_code,
            value=value,
            status=WRITE_STATUS_SAVED,
            reason=None,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )

    # ── Legacy extraction path (backward compat) ───────────────────────────────

    def apply_extraction(
        self,
        extracted_answers: list[ExtractedAnswer],
        profile: BaseProfile,
        confidence_threshold: float = 0.7,
    ) -> None:
        """Merge newly extracted answers into the completion ledger.

        LEGACY — kept for backward compatibility with existing tests and scripts.
        New code should use handle_save_text_field / handle_save_enum_field /
        handle_save_quantitative_field directly.

        Rules:
        - ``field_hint`` is matched against profile field codes and descriptions
          using simple substring matching (case-insensitive). This is intentionally
          fuzzy — the LLM returns natural language hints, not codes.
        - For each matched field, we keep only the highest-confidence extraction.
        - Answers below ``confidence_threshold`` are still stored but will show
          as "missing" in ``compute_missing_fields`` until the threshold is met.

        Args:
            extracted_answers:    Answers returned by the LLM tool call.
            profile:              Active profile (provides field definitions).
            confidence_threshold: Minimum confidence to consider a field captured.
        """
        turn_index = len(self.conversation_history)

        has_assistant_msgs = any(m["role"] == "assistant" for m in self.conversation_history)

        for answer in extracted_answers:
            matched_code = self._match_field(answer.field_hint, profile)
            if not matched_code:
                continue

            answer.field_code = matched_code
            field_def = profile.get_field_by_code(matched_code)

            # Validation check: Ensure the field was actually asked about or the user explicitly provided it.
            # In legacy extraction, we accept without asking if confidence >= 0.9 (explicit correction/provision)
            if field_def and not self.captured.get(matched_code):
                asked = False
                if not has_assistant_msgs:
                    asked = True
                else:
                    code_words = set(field_def.code.replace("_", " ").split())
                    desc_words = set(field_def.description.lower().split())
                    stop_words = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "on", "that", "by", "this", "with"}
                    desc_words = desc_words - stop_words

                    for msg in self.conversation_history:
                        if msg["role"] == "assistant":
                            content = msg["content"].lower()
                            if any(w in content for w in code_words if len(w) > 3) or \
                               any(w in content for w in desc_words if len(w) > 4):
                                asked = True
                                break

                # Only reject unasked fields if confidence < 0.9
                if not asked and answer.confidence < 0.9:
                    logger.warning(
                        "Data-integrity issue: Field extracted without being asked",
                        extra={"field_code": matched_code, "value": answer.value}
                    )
                    continue

            # Enum validation
            field_def = profile.get_field_by_code(matched_code)
            if field_def and field_def.enum_values:
                value_lower = answer.value.lower().strip()
                allowed = [v.lower() for v in field_def.enum_values]
                if value_lower not in allowed:
                    normalized = value_lower.replace(" ", "_").replace("-", "_")
                    if normalized in allowed:
                        answer.value = normalized
                    else:
                        logger.debug(
                            "Enum validation failed: value not in allowed list",
                            extra={
                                "field_code": matched_code,
                                "value": answer.value,
                                "allowed": field_def.enum_values[:5],
                            },
                        )
                        answer.confidence = 0.0

            existing = self.captured.get(matched_code)

            should_update = False
            if existing is None:
                should_update = True
            elif answer.confidence >= existing.confidence:
                should_update = True
            elif answer.confidence >= 0.8 and turn_index > existing.turn_index:
                should_update = True

            if should_update and answer.confidence >= confidence_threshold:
                self.captured[matched_code] = CapturedField(
                    field_code=matched_code,
                    value=answer.value,
                    confidence=answer.confidence,
                    turn_index=turn_index,
                )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _match_field(hint: str, profile: BaseProfile) -> str | None:
        """Fuzzy-match a field hint to a profile field code.

        Matching priority:
        1. Exact field code match (e.g. hint == "client_name")
        2. Partial field code match (hint contains or is contained by code)
        3. Partial description match (hint words appear in field description)

        Returns:
            Matched field code, or None if no match found.
        """
        hint_lower = hint.lower().strip()

        # Pass 1: exact code match
        for field in profile.required_fields:
            if field.code == hint_lower:
                return field.code

        # Pass 2: partial code match
        for field in profile.required_fields:
            code = field.code.replace("_", " ")
            if code in hint_lower or hint_lower in code:
                return field.code

        # Pass 3: significant word overlap with description
        hint_words = set(hint_lower.split())
        best_match: tuple[int, str | None] = (0, None)
        for field in profile.required_fields:
            desc_words = set(field.description.lower().split())
            code_words = set(field.code.replace("_", " ").split())
            overlap = len(hint_words & (desc_words | code_words))
            if overlap > best_match[0]:
                best_match = (overlap, field.code)

        if best_match[0] >= 2:
            return best_match[1]

        return None
