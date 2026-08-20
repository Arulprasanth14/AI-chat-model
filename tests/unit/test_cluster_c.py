"""
tests/unit/test_cluster_c.py
──────────────────────────────
Cluster C tests: Proactive Conversation Quality (Bugs 2, 5).

Tests:
  C1 — Gate returns False when required fields are missing
  C2 — Gate returns True when all required fields are present
  C3 — Gate rule string contains the missing field codes
  C4 — ColorSuggestion rejects missing hex
  C5 — ColorSuggestion rejects 3-digit shorthand
  C6 — ColorSuggestion rejects invalid characters
  C7 — ColorSuggestion accepts a valid 6-digit HEX
  C8 — to_stored_string() produces correct format
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.conversation.color_suggestion import ColorSuggestion, parse_color_suggestion
from app.domain.conversation.state import ConversationState, CapturedField
from app.domain.conversation.suggestion_gate import (
    can_generate_suggestions,
    format_gate_rule,
    REQUIRED_BEFORE_SUGGESTIONS,
)
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_profile() -> BaseProfile:
    """Profile containing all three gate-required fields."""
    return BaseProfile(
        profile_id="test_cluster_c",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="brand_personality",
                description="Brand personality",
                required=True,
                input_type="list",
                enum_values=["bold_disruptive", "premium_minimal"],
            ),
            FieldDefinition(
                code="distribution_channels",
                description="Distribution channels",
                required=True,
                input_type="list",
                enum_values=["instagram", "print"],
            ),
            FieldDefinition(
                code="brand_identity_preference",
                description="Brand identity preference",
                required=True,
                enum_values=["customize_this_project", "use_saved_brand_kit"],
            ),
            FieldDefinition(code="brand_name", description="Brand name", required=True),
        ],
        knowledge_namespace="test_ns",
    )


def _make_state_with_fields(profile_id: str, captured: dict[str, str]) -> ConversationState:
    """Create a state with pre-populated captured fields at confidence 0.9."""
    state = ConversationState(profile_id=profile_id)
    for code, value in captured.items():
        state.captured[code] = CapturedField(
            field_code=code, value=value, confidence=0.9, turn_index=0
        )
    return state


# ── C1: Gate returns False when required fields missing ───────────────────────

def test_suggestions_blocked_until_required_fields_present(full_profile: BaseProfile) -> None:
    """Bug 2 fix: gate returns (False, missing_codes) when required context is absent."""
    state = ConversationState(profile_id="test_cluster_c")  # empty — nothing captured

    allowed, missing = can_generate_suggestions(state, full_profile, confidence_threshold=0.7)

    assert allowed is False, "Gate should block suggestions when required fields are missing"
    assert len(missing) > 0, "Gate should return missing field codes"
    # At minimum, all three required fields should be in missing
    for code in REQUIRED_BEFORE_SUGGESTIONS:
        if full_profile.get_field_by_code(code) is not None:
            assert code in missing, f"{code} should be in missing list"


def test_suggestions_blocked_with_partial_fields(full_profile: BaseProfile) -> None:
    """Gate returns False even if 2 of 3 required fields are present."""
    state = _make_state_with_fields("test_cluster_c", {
        "brand_personality": "bold_disruptive",
        "distribution_channels": "instagram",
        # brand_identity_preference is missing
    })

    allowed, missing = can_generate_suggestions(state, full_profile, confidence_threshold=0.7)

    assert allowed is False
    assert "brand_identity_preference" in missing


# ── C2: Gate returns True when all required fields present ────────────────────

def test_suggestions_allowed_after_required_fields(full_profile: BaseProfile) -> None:
    """Bug 2 fix: gate returns (True, []) when all required context is captured."""
    state = _make_state_with_fields("test_cluster_c", {
        "brand_personality": "bold_disruptive",
        "distribution_channels": "instagram",
        "brand_identity_preference": "customize_this_project",
    })

    allowed, missing = can_generate_suggestions(state, full_profile, confidence_threshold=0.7)

    assert allowed is True, "Gate should allow suggestions when all required fields are captured"
    assert missing == [], f"No fields should be missing, got: {missing}"


# ── C3: Gate rule string ──────────────────────────────────────────────────────

def test_gate_rule_contains_missing_field_codes() -> None:
    """format_gate_rule() should include the missing field codes."""
    missing = ["brand_personality", "distribution_channels"]
    rule = format_gate_rule(missing)

    assert "brand_personality" in rule
    assert "distribution_channels" in rule
    # Should contain language about NOT suggesting
    assert any(word in rule.lower() for word in ["not", "must not", "do not"]), (
        f"Gate rule should contain prohibition language. Got: {rule!r}"
    )


# ── C4–C7: ColorSuggestion HEX validation ────────────────────────────────────

def test_color_suggestion_rejects_missing_hex() -> None:
    """Bug 5 fix: ColorSuggestion should reject a call with no hex field."""
    with pytest.raises(ValidationError):
        ColorSuggestion(name="Deep Navy", rationale="Suits the brand.")  # type: ignore
        # hex is missing — should raise ValidationError


def test_color_suggestion_rejects_three_digit_shorthand() -> None:
    """Bug 5 fix: 3-digit HEX shorthand (#RGB) should be rejected."""
    with pytest.raises(ValidationError) as exc_info:
        ColorSuggestion(name="Navy", hex="#1A2", rationale="Suits the brand.")

    assert "RRGGBB" in str(exc_info.value) or "6" in str(exc_info.value) or "Invalid" in str(exc_info.value)


def test_color_suggestion_rejects_invalid_characters() -> None:
    """Bug 5 fix: HEX code with invalid characters should be rejected."""
    with pytest.raises(ValidationError):
        ColorSuggestion(name="Navy", hex="#GGHHII", rationale="Suits the brand.")


def test_color_suggestion_rejects_missing_hash() -> None:
    """Bug 5 fix: HEX code without leading # should be rejected."""
    with pytest.raises(ValidationError):
        ColorSuggestion(name="Navy", hex="1A2B3C", rationale="Suits the brand.")


def test_color_suggestion_accepts_valid_hex() -> None:
    """Bug 5 fix: A properly formatted 6-digit HEX code should be accepted."""
    color = ColorSuggestion(name="Deep Navy", hex="#1A2B3C", rationale="Cool and professional.")
    assert color.hex == "#1A2B3C"  # should be uppercased


def test_color_suggestion_uppercases_valid_hex() -> None:
    """Valid lowercase HEX should be normalised to uppercase."""
    color = ColorSuggestion(name="Warm Red", hex="#ff5733", rationale="Energetic.")
    assert color.hex == "#FF5733"


# ── C8: to_stored_string format ───────────────────────────────────────────────

def test_color_suggestion_stored_string_format() -> None:
    """to_stored_string() should produce '{name} ({hex}): {rationale}'."""
    color = ColorSuggestion(name="Deep Navy", hex="#1A2B3C", rationale="Cool and professional.")
    stored = color.to_stored_string()

    assert "Deep Navy" in stored
    assert "#1A2B3C" in stored
    assert "Cool and professional" in stored


# ── C9: parse_color_suggestion helper ────────────────────────────────────────

def test_parse_color_suggestion_validates_from_dict() -> None:
    """parse_color_suggestion() should accept a valid dict and reject an invalid one."""
    valid = parse_color_suggestion({"name": "Coral", "hex": "#FF6B6B", "rationale": "Warm and friendly."})
    assert valid.hex == "#FF6B6B"

    with pytest.raises((ValidationError, ValueError)):
        parse_color_suggestion({"name": "Bad", "hex": "not-a-hex", "rationale": "..."})
