"""
tests/unit/test_cluster_d.py
──────────────────────────────
Cluster D tests: Structured Input UI (Bugs 11, 12, 13, 14, 15, 16).

Tests:
  D1 — get_field_spec returns multi_select for list-type fields
  D2 — get_field_spec returns single_select for enum-type fields
  D3 — get_field_spec returns text for plain text fields
  D4 — get_field_spec returns file_upload for existing_assets (override)
  D5 — get_field_spec returns aspect_ratio_picker for aspect_ratio (override)
  D6 — FieldSpec model serialises correctly (model_dump roundtrip)
  D7 — direct_field_write persists at confidence 1.0 (no LLM)
  D8 — direct_field_write is idempotent for same value
  D9 — next_field_spec returns None when session is complete
"""
from __future__ import annotations

import pytest

from app.domain.conversation.color_suggestion import ColorSuggestion
from app.domain.conversation.field_spec import FieldSpec
from app.domain.conversation.field_spec_registry import get_field_spec
from app.domain.conversation.state import ConversationState, CapturedField
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_ui_profile() -> BaseProfile:
    return BaseProfile(
        profile_id="test_cluster_d",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="brand_personality",
                description="Brand personality traits",
                required=True,
                input_type="list",
                enum_values=["bold_disruptive", "premium_minimal", "warm_approachable"],
            ),
            FieldDefinition(
                code="brand_name",
                description="Brand name",
                required=True,
                input_type="text",
            ),
            FieldDefinition(
                code="content_type",
                description="Type of content",
                required=True,
                input_type="text",
                enum_values=["static_post", "reel", "carousel"],
            ),
            FieldDefinition(
                code="existing_assets",
                description="Logo and brand files",
                required=False,
                input_type="file_upload",
            ),
            FieldDefinition(
                code="aspect_ratio",
                description="Target aspect ratio",
                required=False,
                input_type="text",
            ),
        ],
        knowledge_namespace="test_ns",
    )


# ── D1: multi_select for list fields ─────────────────────────────────────────

def test_get_field_spec_multi_select_for_list_field(full_ui_profile: BaseProfile) -> None:
    """Bug 11 fix: list-type fields with enum_values should return multi_select FieldSpec."""
    field_def = full_ui_profile.get_field_by_code("brand_personality")
    assert field_def is not None

    spec = get_field_spec(field_def)

    assert spec.input_type == "multi_select", (
        f"List field with enum_values should return multi_select, got: {spec.input_type}"
    )
    assert spec.options == field_def.enum_values, "Options should match field enum_values"
    assert spec.field_code == "brand_personality"


# ── D2: single_select for enum fields ────────────────────────────────────────

def test_get_field_spec_single_select_for_enum_field(full_ui_profile: BaseProfile) -> None:
    """Bug 14 fix: text-type fields with enum_values should return single_select."""
    field_def = full_ui_profile.get_field_by_code("content_type")
    assert field_def is not None

    spec = get_field_spec(field_def)

    assert spec.input_type == "single_select", (
        f"Enum field should return single_select, got: {spec.input_type}"
    )
    assert spec.options is not None
    assert "static_post" in spec.options


# ── D3: text for plain text fields ───────────────────────────────────────────

def test_get_field_spec_text_for_plain_text_field(full_ui_profile: BaseProfile) -> None:
    """Plain text fields without enum_values should default to text input."""
    field_def = full_ui_profile.get_field_by_code("brand_name")
    assert field_def is not None

    spec = get_field_spec(field_def)

    assert spec.input_type == "text"
    assert spec.options is None


# ── D4: file_upload override for existing_assets ─────────────────────────────

def test_get_field_spec_file_upload_for_existing_assets(full_ui_profile: BaseProfile) -> None:
    """Bug 13 fix: existing_assets should always use file_upload (hard override)."""
    field_def = full_ui_profile.get_field_by_code("existing_assets")
    assert field_def is not None

    spec = get_field_spec(field_def)

    assert spec.input_type == "file_upload", (
        f"existing_assets should always be file_upload, got: {spec.input_type}"
    )


# ── D5: aspect_ratio_picker override ─────────────────────────────────────────

def test_get_field_spec_aspect_ratio_picker(full_ui_profile: BaseProfile) -> None:
    """Bug 12 fix: aspect_ratio field should use aspect_ratio_picker (hard override)."""
    field_def = full_ui_profile.get_field_by_code("aspect_ratio")
    assert field_def is not None

    spec = get_field_spec(field_def)

    assert spec.input_type == "aspect_ratio_picker", (
        f"aspect_ratio should return aspect_ratio_picker, got: {spec.input_type}"
    )
    # Should include standard aspect ratios
    assert spec.options is not None
    assert "9:16" in spec.options or "16:9" in spec.options


# ── D6: FieldSpec model_dump roundtrip ───────────────────────────────────────

def test_field_spec_model_dump_roundtrip() -> None:
    """FieldSpec should serialise cleanly for the API response."""
    spec = FieldSpec(
        field_code="brand_personality",
        input_type="multi_select",
        options=["bold_disruptive", "premium_minimal"],
        label="Brand Personality",
    )
    dumped = spec.model_dump()

    assert dumped["field_code"] == "brand_personality"
    assert dumped["input_type"] == "multi_select"
    assert dumped["options"] == ["bold_disruptive", "premium_minimal"]
    assert dumped["label"] == "Brand Personality"

    # Deserialise back
    restored = FieldSpec.model_validate(dumped)
    assert restored == spec


# ── D7: direct_field_write persists at confidence 1.0 ────────────────────────

def test_direct_field_write_at_max_confidence(full_ui_profile: BaseProfile) -> None:
    """Bug 16 fix: UI-selected values should persist at confidence 1.0."""
    state = ConversationState(profile_id="test_cluster_d")

    # Simulate what /direct_field_write does
    result = state.handle_save_text_field(
        field_code="brand_name",
        value="Luminary",
        confidence=1.0,
        profile=full_ui_profile,
        confidence_threshold=0.0,
    )

    assert result.status == "saved"
    assert state.captured["brand_name"].confidence == 1.0, (
        "Direct UI write must persist at confidence 1.0"
    )


# ── D8: direct_field_write is idempotent ─────────────────────────────────────

def test_direct_field_write_idempotent(full_ui_profile: BaseProfile) -> None:
    """Writing the same value twice via direct write should not corrupt state."""
    state = ConversationState(profile_id="test_cluster_d")

    state.handle_save_text_field(
        field_code="brand_name",
        value="Luminary",
        confidence=1.0,
        profile=full_ui_profile,
        confidence_threshold=0.0,
    )

    # Write the same value again at same confidence
    result2 = state.handle_save_text_field(
        field_code="brand_name",
        value="Luminary",
        confidence=1.0,
        profile=full_ui_profile,
        confidence_threshold=0.0,
    )

    # Both writes should succeed (>=confidence means update is allowed)
    assert state.captured["brand_name"].value == "Luminary", "Value should remain 'Luminary'"
    assert state.captured["brand_name"].confidence == 1.0


# ── D9: next_field_spec returns null when complete ───────────────────────────

def test_compute_missing_fields_empty_when_complete(full_ui_profile: BaseProfile) -> None:
    """next_field_spec returns None when all required fields are captured."""
    state = ConversationState(profile_id="test_cluster_d")

    # Fill all required fields
    for field_def in full_ui_profile.required_fields:
        if field_def.required:
            state.captured[field_def.code] = CapturedField(
                field_code=field_def.code,
                value="some_value",
                confidence=0.95,
                turn_index=0,
            )

    missing = state.compute_missing_fields(full_ui_profile, confidence_threshold=0.7)
    assert len(missing) == 0, (
        f"compute_missing_fields should return empty when all required fields captured, got: {missing}"
    )
    # This mirrors what /next_field_spec checks — if missing is empty, it returns null


# ── D10: distribution_channels uses UI-friendly labels ───────────────────────

def test_distribution_channels_uses_human_readable_labels() -> None:
    """Bug 11 fix: the override for distribution_channels should use human-readable labels."""
    from app.domain.conversation.field_spec_registry import _FIELD_SPEC_OVERRIDES

    if "distribution_channels" in _FIELD_SPEC_OVERRIDES:
        spec = _FIELD_SPEC_OVERRIDES["distribution_channels"]
        assert spec.options is not None
        # Should have human-readable labels (not underscored codes)
        assert "Instagram" in spec.options or "instagram" in spec.options[0]
        assert spec.input_type == "multi_select"
