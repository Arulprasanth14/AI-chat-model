"""
tests/unit/test_cluster_b.py
──────────────────────────────
Cluster B tests: Field Extraction Accuracy (Bugs 6, 7, 8, 9).

Tests:
  B1 — List field merge preserves earlier values across turns
  B2 — List field deduplication (same value added twice → stored once)
  B3 — Non-list fields still replace (regression guard)
  B4 — brand_identity_preference enum distinct from brand_personality enum
  B5 — hybrid_notes is captured when brand_identity_preference = customize_this_project
"""
from __future__ import annotations

import pytest

from app.domain.conversation.state import ConversationState
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def list_field_profile() -> BaseProfile:
    """Profile with one list-type field (brand_personality) and one text field."""
    return BaseProfile(
        profile_id="test_cluster_b",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="brand_personality",
                description="Brand personality traits",
                required=True,
                input_type="list",
                enum_values=["bold_disruptive", "premium_minimal", "warm_approachable", "playful_youthful"],
            ),
            FieldDefinition(
                code="brand_name",
                description="The brand name",
                required=True,
                input_type="text",
            ),
            FieldDefinition(
                code="deliverables",
                description="Deliverables",
                required=True,
                input_type="list",
            ),
            FieldDefinition(
                code="distribution_channels",
                description="Distribution channels",
                required=True,
                input_type="list",
                enum_values=["instagram", "linkedin", "print", "ooh"],
            ),
        ],
        knowledge_namespace="test_ns",
    )


# ── B1: List field additive merge ─────────────────────────────────────────────

@pytest.mark.parametrize("field_code,first_value,second_value,both_present", [
    ("brand_personality", "bold_disruptive", "premium_minimal", True),
    ("deliverables", "10 social media graphics", "1 hero video", True),
    ("distribution_channels", "instagram", "print", True),
])
def test_list_field_merge_preserves_earlier_values(
    list_field_profile: BaseProfile,
    field_code: str,
    first_value: str,
    second_value: str,
    both_present: bool,
) -> None:
    """Bug 6/7/8 fix: A second tool call for a list field appends rather than replaces.

    This tests the exact bug: on turn 1 the user says 'bold and disruptive', on turn 2
    they add 'premium and minimal'. Without the fix, only 'premium_minimal' is stored.
    With the fix, both are stored.
    """
    state = ConversationState(profile_id="test_cluster_b")

    # Turn 1: first value
    result1 = state.handle_save_text_field(
        field_code=field_code,
        value=first_value,
        confidence=0.9,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )
    assert result1.status == "saved", f"First save should succeed, got: {result1.status}"
    assert first_value in state.captured[field_code].value

    # Turn 2: second value (additive merge, not replace)
    result2 = state.handle_save_text_field(
        field_code=field_code,
        value=second_value,
        confidence=0.85,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )
    assert result2.status == "saved", f"Second save should succeed, got: {result2.status}"

    stored = state.captured[field_code].value

    if both_present:
        assert first_value in stored, (
            f"First value '{first_value}' should still be in merged value '{stored}'"
        )
        assert second_value in stored, (
            f"Second value '{second_value}' should be in merged value '{stored}'"
        )


def test_list_field_no_duplicates(list_field_profile: BaseProfile) -> None:
    """Bug 6 fix: Adding the same value twice to a list field should store it only once."""
    state = ConversationState(profile_id="test_cluster_b")

    state.handle_save_text_field(
        field_code="brand_personality",
        value="bold_disruptive",
        confidence=0.9,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )

    # Add the same value again
    state.handle_save_text_field(
        field_code="brand_personality",
        value="bold_disruptive",
        confidence=0.85,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )

    stored = state.captured["brand_personality"].value
    items = [v.strip() for v in stored.split(",") if v.strip()]
    assert items.count("bold_disruptive") == 1, (
        f"'bold_disruptive' should appear exactly once, got: {items}"
    )


# ── B2: Multi-value merge in one call ─────────────────────────────────────────

def test_list_field_multi_value_single_call(list_field_profile: BaseProfile) -> None:
    """Comma-joined values from a single LLM call should all be stored."""
    state = ConversationState(profile_id="test_cluster_b")

    # LLM extracts multiple values in one tool call (comma-joined)
    state.handle_save_text_field(
        field_code="brand_personality",
        value="bold_disruptive, premium_minimal",
        confidence=0.9,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )

    stored = state.captured["brand_personality"].value
    assert "bold_disruptive" in stored
    assert "premium_minimal" in stored


# ── B3: Non-list fields still replace (regression guard) ──────────────────────

def test_non_list_field_replaces_on_update(list_field_profile: BaseProfile) -> None:
    """Regression: text-type fields should still overwrite (not merge) on confidence >= existing."""
    state = ConversationState(profile_id="test_cluster_b")

    state.handle_save_text_field(
        field_code="brand_name",
        value="Luminary",
        confidence=0.85,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )

    # Higher confidence update should replace
    state.handle_save_text_field(
        field_code="brand_name",
        value="Luminary Labs",
        confidence=0.95,
        profile=list_field_profile,
        confidence_threshold=0.7,
    )

    assert state.captured["brand_name"].value == "Luminary Labs", (
        "text-type field should be replaced by higher-confidence value, not merged"
    )


# ── B4: brand_identity_preference vs brand_personality distinction ─────────────

def test_brand_identity_profile_has_structural_enum(list_field_profile: BaseProfile) -> None:
    """Bug 9 fix: brand_identity_preference should have structural enum values, distinct from personality."""
    # The real picasso profile
    from app.project_profiles.base_profile import BaseProfile as BP
    import yaml
    from pathlib import Path

    profile_path = Path("app/project_profiles/picasso_fusion/profile.yaml")
    if not profile_path.exists():
        pytest.skip("Profile YAML not found — skipping integration-level test")

    profile = BP.from_yaml(profile_path)
    bip_field = profile.get_field_by_code("brand_identity_preference")
    bp_field = profile.get_field_by_code("brand_personality")

    assert bip_field is not None, "brand_identity_preference field must exist"
    assert bp_field is not None, "brand_personality field must exist"

    # The two fields should have DIFFERENT enum_values (structural vs personality)
    bip_enums = set(bip_field.enum_values or [])
    bp_enums = set(bp_field.enum_values or [])
    overlap = bip_enums & bp_enums
    assert len(overlap) == 0, (
        f"brand_identity_preference and brand_personality should have non-overlapping enums. "
        f"Overlap: {overlap}"
    )

    # brand_identity_preference should have structural values
    structural_values = {"use_saved_brand_kit", "upload_brand_guidelines", "upload_logo_assets", "customize_this_project"}
    assert structural_values.issubset(bip_enums), (
        f"brand_identity_preference should contain structural values. Got: {bip_enums}"
    )


# ── B5: hybrid_notes captured independently ───────────────────────────────────

def test_hybrid_notes_captured_as_optional_field() -> None:
    """Bug 9 fix: hybrid_notes should be capturable as a separate optional field."""
    profile = BaseProfile(
        profile_id="test_hybrid",
        persona_prompt="You are a test assistant.",
        required_fields=[
            FieldDefinition(
                code="brand_identity_preference",
                description="Structural brand identity choice",
                required=True,
                enum_values=["customize_this_project", "use_saved_brand_kit"],
            ),
            FieldDefinition(
                code="hybrid_notes",
                description="Notes for customized approach",
                required=False,
                input_type="text",
            ),
        ],
        knowledge_namespace="test_ns",
    )
    state = ConversationState(profile_id="test_hybrid")

    # Save brand_identity_preference
    result1 = state.handle_save_enum_field(
        field_code="brand_identity_preference",
        value="customize_this_project",
        confidence=0.95,
        profile=profile,
        confidence_threshold=0.7,
    )
    assert result1.status == "saved"

    # Save hybrid_notes separately
    result2 = state.handle_save_text_field(
        field_code="hybrid_notes",
        value="keep existing colors but update logo style",
        confidence=0.90,
        profile=profile,
        confidence_threshold=0.7,
    )
    assert result2.status == "saved"

    assert "brand_identity_preference" in state.captured
    assert "hybrid_notes" in state.captured
    assert "keep existing colors" in state.captured["hybrid_notes"].value
