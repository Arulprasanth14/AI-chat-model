"""
app/domain/conversation/field_spec_registry.py
───────────────────────────────────────────────
Maps FieldDefinition → FieldSpec using a derivation-first, override-second strategy.

Logic:
  1. If the field_code has a hardcoded override (e.g. aspect_ratio_picker for
     aspect_ratio, file_upload for existing_assets), use that override.
  2. Else, derive the FieldSpec from field_def.input_type + field_def.enum_values:
       - input_type="list"  + enum_values → multi_select
       - input_type="list"  + no enum_values → text (with hint that multiple OK)
       - input_type="text"  + enum_values → single_select
       - input_type="text"  + no enum_values → text
       - input_type="file_upload" → file_upload

Fixes Bugs 11 (no platform choices), 12 (no aspect ratio picker),
13 (no logo upload button), 14 (over-reliance on free text),
15 (no context-aware input control).
"""
from __future__ import annotations

from app.domain.conversation.field_spec import FieldSpec
from app.project_profiles.base_profile import FieldDefinition


# ── Hardcoded overrides for fields that need special UI treatment ──────────────
#
# These take precedence over the derivation logic below. Add new overrides here
# for any field that cannot be correctly inferred from its input_type alone.

_FIELD_SPEC_OVERRIDES: dict[str, FieldSpec] = {
    # Bug 12: aspect ratio is a visual picker, not a plain select
    "aspect_ratio": FieldSpec(
        field_code="aspect_ratio",
        input_type="aspect_ratio_picker",
        options=["1:1", "4:5", "9:16", "16:9"],
        label="Aspect Ratio",
    ),
    # Bug 13: existing_assets is always a file upload (logo, brand kit, etc.)
    "existing_assets": FieldSpec(
        field_code="existing_assets",
        input_type="file_upload",
        label="Upload Brand Assets / Logo",
    ),
}


def get_field_spec(field_def: FieldDefinition) -> FieldSpec:
    """Return the FieldSpec for the given FieldDefinition.

    Args:
        field_def: A profile field definition (from BaseProfile.required_fields).

    Returns:
        FieldSpec describing how this field should be rendered in the UI.
    """
    # 1. Check hardcoded overrides first
    if field_def.code in _FIELD_SPEC_OVERRIDES:
        return _FIELD_SPEC_OVERRIDES[field_def.code]

    # 2. Derive from input_type + enum_values
    if field_def.input_type == "file_upload":
        return FieldSpec(
            field_code=field_def.code,
            input_type="file_upload",
            label=field_def.code.replace("_", " ").title(),
        )

    if field_def.input_type == "list":
        if field_def.enum_values:
            return FieldSpec(
                field_code=field_def.code,
                input_type="multi_select",
                options=field_def.enum_values,
                label=field_def.code.replace("_", " ").title(),
            )
        # List field with no enum — free text that accepts comma-joined values
        return FieldSpec(
            field_code=field_def.code,
            input_type="text",
            label=field_def.code.replace("_", " ").title(),
        )

    # Single-value enum field
    if field_def.enum_values:
        return FieldSpec(
            field_code=field_def.code,
            input_type="single_select",
            options=field_def.enum_values,
            label=field_def.code.replace("_", " ").title(),
        )

    # Default: free text
    return FieldSpec(
        field_code=field_def.code,
        input_type="text",
        label=field_def.code.replace("_", " ").title(),
    )
