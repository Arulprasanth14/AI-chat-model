"""
app/domain/conversation/field_spec.py
────────────────────────────────────────
FieldSpec: describes how a given profile field should be rendered in the UI.

This is the backend contract that drives the frontend input control renderer.
Instead of every field defaulting to a free-text box, the frontend calls
GET /conversation/session/{session_id}/next_field_spec and receives a FieldSpec
that specifies exactly which control to render and what options to show.

Fixes Bugs 11 (platform choices), 12 (aspect ratio picker), 13 (upload button),
14 (over-reliance on free text), 15 (no context-aware input control).

input_type values:
    text             — Free-text input (default for unstructured fields)
    multi_select     — Multi-value selection (maps to list-type fields)
    single_select    — Single value from a fixed list (maps to enum fields)
    file_upload      — File upload button (for logo/asset fields)
    aspect_ratio_picker — Aspect-ratio visual picker (special UI widget)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    """Describes how a profile field should be rendered in the frontend.

    Attributes:
        field_code:  The profile field code this spec targets.
        input_type:  The UI control to render.
        options:     Allowed values (for select/multi-select types).
        label:       Human-readable display label for the field.
    """

    field_code: str = Field(..., description="Profile field code this spec targets")
    input_type: Literal[
        "text",
        "multi_select",
        "single_select",
        "file_upload",
        "aspect_ratio_picker",
    ] = Field(  # type: ignore
        default="text",
        description="UI control type to render for this field",
    )
    options: list[str] | None = Field(
        default=None,
        description="Allowed values (for multi_select and single_select types)",
    )
    label: str = Field(
        default="",
        description="Human-readable display label for the field",
    )
