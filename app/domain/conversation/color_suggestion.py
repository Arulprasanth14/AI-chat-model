"""
app/domain/conversation/color_suggestion.py
────────────────────────────────────────────
Pydantic model for structured color suggestions.

Bug 5 fix: Without a structured output schema and HEX validation, the LLM
generates color suggestions as free text, omitting HEX codes or inventing
invalid ones. This model enforces HEX presence and format before any color
suggestion is returned to the user.

Used by the save_color_suggestion Phase A tool handler in the orchestrator.
The orchestrator rejects tool calls where hex is missing or malformed and
instructs Phase B to re-ask for a corrected color with a valid HEX code.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator


class ColorSuggestion(BaseModel):
    """A validated color suggestion with mandatory HEX code.

    Attributes:
        name:       Human-readable color name (e.g. 'Deep Navy').
        hex:        6-digit HEX code in #RRGGBB format (e.g. '#1A2B3C').
        rationale:  Why this color suits the brand (1–2 sentences).
    """

    name: str = Field(..., min_length=1, description="Human-readable color name")
    hex: str = Field(..., description="6-digit HEX code in #RRGGBB format")
    rationale: str = Field(
        ...,
        min_length=1,
        description="Why this color suits the brand (1-2 sentences)",
    )

    @field_validator("hex")
    @classmethod
    def validate_hex(cls, v: str) -> str:
        """Enforce strict 6-digit HEX format (#RRGGBB).

        3-digit shorthand (#RGB) is rejected to prevent ambiguity.
        Leading '#' is required.

        Args:
            v: Raw hex string from LLM.

        Returns:
            Uppercased, validated HEX string.

        Raises:
            ValueError: If the HEX code is missing, malformed, or 3-digit.
        """
        v = v.strip()
        if not re.match(r"^#[0-9A-Fa-f]{6}$", v):
            raise ValueError(
                f"Invalid HEX code {v!r}. Must be exactly 6 hex digits with leading '#' "
                f"(e.g. '#FF5733'). 3-digit shorthand (#RGB) is not accepted."
            )
        return v.upper()

    def to_stored_string(self) -> str:
        """Return a compact string suitable for storage in the captured field value.

        Format: "{name} ({hex}): {rationale}"

        This is the format written to the ``custom_colors_and_fonts`` field
        in the completion ledger.
        """
        return f"{self.name} ({self.hex}): {self.rationale}"


def parse_color_suggestion(arguments: dict) -> ColorSuggestion:
    """Parse and validate a color suggestion from LLM tool call arguments.

    Args:
        arguments: Parsed JSON arguments from the save_color_suggestion tool call.

    Returns:
        Validated ColorSuggestion.

    Raises:
        ValueError: If required fields are missing or HEX is invalid.
    """
    return ColorSuggestion.model_validate(arguments)
