"""
app/domain/conversation/field_set_loader.py
──────────────────────────────────────────────
Loads a field set YAML file and converts it into a list of FieldDefinition
objects compatible with BaseProfile.required_fields.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.project_profiles.base_profile import FieldDefinition

logger = logging.getLogger(__name__)


def load_field_set(yaml_path: Path) -> list[FieldDefinition] | None:
    """Load a field set YAML into FieldDefinition objects.

    Args:
        yaml_path: Path to the field set YAML file.

    Returns:
        List of FieldDefinition objects, or None if loading fails.
    """
    if not yaml_path.exists():
        logger.warning("Field set YAML not found", extra={"path": str(yaml_path)})
        return None

    try:
        with yaml_path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error(
            "Failed to parse field set YAML",
            extra={"path": str(yaml_path), "error": str(exc)},
        )
        return None

    raw_fields = data.get("fields", [])
    fields: list[FieldDefinition] = []

    for f in raw_fields:
        code = f.get("code")
        if not code:
            continue

        # Build a richer description so the LLM understands the field's context
        # and can calibrate confidence more accurately.
        kind = f.get("kind", "text")
        options = f.get("options", [])
        required = f.get("required", True)
        section_name = f.get("section_name", "")

        desc_parts: list[str] = []

        # Section context helps the LLM understand what topic this belongs to
        if section_name:
            desc_parts.append(f"Section: {section_name}.")

        # Kind context: distinguish radio (single), checkbox (multi), text, file
        kind_label = {
            "radio": "Choose one option",
            "checkbox": "Choose one or more options",
            "text": "Free-text answer",
            "file": "File upload",
        }.get(kind, kind)
        desc_parts.append(f"Type: {kind_label}.")

        if options:
            # Show all options (not just 5) so the LLM knows the full enum
            opts_str = ", ".join(f"'{o['label']}'" for o in options)
            desc_parts.append(f"Options: {opts_str}.")
            if kind == "radio":
                desc_parts.append(
                    "Extract the matching option value exactly. "
                    "Confidence should be 0.95+ only if the user stated one of these options directly."
                )
            elif kind == "checkbox":
                desc_parts.append(
                    "Multiple selections allowed. Extract all that the user mentioned."
                )
        elif kind == "text":
            desc_parts.append(
                "Extract the user's exact words. Confidence should reflect how directly "
                "they answered — 0.9+ for a clear direct statement, 0.6-0.8 for inferred."
            )

        fields.append(
            FieldDefinition(
                code=code,
                description=" ".join(desc_parts),
                required=required,
            )
        )

    return fields
