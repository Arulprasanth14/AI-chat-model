"""
app/domain/conversation/template_resolver.py
──────────────────────────────────────────────
Pure deterministic resolver for multi-template brief configurations.

Scans the available field-set YAMLs in a given vertical folder and finds the
best matching template based on a natural-language template hint. Matching
is purely substring-based against the templateNameMatch values defined in
each spec — no embeddings or LLMs are involved.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ResolvedTemplate:
    vertical: str
    template_key: str
    template_label: str
    field_set_path: Path


def resolve(
    vertical_hint: str | None,
    template_hint: str | None,
    field_sets_root: Path,
) -> ResolvedTemplate | None:
    """Resolve a vertical and template hint to a specific field set.

    Args:
        vertical_hint:   The vertical extracted from the session (e.g. 'restaurant').
                         Must exactly match a folder name in field_sets_root.
        template_hint:   Free-text hint of what the user wants to create
                         (e.g. 'I need a static post').
        field_sets_root: Path to the root field_sets directory.

    Returns:
        ResolvedTemplate if a confident match is found, otherwise None.
        If None, the session should fall back to the generic profile defaults
        and ask the user for clarification.
    """
    if not vertical_hint or not template_hint:
        return None

    vertical_dir = field_sets_root / vertical_hint
    if not vertical_dir.exists() or not vertical_dir.is_dir():
        logger.debug("Vertical directory not found", extra={"vertical": vertical_hint})
        return None

    hint_lower = template_hint.lower().strip()
    best_match: ResolvedTemplate | None = None
    best_match_len = -1

    for yaml_path in vertical_dir.glob("*.yaml"):
        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                data: dict[str, Any] = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.warning(
                "Failed to parse field set YAML",
                extra={"path": str(yaml_path), "error": str(exc)},
            )
            continue

        template_key = str(data.get("template_key", ""))
        if not template_key:
            continue
        template_label = str(data.get("template_label", template_key))
        name_matches: list[str] = data.get("template_name_match", [])

        # Substring match against any of the registered aliases
        for match_str in name_matches:
            match_lower = match_str.lower().strip()
            # If the user's hint contains the match string (or vice versa),
            # we consider it a hit. We prefer the longest matching string
            # if multiple templates match.
            if match_lower in hint_lower or hint_lower in match_lower:
                if len(match_lower) > best_match_len:
                    best_match_len = len(match_lower)
                    best_match = ResolvedTemplate(
                        vertical=vertical_hint,
                        template_key=template_key,
                        template_label=template_label,
                        field_set_path=yaml_path,
                    )

    if best_match:
        logger.info(
            "Resolved template",
            extra={
                "vertical": vertical_hint,
                "hint": template_hint,
                "resolved_key": best_match.template_key,
            },
        )
    else:
        logger.debug(
            "No template match found",
            extra={"vertical": vertical_hint, "hint": template_hint},
        )

    return best_match
