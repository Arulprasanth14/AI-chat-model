"""
app/domain/conversation/brief_renderer.py
──────────────────────────────────────────
Deterministic brief summary renderer.

This module produces a fixed, template-driven markdown summary of a captured
brief session. It is the single source of truth for brief display — used both
for the AI's "show me the brief" responses and the UI's "View Full Brief" modal.

Design principles:
  - Zero LLM involvement: output is 100% deterministic for a given set of
    captured fields. The same inputs always produce the same output.
  - Reads section/question labels from the resolved field-set YAML (V1 format).
  - Groups captured fields by section_order, matching V1's section structure.
  - Unknown fields (no matching YAML entry) are rendered in an appendix.
  - Missing required fields are marked [Not provided] rather than omitted.
  - Missing optional fields are silently omitted.

Usage:
    from app.domain.conversation.brief_renderer import BriefRenderer, render_brief

    # Simple function call
    summary = render_brief(
        captured=state.captured,
        field_set_yaml_path=Path(".../restaurant_cafe_static_post.yaml"),
        profile=active_profile,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from app.domain.conversation.state import CapturedField
from app.project_profiles.base_profile import BaseProfile

logger = logging.getLogger(__name__)


# ── Internal data structures ───────────────────────────────────────────────────

class _FieldEntry:
    """One rendered field entry within a section."""
    def __init__(
        self,
        code: str,
        question: str,
        value: str | None,
        confidence: float,
        required: bool,
        section_order: int,
        section_name: str,
        section_field_order: int,
    ) -> None:
        self.code = code
        self.question = question
        self.value = value
        self.confidence = confidence
        self.required = required
        self.section_order = section_order
        self.section_name = section_name
        self.section_field_order = section_field_order


# ── Public API ─────────────────────────────────────────────────────────────────

def render_brief(
    captured: dict[str, CapturedField],
    field_set_yaml_path: Path | None,
    profile: BaseProfile,
    include_confidence: bool = False,
) -> str:
    """Render a deterministic markdown summary of captured brief fields.

    Args:
        captured:             The state's captured fields dict
                              {field_code: CapturedField}.
        field_set_yaml_path:  Path to the resolved field-set YAML (e.g.
                              restaurant_cafe_static_post.yaml).
                              If None, falls back to the base-profile field list.
        profile:              Active BaseProfile (for field definitions and
                              required/optional status when no YAML is resolved).
        include_confidence:   If True, appends confidence % to each value line.
                              Useful for debugging; False for client-facing output.

    Returns:
        Markdown-formatted brief summary string. Sections match V1's order.
        Same input always produces the same output (deterministic).
    """
    renderer = BriefRenderer(
        captured=captured,
        field_set_yaml_path=field_set_yaml_path,
        profile=profile,
        include_confidence=include_confidence,
    )
    return renderer.render()


class BriefRenderer:
    """Builds the brief summary from the captured state + field-set YAML.

    Instantiate once per render call; do not reuse across sessions.
    """

    SECTION_ICON = {
        "Post Purpose & Product": "🎯",
        "Goal & Audience": "👥",
        "Offer": "🏷️",
        "Copy": "✍️",
        "Design Requirements": "🎨",
        "Images & Assets": "📸",
        "Brand Assets": "🏷️",
        "Style & References": "✨",
        "Publishing & Language": "📱",
        "Additional Notes": "📝",
    }

    def __init__(
        self,
        captured: dict[str, CapturedField],
        field_set_yaml_path: Path | None,
        profile: BaseProfile,
        include_confidence: bool = False,
    ) -> None:
        self._captured = captured
        self._yaml_path = field_set_yaml_path
        self._profile = profile
        self._include_confidence = include_confidence

    def render(self) -> str:
        """Build and return the full markdown summary."""
        if self._yaml_path and self._yaml_path.exists():
            return self._render_from_field_set()
        return self._render_from_base_profile()

    # ── Rendering from resolved field-set YAML (structured, V1-compatible) ────

    def _render_from_field_set(self) -> str:
        """Render using the structured field-set YAML (V1 section order)."""
        if not self._yaml_path:
            return self._render_from_base_profile()

        try:
            raw: dict[str, Any] = yaml.safe_load(
                self._yaml_path.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning(
                "Brief renderer: could not read field-set YAML, falling back",
                extra={"path": str(self._yaml_path), "error": str(exc)},
            )
            return self._render_from_base_profile()

        template_label = raw.get("template_label", raw.get("template_key", ""))
        vertical = raw.get("vertical", "")

        # Build a lookup: field_code → entry with section info
        fields_meta: dict[str, _FieldEntry] = {}
        for f in raw.get("fields", []):
            code = f.get("code")
            if not code:
                continue

            # Resolve the question label: prefer a human label from the
            # JSON spec if present; else construct from section + code
            question = f.get("question") or f.get("section_name") or code

            fields_meta[code] = _FieldEntry(
                code=code,
                question=question,
                value=self._captured.get(code, CapturedField(
                    field_code=code, value="", confidence=0.0, turn_index=0
                )).value if code in self._captured else None,
                confidence=self._captured[code].confidence if code in self._captured else 0.0,
                required=f.get("required", True),
                section_order=f.get("section_order", 99),
                section_name=f.get("section_name", "Other"),
                section_field_order=f.get("section_field_order", 0),
            )

        # Group by section_order → section_name
        sections: dict[int, list[_FieldEntry]] = {}
        section_names: dict[int, str] = {}
        for entry in fields_meta.values():
            order = entry.section_order
            sections.setdefault(order, []).append(entry)
            section_names[order] = entry.section_name

        # Sort sections by order, fields within each section by field_order
        lines: list[str] = []

        # Header
        header_parts = []
        if vertical:
            header_parts.append(vertical.replace("_", " ").title())
        if template_label:
            header_parts.append(template_label)
        header = " · ".join(header_parts) if header_parts else "Creative Brief"
        lines.append(f"## 📋 {header} — Captured Brief\n")

        captured_count = sum(1 for e in fields_meta.values() if e.value is not None)
        total_required = sum(1 for e in fields_meta.values() if e.required)
        captured_required = sum(
            1 for e in fields_meta.values() if e.required and e.value is not None
        )
        lines.append(
            f"*{captured_required}/{total_required} required fields captured · "
            f"{captured_count}/{len(fields_meta)} total fields*\n"
        )

        for order in sorted(sections.keys()):
            section_entries = sorted(
                sections[order], key=lambda e: e.section_field_order
            )
            name = section_names[order]
            icon = self.SECTION_ICON.get(name, "📌")
            lines.append(f"\n### {icon} {name}")

            for entry in section_entries:
                if entry.value:
                    conf_str = (
                        f" *(confidence: {entry.confidence:.0%})*"
                        if self._include_confidence
                        else ""
                    )
                    lines.append(f"- **{entry.question}**: {entry.value}{conf_str}")
                elif entry.required:
                    lines.append(f"- **{entry.question}**: *(not yet provided)*")
                # Optional + not captured: silently omit

        # Appendix: captured fields not in the YAML (base-profile fallback fields)
        unknown_captured = {
            code: cf
            for code, cf in self._captured.items()
            if code not in fields_meta
        }
        if unknown_captured:
            lines.append("\n### 📌 Additional Captured Information")
            for code, cf in unknown_captured.items():
                field_def = self._profile.get_field_by_code(code)
                label = field_def.description if field_def else code
                conf_str = (
                    f" *(confidence: {cf.confidence:.0%})*"
                    if self._include_confidence
                    else ""
                )
                lines.append(f"- **{label}**: {cf.value}{conf_str}")

        return "\n".join(lines)

    # ── Fallback: render from base profile field list ─────────────────────────

    def _render_from_base_profile(self) -> str:
        """Render a simple flat list using base-profile field definitions."""
        lines: list[str] = ["## 📋 Creative Brief — Captured Information\n"]

        captured_count = len(self._captured)
        required_fields = [f for f in self._profile.required_fields if f.required]
        captured_required = sum(
            1 for f in required_fields if f.code in self._captured
        )
        lines.append(
            f"*{captured_required}/{len(required_fields)} required fields captured*\n"
        )

        for field_def in self._profile.required_fields:
            cf = self._captured.get(field_def.code)
            if cf and cf.value:
                conf_str = (
                    f" *(confidence: {cf.confidence:.0%})*"
                    if self._include_confidence
                    else ""
                )
                lines.append(f"- **{field_def.code}**: {cf.value}{conf_str}")
            elif field_def.required:
                lines.append(f"- **{field_def.code}**: *(not yet provided)*")
            # Optional + not captured: silently omit

        return "\n".join(lines)
