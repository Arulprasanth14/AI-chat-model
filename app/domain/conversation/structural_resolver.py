"""Deterministic section guidance lookup for resolved brief templates.

Once a session has a resolved vertical and template, field-set section metadata
identifies the next unanswered section.  Its generated knowledge document is
read directly from disk, avoiding embedding and vector-store work.  Any state
that cannot be mapped unambiguously returns ``None`` so the caller can use the
existing vector retrieval fallback.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from app.domain.conversation.state import ConversationState, MissingField
from app.domain.llm.prompt_builder import RetrievedChunk

logger = logging.getLogger(__name__)


def _slugify(section_name: str) -> str:
    """Mirror the converter's section-to-filename mapping."""
    return re.sub(r"[^a-z0-9]+", "_", section_name.lower()).strip("_")


class StructuralResolver:
    """Resolve the next unanswered generated section without vector search."""

    def __init__(self, field_sets_root: Path, knowledge_docs_root: Path) -> None:
        self._field_sets_root = field_sets_root
        self._knowledge_docs_root = knowledge_docs_root

    def resolve(
        self,
        state: ConversationState,
        missing_fields: Iterable[MissingField],
    ) -> RetrievedChunk | None:
        """Return direct guidance for the earliest section still missing fields.

        A structural result is only safe when every currently missing field has
        exactly one valid section mapping in the resolved field set.  This is
        deliberately conservative: unknown/legacy field-set formats and absent
        documents leave the existing vector path in charge.
        """
        vertical = state.resolved_vertical
        template_key = state.resolved_template_key
        if not vertical or not template_key:
            return None

        field_set_path = self._field_sets_root / vertical / f"{template_key}.yaml"
        try:
            raw_data: dict[str, Any] = yaml.safe_load(
                field_set_path.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.debug(
                "Structural guidance unavailable: field set unreadable",
                extra={"path": str(field_set_path), "error": str(exc)},
            )
            return None

        sections_by_code: dict[str, tuple[int, str]] = {}
        for field in raw_data.get("fields", []):
            if not isinstance(field, dict):
                return None
            code = field.get("code")
            section_name = field.get("section_name")
            section_order = field.get("section_order")
            if (
                not isinstance(code, str)
                or not isinstance(section_name, str)
                or not isinstance(section_order, int)
            ):
                return None
            if code in sections_by_code:
                return None
            sections_by_code[code] = (section_order, section_name)

        missing_codes = [field.field_code for field in missing_fields]
        if not missing_codes or any(code not in sections_by_code for code in missing_codes):
            return None

        next_section_order, next_section_name = min(
            (sections_by_code[code] for code in missing_codes), key=lambda item: item[0]
        )
        doc_path = self._knowledge_docs_root / vertical / (
            f"{template_key}__{_slugify(next_section_name)}.md"
        )
        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.debug(
                "Structural guidance unavailable: document unreadable",
                extra={"path": str(doc_path), "error": str(exc)},
            )
            return None

        return RetrievedChunk(
            content=content,
            doc_type="question_guidance",
            score=1.0,
            field_code=None,
        )
