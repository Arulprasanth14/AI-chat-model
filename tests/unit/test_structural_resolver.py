"""Unit tests for deterministic field-set section guidance lookup."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.domain.conversation.state import ConversationState, MissingField
from app.domain.conversation.structural_resolver import StructuralResolver


@pytest.fixture
def resolver_roots(tmp_path: Path) -> tuple[Path, Path]:
    field_sets_root = tmp_path / "field_sets"
    knowledge_docs_root = tmp_path / "knowledge_docs"
    vertical = "restaurant"
    template = "restaurant_static_post"
    field_set_dir = field_sets_root / vertical
    knowledge_dir = knowledge_docs_root / vertical
    field_set_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)

    yaml.safe_dump(
        {
            "template_key": template,
            "fields": [
                {
                    "code": "purpose",
                    "required": True,
                    "section_name": "Purpose",
                    "section_order": 0,
                    "section_field_order": 0,
                },
                {
                    "code": "audience",
                    "required": True,
                    "section_name": "Purpose",
                    "section_order": 0,
                    "section_field_order": 1,
                },
                {
                    "code": "visuals",
                    "required": True,
                    "section_name": "Visual Direction",
                    "section_order": 1,
                    "section_field_order": 0,
                },
            ],
        },
        (field_set_dir / f"{template}.yaml").open("w", encoding="utf-8"),
        sort_keys=False,
    )
    (knowledge_dir / f"{template}__purpose.md").write_text(
        "Purpose guidance", encoding="utf-8"
    )
    (knowledge_dir / f"{template}__visual_direction.md").write_text(
        "Visual guidance", encoding="utf-8"
    )
    return field_sets_root, knowledge_docs_root


def _state() -> ConversationState:
    return ConversationState(
        profile_id="test",
        resolved_vertical="restaurant",
        resolved_template_key="restaurant_static_post",
    )


def test_resolves_earliest_section_with_missing_fields(resolver_roots: tuple[Path, Path]) -> None:
    resolver = StructuralResolver(*resolver_roots)
    result = resolver.resolve(
        _state(),
        [
            MissingField(field_code="audience", description="Audience"),
            MissingField(field_code="visuals", description="Visuals"),
        ],
    )
    assert result is not None
    assert result.content == "Purpose guidance"


def test_returns_none_without_resolved_template(resolver_roots: tuple[Path, Path]) -> None:
    resolver = StructuralResolver(*resolver_roots)
    state = ConversationState(profile_id="test")
    assert resolver.resolve(
        state, [MissingField(field_code="purpose", description="Purpose")]
    ) is None


def test_advances_when_current_section_is_captured(resolver_roots: tuple[Path, Path]) -> None:
    resolver = StructuralResolver(*resolver_roots)
    result = resolver.resolve(_state(), [MissingField(field_code="visuals", description="Visuals")])
    assert result is not None
    assert result.content == "Visual guidance"
