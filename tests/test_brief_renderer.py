from pathlib import Path
from typing import Dict, Any

import pytest
from app.domain.conversation.brief_renderer import BriefRenderer
from app.domain.conversation.state import ConversationState, CapturedField
from app.project_profiles.base_profile import BaseProfile, FieldDefinition

class DummyProfileSettings(BaseProfile):
    profile_id: str = "test_profile"
    name: str = "Test Profile"
    description: str = "A test profile"
    persona_prompt: str = "You are a test profile"
    knowledge_namespace: str = "test"
    system_prompt: str = "test prompt"
    required_fields: list[FieldDefinition] = [
        FieldDefinition(code="brand_name", description="The brand name", required=True),
        FieldDefinition(code="core_message", description="The main message", required=True),
        FieldDefinition(code="target_audience", description="Who is this for?", required=True),
        FieldDefinition(code="cta", description="What to do next?", required=True)
    ]


@pytest.fixture
def dummy_profile():
    return DummyProfileSettings()

import yaml

@pytest.fixture
def mock_yaml_file(tmp_path):
    yaml_content = {
        "template_key": "test_template",
        "name": "Test Template",
        "description": "A template for testing",
        "format": "Instagram Post",
        "platform": "Instagram",
        "field_groups": [
            {
                "group_name": "Core Identity",
                "fields": [{"code": "brand_name", "question": "What is the brand name?"}]
            },
            {
                "group_name": "Messaging Strategy",
                "fields": [
                    {"code": "core_message", "question": "What is the core message?"},
                    {"code": "target_audience", "question": "Who is the target audience?"},
                    {"code": "cta", "question": "What is the CTA?"}
                ]
            }
        ],
        "fields": [
            {"code": "brand_name", "question": "What is the brand name?", "section_order": 1, "section_name": "Core Identity"},
            {"code": "core_message", "question": "What is the core message?", "section_order": 2, "section_name": "Messaging Strategy"},
            {"code": "target_audience", "question": "Who is the target audience?", "section_order": 2, "section_name": "Messaging Strategy"},
            {"code": "cta", "question": "What is the CTA?", "section_order": 2, "section_name": "Messaging Strategy"}
        ]
    }
    file_path = tmp_path / "test_template.yaml"
    file_path.write_text(yaml.dump(yaml_content))
    return file_path

def test_renderer_with_resolved_template(dummy_profile, mock_yaml_file):
    # Setup state
    state = ConversationState(session_id="test1", profile_id=dummy_profile.profile_id)
    state.resolved_vertical = "test_vertical"
    state.resolved_template_key = "test_template"
    state.captured = {
        "brand_name": CapturedField(field_code="brand_name", value="Acme Corp", confidence=0.95, turn_index=1),
        "core_message": CapturedField(field_code="core_message", value="We make the best anvils.", confidence=0.9, turn_index=2),
        "target_audience": CapturedField(field_code="target_audience", value="Coyotes", confidence=0.85, turn_index=3),
        "cta": CapturedField(field_code="cta", value="Buy now!", confidence=0.99, turn_index=4)
    }

    # Render
    renderer = BriefRenderer(state.captured, mock_yaml_file, dummy_profile)
    md = renderer.render()

    assert "test_template — Captured Brief" in md
    assert "## 📌 Core Identity" in md
    assert "What is the brand name?" in md
    assert "Acme Corp" in md
    assert "## 📌 Messaging Strategy" in md
    assert "What is the core message?" in md
    assert "We make the best anvils." in md

def test_renderer_fallback_to_base_profile(dummy_profile):
    # Setup state without yaml_data
    state = ConversationState(session_id="test2", profile_id=dummy_profile.profile_id)
    state.captured = {
        "brand_name": CapturedField(field_code="brand_name", value="Globex", confidence=0.99, turn_index=1),
        "core_message": CapturedField(field_code="core_message", value="Innovation.", confidence=0.88, turn_index=2),
    }

    # Render without YAML data (fallback)
    renderer = BriefRenderer(state.captured, None, dummy_profile)
    md = renderer.render()

    assert "# 📋 Creative Brief — Captured Information" in md
    assert "**brand_name**: Globex" in md
    assert "**core_message**: Innovation." in md
    # Missing fields should be shown as pending or empty
    assert "**target_audience**: *(not yet provided)*" in md
    assert "**cta**: *(not yet provided)*" in md

def test_renderer_with_missing_yaml_fields(dummy_profile, mock_yaml_file):
    # Setup state with missing captured answers
    state = ConversationState(session_id="test3", profile_id=dummy_profile.profile_id)
    state.captured = {
        "brand_name": CapturedField(field_code="brand_name", value="Initech", confidence=0.9, turn_index=1),
        # missing core_message, target_audience, cta
    }

    renderer = BriefRenderer(state.captured, mock_yaml_file, dummy_profile)
    md = renderer.render()

    assert "Initech" in md
    assert "What is the core message?**: *(not yet provided)*" in md

def test_renderer_with_extra_captured_fields(dummy_profile, mock_yaml_file):
    # Setup state where model captured something NOT in the template
    state = ConversationState(session_id="test4", profile_id=dummy_profile.profile_id)
    state.captured = {
        "brand_name": CapturedField(field_code="brand_name", value="Stark Industries", confidence=0.9, turn_index=1),
        "rogue_field": CapturedField(field_code="rogue_field", value="I am Iron Man", confidence=0.99, turn_index=2)
    }

    renderer = BriefRenderer(state.captured, mock_yaml_file, dummy_profile)
    md = renderer.render()

    assert "Stark Industries" in md
    assert "## 📌 Additional Captured Information" in md
    assert "rogue_field**: I am Iron Man" in md

