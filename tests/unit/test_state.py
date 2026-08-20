"""
tests/unit/test_state.py
─────────────────────────
Unit tests for ConversationState completion ledger logic.

All tests run without LLM calls, database connections, or any external IO.
The profile is constructed inline using the BaseProfile schema.
"""
from __future__ import annotations

import pytest

from app.domain.conversation.state import ConversationState, ExtractedAnswer, MissingField
from app.project_profiles.base_profile import BaseProfile, FieldDefinition


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_profile() -> BaseProfile:
    """A minimal profile with 3 required fields and 1 optional."""
    return BaseProfile(
        profile_id="test_profile",
        persona_prompt="You are a test assistant.",
        knowledge_namespace="test_profile",
        required_fields=[
            FieldDefinition(code="client_name", description="Name of the client or business", required=True),
            FieldDefinition(code="project_type", description="Type of creative project requested", required=True),
            FieldDefinition(code="budget_range", description="Approximate budget or budget tier", required=True),
            FieldDefinition(code="competitors", description="Competitor brands or references", required=False),
        ],
    )


@pytest.fixture
def empty_state(simple_profile: BaseProfile) -> ConversationState:
    """Fresh ConversationState with no history or extractions."""
    return ConversationState(profile_id=simple_profile.profile_id)


# ── Test: compute_missing_fields ───────────────────────────────────────────────

class TestComputeMissingFields:
    """Tests for the completion ledger's missing field computation."""

    def test_all_fields_missing_on_empty_state(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """An empty state should report all required fields as missing."""
        missing = empty_state.compute_missing_fields(simple_profile)
        missing_codes = {mf.field_code for mf in missing}
        assert missing_codes == {"client_name", "project_type", "budget_range"}
        # Optional field should NOT appear in missing
        assert "competitors" not in missing_codes

    def test_no_missing_fields_when_all_captured(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """State with all required fields captured should have no missing fields."""
        answers = [
            ExtractedAnswer(field_hint="client_name", value="Acme Corp", confidence=0.95),
            ExtractedAnswer(field_hint="project_type", value="brand identity", confidence=0.9),
            ExtractedAnswer(field_hint="budget_range", value="$20,000–$30,000", confidence=0.85),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        missing = empty_state.compute_missing_fields(simple_profile)
        assert missing == []

    def test_partial_extraction_leaves_correct_missing(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """Only captured fields should be removed from missing list."""
        answers = [
            ExtractedAnswer(field_hint="client_name", value="Startup X", confidence=0.9),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        missing = empty_state.compute_missing_fields(simple_profile)
        missing_codes = {mf.field_code for mf in missing}
        assert "client_name" not in missing_codes
        assert "project_type" in missing_codes
        assert "budget_range" in missing_codes

    def test_low_confidence_field_remains_missing(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """Fields extracted below confidence threshold remain in missing list."""
        answers = [
            ExtractedAnswer(field_hint="budget_range", value="maybe $5k", confidence=0.3),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        # Default threshold is 0.7 — 0.3 confidence should not count
        missing = empty_state.compute_missing_fields(simple_profile, confidence_threshold=0.7)
        missing_codes = {mf.field_code for mf in missing}
        assert "budget_range" in missing_codes

    def test_field_at_exact_threshold_counts_as_captured(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A field at exactly the threshold value should count as captured."""
        answers = [
            ExtractedAnswer(field_hint="budget_range", value="around 10k", confidence=0.7),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        missing = empty_state.compute_missing_fields(simple_profile, confidence_threshold=0.7)
        missing_codes = {mf.field_code for mf in missing}
        assert "budget_range" not in missing_codes


# ── Test: apply_extraction ─────────────────────────────────────────────────────

class TestApplyExtraction:
    """Tests for the extraction merging logic."""

    def test_higher_confidence_replaces_lower(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A higher-confidence extraction should replace a lower-confidence one."""
        low = [ExtractedAnswer(field_hint="client_name", value="Acme", confidence=0.75)]
        high = [ExtractedAnswer(field_hint="client_name", value="Acme Corporation", confidence=0.9)]

        empty_state.apply_extraction(low, simple_profile)
        assert empty_state.captured["client_name"].confidence == 0.75

        empty_state.apply_extraction(high, simple_profile)
        assert empty_state.captured["client_name"].confidence == 0.9
        assert empty_state.captured["client_name"].value == "Acme Corporation"

    def test_lower_confidence_does_not_replace_higher(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A lower-confidence extraction should NOT replace a higher one."""
        high = [ExtractedAnswer(field_hint="project_type", value="brand identity", confidence=0.9)]
        low = [ExtractedAnswer(field_hint="project_type", value="logo", confidence=0.75)]

        empty_state.apply_extraction(high, simple_profile)
        empty_state.apply_extraction(low, simple_profile)
        assert empty_state.captured["project_type"].confidence == 0.9
        assert empty_state.captured["project_type"].value == "brand identity"

    def test_unmatched_hint_is_ignored(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A field_hint that doesn't match any profile field should be ignored."""
        answers = [
            ExtractedAnswer(field_hint="xyzzy_does_not_exist", value="irrelevant", confidence=0.9),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        assert len(empty_state.captured) == 0

    def test_multi_field_extraction_in_one_call(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """Multiple fields can be extracted in a single apply_extraction call."""
        answers = [
            ExtractedAnswer(field_hint="client name", value="Brand Co", confidence=0.9),
            ExtractedAnswer(field_hint="project type", value="campaign", confidence=0.8),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        assert "client_name" in empty_state.captured
        assert "project_type" in empty_state.captured

    def test_post_completion_correction_overwrites(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A correction in a newer turn should overwrite existing value if confidence >= 0.8."""
        empty_state.add_turn("assistant", "What is the budget?")
        empty_state.add_turn("user", "10k")
        empty_state.apply_extraction([ExtractedAnswer(field_hint="budget_range", value="10k", confidence=0.9)], simple_profile)
        
        # Correction in a later turn with slightly lower but still high confidence
        empty_state.add_turn("assistant", "Got it, 10k.")
        empty_state.add_turn("user", "Actually make it 15k")
        empty_state.apply_extraction([ExtractedAnswer(field_hint="budget_range", value="15k", confidence=0.85)], simple_profile)
        
        assert empty_state.captured["budget_range"].value == "15k"

    def test_unasked_field_is_rejected_unless_high_confidence(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """A field that was never asked should be rejected (confidence set to 0.0) unless it's a 1.0 confidence correction."""
        empty_state.add_turn("assistant", "Tell me about the project type.")
        empty_state.add_turn("user", "It's a brand identity. The client is Acme.")
        
        empty_state.apply_extraction([
            ExtractedAnswer(field_hint="project_type", value="brand identity", confidence=0.9),
            ExtractedAnswer(field_hint="client_name", value="Acme", confidence=0.8),
        ], simple_profile)
        
        assert "project_type" in empty_state.captured
        assert "client_name" not in empty_state.captured


# ── Test: is_complete ──────────────────────────────────────────────────────────

class TestIsComplete:
    def test_not_complete_on_empty_state(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        assert empty_state.is_complete(simple_profile) is False

    def test_complete_when_all_required_captured(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        answers = [
            ExtractedAnswer(field_hint="client_name", value="Brand Co", confidence=0.9),
            ExtractedAnswer(field_hint="project_type", value="brand identity", confidence=0.9),
            ExtractedAnswer(field_hint="budget_range", value="$25k", confidence=0.9),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        assert empty_state.is_complete(simple_profile) is True

    def test_optional_field_missing_does_not_block_completion(
        self, empty_state: ConversationState, simple_profile: BaseProfile
    ) -> None:
        """Completion is True even if optional 'competitors' field is missing."""
        answers = [
            ExtractedAnswer(field_hint="client_name", value="Brand Co", confidence=0.9),
            ExtractedAnswer(field_hint="project_type", value="campaign", confidence=0.9),
            ExtractedAnswer(field_hint="budget_range", value="$10k", confidence=0.9),
        ]
        empty_state.apply_extraction(answers, simple_profile)
        assert empty_state.is_complete(simple_profile) is True


# ── Test: add_turn / get_recent_history ───────────────────────────────────────

class TestHistory:
    def test_add_turn_appends_to_history(self, empty_state: ConversationState) -> None:
        empty_state.add_turn("user", "Hello")
        empty_state.add_turn("assistant", "Hi there!")
        assert len(empty_state.conversation_history) == 2
        assert empty_state.conversation_history[0]["role"] == "user"
        assert empty_state.conversation_history[1]["content"] == "Hi there!"

    def test_get_recent_history_respects_window(self, empty_state: ConversationState) -> None:
        for i in range(30):
            empty_state.add_turn("user", f"message {i}")
        recent = empty_state.get_recent_history(window=10)
        assert len(recent) == 10
        assert recent[-1]["content"] == "message 29"

    def test_get_recent_history_window_larger_than_history(
        self, empty_state: ConversationState
    ) -> None:
        empty_state.add_turn("user", "only message")
        recent = empty_state.get_recent_history(window=100)
        assert len(recent) == 1
