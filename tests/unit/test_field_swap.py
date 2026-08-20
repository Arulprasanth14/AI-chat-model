"""
tests/unit/test_field_swap.py
──────────────────────────────
Tests for the key_messages ↔ success_metrics field-swap bug.

These tests verify that the field-matching logic in state._match_field
and state.apply_extraction correctly routes values to the right fields
given the rewritten profile.yaml descriptions.

Three scenarios:
  1. Single turn containing both a key message AND a success metric.
  2. Multi-turn: key_messages first, success_metrics on the next turn.
  3. Edge case: a success metric stated without any number (e.g. 'we
     want to become the most trusted brand').  We assert the LLM puts
     it somewhere and we report what happened — we do NOT assert a fixed
     expectation because this is genuinely ambiguous.
"""
from __future__ import annotations

import warnings
import logging
from pathlib import Path

import pytest

from app.domain.conversation.state import ConversationState, ExtractedAnswer
from app.project_profiles.base_profile import BaseProfile

# ── Fixture helpers ───────────────────────────────────────────────────────────

PROFILE_PATH = (
    Path(__file__).parent.parent.parent
    / "app/project_profiles/picasso_fusion/profile.yaml"
)


@pytest.fixture()
def profile() -> BaseProfile:
    """Load the live picasso_fusion profile from disk."""
    return BaseProfile.from_yaml(PROFILE_PATH)


@pytest.fixture()
def state(profile: BaseProfile) -> ConversationState:
    """Return a fresh state object with one assistant turn (so unasked-guard passes)."""
    s = ConversationState(profile_id=profile.profile_id)
    # Add a synthetic assistant message that mentions both topics so the
    # unasked-field guard in apply_extraction considers them 'asked'.
    s.add_turn("assistant", (
        "Great! Could you tell me the key messages for the campaign, "
        "and also what success metrics you are targeting?"
    ))
    return s


# ── Test 1: Single turn — both fields in one message ─────────────────────────

class TestSingleTurnBothFields:
    """User provides both key_messages and success_metrics in one turn."""

    def test_key_messages_lands_correctly(self, state: ConversationState, profile: BaseProfile) -> None:
        """'affordability and trust' must land in key_messages, not success_metrics."""
        answers = [
            ExtractedAnswer(field_hint="key messages", value="affordability and trust", confidence=0.9),
            ExtractedAnswer(field_hint="success metrics", value="20% increase in engagement", confidence=0.9),
        ]
        state.apply_extraction(answers, profile)

        assert "key_messages" in state.captured, "key_messages was not captured at all"
        assert state.captured["key_messages"].value == "affordability and trust", (
            f"key_messages got wrong value: {state.captured.get('key_messages')}"
        )

    def test_success_metrics_lands_correctly(self, state: ConversationState, profile: BaseProfile) -> None:
        """'20% increase in engagement' must land in success_metrics, not key_messages."""
        answers = [
            ExtractedAnswer(field_hint="key messages", value="affordability and trust", confidence=0.9),
            ExtractedAnswer(field_hint="success metrics", value="20% increase in engagement", confidence=0.9),
        ]
        state.apply_extraction(answers, profile)

        assert "success_metrics" in state.captured, "success_metrics was not captured at all"
        assert state.captured["success_metrics"].value == "20% increase in engagement", (
            f"success_metrics got wrong value: {state.captured.get('success_metrics')}"
        )

    def test_no_cross_contamination(self, state: ConversationState, profile: BaseProfile) -> None:
        """The qualitative phrase must NOT appear in success_metrics, and the
        quantitative phrase must NOT appear in key_messages."""
        answers = [
            ExtractedAnswer(field_hint="key messages", value="affordability and trust", confidence=0.9),
            ExtractedAnswer(field_hint="success metrics", value="20% increase in engagement", confidence=0.9),
        ]
        state.apply_extraction(answers, profile)

        km = state.captured.get("key_messages")
        sm = state.captured.get("success_metrics")

        if km:
            assert "20%" not in km.value, f"Quantitative metric leaked into key_messages: {km.value!r}"
        if sm:
            assert "affordability" not in sm.value, (
                f"Qualitative message leaked into success_metrics: {sm.value!r}"
            )


# ── Test 2: Multi-turn — no cross-contamination across turns ─────────────────

class TestMultiTurnNoContamination:
    """key_messages is captured on turn 1; success_metrics on turn 2.
    Neither should overwrite or bleed into the other."""

    def test_first_turn_captures_key_messages(self, profile: BaseProfile) -> None:
        s = ConversationState(profile_id=profile.profile_id)
        s.add_turn("assistant", "What key message do you want to communicate in the campaign?")
        s.add_turn("user", "Our key message is that we care about affordability.")

        answers_t1 = [
            ExtractedAnswer(field_hint="key messages", value="we care about affordability", confidence=0.9),
        ]
        s.apply_extraction(answers_t1, profile)

        assert "key_messages" in s.captured
        assert s.captured["key_messages"].value == "we care about affordability"
        assert "success_metrics" not in s.captured, (
            "success_metrics should not have been captured on turn 1"
        )

    def test_second_turn_captures_success_metrics_without_overwriting(self, profile: BaseProfile) -> None:
        s = ConversationState(profile_id=profile.profile_id)
        s.add_turn("assistant", "What key message do you want to communicate in the campaign?")
        s.add_turn("user", "Our key message is that we care about affordability.")

        answers_t1 = [
            ExtractedAnswer(field_hint="key messages", value="we care about affordability", confidence=0.9),
        ]
        s.apply_extraction(answers_t1, profile)

        s.add_turn("assistant", "Great! And what are your success metrics?")
        s.add_turn("user", "We want a 15% lift in conversions over 3 months.")

        answers_t2 = [
            ExtractedAnswer(field_hint="success metrics", value="15% lift in conversions over 3 months", confidence=0.9),
        ]
        s.apply_extraction(answers_t2, profile)

        # key_messages must be unchanged from turn 1
        assert s.captured["key_messages"].value == "we care about affordability", (
            f"key_messages was overwritten by turn 2: {s.captured['key_messages'].value!r}"
        )
        # success_metrics must be the turn 2 value
        assert "success_metrics" in s.captured
        assert "15%" in s.captured["success_metrics"].value, (
            f"success_metrics did not receive turn-2 value: {s.captured['success_metrics'].value!r}"
        )


# ── Test 3: Edge case — qualitative success metric (no numbers) ───────────────

class TestAmbiguousSuccessMetric:
    """'We want to become the most trusted brand in the region.'

    This is genuinely ambiguous:
      - It sounds like a key_message (qualitative aspiration).
      - But it is framed as a goal/outcome → could be success_metrics.
      - We do NOT assert a fixed field here; we report where it lands and
        whether the warning heuristic fires.

    The test is a diagnostic test, not a correctness assertion.
    """

    AMBIGUOUS_VALUE = "we want to become the most trusted brand in the region"

    def test_field_assigned_and_warning_fires_if_sm(
        self, state: ConversationState, profile: BaseProfile, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If the LLM routes this to success_metrics, the warning heuristic
        should fire because the value contains no quantitative signal."""
        with caplog.at_level(logging.WARNING, logger="app.domain.conversation.state"):
            result = state.handle_save_quantitative_field(
                field_code="success_metrics",
                value=self.AMBIGUOUS_VALUE,
                confidence=0.75,
                profile=profile,
            )

        # In explicit tool calling, it is rejected entirely
        assert result.status == "rejected_qualitative"
        
        warning_fired = any("no quantitative signal" in r.message for r in caplog.records)
        assert warning_fired, "Quantitative-signal warning did NOT fire. Regression guard broken."

    def test_warning_fires_for_qualitative_sm_extraction(
        self, state: ConversationState, profile: BaseProfile, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Directly assert: if ANY qualitative-only value is written to
        success_metrics, a WARNING is logged. This is the regression trap."""
        with caplog.at_level(logging.WARNING, logger="app.domain.conversation.state"):
            result = state.handle_save_quantitative_field(
                field_code="success_metrics",
                value="make our brand more recognisable",
                confidence=0.9,
                profile=profile,
            )

        assert result.status == "rejected_qualitative"
        warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("no quantitative signal" in m for m in warning_msgs), (
            "A clearly qualitative value did not trigger the warning heuristic."
        )

    def test_no_warning_for_quantitative_sm(
        self, state: ConversationState, profile: BaseProfile, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sanity: a properly quantitative value must NOT trigger the warning."""
        quantitative = ExtractedAnswer(
            field_hint="success metrics",
            value="25% increase in website traffic",
            confidence=0.9,
        )
        with caplog.at_level(logging.WARNING, logger="app.domain.conversation.state"):
            state.apply_extraction([quantitative], profile)

        if "success_metrics" in state.captured:
            warning_msgs = [r.message for r in caplog.records if r.levelname == "WARNING"]
            false_positive = any("no quantitative signal" in m for m in warning_msgs)
            assert not false_positive, (
                "The quantitative-signal heuristic fired a false-positive warning "
                f"for value {quantitative.value!r}. Regex needs tuning."
            )
