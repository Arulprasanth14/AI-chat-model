"""
app/domain/conversation/suggestion_gate.py
────────────────────────────────────────────
Pre-generation gate that blocks creative suggestions until the required
context fields are captured.

Bug 2 fix: Without this gate, the LLM can generate creative concepts,
color palettes, or design directions before knowing the brand personality,
channels, or brand identity preference — producing irrelevant suggestions
the user must then override.

The gate injects a hard prohibition rule into the Phase A/B system message
at the *code level* (not as a prompt suggestion), because Cluster A established
that LLMs won't reliably self-police purely suggestion-based restrictions.

Usage in the orchestrator / prompt builder:
    can_suggest, missing = can_generate_suggestions(state, profile, threshold)
    if not can_suggest:
        # inject gate rule string into system message
        gate_rule = format_gate_rule(missing)
"""
from __future__ import annotations

from app.domain.conversation.state import ConversationState
from app.project_profiles.base_profile import BaseProfile

# Fields that must be captured before the LLM is permitted to suggest
# creative concepts, color palettes, or design directions.
# These represent the minimum context needed for suggestions to be relevant.
REQUIRED_BEFORE_SUGGESTIONS: list[str] = [
    "brand_personality",
    "distribution_channels",
    "brand_identity_preference",
]


def can_generate_suggestions(
    state: ConversationState,
    profile: BaseProfile,
    confidence_threshold: float = 0.7,
) -> tuple[bool, list[str]]:
    """Check whether the LLM is allowed to generate creative suggestions.

    Args:
        state:                Current conversation state.
        profile:              Active project profile (used to validate field codes).
        confidence_threshold: Minimum confidence for a field to count as captured.

    Returns:
        Tuple of (allowed: bool, missing_field_codes: list[str]).
        If allowed is False, the caller should inject ``format_gate_rule(missing)``
        into the system message to hard-block suggestions.
    """
    # Only gate on fields that actually exist in the profile
    profile_codes = {f.code for f in profile.required_fields}
    relevant_required = [
        code for code in REQUIRED_BEFORE_SUGGESTIONS
        if code in profile_codes
    ]

    missing: list[str] = []
    for field_code in relevant_required:
        captured = state.captured.get(field_code)
        if captured is None or captured.confidence < confidence_threshold:
            missing.append(field_code)

    return len(missing) == 0, missing


def format_gate_rule(missing_field_codes: list[str]) -> str:
    """Format the hard gate rule string for injection into the system message.

    Args:
        missing_field_codes: Field codes that must be captured first.

    Returns:
        String to inject into the prompt as a hard prohibition rule.
    """
    missing_str = ", ".join(f"`{code}`" for code in missing_field_codes)
    return (
        f"CREATIVE SUGGESTION GATE (enforced — do not override):\\n"
        f"You MUST NOT suggest creative concepts, color palettes, design directions, "
        f"or any specific visual ideas until these fields are captured: {missing_str}.\\n"
        f"If the user asks for suggestions, explain warmly that you need a few more "
        f"details first, then ask about one of the missing fields listed above."
    )
