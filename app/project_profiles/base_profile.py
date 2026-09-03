"""
app/project_profiles/base_profile.py
──────────────────────────────────────
Defines the reusability contract for all project profiles.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  REUSABILITY BOUNDARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  A new integration requires ONLY:
    1. A new folder under app/project_profiles/<your_profile>/
    2. A profile.yaml implementing the BaseProfile schema below
    3. A knowledge_docs/ folder with domain documents

  Zero core code changes are needed. The orchestrator, retriever,
  prompt builder, and state ledger are all profile-agnostic — they
  consume BaseProfile fields at runtime.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class ShowIfCondition(BaseModel):
    """Conditional logic for showing a field based on another field's value."""
    field_code: str = Field(alias="field")
    in_: list[str] | None = Field(default=None, alias="in")
    not_in: list[str] | None = Field(default=None)


class FieldDefinition(BaseModel):
    """Describes a single field the LLM should try to extract.

    Attributes:
        code:        Machine-readable identifier (e.g. "client_name").
                     Used as the key in extraction results and the completion ledger.
        description: Human-readable description sent to the LLM as context.
                     This is what the LLM uses to understand what to look for —
                     it is NOT a hardcoded question string.
        required:    Whether this field must be captured before the session is
                     considered complete (drives the missing_fields gate).
    """

    code: str = Field(..., description="Unique machine-readable field identifier")
    description: str = Field(
        ...,
        description="Human-readable description of the field (fed to LLM as context)",
    )
    required: bool = Field(
        default=True,
        description="Whether this field must be captured for completion",
    )
    enum_values: list[str] | None = Field(
        default=None,
        description=(
            "If set, extracted values must belong to this list. "
            "Values not in the list are rejected (confidence zeroed) so the LLM "
            "re-asks rather than silently capturing bad data."
        ),
    )
    enum_options: list[dict[str, str]] | None = Field(
        default=None,
        description="Full option dicts containing both 'label' and 'value'. Used for LLM prompt generation.",
    )
    input_type: str = Field(
        default="text",
        description=(
            "The expected input modality for this field. Controls both merge semantics "
            "(list fields merge additively) and UI rendering (via /next_field_spec). "
            "Values: 'text' (free text, replaces on update), "
            "'list' (comma-joined multi-value, additive merge), "
            "'enum' (single value from enum_values), "
            "'quantitative' (must contain numeric/KPI signal), "
            "'file_upload' (handled by the /logo or /document endpoints)."
        ),
    )
    show_if: ShowIfCondition | None = Field(
        default=None,
        description="Optional conditional display logic.",
    )


class BaseProfile(BaseModel):
    """Abstract base class for all project profiles.

    ─────────────────────────────────────────────────────────────────────
    REUSABILITY CONTRACT:
      Every profile.yaml MUST provide values for all required fields
      defined here. Optional fields have sensible defaults.

      The orchestrator, state ledger, prompt builder, and retriever
      operate entirely on BaseProfile instances — they never reference
      concrete profile implementations directly.
    ─────────────────────────────────────────────────────────────────────
    """

    profile_id: str = Field(
        ...,
        description=(
            "Unique identifier for this profile. Used as the namespace key "
            "for vector store filtering (knowledge_namespace) and session tagging."
        ),
    )
    persona_prompt: str = Field(
        ...,
        description=(
            "System-level persona/instruction text sent to the LLM. "
            "Defines the AI's role, tone, and high-level objectives. "
            "Should NOT contain hardcoded question scripts."
        ),
    )
    required_fields: list[FieldDefinition] = Field(
        ...,
        description=(
            "Ordered list of fields the session must collect. "
            "The state ledger uses this list to compute missing_fields. "
            "The LLM uses field descriptions (not codes) as extraction hints."
        ),
    )
    knowledge_namespace: str = Field(
        ...,
        description=(
            "Namespace used to partition vector store chunks for this profile. "
            "Typically matches profile_id. Used as the profile_id filter in retrieval."
        ),
    )
    industries: list[str] = Field(
        default_factory=list,
        description="Supported industry tags used for optional retrieval filtering",
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for the LLM chat completion calls",
    )

    @field_validator("required_fields")
    @classmethod
    def _validate_unique_codes(cls, fields: list[FieldDefinition]) -> list[FieldDefinition]:
        codes = [f.code for f in fields]
        if len(codes) != len(set(codes)):
            duplicates = {c for c in codes if codes.count(c) > 1}
            raise ValueError(f"Duplicate field codes in profile: {duplicates}")
        return fields

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaseProfile":
        """Load a profile from a YAML file.

        Args:
            path: Absolute or relative path to profile.yaml.

        Returns:
            A validated BaseProfile instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValidationError: If the YAML content does not match the schema.

        Example:
            >>> profile = BaseProfile.from_yaml("app/project_profiles/picasso_fusion/profile.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile YAML not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)

        return cls.model_validate(data)

    def get_required_field_codes(self) -> list[str]:
        """Return codes of all required fields."""
        return [f.code for f in self.required_fields if f.required]

    def get_field_by_code(self, code: str) -> FieldDefinition | None:
        """Look up a FieldDefinition by its code."""
        return next((f for f in self.required_fields if f.code == code), None)
