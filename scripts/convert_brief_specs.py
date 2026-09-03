"""
scripts/convert_brief_specs.py
──────────────────────────────
Converts legacy V1 _source_brief_specs/ JSON files into two artefacts:

  (a) Field-set YAMLs  →  field_sets/<vertical>/<key>.yaml
      Deterministic field requirements: code, kind, required, options.
      These feed the completion ledger directly — never touch the vector store.

  (b) Prose knowledge_doc markdowns  →  knowledge_docs/<vertical>/<key>__<section_slug>.md
      One markdown per section, with YAML frontmatter carrying the metadata
      columns the ingestion pipeline reads (industry, brief_type, doc_type, section).
      These are embedded and retrieved through the existing RAG pipeline.

Usage
-----
    python scripts/convert_brief_specs.py --verticals restaurant,realestate

    # Dry-run (show what would be written without writing):
    python scripts/convert_brief_specs.py --verticals restaurant --dry-run

Idempotent: re-running overwrites existing files cleanly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────────

_SCRIPTS_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_PROFILE_DIR = _PROJECT_ROOT / "app" / "project_profiles" / "picasso_fusion"
_SOURCE_DIR = _PROFILE_DIR / "_source_brief_specs"
_FIELD_SETS_ROOT = _PROFILE_DIR / "field_sets"
_KNOWLEDGE_DOCS_ROOT = _PROFILE_DIR / "knowledge_docs"


# ── Section → slug helper ──────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert a section name to a filesystem-safe slug.

    Examples:
        "Post Purpose & Product" → "post_purpose_product"
        "Images & Assets"        → "images_assets"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


# ── Prose generation ───────────────────────────────────────────────────────────

def _option_labels(options: list[dict[str, str]]) -> list[str]:
    """Extract human-readable labels from an options list, excluding 'Other'."""
    return [o["label"] for o in options if o["label"].lower() != "other"]


def _prose_for_section(
    section_name: str,
    questions: list[dict[str, Any]],
    template_label: str,
    vertical_key: str,
) -> str:
    """Generate natural-language prose for a section from its questions.

    Combines the question text and hints into readable paragraphs.
    Field codes and JSON syntax are never included in the output.

    The prose covers:
    - What to ask about in this section
    - Any hint content from the source (phrased naturally)
    - Option context for radio/checkbox questions (labels only, as examples)
    """
    vertical_display = vertical_key.replace("realestate", "real estate").replace("_", " ").title()
    lines: list[str] = []

    lines.append(f"## {section_name}")
    lines.append("")

    # Opening sentence for the section
    req_questions = [q for q in questions if q.get("required", True)]
    opt_questions = [q for q in questions if not q.get("required", True)]

    intro = (
        f"When conducting the {template_label} brief for a {vertical_display} client, "
        f"the **{section_name}** section covers the following key information."
    )
    lines.append(intro)
    lines.append("")

    for q in questions:
        question_text = q["question"]
        hint = q.get("hint", "")
        kind = q.get("kind", "text")
        options = q.get("options", [])
        required = q.get("required", True)

        # Build the paragraph for this question
        paragraph_parts: list[str] = []

        # Question as a topic sentence
        if required:
            paragraph_parts.append(f"Ask the client: {question_text}")
        else:
            paragraph_parts.append(
                f"Optionally, ask the client: {question_text} (this is not required)"
            )

        # Expand hint naturally
        if hint:
            hint_clean = hint.strip().rstrip(".")
            if hint_clean.lower().startswith("only if"):
                paragraph_parts.append(f"Note: {hint_clean}.")
            elif hint_clean.lower().startswith("only ask"):
                paragraph_parts.append(f"Note: {hint_clean}.")
            else:
                paragraph_parts.append(f"{hint_clean}.")

        # For radio/checkbox with options, mention the choices naturally
        if kind in ("radio", "checkbox") and options:
            labels = _option_labels(options)
            if labels:
                if kind == "radio":
                    examples = ", ".join(f'"{l}"' for l in labels[:6])
                    if len(labels) > 6:
                        examples += f", or other options"
                    paragraph_parts.append(
                        f"The available choices include: {examples}."
                    )
                else:  # checkbox
                    examples = ", ".join(f'"{l}"' for l in labels[:5])
                    if len(labels) > 5:
                        examples += ", among others"
                    paragraph_parts.append(
                        f"The client may select one or more from: {examples}."
                    )

        lines.append(" ".join(paragraph_parts))
        lines.append("")

    # Summary of required vs optional in this section
    if req_questions and opt_questions:
        lines.append(
            f"In this section, {len(req_questions)} question(s) are required and "
            f"{len(opt_questions)} are optional — follow up on optional items only "
            f"when contextually appropriate."
        )
        lines.append("")
    elif opt_questions and not req_questions:
        lines.append(
            f"All questions in this section are optional. Gather this information "
            f"only when the client has relevant context to share."
        )
        lines.append("")

    return "\n".join(lines)


def _build_knowledge_doc(
    section_name: str,
    questions: list[dict[str, Any]],
    spec: dict[str, Any],
) -> str:
    """Build a complete markdown knowledge_doc for one section.

    Includes YAML frontmatter with the metadata columns the ingestion pipeline
    reads (industry, brief_type, doc_type, section).
    """
    vertical_key: str = spec.get("verticalKey", "")
    template_key: str = spec["key"]
    template_label: str = spec["label"]
    section_slug = _slugify(section_name)

    frontmatter = {
        "doc_type": "question_guidance",
        "industry": vertical_key,
        "brief_type": template_key,
        "section": section_name,
        "template_label": template_label,
    }
    fm_str = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True).strip()

    prose = _prose_for_section(section_name, questions, template_label, vertical_key)

    return f"---\n{fm_str}\n---\n\n{prose}"


# ── Field-set YAML builder ─────────────────────────────────────────────────────

def _build_field_set(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the field-set YAML data structure from a BriefSpec.

    Preserves exactly:
    - template_key, template_label, vertical
    - template_name_match (for resolver)
    - fields: code, kind, required, options (if present), and source section
      identity/order for deterministic structural guidance lookup

    Nothing is inferred or dropped.
    """
    fields: list[dict[str, Any]] = []
    for section_order, section in enumerate(spec.get("sections", [])):
        section_name = section["name"]
        for section_field_order, q in enumerate(section.get("questions", [])):
            field: dict[str, Any] = {
                "code": q["code"],
                "kind": q["kind"],
                "required": bool(q.get("required", True)),
                "section_name": section_name,
                "section_order": section_order,
                "section_field_order": section_field_order,
            }
            if "question" in q:
                field["question"] = q["question"]
            if "hint" in q:
                field["hint"] = q["hint"]
            if q.get("options"):
                field["options"] = [
                    {"label": o["label"], "value": o["value"]}
                    for o in q["options"]
                ]
            fields.append(field)

    return {
        "template_key": spec["key"],
        "template_label": spec["label"],
        "vertical": spec.get("verticalKey", ""),
        "template_name_match": spec.get("templateNameMatch", []),
        "fields": fields,
    }


# ── Per-spec conversion ────────────────────────────────────────────────────────

def convert_spec(
    spec_path: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Convert one JSON spec file into field_set + knowledge_docs.

    Returns:
        (field_sets_written, knowledge_docs_written)
    """
    with spec_path.open("r", encoding="utf-8") as fh:
        spec: dict[str, Any] = json.load(fh)

    vertical_key: str = spec.get("verticalKey", "")
    template_key: str = spec["key"]

    # ── Emit field-set YAML ────────────────────────────────────────────────────
    field_set_dir = _FIELD_SETS_ROOT / vertical_key
    field_set_path = field_set_dir / f"{template_key}.yaml"

    field_set_data = _build_field_set(spec)
    field_set_yaml = yaml.dump(
        field_set_data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )

    if not dry_run:
        field_set_dir.mkdir(parents=True, exist_ok=True)
        field_set_path.write_text(field_set_yaml, encoding="utf-8")

    field_sets_written = 1

    # ── Emit knowledge_doc markdowns (one per section) ────────────────────────
    knowledge_docs_written = 0
    knowledge_dir = _KNOWLEDGE_DOCS_ROOT / vertical_key

    if not dry_run:
        knowledge_dir.mkdir(parents=True, exist_ok=True)

    for section in spec.get("sections", []):
        section_name: str = section["name"]
        section_slug = _slugify(section_name)
        doc_filename = f"{template_key}__{section_slug}.md"
        doc_path = knowledge_dir / doc_filename

        doc_content = _build_knowledge_doc(section_name, section["questions"], spec)

        if not dry_run:
            doc_path.write_text(doc_content, encoding="utf-8")
        knowledge_docs_written += 1

    return field_sets_written, knowledge_docs_written


# ── Vertical filtering ─────────────────────────────────────────────────────────

def find_specs_for_verticals(verticals: list[str]) -> list[Path]:
    """Return all JSON spec files whose verticalKey is in the requested list."""
    if not _SOURCE_DIR.exists():
        print(f"ERROR: source dir not found: {_SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    matches: list[Path] = []
    for json_path in sorted(_SOURCE_DIR.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {json_path.name} — {exc}", file=sys.stderr)
            continue

        vertical_key = data.get("verticalKey", "")
        if vertical_key in verticals:
            matches.append(json_path)

    return matches


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert _source_brief_specs/ JSON files into field-set YAMLs "
            "and prose knowledge_doc markdowns."
        )
    )
    parser.add_argument(
        "--verticals",
        required=True,
        help=(
            "Comma-separated list of verticalKey values to process. "
            "Example: restaurant,realestate"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without writing any files.",
    )
    args = parser.parse_args()

    requested = [v.strip() for v in args.verticals.split(",") if v.strip()]
    dry_run: bool = args.dry_run

    if dry_run:
        print("DRY RUN — no files will be written.\n")

    print(f"Processing verticals: {', '.join(requested)}\n")

    # Summary counters per vertical
    summary: dict[str, dict[str, int]] = {}

    spec_paths = find_specs_for_verticals(requested)
    if not spec_paths:
        print("No matching spec files found for the requested verticals.")
        sys.exit(0)

    for spec_path in spec_paths:
        # Peek at verticalKey for grouping
        with spec_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        vk = data.get("verticalKey", "unknown")
        tk = data.get("key", spec_path.stem)

        fs_count, kd_count = convert_spec(spec_path, dry_run=dry_run)

        summary.setdefault(vk, {"field_sets": 0, "knowledge_docs": 0})
        summary[vk]["field_sets"] += fs_count
        summary[vk]["knowledge_docs"] += kd_count

        verb = "Would write" if dry_run else "Wrote"
        print(
            f"  [{vk}] {tk}: "
            f"{verb} 1 field_set YAML + {kd_count} knowledge_doc(s)"
        )

    # Print summary table
    print("\n── Summary ──────────────────────────────────────────────")
    print(f"{'Vertical':<20} {'Field Sets':>12} {'Knowledge Docs':>15}")
    print("-" * 50)
    for vk in sorted(summary):
        row = summary[vk]
        print(f"{vk:<20} {row['field_sets']:>12} {row['knowledge_docs']:>15}")
    total_fs = sum(r["field_sets"] for r in summary.values())
    total_kd = sum(r["knowledge_docs"] for r in summary.values())
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_fs:>12} {total_kd:>15}")
    print()

    if not dry_run:
        print(
            "Done. Next step: run scripts/ingest_knowledge.py to embed "
            "the new knowledge_docs into the vector store."
        )


if __name__ == "__main__":
    main()
