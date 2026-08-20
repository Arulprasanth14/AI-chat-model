"""
tests/unit/test_convert_brief_specs.py
──────────────────────────────────────
Golden-file comparison test for the conversion script.
Uses restaurant-cafe-static-post.json as the golden source.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

# Import the convert_spec function from our script
# Note: we need to adjust sys.path or import it directly if it's treated as a script,
# but it's well-formed so we can import it.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.convert_brief_specs import convert_spec


@pytest.fixture
def mock_script_paths(tmp_path: Path):
    """Patch the global paths in convert_brief_specs to point to tmp_path."""
    fs_root = tmp_path / "field_sets"
    kd_root = tmp_path / "knowledge_docs"
    
    with mock.patch("scripts.convert_brief_specs._FIELD_SETS_ROOT", fs_root), \
         mock.patch("scripts.convert_brief_specs._KNOWLEDGE_DOCS_ROOT", kd_root):
        yield fs_root, kd_root


def test_convert_restaurant_static_post(mock_script_paths: tuple[Path, Path]) -> None:
    fs_root, kd_root = mock_script_paths
    
    # Grab the real source JSON
    source_json_path = (
        Path(__file__).parent.parent.parent
        / "app" / "project_profiles" / "picasso_fusion" / "_source_brief_specs"
        / "restaurant-cafe-static-post.json"
    )
    
    assert source_json_path.exists(), "Test requires the real source JSON file"
    
    fs_written, kd_written = convert_spec(source_json_path, dry_run=False)
    
    assert fs_written == 1
    # Expect 10 sections = 10 knowledge docs
    assert kd_written == 10
    
    # 1. Check Field Set YAML
    fs_yaml_path = fs_root / "restaurant" / "restaurant_cafe_static_post.yaml"
    assert fs_yaml_path.exists()
    
    with fs_yaml_path.open() as f:
        fs_data = yaml.safe_load(f)
        
    assert fs_data["template_key"] == "restaurant_cafe_static_post"
    assert fs_data["vertical"] == "restaurant"
    assert len(fs_data["fields"]) == 21  # SP1 to SP21
    
    sp1 = next((f for f in fs_data["fields"] if f["code"] == "SP1"), None)
    assert sp1 is not None
    assert sp1["kind"] == "radio"
    assert sp1["required"] is True
    assert len(sp1["options"]) == 14
    assert sp1["section_name"] == "Post Purpose & Product"
    assert sp1["section_order"] == 0
    assert sp1["section_field_order"] == 0

    sp2 = next((f for f in fs_data["fields"] if f["code"] == "SP2"), None)
    assert sp2 is not None
    assert sp2["section_name"] == "Post Purpose & Product"
    assert sp2["section_field_order"] == 1
    
    # 2. Check Knowledge Docs
    kd_dir = kd_root / "restaurant"
    md_files = list(kd_dir.glob("*.md"))
    assert len(md_files) == 10
    
    # Grab the 'Post Purpose & Product' section doc
    purpose_doc = kd_dir / "restaurant_cafe_static_post__post_purpose_product.md"
    assert purpose_doc.exists()
    
    content = purpose_doc.read_text(encoding="utf-8")
    
    # Check frontmatter
    assert "industry: restaurant" in content
    assert "brief_type: restaurant_cafe_static_post" in content
    assert "doc_type: question_guidance" in content
    
    # Check that no raw field codes leaked into the markdown
    for i in range(1, 22):
        assert f"SP{i}" not in content
    
    # Check that prose expands nicely (e.g. mentions options)
    assert "Featured Dish" in content
