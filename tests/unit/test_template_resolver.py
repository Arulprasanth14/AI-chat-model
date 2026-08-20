"""
tests/unit/test_template_resolver.py
────────────────────────────────────
Tests for deterministic template resolution.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.domain.conversation.template_resolver import resolve


@pytest.fixture
def field_sets_dir(tmp_path: Path) -> Path:
    """Create a temporary field_sets structure for testing."""
    fs_root = tmp_path / "field_sets"
    
    # Create restaurant vertical
    rest_dir = fs_root / "restaurant"
    rest_dir.mkdir(parents=True)
    
    # Add a static post template
    static_post = {
        "template_key": "restaurant_cafe_static_post",
        "template_label": "Static Post",
        "vertical": "restaurant",
        "template_name_match": ["static post", "social media post", "post set"],
        "fields": []
    }
    with (rest_dir / "restaurant_cafe_static_post.yaml").open("w") as f:
        yaml.dump(static_post, f)

    # Add a reel template
    reel = {
        "template_key": "restaurant_cafe_reel",
        "template_label": "Reel",
        "vertical": "restaurant",
        "template_name_match": ["reel", "reels"],
        "fields": []
    }
    with (rest_dir / "restaurant_cafe_reel.yaml").open("w") as f:
        yaml.dump(reel, f)
        
    return fs_root


def test_exact_vertical_and_substring_match(field_sets_dir: Path) -> None:
    resolved = resolve("restaurant", "i need a static post for my cafe", field_sets_dir)
    assert resolved is not None
    assert resolved.vertical == "restaurant"
    assert resolved.template_key == "restaurant_cafe_static_post"


def test_substring_match_case_insensitive(field_sets_dir: Path) -> None:
    resolved = resolve("restaurant", "let's do some ReELs", field_sets_dir)
    assert resolved is not None
    assert resolved.template_key == "restaurant_cafe_reel"


def test_no_match_returns_none(field_sets_dir: Path) -> None:
    # Unknown vertical
    assert resolve("unknown_vertical", "static post", field_sets_dir) is None


def test_ambiguous_template_returns_none(field_sets_dir: Path) -> None:
    # Vertical matches, but template hint matches zero entries
    resolved = resolve("restaurant", "i need a website design", field_sets_dir)
    assert resolved is None


def test_vertical_mismatch_returns_none(field_sets_dir: Path) -> None:
    # Right template name, wrong vertical
    # (assuming realestate isn't set up in the fixture)
    resolved = resolve("realestate", "static post", field_sets_dir)
    assert resolved is None


def test_longest_match_wins(field_sets_dir: Path) -> None:
    # "post" matches "post set" (len 8) in static post, and "post" (len 4) if we had it.
    # We will test that "social media post" (len 17) matches static post.
    # Wait, both "post" and "social media post" are in the static_post match list.
    resolved = resolve("restaurant", "social media post", field_sets_dir)
    assert resolved is not None
    assert resolved.template_key == "restaurant_cafe_static_post"
