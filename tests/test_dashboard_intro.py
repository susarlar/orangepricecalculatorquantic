"""Unit tests for the dashboard Welcome / About intro page."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_intro_harbor_jpeg_is_committed_to_repo_root():
    """finikeliman.jpeg must ship in the Render deploy artifact (AC2)."""
    assert (ROOT / "finikeliman.jpeg").exists(), (
        "finikeliman.jpeg missing from project root — Render will render a "
        "broken-image placeholder on the intro page."
    )


def test_intro_orange_jpeg_is_committed_to_repo_root():
    """portakal.jpeg must ship in the Render deploy artifact (AC2)."""
    assert (ROOT / "portakal.jpeg").exists(), (
        "portakal.jpeg missing from project root — Render will render a "
        "broken-image placeholder on the intro page."
    )


def _read_constant_from_dashboard(name: str) -> str:
    source = (ROOT / "dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Constant {name} not found in dashboard.py")


def test_intro_copy_is_at_most_160_words():
    """AC3: intro copy ≤ 150 words; 10-word tolerance per the functional plan."""
    copy = _read_constant_from_dashboard("INTRO_COPY")
    word_count = len(copy.split())
    assert word_count <= 160, (
        f"INTRO_COPY is {word_count} words; cap is 160 (target 150, +10 tolerance)."
    )


def test_intro_copy_names_finike_antalya_and_oranges():
    """AC3: copy must name Finike's location and the produce."""
    copy = _read_constant_from_dashboard("INTRO_COPY").lower()
    assert "finike" in copy
    assert "antalya" in copy
    assert "turkey" in copy
    assert "orange" in copy


def test_intro_copy_names_the_audience():
    """AC3: copy must name farmers, traders, exporters, analysts."""
    copy = _read_constant_from_dashboard("INTRO_COPY").lower()
    for audience in ("farmer", "trader", "exporter", "analyst"):
        assert audience in copy, f"Intro copy is missing audience term: {audience}"


def test_intro_copy_uses_american_spelling():
    """AC3: prefer American English (no -our, no -ise)."""
    copy = _read_constant_from_dashboard("INTRO_COPY").lower()
    british_spellings = [
        "colour", "favour", "behaviour", "organise", "analyse",
        "centre", "metre", "kilometre",
    ]
    found = [w for w in british_spellings if w in copy]
    assert not found, f"Intro copy contains British spellings: {found}"


def test_intro_attribution_spells_su_sarlar_and_quantic_exactly():
    """AC4: 'Su Sarlar' and 'Quantic' are spelled exactly as specified."""
    attribution = _read_constant_from_dashboard("INTRO_ATTRIBUTION")
    assert "Su Sarlar" in attribution
    assert "Quantic" in attribution
    assert "MSc Software Engineering Capstone" in attribution
