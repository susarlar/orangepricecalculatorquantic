"""Contract tests for src.utils.labels — display-layer Americanization."""
from __future__ import annotations

from unittest.mock import MagicMock

from src.utils.labels import (
    GLOSSARY,
    LABELS,
    assert_no_turkish_chrome,
    render_glossary_expander,
)


def test_labels_dict_shape():
    """Every LABELS entry is a non-empty string keyed by a string."""
    assert isinstance(LABELS, dict)
    assert LABELS, "LABELS dict must not be empty"
    for key, value in LABELS.items():
        assert isinstance(key, str) and key, f"Bad key: {key!r}"
        assert isinstance(value, str) and value, f"Bad value for {key!r}: {value!r}"


def test_glossary_has_six_terms():
    """Glossary must contain exactly the six terms named in the functional plan."""
    expected = {"Hal", "Narenciye", "TCMB", "IBB", "Antalya/Istanbul Hal", "Hal commission"}
    assert set(GLOSSARY.keys()) == expected
    for term, definition in GLOSSARY.items():
        assert isinstance(definition, str) and len(definition) > 10, (
            f"Glossary entry for {term!r} is too short"
        )


def test_labels_pass_no_turkish_guard():
    """Every value in LABELS must pass the no-Turkish-chrome guard."""
    offenders = [k for k, v in LABELS.items() if not assert_no_turkish_chrome(v)]
    assert not offenders, f"LABELS values contain bare Turkish chrome: {offenders}"


def test_assert_no_turkish_chrome_rejects_bare_token():
    assert assert_no_turkish_chrome("Antalya Hal Price") is False


def test_assert_no_turkish_chrome_accepts_parenthetical_gloss():
    assert assert_no_turkish_chrome("Antalya Wholesale Price (Hal)") is True


def test_assert_no_turkish_chrome_accepts_pure_english():
    assert assert_no_turkish_chrome("Orange Price Forecast (TRY/kg)") is True


def test_render_glossary_expander_uses_sidebar_expander_named_terms():
    """The expander helper must hit st.sidebar.expander('Terms', ...) and write one line per glossary entry."""
    st = MagicMock()
    expander_cm = MagicMock()
    st.sidebar.expander.return_value.__enter__.return_value = expander_cm

    render_glossary_expander(st)

    st.sidebar.expander.assert_called_once_with("Terms", expanded=False)
    assert st.markdown.call_count == len(GLOSSARY), (
        f"Expected {len(GLOSSARY)} markdown calls, got {st.markdown.call_count}"
    )
