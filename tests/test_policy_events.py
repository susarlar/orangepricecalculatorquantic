"""Unit tests for policy event features."""
import pandas as pd

from src.data.policy_events import POLICY_EVENTS, build_policy_events_df, build_policy_features


def test_policy_events_have_required_fields():
    for date, event_type, desc, direction, magnitude in POLICY_EVENTS:
        # Date parses
        assert pd.Timestamp(date)
        # Magnitude is in 1..3
        assert 1 <= magnitude <= 3
        # Direction is up or down
        assert direction in ("up", "down")
        # Description is non-empty
        assert desc.strip()


def test_policy_events_are_in_english():
    """Capstone deliverable: no Turkish-only descriptions remain in the seed data."""
    turkish_only_signals = ["fiyatları", "narenciye", "ihracatı", "değer kaybı", "tarımı"]
    for _, _, desc, _, _ in POLICY_EVENTS:
        for word in turkish_only_signals:
            assert word not in desc.lower(), f"Untranslated descriptor: {desc!r}"


def test_build_policy_events_df_returns_sorted_dataframe():
    df = build_policy_events_df()
    assert len(df) == len(POLICY_EVENTS)
    assert df["date"].is_monotonic_increasing
    assert {"date", "event_type", "description", "impact_direction",
            "impact_magnitude", "impact_sign"}.issubset(df.columns)


def test_impact_sign_matches_direction():
    df = build_policy_events_df()
    up = df[df["impact_direction"] == "up"]
    down = df[df["impact_direction"] == "down"]
    assert (up["impact_sign"] == 1).all()
    assert (down["impact_sign"] == -1).all()


def test_build_policy_features_has_one_row_per_day():
    df = build_policy_features(start_date="2024-01-01", end_date="2024-01-31")
    assert len(df) == 31
    expected = {"policy_impact_score", "policy_event_count_30d",
                "policy_event_count_90d", "policy_impact_30d_avg",
                "policy_impact_90d_avg"}
    assert expected.issubset(df.columns)


def test_policy_events_csv_matches_english_source():
    """The on-disk CSV must contain the same English descriptions as the source list.

    Prevents the drift bug that motivated the news-translation plan: the source
    POLICY_EVENTS was English while the committed CSV held an older Turkish payload.
    """
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "policy_events.csv"
    if not csv_path.exists():
        return  # CSV not regenerated yet (e.g. CI cold start); skip without failing.

    df_disk = pd.read_csv(csv_path)
    df_source = build_policy_events_df()

    source_pairs = set(zip(df_source["date"].dt.strftime("%Y-%m-%d"), df_source["description"]))
    disk_pairs = set(zip(df_disk["date"].astype(str).str[:10], df_disk["description"]))

    missing_from_disk = source_pairs - disk_pairs
    assert not missing_from_disk, (
        f"policy_events.csv is stale — missing rows from POLICY_EVENTS source: "
        f"{list(missing_from_disk)[:3]}"
    )
