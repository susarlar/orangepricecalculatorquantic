"""Unit tests for config constants."""
from src import config


def test_finike_coordinates_are_in_southwestern_turkey():
    """Finike sits on the Mediterranean coast around 36.3°N, 30.1°E."""
    assert 36.0 < config.FINIKE_LAT < 37.0
    assert 29.5 < config.FINIKE_LON < 30.5


def test_finike_bbox_contains_finike_point():
    bbox = config.FINIKE_BBOX
    assert bbox["min_lat"] <= config.FINIKE_LAT <= bbox["max_lat"]
    assert bbox["min_lon"] <= config.FINIKE_LON <= bbox["max_lon"]


def test_prediction_horizons_are_sorted_and_positive():
    horizons = config.PREDICTION_HORIZONS
    assert all(h > 0 for h in horizons)
    assert horizons == sorted(horizons)


def test_alert_thresholds_have_expected_keys():
    expected = {"frost_mild", "frost_severe", "drought_days",
                "ndvi_drop_pct", "fx_spike_pct", "competitor_export_surge_pct"}
    assert expected.issubset(config.ALERT_THRESHOLDS.keys())


def test_severe_frost_is_lower_than_mild_frost():
    """Severe frost threshold must be a colder temperature than mild."""
    thr = config.ALERT_THRESHOLDS
    assert thr["frost_severe"] < thr["frost_mild"]


def test_competitor_countries_have_currencies():
    for country, info in config.COMPETITOR_COUNTRIES.items():
        assert "currency" in info, f"{country} is missing a currency"
        assert "harvest_start" in info
        assert "harvest_end" in info
        assert 1 <= info["harvest_start"] <= 12
        assert 1 <= info["harvest_end"] <= 12
