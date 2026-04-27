"""Unit tests for the farmer decision-support model."""
import pytest

from src.models.farmer import DEFAULT_COSTS, _decide, _season_phase, compute_breakeven


def test_default_costs_are_all_positive():
    for k, v in DEFAULT_COSTS.items():
        assert v >= 0, f"{k} should be non-negative"


def test_breakeven_above_sum_of_fixed_costs():
    """With a non-zero commission, breakeven must exceed the simple cost sum."""
    fixed_sum = sum(
        v for k, v in DEFAULT_COSTS.items()
        if k not in ("commission_pct", "cold_storage_daily")
    )
    breakeven = compute_breakeven()
    assert breakeven > fixed_sum


def test_breakeven_no_commission_equals_fixed_sum():
    costs = dict(DEFAULT_COSTS)
    costs["commission_pct"] = 0.0
    fixed_sum = sum(
        v for k, v in costs.items()
        if k not in ("commission_pct", "cold_storage_daily")
    )
    assert compute_breakeven(costs) == pytest.approx(fixed_sum, rel=1e-3)


def test_season_phase_groups_are_correct():
    # Off-season: Jun–Sep
    for m in [6, 7, 8, 9]:
        assert _season_phase(m) == 0
    # Early harvest: Oct–Nov
    for m in [10, 11]:
        assert _season_phase(m) == 1
    # Peak harvest: Dec–Feb
    for m in [12, 1, 2]:
        assert _season_phase(m) == 2
    # Late harvest: Mar–May
    for m in [3, 4, 5]:
        assert _season_phase(m) == 3


def test_decide_recommends_wait_when_below_breakeven_with_no_upside():
    # current price below breakeven, no forecast above current
    forecasts = {7: {"price": 8.0, "change_pct": -10}, 30: {"price": 7.5, "change_pct": -15}}
    rec = _decide(
        current_price=10.0,
        forecasts=forecasts,
        breakeven=15.0,
        costs=DEFAULT_COSTS,
        current_row=None,
    )
    assert rec["action"] == "WAIT"
    assert rec["urgency"] == "low"


def test_decide_recommends_sell_now_when_prices_dropping():
    forecasts = {
        7: {"price": 18.0, "change_pct": -10},
        14: {"price": 17.5, "change_pct": -12.5},
    }
    rec = _decide(
        current_price=20.0,
        forecasts=forecasts,
        breakeven=15.0,
        costs=DEFAULT_COSTS,
        current_row=None,
    )
    assert rec["action"] == "SELL NOW"
    assert rec["urgency"] == "high"


def test_decide_recommends_cold_storage_for_strong_future_upside():
    # 5% storage cost, 50% forecast upside → COLD STORAGE
    forecasts = {30: {"price": 30.0, "change_pct": 50}}
    rec = _decide(
        current_price=20.0,
        forecasts=forecasts,
        breakeven=15.0,
        costs=DEFAULT_COSTS,
        current_row=None,
    )
    assert rec["action"] == "COLD STORAGE"
    assert rec["best_sell_horizon"] == 30
