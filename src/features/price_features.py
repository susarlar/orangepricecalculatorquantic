"""
Feature engineering from price data.

Creates lag features, rolling statistics, seasonal indicators,
and price momentum signals.
"""
import numpy as np
import pandas as pd

from src.config import LAG_DAYS, ROLLING_WINDOWS


def create_price_features(df: pd.DataFrame, price_col: str = "avg_price") -> pd.DataFrame:
    """Generate all price-derived features.

    Args:
        df: DataFrame with 'date' and price columns, sorted by date.
        price_col: Name of the price column.

    Returns:
        DataFrame with added feature columns.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    # ── Lag features ──
    for lag in LAG_DAYS:
        df[f"price_lag_{lag}d"] = df[price_col].shift(lag)

    # ── Rolling statistics ──
    for window in ROLLING_WINDOWS:
        df[f"price_roll_mean_{window}d"] = df[price_col].rolling(window, min_periods=1).mean()
        df[f"price_roll_std_{window}d"] = df[price_col].rolling(window, min_periods=1).std()
        df[f"price_roll_min_{window}d"] = df[price_col].rolling(window, min_periods=1).min()
        df[f"price_roll_max_{window}d"] = df[price_col].rolling(window, min_periods=1).max()

    # ── Price momentum ──
    for period in [7, 14, 30]:
        df[f"price_change_{period}d"] = df[price_col].pct_change(periods=period)

    # ── Volatility ──
    df["price_volatility_30d"] = df[price_col].rolling(30, min_periods=7).std() / \
                                  df[price_col].rolling(30, min_periods=7).mean()

    # ── Year-over-year ──
    df["price_yoy_change"] = df[price_col].pct_change(periods=365)

    # ── Seasonal features ──
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.fillna(0).astype(int)
        df["is_harvest_season"] = df["month"].isin([11, 12, 1, 2, 3, 4, 5]).astype(int)
        df["season_phase"] = df["month"].map(_get_season_phase)

        # Cyclical encoding for month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Price relative to rolling average ──
    df["price_vs_30d_avg"] = df[price_col] / df["price_roll_mean_30d"] - 1
    df["price_vs_90d_avg"] = df[price_col] / df["price_roll_mean_90d"] - 1

    # ── Min/max price spread (if available) ──
    if "min_price" in df.columns and "max_price" in df.columns:
        df["price_spread"] = df["max_price"] - df["min_price"]
        df["price_spread_pct"] = df["price_spread"] / df[price_col]

    return df


def create_multi_market_features(
    dfs: dict[str, pd.DataFrame],
    target_market: str = "finike",
    price_col: str = "avg_price",
) -> pd.DataFrame:
    """Create features from multiple market prices.

    Args:
        dfs: Dict of {market_name: DataFrame}.
        target_market: The market we're predicting.
        price_col: Price column name.

    Returns:
        DataFrame with cross-market features.
    """
    if target_market not in dfs:
        raise ValueError(f"Target market '{target_market}' not in provided data")

    base = dfs[target_market][["date", price_col]].copy()
    base = base.rename(columns={price_col: f"{target_market}_price"})

    for market, df in dfs.items():
        if market == target_market:
            continue

        market_df = df[["date", price_col]].copy()
        market_df = market_df.rename(columns={price_col: f"{market}_price"})
        base = base.merge(market_df, on="date", how="left")

        # Price differential
        base[f"spread_{target_market}_vs_{market}"] = (
            base[f"{target_market}_price"] - base[f"{market}_price"]
        )

        # Ratio
        base[f"ratio_{target_market}_vs_{market}"] = (
            base[f"{target_market}_price"] / base[f"{market}_price"]
        )

    return base


def _get_season_phase(month: int) -> int:
    """Map month to orange season phase.

    0 = off-season (June-September)
    1 = early harvest (October-November)
    2 = peak harvest (December-February)
    3 = late harvest (March-May)
    """
    if month in [6, 7, 8, 9]:
        return 0
    elif month in [10, 11]:
        return 1
    elif month in [12, 1, 2]:
        return 2
    else:  # 3, 4, 5
        return 3
