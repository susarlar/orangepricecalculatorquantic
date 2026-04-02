"""
Track predictions vs actual prices over time.

Saves each day's predictions to a history log. When actual prices become
available for a prediction's target date, computes the error and updates
the log. This lets you monitor model accuracy over time.

Usage:
    # Called automatically by auto_refresh after generating predictions:
    from src.prediction_tracker import log_predictions, evaluate_predictions

    log_predictions(pred_df)        # Save today's predictions
    evaluate_predictions()          # Score past predictions that now have actuals
    report = accuracy_report()      # Get rolling accuracy stats
"""
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

PREDICTION_HISTORY_PATH = PROCESSED_DIR / "prediction_history.csv"
ACCURACY_REPORT_PATH = PROCESSED_DIR / "accuracy_report.csv"

HISTORY_COLUMNS = [
    "date_generated",   # when the prediction was made
    "horizon_days",     # 30, 60, or 90
    "target_date",      # the date the prediction is for
    "predicted_price",  # model output (TL/kg)
    "pred_lower",       # lower bound (if available)
    "pred_upper",       # upper bound (if available)
    "current_price",    # price at time of prediction
    "actual_price",     # filled in later when target_date arrives
    "error",            # predicted - actual
    "abs_error",        # |error|
    "pct_error",        # |error| / actual * 100
    "evaluated",        # True when actual has been filled in
]


def _load_history() -> pd.DataFrame:
    """Load prediction history, creating the file if needed."""
    if PREDICTION_HISTORY_PATH.exists():
        df = pd.read_csv(PREDICTION_HISTORY_PATH)
        df["date_generated"] = pd.to_datetime(df["date_generated"])
        df["target_date"] = pd.to_datetime(df["target_date"])
        return df
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _save_history(df: pd.DataFrame):
    """Save prediction history."""
    PREDICTION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREDICTION_HISTORY_PATH, index=False)


def log_predictions(pred_df: pd.DataFrame):
    """Append today's predictions to the history log.

    Args:
        pred_df: DataFrame from generate_predictions() with columns:
            horizon_days, prediction, date_generated, target_date, current_price,
            and optionally pred_lower, pred_upper.
    """
    if pred_df.empty:
        return

    history = _load_history()

    new_rows = []
    for _, row in pred_df.iterrows():
        gen_date = pd.to_datetime(row["date_generated"])
        horizon = row["horizon_days"]

        # Skip if we already logged this exact prediction
        if not history.empty:
            exists = history[
                (history["date_generated"].dt.date == gen_date.date())
                & (history["horizon_days"] == horizon)
            ]
            if not exists.empty:
                continue

        new_rows.append({
            "date_generated": gen_date,
            "horizon_days": int(horizon),
            "target_date": pd.to_datetime(row["target_date"]),
            "predicted_price": round(row["prediction"], 4),
            "pred_lower": round(row.get("pred_lower", np.nan), 4) if pd.notna(row.get("pred_lower")) else np.nan,
            "pred_upper": round(row.get("pred_upper", np.nan), 4) if pd.notna(row.get("pred_upper")) else np.nan,
            "current_price": round(row["current_price"], 4),
            "actual_price": np.nan,
            "error": np.nan,
            "abs_error": np.nan,
            "pct_error": np.nan,
            "evaluated": False,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        history = pd.concat([history, new_df], ignore_index=True)
        _save_history(history)
        logger.info(f"Logged {len(new_rows)} new predictions to history")
    else:
        logger.info("No new predictions to log (already recorded today)")


def evaluate_predictions():
    """Score past predictions whose target date has passed.

    Looks up the actual price on each target_date from hal_prices.csv
    and computes error metrics.
    """
    history = _load_history()
    if history.empty:
        logger.info("No prediction history to evaluate")
        return

    unevaluated = history[history["evaluated"] != True]  # noqa: E712
    if unevaluated.empty:
        logger.info("All predictions already evaluated")
        return

    # Load actual prices
    prices_path = RAW_DIR / "hal_prices.csv"
    if not prices_path.exists():
        logger.warning("No price data to evaluate against")
        return

    prices = pd.read_csv(prices_path, parse_dates=["date"])
    prices = prices[["date", "avg_price"]].dropna()
    price_by_date = dict(zip(prices["date"].dt.date, prices["avg_price"]))

    today = datetime.now().date()
    evaluated_count = 0

    for idx in unevaluated.index:
        target = history.loc[idx, "target_date"]
        if isinstance(target, str):
            target = pd.to_datetime(target)
        target_date = target.date()

        # Only evaluate if target date has passed
        if target_date > today:
            continue

        # Find the closest actual price within +/- 3 days of target
        actual = None
        for offset in range(4):
            for delta in [offset, -offset]:
                check_date = target_date + pd.Timedelta(days=delta)
                if check_date in price_by_date:
                    actual = price_by_date[check_date]
                    break
            if actual is not None:
                break

        if actual is None:
            continue

        predicted = history.loc[idx, "predicted_price"]
        error = predicted - actual
        abs_error = abs(error)
        pct_error = (abs_error / actual) * 100 if actual != 0 else np.nan

        history.loc[idx, "actual_price"] = round(actual, 4)
        history.loc[idx, "error"] = round(error, 4)
        history.loc[idx, "abs_error"] = round(abs_error, 4)
        history.loc[idx, "pct_error"] = round(pct_error, 2)
        history.loc[idx, "evaluated"] = True
        evaluated_count += 1

    if evaluated_count > 0:
        _save_history(history)
        logger.info(f"Evaluated {evaluated_count} predictions against actuals")
    else:
        logger.info("No predictions ready for evaluation yet (target dates in future)")


def accuracy_report() -> pd.DataFrame:
    """Generate accuracy stats grouped by horizon.

    Returns:
        DataFrame with MAE, MAPE, RMSE, hit_rate (within prediction interval)
        per horizon, plus overall.
    """
    history = _load_history()
    evaluated = history[history["evaluated"] == True]  # noqa: E712

    if evaluated.empty:
        logger.info("No evaluated predictions yet — report will be available after target dates pass")
        return pd.DataFrame()

    rows = []
    for horizon in sorted(evaluated["horizon_days"].unique()):
        subset = evaluated[evaluated["horizon_days"] == horizon]
        n = len(subset)
        mae = subset["abs_error"].mean()
        mape = subset["pct_error"].mean()
        rmse = np.sqrt((subset["error"] ** 2).mean())

        # Hit rate: was actual within [pred_lower, pred_upper]?
        has_interval = subset["pred_lower"].notna() & subset["pred_upper"].notna()
        if has_interval.any():
            interval_subset = subset[has_interval]
            hits = (
                (interval_subset["actual_price"] >= interval_subset["pred_lower"])
                & (interval_subset["actual_price"] <= interval_subset["pred_upper"])
            ).sum()
            hit_rate = (hits / len(interval_subset)) * 100
        else:
            hit_rate = np.nan

        # Direction accuracy: did we correctly predict up/down?
        direction_correct = (
            (subset["predicted_price"] > subset["current_price"])
            == (subset["actual_price"] > subset["current_price"])
        ).sum()
        direction_pct = (direction_correct / n) * 100

        rows.append({
            "horizon_days": horizon,
            "n_predictions": n,
            "mae": round(mae, 4),
            "mape_pct": round(mape, 2),
            "rmse": round(rmse, 4),
            "interval_hit_rate_pct": round(hit_rate, 1) if pd.notna(hit_rate) else np.nan,
            "direction_accuracy_pct": round(direction_pct, 1),
        })

    report_df = pd.DataFrame(rows)

    # Save report
    ACCURACY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(ACCURACY_REPORT_PATH, index=False)
    logger.info(f"Accuracy report saved to {ACCURACY_REPORT_PATH}")

    # Log summary
    for _, row in report_df.iterrows():
        logger.info(
            f"  {int(row['horizon_days'])}d: MAE={row['mae']:.2f} TL/kg, "
            f"MAPE={row['mape_pct']:.1f}%, Direction={row['direction_accuracy_pct']:.0f}% "
            f"(n={int(row['n_predictions'])})"
        )

    return report_df
