"""
Baseline ML models for orange price prediction.

Implements multiple models with time-series cross-validation:
- Seasonal Naive (benchmark)
- Linear Regression
- Random Forest
- XGBoost
- LightGBM
"""
import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelResult:
    """Container for model evaluation results."""
    name: str
    horizon: int
    mae: float
    mape: float
    rmse: float
    r2: float
    predictions: np.ndarray = None
    feature_importance: pd.DataFrame = None


# ─── Feature selection helpers ──────────────────────────────────────────────────

EXCLUDE_COLS = {"date", "product", "market", "unit", "source", "vegetation_health",
                "is_forecast", "month"}


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Get numeric feature columns, excluding targets and metadata."""
    target_cols = {c for c in df.columns if c.startswith("target_")}
    exclude = EXCLUDE_COLS | target_cols | {target_col}

    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    return feature_cols


# ─── Seasonal Naive ─────────────────────────────────────────────────────────────

def seasonal_naive_predict(
    df: pd.DataFrame,
    price_col: str = "avg_price",
    horizon: int = 30,
) -> ModelResult:
    """Seasonal naive: predict using same period last year.

    This is the benchmark — any useful model must beat this.
    """
    df = df.copy()
    df["prediction"] = df[price_col].shift(365)

    target_col = f"target_{horizon}d"
    valid = df.dropna(subset=[target_col, "prediction"])

    if len(valid) < 10:
        logger.warning("Not enough data for seasonal naive evaluation")
        return ModelResult("seasonal_naive", horizon, np.nan, np.nan, np.nan, np.nan)

    y_true = valid[target_col].values
    y_pred = valid["prediction"].values

    return ModelResult(
        name="seasonal_naive",
        horizon=horizon,
        mae=mean_absolute_error(y_true, y_pred),
        mape=mean_absolute_percentage_error(y_true, y_pred),
        rmse=np.sqrt(np.mean((y_true - y_pred) ** 2)),
        r2=r2_score(y_true, y_pred),
        predictions=y_pred,
    )


# ─── Model training with time-series CV ─────────────────────────────────────────

def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    horizon: int = 30,
    n_splits: int = 5,
    price_col: str = "avg_price",
) -> ModelResult:
    """Train and evaluate a model using time-series cross-validation.

    Args:
        df: Feature matrix from build_feature_matrix().
        model_type: One of 'linear', 'random_forest', 'xgboost', 'lightgbm'.
        horizon: Prediction horizon in days.
        n_splits: Number of CV folds.
        price_col: Price column name.

    Returns:
        ModelResult with metrics and feature importance.
    """
    target_col = f"target_{horizon}d"
    feature_cols = get_feature_columns(df, target_col)

    # Drop rows with missing target or features
    valid = df.dropna(subset=[target_col] + feature_cols[:5])  # at least first 5 features
    valid = valid.sort_values("date").reset_index(drop=True)

    if len(valid) < 50:
        logger.warning(f"Only {len(valid)} valid rows — need at least 50")
        return ModelResult(model_type, horizon, np.nan, np.nan, np.nan, np.nan)

    X = valid[feature_cols].fillna(0).values
    y = valid[target_col].values

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = _create_model(model_type)
        if model_type == "linear":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Feature importance (train on full data for importance)
    model = _create_model(model_type)
    if model_type == "linear":
        scaler = StandardScaler()
        model.fit(scaler.fit_transform(X), y)
        importance = np.abs(model.coef_)
    else:
        model.fit(X, y)
        importance = model.feature_importances_

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return ModelResult(
        name=model_type,
        horizon=horizon,
        mae=mean_absolute_error(all_y_true, all_y_pred),
        mape=mean_absolute_percentage_error(all_y_true, all_y_pred),
        rmse=np.sqrt(np.mean((all_y_true - all_y_pred) ** 2)),
        r2=r2_score(all_y_true, all_y_pred),
        predictions=all_y_pred,
        feature_importance=fi,
    )


def _create_model(model_type: str):
    """Create a model instance by type."""
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            random_state=42,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ─── Run all models ─────────────────────────────────────────────────────────────

def run_all_models(
    df: pd.DataFrame,
    horizons: list[int] = None,
) -> pd.DataFrame:
    """Run all baseline models across all prediction horizons.

    Args:
        df: Feature matrix.
        horizons: List of prediction horizons in days.

    Returns:
        DataFrame with comparison of all models.
    """
    if horizons is None:
        horizons = [30, 60, 90]

    model_types = ["linear", "random_forest"]
    if HAS_XGBOOST:
        model_types.append("xgboost")
    if HAS_LIGHTGBM:
        model_types.append("lightgbm")

    results = []

    for horizon in horizons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Horizon: {horizon} days")
        logger.info(f"{'='*50}")

        # Seasonal naive benchmark
        naive = seasonal_naive_predict(df, horizon=horizon)
        results.append(naive)
        logger.info(f"  Seasonal Naive — MAE: {naive.mae:.2f}, MAPE: {naive.mape:.2%}")

        # ML models
        for model_type in model_types:
            logger.info(f"  Training {model_type}...")
            result = train_and_evaluate(df, model_type=model_type, horizon=horizon)
            results.append(result)
            logger.info(
                f"  {model_type} — MAE: {result.mae:.2f}, "
                f"MAPE: {result.mape:.2%}, R²: {result.r2:.3f}"
            )

    # Summary table
    summary = pd.DataFrame([
        {
            "model": r.name,
            "horizon_days": r.horizon,
            "MAE": round(r.mae, 4),
            "MAPE": round(r.mape, 4),
            "RMSE": round(r.rmse, 4),
            "R2": round(r.r2, 4),
        }
        for r in results
    ])

    return summary
