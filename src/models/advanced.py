"""
Advanced ML models for orange price prediction.

Improvements over baseline:
1. Log-transformed target (handles exponential price trend)
2. Hyperparameter-tuned XGBoost/LightGBM
3. Stacking ensemble
4. Prediction intervals via quantile regression
5. SHAP explainability
"""
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─── Feature selection ───────────────────────────────────────────────────────────

EXCLUDE_COLS = {
    "date", "product", "market", "unit", "source", "vegetation_health",
    "is_forecast", "eu_orange_origin",
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding targets and metadata."""
    target_cols = {c for c in df.columns if c.startswith("target_")}
    exclude = EXCLUDE_COLS | target_cols

    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def prepare_data(df: pd.DataFrame, horizon: int = 30, use_log: bool = True):
    """Prepare X, y arrays with proper handling.

    Args:
        df: Feature matrix.
        horizon: Prediction horizon in days.
        use_log: If True, predict log(price) to handle exponential trend.

    Returns:
        X, y, feature_cols, valid_df
    """
    target_col = f"target_{horizon}d"
    feature_cols = get_feature_cols(df)

    valid = df.dropna(subset=[target_col]).copy()
    valid = valid.sort_values("date").reset_index(drop=True)

    X = valid[feature_cols].fillna(0).values
    y = valid[target_col].values

    if use_log:
        y = np.log1p(y)

    return X, y, feature_cols, valid


# ─── Tuned XGBoost ───────────────────────────────────────────────────────────────

def train_tuned_xgboost(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Train XGBoost with log-transform and tuned hyperparams."""
    if not HAS_XGB:
        raise ImportError("xgboost required")

    X, y, feature_cols, valid = prepare_data(df, horizon, use_log=True)

    if len(valid) < 100:
        logger.warning(f"Only {len(valid)} rows — insufficient for tuned model")
        return {}

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)
    all_true, all_pred = [], []

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        true = np.expm1(y_test)

        all_true.extend(true)
        all_pred.extend(pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Train final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    final_model.fit(X, y)

    metrics = {
        "model": "xgboost_tuned",
        "horizon": horizon,
        "mae": mean_absolute_error(all_true, all_pred),
        "mape": mean_absolute_percentage_error(all_true, all_pred),
        "rmse": np.sqrt(np.mean((all_true - all_pred) ** 2)),
        "r2": r2_score(all_true, all_pred),
    }

    logger.info(f"Tuned XGBoost ({horizon}d): MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

    return {
        "model": final_model,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "use_log": True,
        "predictions": all_pred,
        "actuals": all_true,
    }


# ─── Tuned LightGBM ──────────────────────────────────────────────────────────────

def train_tuned_lightgbm(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Train LightGBM with log-transform and tuned hyperparams."""
    if not HAS_LGB:
        raise ImportError("lightgbm required")

    X, y, feature_cols, valid = prepare_data(df, horizon, use_log=True)

    tscv = TimeSeriesSplit(n_splits=5)
    all_true, all_pred = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        pred = np.expm1(model.predict(X_test))
        true = np.expm1(y_test)
        all_true.extend(true)
        all_pred.extend(pred)

    all_true, all_pred = np.array(all_true), np.array(all_pred)

    final_model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=-1,
    )
    final_model.fit(X, y)

    metrics = {
        "model": "lightgbm_tuned",
        "horizon": horizon,
        "mae": mean_absolute_error(all_true, all_pred),
        "mape": mean_absolute_percentage_error(all_true, all_pred),
        "rmse": np.sqrt(np.mean((all_true - all_pred) ** 2)),
        "r2": r2_score(all_true, all_pred),
    }

    logger.info(f"Tuned LightGBM ({horizon}d): MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

    return {
        "model": final_model,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "use_log": True,
    }


# ─── Stacking Ensemble ──────────────────────────────────────────────────────────

def train_ensemble(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Train a stacking ensemble of XGBoost + LightGBM + Ridge."""
    X, y, feature_cols, valid = prepare_data(df, horizon, use_log=True)

    estimators = [
        ("ridge", Ridge(alpha=1.0)),
    ]

    if HAS_XGB:
        estimators.append(("xgb", xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, verbosity=0, random_state=42,
        )))

    if HAS_LGB:
        estimators.append(("lgb", lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, verbosity=-1, random_state=42,
        )))

    # Evaluate with outer CV using simple averaging ensemble
    tscv = TimeSeriesSplit(n_splits=5)
    all_true, all_pred = [], []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        preds = []
        for name, est in estimators:
            import copy
            m = copy.deepcopy(est)
            m.fit(X_train, y_train)
            preds.append(m.predict(X_test))

        avg_pred = np.expm1(np.mean(preds, axis=0))
        true = np.expm1(y_test)
        all_true.extend(true)
        all_pred.extend(avg_pred)

    all_true, all_pred = np.array(all_true), np.array(all_pred)

    # Final stacking model
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=0.5),
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X, y)

    metrics = {
        "model": "ensemble_stack",
        "horizon": horizon,
        "mae": mean_absolute_error(all_true, all_pred),
        "mape": mean_absolute_percentage_error(all_true, all_pred),
        "rmse": np.sqrt(np.mean((all_true - all_pred) ** 2)),
        "r2": r2_score(all_true, all_pred),
    }

    logger.info(f"Ensemble ({horizon}d): MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

    return {
        "model": stack,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "use_log": True,
    }


# ─── Prediction Intervals ───────────────────────────────────────────────────────

def train_quantile_model(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Train quantile regression models for prediction intervals.

    Returns median + 10th/90th percentile predictions (80% interval).
    """
    X, y_log, feature_cols, valid = prepare_data(df, horizon, use_log=True)
    y = np.expm1(y_log)

    quantiles = {"lower": 0.10, "median": 0.50, "upper": 0.90}
    models = {}

    for name, alpha in quantiles.items():
        model = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            loss="quantile", alpha=alpha, random_state=42,
        )
        model.fit(X, y)
        models[name] = model

    return {
        "models": models,
        "feature_cols": feature_cols,
        "use_log": False,
    }


def predict_with_intervals(quantile_result: dict, X: np.ndarray) -> pd.DataFrame:
    """Generate predictions with confidence intervals."""
    models = quantile_result["models"]

    return pd.DataFrame({
        "pred_lower": models["lower"].predict(X),
        "pred_median": models["median"].predict(X),
        "pred_upper": models["upper"].predict(X),
    })


# ─── SHAP Explainability ────────────────────────────────────────────────────────

def compute_shap_values(model_result: dict, df: pd.DataFrame, max_samples: int = 500) -> dict:
    """Compute SHAP values for model explainability.

    Args:
        model_result: Dict from train_tuned_xgboost or similar.
        df: Feature matrix.
        max_samples: Max samples for SHAP computation.

    Returns:
        Dict with shap_values, feature_importance, expected_value.
    """
    if not HAS_SHAP:
        raise ImportError("shap required: pip install shap")

    model = model_result["model"]
    feature_cols = model_result["feature_cols"]

    X = df[feature_cols].fillna(0).values
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Global feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "shap_importance": np.abs(shap_values).mean(axis=0),
    }).sort_values("shap_importance", ascending=False)

    logger.info(f"SHAP computed for {len(X_sample)} samples, {len(feature_cols)} features")

    return {
        "shap_values": shap_values,
        "X_sample": X_sample,
        "feature_cols": feature_cols,
        "importance": importance,
        "expected_value": explainer.expected_value,
    }


# ─── Save/Load Models ───────────────────────────────────────────────────────────

def save_model(model_result: dict, name: str):
    """Save a trained model to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model_result, path)
    logger.info(f"Model saved to {path}")


def load_model(name: str) -> dict:
    """Load a trained model from disk."""
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


# ─── Run all advanced models ─────────────────────────────────────────────────────

def run_advanced_models(df: pd.DataFrame, horizons: list[int] = None) -> pd.DataFrame:
    """Train all advanced models and return comparison table."""
    if horizons is None:
        horizons = [30, 60, 90]

    results = []

    for horizon in horizons:
        logger.info(f"\n{'='*50}")
        logger.info(f"Advanced models — Horizon: {horizon} days")
        logger.info(f"{'='*50}")

        # Tuned XGBoost
        if HAS_XGB:
            xgb_result = train_tuned_xgboost(df, horizon)
            if xgb_result:
                results.append(xgb_result["metrics"])
                save_model(xgb_result, f"xgboost_tuned_{horizon}d")

        # Tuned LightGBM
        if HAS_LGB:
            lgb_result = train_tuned_lightgbm(df, horizon)
            if lgb_result:
                results.append(lgb_result["metrics"])
                save_model(lgb_result, f"lightgbm_tuned_{horizon}d")

        # Stacking ensemble
        ens_result = train_ensemble(df, horizon)
        if ens_result:
            results.append(ens_result["metrics"])
            save_model(ens_result, f"ensemble_{horizon}d")

        # Quantile model for intervals
        q_result = train_quantile_model(df, horizon)
        save_model(q_result, f"quantile_{horizon}d")

    summary = pd.DataFrame(results)
    summary = summary.rename(columns={"model": "model", "horizon": "horizon_days",
                                       "mae": "MAE", "mape": "MAPE", "rmse": "RMSE", "r2": "R2"})
    return summary
