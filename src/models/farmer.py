"""
Farmer-focused orange price prediction and decision support.

Target audience: Finike portakal üreticisi (orange grower)
Target variable: Antalya Hal portakal prices (nearest market to Finike)

Key decisions this model supports:
1. WHEN to sell — optimal timing within the season
2. HOLD or SELL — should I wait for higher prices?
3. COLD STORE — worth the cost to store and sell later?
4. Price forecast with confidence intervals
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR

logger = logging.getLogger(__name__)

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

# ─── Farmer cost parameters ─────────────────────────────────────────────────────

# Typical Finike portakal üretim maliyeti (2025-2026 sezonu, TL/kg)
DEFAULT_COSTS = {
    "hasat_iscilik": 5.0,       # Harvest labor
    "nakliye_hale": 2.0,        # Transport to hal (Finike→Antalya ~80km)
    "komisyon_pct": 8.0,        # Hal commission (% of sale price)
    "ambalaj": 1.5,             # Packaging
    "ilac_gubre": 3.0,          # Pesticide + fertilizer (amortized per kg)
    "sulama": 1.0,              # Irrigation
    "soguk_hava_gunluk": 0.15,  # Cold storage per kg per day
}


def compute_breakeven(costs: dict = None) -> float:
    """Compute breakeven price (TL/kg) for Finike farmer.

    This is the minimum hal price needed to cover costs.
    Commission is applied on top, so:
        breakeven = fixed_costs / (1 - commission_pct/100)
    """
    if costs is None:
        costs = DEFAULT_COSTS

    fixed = sum(v for k, v in costs.items() if k != "komisyon_pct" and k != "soguk_hava_gunluk")
    commission = costs.get("komisyon_pct", 8.0) / 100
    breakeven = fixed / (1 - commission)
    return round(breakeven, 2)


# ─── Data preparation ────────────────────────────────────────────────────────────

def build_farmer_features() -> pd.DataFrame:
    """Build feature matrix centered on Antalya Hal portakal prices.

    Merges:
    - Antalya portakal prices (target)
    - Istanbul portakal prices (downstream signal)
    - Weather (Finike region)
    - FX rates
    - Policy events
    - Demand features (Ramadan, tourism)
    - Seasonal features
    """
    # Load Antalya prices — aggregate both varieties to daily avg
    antalya = pd.read_csv(RAW_DIR / "antalya_hal_prices.csv", parse_dates=["date"])
    portakal = antalya[antalya["product"].str.contains("Portakal", case=False)].copy()

    # Daily aggregate across varieties
    daily = portakal.groupby("date").agg(
        antalya_min=("min_price", "min"),
        antalya_max=("max_price", "max"),
        antalya_avg=("avg_price", "mean"),
        n_varieties=("product", "nunique"),
    ).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)

    # Per-variety prices as separate features
    for variety in ["Sıkmalık", "Valencia"]:
        var_data = portakal[portakal["product"].str.contains(variety, case=False)]
        var_daily = var_data.groupby("date")["avg_price"].mean().reset_index()
        var_daily = var_daily.rename(columns={"avg_price": f"antalya_{variety.lower()}_price"})
        daily = daily.merge(var_daily, on="date", how="left")

    # ── Price features ──
    price_col = "antalya_avg"
    for lag in [1, 3, 7, 14, 30]:
        daily[f"price_lag_{lag}d"] = daily[price_col].shift(lag)

    for window in [7, 14, 30]:
        daily[f"price_ma_{window}d"] = daily[price_col].rolling(window, min_periods=1).mean()
        daily[f"price_std_{window}d"] = daily[price_col].rolling(window, min_periods=1).std()

    for period in [7, 14, 30]:
        daily[f"price_change_{period}d"] = daily[price_col].pct_change(period)

    daily["price_spread"] = daily["antalya_max"] - daily["antalya_min"]
    daily["price_spread_pct"] = daily["price_spread"] / daily[price_col]

    # ── Istanbul prices as feature (downstream market) ──
    istanbul = pd.read_csv(RAW_DIR / "hal_prices.csv", parse_dates=["date"])
    if "avg_price" not in istanbul.columns:
        istanbul["avg_price"] = (istanbul["min_price"] + istanbul["max_price"]) / 2
    ist_daily = istanbul.groupby("date")["avg_price"].mean().reset_index()
    ist_daily = ist_daily.rename(columns={"avg_price": "istanbul_price"})
    daily = daily.merge(ist_daily, on="date", how="left")

    # Antalya→Istanbul spread (transport + margin signal)
    daily["antalya_istanbul_spread"] = daily["istanbul_price"] - daily[price_col]
    daily["antalya_istanbul_ratio"] = daily["istanbul_price"] / daily[price_col]

    # ── Seasonal features ──
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

    # Season phase
    daily["is_harvest"] = daily["month"].isin([11, 12, 1, 2, 3, 4, 5]).astype(int)
    daily["season_phase"] = daily["month"].map(_season_phase)

    # ── Weather (Finike) ──
    weather_path = RAW_DIR / "weather_finike.csv"
    if weather_path.exists():
        weather = pd.read_csv(weather_path, parse_dates=["date"])
        w_cols = ["date", "temp_max", "temp_min", "temp_mean", "precipitation", "humidity"]
        w_cols = [c for c in w_cols if c in weather.columns]
        daily = daily.merge(weather[w_cols], on="date", how="left")

        if "temp_min" in daily.columns:
            daily["frost"] = (daily["temp_min"] < 0).astype(int)
            daily["frost_7d"] = daily["frost"].rolling(7, min_periods=1).sum()

        if "precipitation" in daily.columns:
            daily["precip_7d"] = daily["precipitation"].rolling(7, min_periods=1).sum()
            daily["precip_30d"] = daily["precipitation"].rolling(30, min_periods=1).sum()

    # ── FX rates ──
    fx_path = RAW_DIR / "fx_rates.csv"
    if fx_path.exists():
        fx = pd.read_csv(fx_path, parse_dates=["date"])
        usd_col = "TRY_per_USD" if "TRY_per_USD" in fx.columns else None
        if usd_col:
            daily = daily.merge(fx[["date", usd_col]], on="date", how="left")
            daily[usd_col] = daily[usd_col].ffill()

    # ── Policy events ──
    policy_path = RAW_DIR / "policy_features.csv"
    if policy_path.exists():
        policy = pd.read_csv(policy_path, parse_dates=["date"])
        p_cols = ["date", "policy_impact_score", "policy_impact_30d_avg",
                  "event_frost_active", "event_economic_active"]
        p_cols = [c for c in p_cols if c in policy.columns]
        daily = daily.merge(policy[p_cols], on="date", how="left")

    # ── Demand features ──
    demand_path = RAW_DIR / "demand_features.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path, parse_dates=["date"])
        d_cols = ["date", "ramadan_active", "input_cost_index", "tourism_intensity", "cpi_index"]
        d_cols = [c for c in d_cols if c in demand.columns]
        daily = daily.merge(demand[d_cols], on="date", how="left")

    # ── Target variables ──
    for horizon in [7, 14, 30, 60, 90]:
        daily[f"target_{horizon}d"] = daily[price_col].shift(-horizon)
        daily[f"target_{horizon}d_change_pct"] = (
            (daily[f"target_{horizon}d"] / daily[price_col]) - 1
        ) * 100

    # Optimal sell signal: price will drop >5% in next 14 days
    daily["sell_signal_14d"] = (daily["target_14d_change_pct"] < -5).astype(int)
    # Hold signal: price will rise >5% in next 14 days
    daily["hold_signal_14d"] = (daily["target_14d_change_pct"] > 5).astype(int)

    logger.info(f"Farmer features: {daily.shape[0]} rows, {daily.shape[1]} columns")
    return daily


def _season_phase(month: int) -> int:
    """0=off-season, 1=early harvest, 2=peak harvest, 3=late harvest."""
    if month in [6, 7, 8, 9]:
        return 0
    elif month in [10, 11]:
        return 1
    elif month in [12, 1, 2]:
        return 2
    else:
        return 3


# ─── Model training ──────────────────────────────────────────────────────────────

EXCLUDE = {"date", "product", "market", "unit", "n_varieties"}


def _get_feature_cols(df, target_col):
    targets = {c for c in df.columns if c.startswith("target_") or c.endswith("_signal_14d")}
    exclude = EXCLUDE | targets | {target_col}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def train_farmer_model(df: pd.DataFrame, horizon: int = 30) -> dict:
    """Train XGBoost model for Antalya portakal price prediction."""
    target_col = f"target_{horizon}d"
    feature_cols = _get_feature_cols(df, target_col)

    valid = df.dropna(subset=[target_col]).copy()
    valid = valid.sort_values("date").reset_index(drop=True)

    if len(valid) < 60:
        logger.warning(f"Only {len(valid)} rows — need more data")
        return {}

    X = valid[feature_cols].fillna(0).values
    y = np.log1p(valid[target_col].values)

    # Time series CV
    n_splits = min(3, max(2, len(valid) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_true, all_pred = [], []

    for train_idx, test_idx in tscv.split(X):
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            random_state=42, verbosity=0,
        ) if HAS_XGB else Ridge(alpha=1.0)

        model.fit(X[train_idx], y[train_idx])
        pred = np.expm1(model.predict(X[test_idx]))
        true = np.expm1(y[test_idx])
        all_true.extend(true)
        all_pred.extend(pred)

    all_true, all_pred = np.array(all_true), np.array(all_pred)

    # Final model
    final = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        random_state=42, verbosity=0,
    ) if HAS_XGB else Ridge(alpha=1.0)
    final.fit(X, y)

    # Quantile models for intervals
    q_models = {}
    for name, alpha in [("lower", 0.10), ("upper", 0.90)]:
        q = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            loss="quantile", alpha=alpha, random_state=42,
        )
        q.fit(X, np.expm1(y))
        q_models[name] = q

    metrics = {
        "mae": mean_absolute_error(all_true, all_pred),
        "mape": mean_absolute_percentage_error(all_true, all_pred),
        "r2": r2_score(all_true, all_pred),
    }

    logger.info(f"Farmer model ({horizon}d): MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")

    result = {
        "model": final,
        "quantile_models": q_models,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "use_log": True,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, MODELS_DIR / f"farmer_{horizon}d.joblib")

    return result


# ─── Decision support ────────────────────────────────────────────────────────────

def generate_farmer_advice(df: pd.DataFrame, costs: dict = None) -> dict:
    """Generate actionable advice for a Finike farmer.

    Returns:
        Dict with current_price, forecasts, recommendation, breakeven, etc.
    """
    if costs is None:
        costs = DEFAULT_COSTS

    breakeven = compute_breakeven(costs)
    current = df.iloc[-1]
    current_price = current["antalya_avg"]

    # Load models and predict
    forecasts = {}
    for horizon in [7, 14, 30, 60, 90]:
        model_path = MODELS_DIR / f"farmer_{horizon}d.joblib"
        if model_path.exists():
            model_data = joblib.load(model_path)
            feature_cols = model_data["feature_cols"]
            X = df.iloc[-1:][feature_cols].fillna(0).values

            # Point prediction
            pred = model_data["model"].predict(X)
            if model_data.get("use_log"):
                pred = np.expm1(pred)

            # Intervals
            q_models = model_data.get("quantile_models", {})
            lower = q_models["lower"].predict(X)[0] if "lower" in q_models else pred[0] * 0.85
            upper = q_models["upper"].predict(X)[0] if "upper" in q_models else pred[0] * 1.15

            forecasts[horizon] = {
                "price": float(pred[0]),
                "lower": float(lower),
                "upper": float(upper),
                "change_pct": float((pred[0] / current_price - 1) * 100),
                "target_date": (datetime.now() + timedelta(days=horizon)).strftime("%Y-%m-%d"),
            }

    # Decision logic
    recommendation = _decide(current_price, forecasts, breakeven, costs, current)

    # Seasonal context
    month = datetime.now().month
    season_info = _season_context(month)

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "current_price": round(current_price, 2),
        "breakeven_price": breakeven,
        "margin_per_kg": round(current_price - breakeven, 2),
        "margin_pct": round((current_price / breakeven - 1) * 100, 1),
        "forecasts": forecasts,
        "recommendation": recommendation,
        "season_info": season_info,
        "costs": costs,
    }


def _decide(current_price, forecasts, breakeven, costs, current_row) -> dict:
    """Generate sell/hold/store recommendation."""
    cold_cost_daily = costs.get("soguk_hava_gunluk", 0.15)

    # Check if selling now is above breakeven
    profitable_now = current_price > breakeven

    # Check future forecasts
    best_horizon = None
    best_price = current_price
    best_net_gain = 0

    for horizon, fc in forecasts.items():
        future_price = fc["price"]
        storage_cost = cold_cost_daily * horizon
        net_future = future_price - storage_cost
        net_gain = net_future - current_price

        if net_gain > best_net_gain:
            best_net_gain = net_gain
            best_horizon = horizon
            best_price = future_price

    # Decision
    if not profitable_now and best_net_gain <= 0:
        action = "BEKLE"
        reason = f"Mevcut fiyat ({current_price:.1f} ₺) maliyet altında ({breakeven:.1f} ₺). Fiyat artışı bekleniyor mu kontrol edin."
        urgency = "low"
    elif best_horizon and best_net_gain > current_price * 0.05:
        storage_cost = cold_cost_daily * best_horizon
        action = "SOĞUK HAVA"
        reason = (
            f"{best_horizon} gün depolayıp satın. "
            f"Beklenen fiyat: {best_price:.1f} ₺/kg, "
            f"depolama maliyeti: {storage_cost:.1f} ₺/kg, "
            f"net kazanç: +{best_net_gain:.1f} ₺/kg."
        )
        urgency = "medium"
    elif profitable_now:
        margin = current_price - breakeven
        if any(fc["change_pct"] < -5 for fc in forecasts.values()):
            action = "ŞİMDİ SAT"
            reason = f"Fiyatlar düşecek. Mevcut marj: {margin:.1f} ₺/kg ({(margin/breakeven)*100:.0f}%). Hemen satmak en karlı."
            urgency = "high"
        else:
            action = "SAT"
            reason = f"Karlı satış mümkün. Marj: {margin:.1f} ₺/kg. Önemli bir artış beklenmiyor."
            urgency = "medium"
    else:
        action = "BEKLE"
        reason = "Piyasa koşulları belirsiz. Fiyat hareketlerini takip edin."
        urgency = "low"

    return {
        "action": action,
        "reason": reason,
        "urgency": urgency,
        "best_sell_horizon": best_horizon,
        "best_expected_price": round(best_price, 2),
    }


def _season_context(month: int) -> dict:
    """Provide seasonal context for the farmer."""
    contexts = {
        1: {"phase": "Hasat Zirvesi", "advice": "Arz yüksek, fiyatlar düşük. Kaliteli ürünle öne çıkın."},
        2: {"phase": "Hasat Zirvesi", "advice": "Washington Navel sezonu. Fiyatlar dipte olabilir."},
        3: {"phase": "Geç Hasat", "advice": "Valencia sezonu başlıyor. Fiyatlar yükselmeye başlayabilir."},
        4: {"phase": "Sezon Sonu", "advice": "Arz azalıyor, fiyatlar yükseliyor. Kalan ürünü değerlendirin."},
        5: {"phase": "Sezon Sonu", "advice": "Son hasat. Depolanabilecek portakal varsa soğuk havaya koyun."},
        6: {"phase": "Sezon Dışı", "advice": "Hasat bitti. Soğuk havadaki ürünü satma zamanı gelebilir."},
        7: {"phase": "Sezon Dışı", "advice": "Fiyatlar yüksek ama arz çok düşük. Depo varsa iyi fiyat."},
        8: {"phase": "Sezon Dışı", "advice": "En yüksek fiyat dönemi yaklaşıyor."},
        9: {"phase": "Sezon Dışı", "advice": "Fiyatlar zirveye yakın. Ekim öncesi son fırsat."},
        10: {"phase": "Erken Hasat", "advice": "Yeni sezon başlıyor. İlk hasatlar yüksek fiyat alabilir."},
        11: {"phase": "Hasat Başlangıcı", "advice": "Hasat yoğunlaşıyor. Erken satış avantajlı olabilir."},
        12: {"phase": "Hasat Zirvesi", "advice": "Arz artıyor, fiyatlar düşüyor. Hızlı satış önemli."},
    }
    return contexts.get(month, {"phase": "Bilinmiyor", "advice": ""})


# ─── Train all horizons ──────────────────────────────────────────────────────────

def train_all_farmer_models() -> pd.DataFrame:
    """Build features and train models for all horizons."""
    logger.info("Building farmer feature matrix...")
    df = build_farmer_features()

    # Save feature matrix
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "farmer_features.csv", index=False)

    results = []
    for horizon in [7, 14, 30, 60, 90]:
        logger.info(f"\nTraining {horizon}-day model...")
        result = train_farmer_model(df, horizon)
        if result:
            results.append({
                "horizon": horizon,
                **result["metrics"],
            })

    summary = pd.DataFrame(results)
    summary.to_csv(PROCESSED_DIR / "farmer_model_results.csv", index=False)

    # Generate current advice
    advice = generate_farmer_advice(df)
    logger.info(f"\n{'='*50}")
    logger.info(f"PORTAKAL ÇİFTÇİ TAVSİYESİ")
    logger.info(f"{'='*50}")
    logger.info(f"Güncel fiyat: {advice['current_price']:.1f} ₺/kg")
    logger.info(f"Maliyet: {advice['breakeven_price']:.1f} ₺/kg")
    logger.info(f"Marj: {advice['margin_per_kg']:.1f} ₺/kg ({advice['margin_pct']:.0f}%)")
    logger.info(f"Sezon: {advice['season_info']['phase']}")
    logger.info(f"Tavsiye: {advice['recommendation']['action']}")
    logger.info(f"Neden: {advice['recommendation']['reason']}")

    for h, fc in advice["forecasts"].items():
        logger.info(f"  {h:>2}d tahmin: {fc['price']:.1f} ₺ ({fc['change_pct']:+.1f}%) [{fc['lower']:.1f}-{fc['upper']:.1f}]")

    # Save advice
    import json
    advice_path = PROCESSED_DIR / "farmer_advice.json"
    with open(advice_path, "w", encoding="utf-8") as f:
        json.dump(advice, f, ensure_ascii=False, indent=2)

    return summary
