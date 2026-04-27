"""
Orange Price Predictor — Interactive Dashboard

Run: streamlit run dashboard.py
"""
import json
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Setup ───────────────────────────────────────────────────────────────────────

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Orange Price Predictor",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=300)
def load_data():
    """Load all data files."""
    data = {}

    prices_path = RAW_DIR / "hal_prices.csv"
    if prices_path.exists():
        df = pd.read_csv(prices_path, parse_dates=["date"])
        if "avg_price" not in df.columns:
            df["avg_price"] = (df["min_price"] + df["max_price"]) / 2
        data["prices"] = df

    weather_path = RAW_DIR / "weather_finike.csv"
    if weather_path.exists():
        data["weather"] = pd.read_csv(weather_path, parse_dates=["date"])

    fx_path = RAW_DIR / "fx_rates.csv"
    if fx_path.exists():
        data["fx"] = pd.read_csv(fx_path, parse_dates=["date"])

    foreign_path = RAW_DIR / "foreign_markets.csv"
    if foreign_path.exists():
        data["foreign"] = pd.read_csv(foreign_path, parse_dates=["date"])

    events_path = RAW_DIR / "policy_events.csv"
    if events_path.exists():
        data["events"] = pd.read_csv(events_path, parse_dates=["date"])

    policy_path = RAW_DIR / "policy_features.csv"
    if policy_path.exists():
        data["policy"] = pd.read_csv(policy_path, parse_dates=["date"])

    features_path = PROCESSED_DIR / "feature_matrix.csv"
    if features_path.exists():
        data["features"] = pd.read_csv(features_path, parse_dates=["date"])

    results_path = PROCESSED_DIR / "model_results.csv"
    if results_path.exists():
        data["results"] = pd.read_csv(results_path)

    shap_path = PROCESSED_DIR / "shap_importance.csv"
    if shap_path.exists():
        data["shap"] = pd.read_csv(shap_path)

    predictions_path = PROCESSED_DIR / "latest_predictions.csv"
    if predictions_path.exists():
        data["predictions"] = pd.read_csv(predictions_path)

    alerts_path = PROCESSED_DIR / "latest_alerts.txt"
    if alerts_path.exists():
        data["alerts_text"] = alerts_path.read_text(encoding="utf-8")

    demand_path = RAW_DIR / "demand_features.csv"
    if demand_path.exists():
        data["demand"] = pd.read_csv(demand_path, parse_dates=["date"])

    trends_path = RAW_DIR / "google_trends.csv"
    if trends_path.exists():
        data["trends"] = pd.read_csv(trends_path, parse_dates=["date"])

    antalya_path = RAW_DIR / "antalya_hal_prices.csv"
    if antalya_path.exists():
        data["antalya"] = pd.read_csv(antalya_path, parse_dates=["date"])

    farmer_advice_path = PROCESSED_DIR / "farmer_advice.json"
    if farmer_advice_path.exists():
        data["farmer_advice"] = json.loads(farmer_advice_path.read_text(encoding="utf-8"))

    farmer_results_path = PROCESSED_DIR / "farmer_model_results.csv"
    if farmer_results_path.exists():
        data["farmer_results"] = pd.read_csv(farmer_results_path)

    pred_history_path = PROCESSED_DIR / "prediction_history.csv"
    if pred_history_path.exists():
        data["pred_history"] = pd.read_csv(pred_history_path, parse_dates=["date_generated", "target_date"])

    accuracy_path = PROCESSED_DIR / "accuracy_report.csv"
    if accuracy_path.exists():
        data["accuracy"] = pd.read_csv(accuracy_path)

    refresh_log_path = PROCESSED_DIR / "refresh_log.csv"
    if refresh_log_path.exists():
        data["refresh_log"] = pd.read_csv(refresh_log_path)

    return data


data = load_data()


def _freshness_summary(data: dict) -> list[tuple[str, pd.Timestamp | None]]:
    """Return (label, last_date) pairs for each data source shown on the dashboard."""
    items: list[tuple[str, pd.Timestamp | None]] = []

    def _last(df_key: str, date_col: str = "date") -> pd.Timestamp | None:
        if df_key in data and date_col in data[df_key].columns:
            col = pd.to_datetime(data[df_key][date_col], errors="coerce")
            if col.notna().any():
                return col.max()
        return None

    items.append(("Istanbul Hal prices", _last("prices")))
    items.append(("Antalya Hal prices", _last("antalya")))
    items.append(("Weather (Finike)", _last("weather")))
    items.append(("FX rates", _last("fx")))
    items.append(("Demand / policy", _last("demand")))
    items.append(("Google Trends", _last("trends")))

    if "farmer_advice" in data and isinstance(data["farmer_advice"], dict):
        adv = data["farmer_advice"]
        last_pd = adv.get("last_price_date") or adv.get("date")
        if last_pd:
            items.append(("Farmer advice (Antalya)", pd.Timestamp(last_pd)))

    return items


def render_freshness_banner(data: dict) -> None:
    """Render a top banner showing per-source last-data date + today."""
    today = pd.Timestamp.today().normalize()
    summary = _freshness_summary(data)
    stale_items = [(label, d) for label, d in summary if d is None or (today - d).days > 2]

    header = f"📅 Today: **{today.strftime('%d %B %Y')}**"
    if not stale_items:
        st.success(f"{header} — All data sources are up to date.")
    else:
        max_lag = max(((today - d).days if d is not None else 999) for _, d in stale_items)
        st.warning(f"{header} — {len(stale_items)} data source(s) stale (up to {max_lag} days).")

    with st.expander("Data Freshness Status", expanded=bool(stale_items)):
        rows = []
        for label, d in summary:
            if d is None:
                status = "❌ Not found"
                last_str = "—"
                age = "—"
            else:
                age_days = (today - d).days
                last_str = d.strftime("%Y-%m-%d")
                age = f"{age_days} days"
                if age_days <= 2:
                    status = "✅ Fresh"
                elif age_days <= 7:
                    status = "⚠️ Slightly stale"
                else:
                    status = "🔴 Stale"
            rows.append({"Source": label, "Last Date": last_str, "Age": age, "Status": status})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


render_freshness_banner(data)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

st.sidebar.title("🍊 Orange Dashboard")
st.sidebar.markdown(f"**Today:** {pd.Timestamp.today().strftime('%Y-%m-%d')}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Page",
    ["Farmer Panel", "Overview", "Price Analysis", "Weather & Environment",
     "Market & Policy", "Demand & Trends", "Model Results", "Forecasts & Alerts"],
)

# Date range filter
if "prices" in data:
    prices = data["prices"]
    min_date = prices["date"].min().date()
    data_max = prices["date"].max().date()
    today_date = pd.Timestamp.today().date()
    max_date = max(data_max, today_date)

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (prices["date"].dt.date >= start_date) & (prices["date"].dt.date <= end_date)
        prices_filtered = prices[mask].copy()
    else:
        prices_filtered = prices.copy()
else:
    prices_filtered = pd.DataFrame()
    st.error("No price data found. Run the pipeline first.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Farmer Panel
# ═════════════════════════════════════════════════════════════════════════════════

if page == "Farmer Panel":
    st.title("🧑‍🌾 Finike Orange Farmer — Decision Support Panel")

    if "farmer_advice" not in data:
        st.warning("Farmer model has not been run yet. `python -c \"from src.models.farmer import train_all_farmer_models; train_all_farmer_models()\"`")
        st.stop()

    advice = data["farmer_advice"]

    advice_date = advice.get("date", "—")
    last_price_date = advice.get("last_price_date", advice_date)
    age_days = advice.get("data_age_days")
    if age_days is None and last_price_date != "—":
        try:
            age_days = (pd.Timestamp.today().normalize() - pd.Timestamp(last_price_date)).days
        except Exception:
            age_days = None

    today_str = pd.Timestamp.today().strftime("%d %B %Y")
    if age_days is None:
        st.caption(f"Today: **{today_str}** · Advice date: {advice_date}")
    elif age_days <= 1:
        st.caption(f"Today: **{today_str}** · Antalya Hal latest data: {last_price_date} (fresh)")
    elif age_days <= 7:
        st.info(f"Today **{today_str}**. Antalya Hal latest price: **{last_price_date}** ({age_days} days ago). Calculations continue with the last price.")
    else:
        st.warning(f"Today **{today_str}**. Antalya Hal data is **{age_days} days** old (last: {last_price_date}). Check that the daily pipeline is running.")

    # ── Top KPIs ──
    col1, col2, col3, col4 = st.columns(4)

    current = advice["current_price"]
    breakeven = advice["breakeven_price"]
    margin = advice["margin_per_kg"]
    rec = advice["recommendation"]

    with col1:
        st.metric("Antalya Hal Price", f"{current:.1f} TRY/kg")
    with col2:
        st.metric("Breakeven Cost", f"{breakeven:.1f} TRY/kg")
    with col3:
        color = "normal" if margin > 0 else "inverse"
        st.metric("Margin", f"{margin:.1f} TRY/kg", delta=f"{advice['margin_pct']:.0f}%")
    with col4:
        action_colors = {"SELL NOW": "🔴", "SELL": "🟡", "COLD STORAGE": "🔵", "WAIT": "⚪"}
        emoji = action_colors.get(rec["action"], "⚪")
        st.metric("Recommendation", f"{emoji} {rec['action']}")

    # ── Recommendation box ──
    urgency_colors = {"high": "error", "medium": "warning", "low": "info"}
    alert_type = urgency_colors.get(rec["urgency"], "info")
    getattr(st, alert_type)(f"**{rec['action']}** — {rec['reason']}")

    st.info(f"**Season:** {advice['season_info']['phase']} — {advice['season_info']['advice']}")

    st.markdown("---")

    # ── Forecasts ──
    st.subheader("Price Forecasts")
    forecasts = advice.get("forecasts", {})

    if forecasts:
        cols = st.columns(len(forecasts))
        for i, (horizon, fc) in enumerate(sorted(forecasts.items(), key=lambda x: int(x[0]))):
            with cols[i]:
                change = fc["change_pct"]
                st.metric(
                    f"{horizon} Days",
                    f"{fc['price']:.1f} TRY/kg",
                    delta=f"{change:+.1f}%",
                )
                st.caption(f"Range: {fc['lower']:.1f} — {fc['upper']:.1f} TRY")

        # Forecast chart
        fig_fc = go.Figure()

        # Historical Antalya prices
        if "antalya" in data:
            ant = data["antalya"]
            oranges = ant[ant["product"].str.contains("Portakal", case=False)]
            daily_ant = oranges.groupby("date")["avg_price"].mean().reset_index()
            fig_fc.add_trace(go.Scatter(
                x=daily_ant["date"], y=daily_ant["avg_price"],
                mode="lines", name="Antalya Hal (actual)",
                line=dict(color="darkorange", width=2),
            ))

        # Forecasts
        last_date = pd.Timestamp(advice["date"])
        for horizon, fc in sorted(forecasts.items(), key=lambda x: int(x[0])):
            target_date = pd.Timestamp(fc["target_date"])
            fig_fc.add_trace(go.Scatter(
                x=[last_date, target_date],
                y=[current, fc["price"]],
                mode="lines+markers",
                name=f"{horizon}d forecast",
                line=dict(dash="dash"),
                marker=dict(size=10),
            ))
            # Interval
            fig_fc.add_trace(go.Scatter(
                x=[target_date, target_date],
                y=[fc["lower"], fc["upper"]],
                mode="lines",
                line=dict(width=6, color="rgba(0,0,0,0.15)"),
                showlegend=False,
            ))

        # Breakeven line
        fig_fc.add_hline(y=breakeven, line_dash="dot", line_color="red",
                          annotation_text=f"Breakeven: {breakeven:.1f} TRY")

        fig_fc.update_layout(
            title="Orange Price Forecasts — Antalya Hal",
            height=450, hovermode="x unified",
            yaxis_title="TRY/kg",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ── Cost breakdown ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Cost Breakdown (TRY/kg)")
        costs = advice.get("costs", {})
        cost_items = {
            "harvest_labor": "Harvest Labor",
            "transport_to_hal": "Transport (Finike→Antalya)",
            "commission_pct": "Hal Commission (%)",
            "packaging": "Packaging",
            "pesticide_fertilizer": "Pesticide & Fertilizer",
            "irrigation": "Irrigation",
            "cold_storage_daily": "Cold Storage (TRY/kg/day)",
        }
        cost_df = pd.DataFrame([
            {"Item": cost_items.get(k, k), "Amount": f"{v:.2f}"}
            for k, v in costs.items()
        ])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        st.markdown("*Edit costs in `src/models/farmer.py` → `DEFAULT_COSTS`*")

    with col_b:
        st.subheader("Cold Storage Scenario")
        cold_cost = costs.get("cold_storage_daily", 0.15)

        storage_days = st.slider("Storage Duration (days)", 0, 120, 30)
        storage_total = cold_cost * storage_days

        # Find forecast closest to selected days
        closest_horizon = min(forecasts.keys(), key=lambda h: abs(int(h) - storage_days)) if forecasts else None
        if closest_horizon:
            fc = forecasts[closest_horizon]
            expected = fc["price"]
            net_gain = expected - current - storage_total

            st.metric("Storage Cost", f"{storage_total:.1f} TRY/kg")
            st.metric("Expected Sale Price", f"{expected:.1f} TRY/kg")
            color = "normal" if net_gain > 0 else "inverse"
            st.metric("Net Gain / Loss", f"{net_gain:+.1f} TRY/kg",
                       delta="Profitable" if net_gain > 0 else "Loss-making")

    # ── Antalya vs Istanbul comparison ──
    if "antalya" in data:
        st.markdown("---")
        st.subheader("Antalya vs Istanbul Hal Prices")

        ant = data["antalya"]
        oranges_ant = ant[ant["product"].str.contains("Portakal", case=False)]
        ant_daily = oranges_ant.groupby("date")["avg_price"].mean().reset_index()
        ant_daily = ant_daily.rename(columns={"avg_price": "Antalya"})

        ist_daily = prices[["date", "avg_price"]].copy()
        ist_daily = ist_daily.rename(columns={"avg_price": "Istanbul"})

        merged = ant_daily.merge(ist_daily, on="date", how="inner")
        merged["Spread (Ist-Ant)"] = merged["Istanbul"] - merged["Antalya"]

        fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Price Comparison", "Istanbul–Antalya Spread (TRY/kg)"),
                                  row_heights=[0.6, 0.4])

        fig_comp.add_trace(go.Scatter(x=merged["date"], y=merged["Antalya"],
                                       name="Antalya Hal", line=dict(color="darkorange")), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=merged["date"], y=merged["Istanbul"],
                                       name="Istanbul Hal", line=dict(color="royalblue")), row=1, col=1)

        fig_comp.add_trace(go.Bar(x=merged["date"], y=merged["Spread (Ist-Ant)"],
                                   marker_color=np.where(merged["Spread (Ist-Ant)"] > 0, "green", "red"),
                                   name="Spread", opacity=0.6), row=2, col=1)

        fig_comp.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)

        avg_spread = merged["Spread (Ist-Ant)"].mean()
        st.info(f"Average Istanbul–Antalya spread: **{avg_spread:.1f} TRY/kg** (transport + commission + margin)")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Overview":
    st.title("🍊 Orange Price Predictor — Overview")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    latest = prices_filtered.iloc[-1]
    prev_30 = prices_filtered[prices_filtered["date"] <= latest["date"] - pd.Timedelta(days=30)]

    with col1:
        st.metric(
            "Latest Price",
            f"{latest['avg_price']:.1f} TRY/kg",
            delta=f"{latest['avg_price'] - prev_30.iloc[-1]['avg_price']:.1f} TRY" if not prev_30.empty else None,
        )
    with col2:
        st.metric("Min Price", f"{latest['min_price']:.1f} TRY/kg")
    with col3:
        st.metric("Max Price", f"{latest['max_price']:.1f} TRY/kg")
    with col4:
        spread = latest["max_price"] - latest["min_price"]
        st.metric("Spread", f"{spread:.1f} TRY")
    with col5:
        st.metric("Records", f"{len(prices_filtered):,}")

    st.markdown("---")

    # Main price chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Orange Hal Price (TRY/kg)", "Daily Spread (Max − Min)"),
    )

    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["max_price"],
            fill=None, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["min_price"],
            fill="tonexty", fillcolor="rgba(255,165,0,0.2)",
            mode="lines", line=dict(width=0),
            name="Min–Max range",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["avg_price"],
            mode="lines", line=dict(color="darkorange", width=1.5),
            name="Average price",
        ),
        row=1, col=1,
    )

    # 30-day moving average
    prices_filtered["ma30"] = prices_filtered["avg_price"].rolling(30, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["ma30"],
            mode="lines", line=dict(color="red", width=1, dash="dash"),
            name="30-day MA",
        ),
        row=1, col=1,
    )

    # Spread
    spread_vals = prices_filtered["max_price"] - prices_filtered["min_price"]
    fig.add_trace(
        go.Bar(
            x=prices_filtered["date"], y=spread_vals,
            marker_color="rgba(255,165,0,0.5)", name="Spread",
        ),
        row=2, col=1,
    )

    fig.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(title_text="TRY/kg", row=1, col=1)
    fig.update_yaxes(title_text="TRY", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Yearly Averages")
        yearly = prices_filtered.copy()
        yearly["year"] = yearly["date"].dt.year
        yearly_stats = yearly.groupby("year")["avg_price"].agg(["mean", "min", "max", "std"]).round(2)
        yearly_stats.columns = ["Mean", "Min", "Max", "Std"]
        st.dataframe(yearly_stats, use_container_width=True)

    with col_b:
        st.subheader("Monthly Seasonality")
        monthly = prices_filtered.copy()
        monthly["month"] = monthly["date"].dt.month
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly_stats = monthly.groupby("month")["avg_price"].mean().round(2)
        monthly_stats.index = monthly_stats.index.map(month_names)
        fig_month = px.bar(
            x=monthly_stats.index, y=monthly_stats.values,
            labels={"x": "Month", "y": "Avg price (TRY/kg)"},
            color=monthly_stats.values,
            color_continuous_scale="Oranges",
        )
        fig_month.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Price Analysis
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Price Analysis":
    st.title("📈 Price Analysis")

    tab1, tab2, tab3 = st.tabs(["Trend & Momentum", "Volatility", "YoY Comparison"])

    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Price + Moving Averages", "7-Day Change (%)", "30-Day Change (%)"),
        )

        fig.add_trace(go.Scatter(x=prices_filtered["date"], y=prices_filtered["avg_price"],
                                  name="Price", line=dict(color="darkorange")), row=1, col=1)

        for window, color in [(7, "blue"), (30, "red"), (90, "green")]:
            ma = prices_filtered["avg_price"].rolling(window, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=prices_filtered["date"], y=ma,
                                      name=f"MA-{window}", line=dict(color=color, dash="dash")), row=1, col=1)

        pct7 = prices_filtered["avg_price"].pct_change(7) * 100
        pct30 = prices_filtered["avg_price"].pct_change(30) * 100

        fig.add_trace(go.Bar(x=prices_filtered["date"], y=pct7,
                              marker_color=np.where(pct7 > 0, "green", "red"), name="7d %", opacity=0.6), row=2, col=1)
        fig.add_trace(go.Bar(x=prices_filtered["date"], y=pct30,
                              marker_color=np.where(pct30 > 0, "green", "red"), name="30d %", opacity=0.6), row=3, col=1)

        fig.update_layout(height=700, hovermode="x unified", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        vol30 = prices_filtered["avg_price"].rolling(30, min_periods=7).std()
        vol_pct = vol30 / prices_filtered["avg_price"].rolling(30, min_periods=7).mean() * 100

        fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=("30-Day Volatility (TRY)", "Volatility (%)"))
        fig_vol.add_trace(go.Scatter(x=prices_filtered["date"], y=vol30,
                                      fill="tozeroy", fillcolor="rgba(255,0,0,0.1)",
                                      line=dict(color="red"), name="Std Dev"), row=1, col=1)
        fig_vol.add_trace(go.Scatter(x=prices_filtered["date"], y=vol_pct,
                                      fill="tozeroy", fillcolor="rgba(128,0,128,0.1)",
                                      line=dict(color="purple"), name="CV%"), row=2, col=1)
        fig_vol.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab3:
        pf = prices_filtered.copy()
        pf["year"] = pf["date"].dt.year
        pf["day_of_year"] = pf["date"].dt.dayofyear

        years = sorted(pf["year"].unique())
        selected_years = st.multiselect("Years", years, default=years[-4:])

        fig_yoy = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, year in enumerate(selected_years):
            yr_data = pf[pf["year"] == year]
            fig_yoy.add_trace(go.Scatter(
                x=yr_data["day_of_year"], y=yr_data["avg_price"],
                mode="lines", name=str(year),
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        fig_yoy.update_layout(
            height=500, xaxis_title="Day of Year", yaxis_title="TRY/kg",
            title="Year-over-Year Price Comparison", hovermode="x unified",
        )
        st.plotly_chart(fig_yoy, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Weather & Environment
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Weather & Environment":
    st.title("🌤️ Weather & Environmental Factors")

    if "weather" not in data:
        st.warning("No weather data found.")
        st.stop()

    weather = data["weather"]
    w_mask = (weather["date"].dt.date >= start_date) & (weather["date"].dt.date <= end_date)
    wf = weather[w_mask].copy()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        frost_days = int(wf["frost"].sum()) if "frost" in wf.columns else 0
        st.metric("Frost Days", frost_days)
    with col2:
        avg_temp = wf["temp_mean"].mean()
        st.metric("Avg. Temperature", f"{avg_temp:.1f}°C")
    with col3:
        total_precip = wf["precipitation"].sum()
        st.metric("Total Precipitation", f"{total_precip:.0f} mm")
    with col4:
        max_wind = wf["wind_speed_max"].max() if "wind_speed_max" in wf.columns else 0
        st.metric("Max Wind", f"{max_wind:.1f} km/h")

    st.markdown("---")

    tab1, tab2 = st.tabs(["Temperature & Precipitation", "Price–Weather Relationship"])

    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Temperature (°C)", "Daily Precipitation (mm)", "Humidity (%)"),
        )

        fig.add_trace(go.Scatter(x=wf["date"], y=wf["temp_max"], name="Max",
                                  line=dict(color="red", width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=wf["date"], y=wf["temp_min"], name="Min",
                                  fill="tonexty", fillcolor="rgba(255,0,0,0.1)",
                                  line=dict(color="blue", width=0.5)), row=1, col=1)
        fig.add_shape(type="line", y0=0, y1=0, x0=wf["date"].min(), x1=wf["date"].max(),
                      line=dict(color="darkblue", dash="dash", width=1), row=1, col=1)

        fig.add_trace(go.Bar(x=wf["date"], y=wf["precipitation"],
                              marker_color="royalblue", name="Precipitation", opacity=0.7), row=2, col=1)

        if "humidity" in wf.columns:
            fig.add_trace(go.Scatter(x=wf["date"], y=wf["humidity"],
                                      line=dict(color="teal", width=0.8), name="Humidity"), row=3, col=1)

        fig.update_layout(height=700, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Merge price and weather
        merged = prices_filtered.merge(wf[["date", "temp_mean", "precipitation", "frost"]], on="date", how="inner")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_scatter = px.scatter(
                merged, x="temp_mean", y="avg_price",
                color="frost", color_discrete_map={0: "orange", 1: "blue"},
                labels={"temp_mean": "Avg. temperature (°C)", "avg_price": "Price (TRY/kg)", "frost": "Frost"},
                title="Temperature vs Price",
                opacity=0.5,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b:
            fig_scatter2 = px.scatter(
                merged, x="precipitation", y="avg_price",
                labels={"precipitation": "Precipitation (mm)", "avg_price": "Price (TRY/kg)"},
                title="Precipitation vs Price",
                opacity=0.3,
                color_discrete_sequence=["royalblue"],
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Market & Policy
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Market & Policy":
    st.title("🌍 Market Dynamics & Policy Effects")

    tab1, tab2, tab3 = st.tabs(["FX & International", "Policy Events", "Competition"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            if "fx" in data:
                fx = data["fx"]
                usd_col = "TRY_per_USD" if "TRY_per_USD" in fx.columns else "USD_TRY"
                fig_fx = px.line(fx, x="date", y=usd_col, title="USD/TRY Rate",
                                 labels={"date": "", usd_col: "TRY per USD"})
                fig_fx.update_traces(line_color="purple")
                fig_fx.update_layout(height=350)
                st.plotly_chart(fig_fx, use_container_width=True)

        with col_b:
            if "foreign" in data:
                foreign = data["foreign"]
                fig_fao = px.line(foreign, x="date", y="fao_fruit_index",
                                  title="FAO Fruit Price Index",
                                  labels={"date": "", "fao_fruit_index": "Index (2014–16=100)"})
                fig_fao.update_traces(line_color="darkgreen")
                fig_fao.update_layout(height=350)
                fig_fao.add_hline(y=100, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_fao, use_container_width=True)

        # Dual axis: Price vs USD/TRY
        if "fx" in data:
            fx = data["fx"]
            usd_col = "TRY_per_USD" if "TRY_per_USD" in fx.columns else "USD_TRY"

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            monthly_p = prices.set_index("date").resample("ME")["avg_price"].mean().reset_index()
            monthly_fx = fx.set_index("date")[usd_col].resample("ME").mean().reset_index()

            fig_dual.add_trace(
                go.Scatter(x=monthly_p["date"], y=monthly_p["avg_price"],
                           name="Orange (TRY/kg)", line=dict(color="darkorange", width=2)),
                secondary_y=False,
            )
            fig_dual.add_trace(
                go.Scatter(x=monthly_fx["date"], y=monthly_fx[usd_col],
                           name="USD/TRY", line=dict(color="purple", width=2)),
                secondary_y=True,
            )
            fig_dual.update_layout(title="Orange Price vs FX Rate (Monthly)", height=400,
                                    hovermode="x unified")
            fig_dual.update_yaxes(title_text="TRY/kg", secondary_y=False)
            fig_dual.update_yaxes(title_text="USD/TRY", secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

    with tab2:
        if "events" in data:
            events = data["events"]

            st.subheader("Policy and Event Timeline")

            # Price chart with event markers
            fig_events = go.Figure()
            fig_events.add_trace(go.Scatter(
                x=prices["date"], y=prices["avg_price"],
                mode="lines", line=dict(color="darkorange", width=1),
                name="Price", opacity=0.7,
            ))

            color_map = {
                "frost": "blue", "sanction": "red", "economic": "purple",
                "regulation": "green", "supply": "brown", "trade": "gray", "pandemic": "black",
            }
            symbol_map = {"up": "triangle-up", "down": "triangle-down"}

            for _, ev in events.iterrows():
                # Find nearest price
                price_at = prices.loc[prices["date"] >= ev["date"], "avg_price"]
                y_val = price_at.iloc[0] if not price_at.empty else 0

                fig_events.add_trace(go.Scatter(
                    x=[ev["date"]], y=[y_val],
                    mode="markers+text",
                    marker=dict(
                        size=ev["impact_magnitude"] * 8,
                        color=color_map.get(ev["event_type"], "gray"),
                        symbol=symbol_map.get(ev["impact_direction"], "circle"),
                        line=dict(width=1, color="white"),
                    ),
                    text=ev["event_type"],
                    textposition="top center",
                    textfont=dict(size=8),
                    name=ev["description"][:40],
                    showlegend=False,
                    hovertemplate=f"<b>{ev['description']}</b><br>Date: {ev['date'].strftime('%Y-%m-%d')}<br>"
                                 f"Type: {ev['event_type']}<br>Impact: {ev['impact_direction']} ({ev['impact_magnitude']})<extra></extra>",
                ))

            fig_events.update_layout(height=500, title="Price + Policy Events", hovermode="closest")
            st.plotly_chart(fig_events, use_container_width=True)

            # Event table
            st.subheader("Event List")
            display_events = events[["date", "event_type", "description", "impact_direction", "impact_magnitude"]].copy()
            display_events["date"] = display_events["date"].dt.strftime("%Y-%m-%d")
            display_events.columns = ["Date", "Type", "Description", "Direction", "Magnitude"]
            st.dataframe(display_events, use_container_width=True, hide_index=True)

        # Policy impact score
        if "policy" in data:
            policy = data["policy"]
            fig_impact = go.Figure()
            pos_mask = policy["policy_impact_score"] > 0
            fig_impact.add_trace(go.Bar(
                x=policy.loc[pos_mask, "date"], y=policy.loc[pos_mask, "policy_impact_score"],
                marker_color="red", name="Price-raising", opacity=0.6,
            ))
            fig_impact.add_trace(go.Bar(
                x=policy.loc[~pos_mask, "date"], y=policy.loc[~pos_mask, "policy_impact_score"],
                marker_color="blue", name="Price-lowering", opacity=0.6,
            ))
            fig_impact.update_layout(title="Policy Impact Score", height=300, hovermode="x unified",
                                      barmode="relative")
            st.plotly_chart(fig_impact, use_container_width=True)

    with tab3:
        if "foreign" in data:
            foreign = data["foreign"]

            col_a, col_b = st.columns(2)

            with col_a:
                if "eu_orange_price_eur_100kg" in foreign.columns:
                    fig_eu = px.line(foreign, x="date", y="eu_orange_price_eur_100kg",
                                     title="EU Orange Price (EUR/100kg)",
                                     labels={"eu_orange_price_eur_100kg": "EUR/100kg"})
                    fig_eu.update_traces(line_color="royalblue")
                    fig_eu.update_layout(height=350)
                    st.plotly_chart(fig_eu, use_container_width=True)

            with col_b:
                if "competition_index" in foreign.columns:
                    comp = foreign.dropna(subset=["competition_index"])
                    fig_comp = px.bar(comp, x="date", y="competition_index",
                                      title="Competition Intensity Index",
                                      labels={"competition_index": "Index"},
                                      color="active_competitors",
                                      color_continuous_scale="Reds")
                    fig_comp.update_layout(height=350)
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Competitor production
            from src.data.foreign_markets import fetch_competitor_production
            prod = fetch_competitor_production()

            st.subheader("Competitor Country Production")
            year_select = st.selectbox("Year", sorted(prod["year"].unique(), reverse=True))
            yr = prod[prod["year"] == year_select].sort_values("production_kt", ascending=True)

            fig_prod = px.bar(yr, x="production_kt", y="country", orientation="h",
                               color="estimated_export_kt", color_continuous_scale="YlOrRd",
                               labels={"production_kt": "Production (k tons)", "country": "",
                                        "estimated_export_kt": "Exports (k tons)"},
                               title=f"{year_select} Orange Production")
            fig_prod.update_layout(height=350)
            st.plotly_chart(fig_prod, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Model Results
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Model Results":
    st.title("🤖 Model Results")

    if "results" not in data:
        st.warning("No model results found. Run the pipeline.")
        st.stop()

    results = data["results"]

    # Best model highlight
    best = results.loc[results["MAE"].idxmin()]
    st.success(f"Best model: **{best['model']}** — MAE: {best['MAE']:.2f} TRY/kg, "
               f"MAPE: {best['MAPE']*100:.1f}%, R²: {best['R2']:.3f} ({int(best['horizon_days'])} days)")

    st.markdown("---")

    # Comparison charts
    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(results, x="model", y="MAE", color="horizon_days",
                          barmode="group", title="MAE Comparison (TRY/kg)",
                          color_continuous_scale="Viridis",
                          labels={"MAE": "MAE (TRY/kg)", "model": "", "horizon_days": "Horizon"})
        fig_mae.update_layout(height=400)
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        fig_r2 = px.bar(results, x="model", y="R2", color="horizon_days",
                          barmode="group", title="R² Comparison",
                          color_continuous_scale="Viridis",
                          labels={"R2": "R²", "model": "", "horizon_days": "Horizon"})
        fig_r2.add_hline(y=0, line_dash="dash", line_color="red")
        fig_r2.update_layout(height=400)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Full results table
    st.subheader("Detailed Results")
    display_results = results.copy()
    display_results["MAPE"] = (display_results["MAPE"] * 100).round(1).astype(str) + "%"
    display_results["MAE"] = display_results["MAE"].round(2)
    display_results["RMSE"] = display_results["RMSE"].round(2)
    display_results["R2"] = display_results["R2"].round(3)
    display_results.columns = ["Model", "Horizon (days)", "MAE", "MAPE", "RMSE", "R²"]
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    # Feature importance (if we can compute it)
    st.markdown("---")
    st.subheader("Feature Matrix Summary")

    if "features" in data:
        features = data["features"]
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        horizon = st.selectbox("Forecast horizon", [30, 60, 90])
        target = f"target_{horizon}d"

        if target in features.columns:
            corr = features[numeric_cols].corrwith(features[target]).abs().sort_values(ascending=False)
            corr = corr.drop([c for c in corr.index if c.startswith("target_")])
            top_20 = corr.head(20)

            fig_fi = px.bar(x=top_20.values, y=top_20.index, orientation="h",
                             title=f"Top 20 Features — Correlation with {horizon}-Day Target",
                             labels={"x": "|Correlation|", "y": ""},
                             color=top_20.values, color_continuous_scale="Oranges")
            fig_fi.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        st.info(f"Total features: **{len(numeric_cols)}** | Rows: **{len(features):,}** | "
                f"Date range: {features['date'].min().strftime('%Y-%m-%d')} — {features['date'].max().strftime('%Y-%m-%d')}")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Demand & Trends
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Demand & Trends":
    st.title("📊 Demand Signals & Google Trends")

    tab1, tab2 = st.tabs(["Google Trends", "Demand Factors"])

    with tab1:
        if "trends" in data:
            trends = data["trends"]
            trend_cols = [c for c in trends.columns if c.startswith("trend_")]

            fig_trends = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, col in enumerate(trend_cols):
                fig_trends.add_trace(go.Scatter(
                    x=trends["date"], y=trends[col],
                    name=col.replace("trend_", "").replace("_", " ").title(),
                    line=dict(color=colors[i % len(colors)]),
                ))

            fig_trends.update_layout(
                title="Google Trends — Orange Search Interest (Turkey)",
                height=400, hovermode="x unified",
                yaxis_title="Search Interest (0–100)",
            )
            st.plotly_chart(fig_trends, use_container_width=True)

            # Overlay with price
            fig_tp = make_subplots(specs=[[{"secondary_y": True}]])
            monthly_p = prices.set_index("date").resample("ME")["avg_price"].mean().reset_index()
            if "trend_portakal_fiyat" in trends.columns:
                fig_tp.add_trace(go.Scatter(x=trends["date"], y=trends["trend_portakal_fiyat"],
                                             name="Trend: orange price", line=dict(color="blue")), secondary_y=False)
            fig_tp.add_trace(go.Scatter(x=monthly_p["date"], y=monthly_p["avg_price"],
                                         name="Price (TRY/kg)", line=dict(color="darkorange")), secondary_y=True)
            fig_tp.update_layout(title="Search Interest vs Price", height=350, hovermode="x unified")
            fig_tp.update_yaxes(title_text="Trend (0–100)", secondary_y=False)
            fig_tp.update_yaxes(title_text="TRY/kg", secondary_y=True)
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("No trend data found.")

    with tab2:
        if "demand" in data:
            demand = data["demand"]
            dm_filtered = demand[(demand["date"].dt.date >= start_date) & (demand["date"].dt.date <= end_date)]

            col1, col2 = st.columns(2)

            with col1:
                # Ramadan periods
                ramadan_data = dm_filtered[dm_filtered["ramadan_active"] == 1]
                st.subheader("Ramadan Periods")
                fig_ram = go.Figure()
                fig_ram.add_trace(go.Scatter(
                    x=prices_filtered["date"], y=prices_filtered["avg_price"],
                    mode="lines", line=dict(color="darkorange"), name="Price",
                ))
                # Shade Ramadan periods
                for _, row in dm_filtered[dm_filtered["ramadan_active"] == 1].groupby(
                    (dm_filtered["ramadan_active"] != dm_filtered["ramadan_active"].shift()).cumsum()
                ).agg({"date": ["min", "max"]}).iterrows():
                    fig_ram.add_vrect(x0=row[("date", "min")], x1=row[("date", "max")],
                                      fillcolor="green", opacity=0.1, line_width=0)
                fig_ram.update_layout(height=300, title="Price + Ramadan Periods (green)")
                st.plotly_chart(fig_ram, use_container_width=True)

            with col2:
                # Input cost index
                st.subheader("Input Cost Index")
                fig_input = px.line(dm_filtered, x="date", y="input_cost_index",
                                     title="Agricultural Input Cost Index (2015=100)")
                fig_input.update_traces(line_color="brown")
                fig_input.update_layout(height=300)
                st.plotly_chart(fig_input, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Tourism Intensity")
                fig_tour = px.line(dm_filtered, x="date", y="tourism_intensity",
                                    title="Antalya Tourism Intensity")
                fig_tour.update_traces(line_color="teal")
                fig_tour.update_layout(height=300)
                st.plotly_chart(fig_tour, use_container_width=True)

            with col4:
                st.subheader("CPI Index")
                fig_cpi = px.line(dm_filtered, x="date", y="cpi_index",
                                   title="Consumer Price Index (2007=100)")
                fig_cpi.update_traces(line_color="purple")
                fig_cpi.update_layout(height=300)
                st.plotly_chart(fig_cpi, use_container_width=True)
        else:
            st.info("No demand data found.")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Forecasts & Alerts
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Forecasts & Alerts":
    st.title("🔮 Forecasts & Alert System")

    tab1, tab2, tab3, tab4 = st.tabs(["Price Forecasts", "Forecast Tracking", "Alerts", "SHAP Analysis"])

    with tab1:
        if "predictions" in data:
            preds = data["predictions"]
            current = prices.iloc[-1]["avg_price"]

            st.subheader(f"Current Price: {current:.1f} TRY/kg")
            st.markdown("---")

            cols = st.columns(len(preds))
            for i, (_, row) in enumerate(preds.iterrows()):
                with cols[i]:
                    horizon = int(row["horizon_days"])
                    pred = row["prediction"]
                    change = pred - current
                    change_pct = (change / current) * 100

                    st.metric(
                        f"{horizon}-Day Forecast",
                        f"{pred:.1f} TRY/kg",
                        delta=f"{change:+.1f} TRY ({change_pct:+.1f}%)",
                    )

                    if "pred_lower" in row and "pred_upper" in row:
                        st.caption(f"Range: {row['pred_lower']:.1f} — {row['pred_upper']:.1f} TRY")

            # Prediction chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=prices.tail(90)["date"], y=prices.tail(90)["avg_price"],
                mode="lines", name="Actual price", line=dict(color="darkorange", width=2),
            ))

            for _, row in preds.iterrows():
                target_date = pd.Timestamp(row.get("target_date", ""))
                fig_pred.add_trace(go.Scatter(
                    x=[prices.iloc[-1]["date"], target_date],
                    y=[current, row["prediction"]],
                    mode="lines+markers",
                    name=f"{int(row['horizon_days'])}d forecast",
                    line=dict(dash="dash"),
                    marker=dict(size=10),
                ))

                if "pred_lower" in row and "pred_upper" in row:
                    fig_pred.add_trace(go.Scatter(
                        x=[target_date, target_date],
                        y=[row["pred_lower"], row["pred_upper"]],
                        mode="lines", line=dict(width=4, color="rgba(0,0,0,0.2)"),
                        showlegend=False,
                    ))

            fig_pred.update_layout(title="Price Forecast", height=400, hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("No prediction data found. Run `python -m src.auto_refresh --predict`.")

    with tab2:
        st.subheader("Forecast Tracking — Predicted vs Actual")

        if "pred_history" in data:
            hist = data["pred_history"]
            evaluated = hist[hist["evaluated"] == True] if "evaluated" in hist.columns else pd.DataFrame()  # noqa: E712

            # ── Data freshness ──
            if "refresh_log" in data:
                rlog = data["refresh_log"]
                st.markdown("**Data Freshness**")
                fresh_cols = st.columns(4)
                sources = ["hal_prices", "weather", "fx_rates", "demand_policy"]
                for i, src in enumerate(sources):
                    src_rows = rlog[rlog["source"] == src] if "source" in rlog.columns else pd.DataFrame()
                    if not src_rows.empty:
                        last = src_rows.iloc[-1]
                        status_icon = "✅" if last.get("status") == "ok" else "⚠️"
                        fresh_cols[i].metric(src, f"{status_icon} {last.get('records_after', '—')}")
                    else:
                        fresh_cols[i].metric(src, "—")
                st.markdown("---")

            # ── Accuracy summary table ──
            if "accuracy" in data and not data["accuracy"].empty:
                acc = data["accuracy"]
                st.markdown("**Model Accuracy by Horizon**")

                acc_cols = st.columns(len(acc))
                for i, (_, row) in enumerate(acc.iterrows()):
                    horizon = int(row["horizon_days"])
                    with acc_cols[i]:
                        label = f"{horizon // 7} Week" if horizon % 7 == 0 and horizon <= 28 else f"{horizon} Days"
                        st.metric(f"{label} MAE", f"{row['mae']:.2f} TRY/kg")
                        st.caption(
                            f"MAPE: {row['mape_pct']:.1f}%\n\n"
                            f"Direction: {row['direction_accuracy_pct']:.0f}%\n\n"
                            f"n={int(row['n_predictions'])}"
                        )
                st.markdown("---")

            # ── Prediction vs Actual chart ──
            if not evaluated.empty:
                st.markdown("**Predicted vs Actual Price**")

                horizon_options = sorted(evaluated["horizon_days"].unique())
                horizon_labels = {h: (f"{h // 7} Week" if h % 7 == 0 and h <= 28 else f"{h} Days") for h in horizon_options}
                selected_horizon = st.selectbox(
                    "Select horizon",
                    horizon_options,
                    format_func=lambda x: horizon_labels[x],
                )

                subset = evaluated[evaluated["horizon_days"] == selected_horizon].sort_values("target_date")

                fig_track = go.Figure()

                # Actual prices
                fig_track.add_trace(go.Scatter(
                    x=subset["target_date"], y=subset["actual_price"],
                    mode="lines+markers", name="Actual price",
                    line=dict(color="darkorange", width=2),
                    marker=dict(size=6),
                ))

                # Predicted prices
                fig_track.add_trace(go.Scatter(
                    x=subset["target_date"], y=subset["predicted_price"],
                    mode="lines+markers", name="Predicted",
                    line=dict(color="royalblue", width=2, dash="dash"),
                    marker=dict(size=6),
                ))

                # Confidence interval
                if subset["pred_lower"].notna().any():
                    fig_track.add_trace(go.Scatter(
                        x=pd.concat([subset["target_date"], subset["target_date"][::-1]]),
                        y=pd.concat([subset["pred_upper"], subset["pred_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(65,105,225,0.1)",
                        line=dict(color="rgba(65,105,225,0)"),
                        name="Confidence interval",
                    ))

                fig_track.update_layout(
                    title=f"Predicted vs Actual — {horizon_labels[selected_horizon]}",
                    xaxis_title="Date", yaxis_title="Price (TRY/kg)",
                    height=400, hovermode="x unified",
                )
                st.plotly_chart(fig_track, use_container_width=True)

                # ── Error over time chart ──
                fig_err = go.Figure()
                fig_err.add_trace(go.Bar(
                    x=subset["target_date"], y=subset["error"],
                    name="Error (TRY/kg)",
                    marker_color=["crimson" if e > 0 else "forestgreen" for e in subset["error"]],
                ))
                fig_err.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_err.update_layout(
                    title="Prediction Error (Positive = Overestimate)",
                    xaxis_title="Date", yaxis_title="Error (TRY/kg)",
                    height=300,
                )
                st.plotly_chart(fig_err, use_container_width=True)

                # ── Detailed table ──
                st.markdown("**Detailed Table**")
                display_cols = ["date_generated", "target_date", "predicted_price",
                                "actual_price", "error", "pct_error"]
                display_cols = [c for c in display_cols if c in subset.columns]
                st.dataframe(
                    subset[display_cols].sort_values("target_date", ascending=False),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "date_generated": st.column_config.DateColumn("Predicted On"),
                        "target_date": st.column_config.DateColumn("Target Date"),
                        "predicted_price": st.column_config.NumberColumn("Predicted (TRY/kg)", format="%.2f"),
                        "actual_price": st.column_config.NumberColumn("Actual (TRY/kg)", format="%.2f"),
                        "error": st.column_config.NumberColumn("Error (TRY)", format="%.2f"),
                        "pct_error": st.column_config.NumberColumn("Error (%)", format="%.1f%%"),
                    },
                )
            else:
                st.info("No evaluated predictions yet. Data will populate as target dates pass.")

            # ── Pending predictions ──
            pending = hist[hist["evaluated"] != True] if "evaluated" in hist.columns else pd.DataFrame()  # noqa: E712
            if not pending.empty:
                with st.expander(f"Pending Forecasts ({len(pending)})"):
                    pending_display = pending[["date_generated", "horizon_days", "target_date",
                                               "predicted_price", "current_price"]].sort_values("target_date")
                    st.dataframe(pending_display, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No prediction history found. Data will accumulate as the daily pipeline runs.\n\n"
                "`python -m src.auto_refresh --full`"
            )

    with tab3:
        if "alerts_text" in data:
            st.code(data["alerts_text"], language=None)
        else:
            # Generate live alerts
            st.subheader("Live Alert Check")
            if st.button("Check Alerts"):
                from src.auto_refresh import run_alerts
                alerts = run_alerts()
                st.success(f"{len(alerts)} alerts checked.")

    with tab4:
        if "shap" in data:
            shap_df = data["shap"]
            top_n = st.slider("Top N Features", 10, 40, 20)
            top = shap_df.head(top_n)

            fig_shap = px.bar(
                top, x="shap_importance", y="feature", orientation="h",
                title=f"SHAP Feature Importance (Top {top_n})",
                labels={"shap_importance": "Mean |SHAP value|", "feature": ""},
                color="shap_importance", color_continuous_scale="Oranges",
            )
            fig_shap.update_layout(height=max(400, top_n * 22), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_shap, use_container_width=True)

            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("No SHAP data found. Run `python -m src.pipeline --advanced`.")


# ─── Footer ──────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data Sources:**\n"
    "- İBB Istanbul Hal\n"
    "- Open-Meteo\n"
    "- Frankfurter (FX)\n"
    "- FAO / Eurostat\n"
    "- USDA FAS"
)
st.sidebar.markdown(f"Last update: {prices['date'].max().strftime('%d.%m.%Y')}")
