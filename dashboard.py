"""
Portakal Fiyat Tahmini — Interactive Dashboard

Run: streamlit run dashboard.py
"""
import json
import os
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
    page_title="Portakal Fiyat Tahmini",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Password Protection ─────────────────────────────────────────────────────────

def check_password():
    """Simple password gate."""
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return True

    password = os.environ.get("DASHBOARD_PASSWORD", "Portakal1996!")

    st.markdown(
        "<div style='max-width:400px;margin:auto;padding-top:15vh;'>",
        unsafe_allow_html=True,
    )
    st.title("🍊 Portakal Fiyat Tahmini")
    entered = st.text_input("Şifre", type="password", key="pw_input")
    if st.button("Giriş", type="primary", use_container_width=True):
        if entered == password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Yanlış şifre.")
    st.markdown("</div>", unsafe_allow_html=True)
    return False


if not check_password():
    st.stop()


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

    return data


data = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

st.sidebar.title("🍊 Portakal Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Sayfa",
    ["Çiftçi Paneli", "Genel Bakış", "Fiyat Analizi", "Hava & Çevre", "Pazar & Politika",
     "Talep & Trendler", "Model Sonuçları", "Tahminler & Uyarılar"],
)

# Date range filter
if "prices" in data:
    prices = data["prices"]
    min_date = prices["date"].min().date()
    max_date = prices["date"].max().date()

    date_range = st.sidebar.date_input(
        "Tarih Aralığı",
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
    st.error("Fiyat verisi bulunamadı. Önce pipeline çalıştırın.")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Çiftçi Paneli
# ═════════════════════════════════════════════════════════════════════════════════

if page == "Çiftçi Paneli":
    st.title("🧑‍🌾 Finike Portakal Çiftçisi — Karar Destek Paneli")

    if "farmer_advice" not in data:
        st.warning("Çiftçi modeli henüz çalıştırılmadı. `python -c \"from src.models.farmer import train_all_farmer_models; train_all_farmer_models()\"`")
        st.stop()

    advice = data["farmer_advice"]

    # ── Top KPIs ──
    col1, col2, col3, col4 = st.columns(4)

    current = advice["current_price"]
    breakeven = advice["breakeven_price"]
    margin = advice["margin_per_kg"]
    rec = advice["recommendation"]

    with col1:
        st.metric("Antalya Hal Fiyatı", f"{current:.1f} ₺/kg")
    with col2:
        st.metric("Maliyet (Kırılma)", f"{breakeven:.1f} ₺/kg")
    with col3:
        color = "normal" if margin > 0 else "inverse"
        st.metric("Marj", f"{margin:.1f} ₺/kg", delta=f"{advice['margin_pct']:.0f}%")
    with col4:
        action_colors = {"ŞİMDİ SAT": "🔴", "SAT": "🟡", "SOĞUK HAVA": "🔵", "BEKLE": "⚪"}
        emoji = action_colors.get(rec["action"], "⚪")
        st.metric("Tavsiye", f"{emoji} {rec['action']}")

    # ── Recommendation box ──
    urgency_colors = {"high": "error", "medium": "warning", "low": "info"}
    alert_type = urgency_colors.get(rec["urgency"], "info")
    getattr(st, alert_type)(f"**{rec['action']}** — {rec['reason']}")

    st.info(f"**Sezon:** {advice['season_info']['phase']} — {advice['season_info']['advice']}")

    st.markdown("---")

    # ── Forecasts ──
    st.subheader("Fiyat Tahminleri")
    forecasts = advice.get("forecasts", {})

    if forecasts:
        cols = st.columns(len(forecasts))
        for i, (horizon, fc) in enumerate(sorted(forecasts.items(), key=lambda x: int(x[0]))):
            with cols[i]:
                change = fc["change_pct"]
                st.metric(
                    f"{horizon} Gün",
                    f"{fc['price']:.1f} ₺/kg",
                    delta=f"{change:+.1f}%",
                )
                st.caption(f"Aralık: {fc['lower']:.1f} — {fc['upper']:.1f} ₺")

        # Forecast chart
        fig_fc = go.Figure()

        # Historical Antalya prices
        if "antalya" in data:
            ant = data["antalya"]
            portakal = ant[ant["product"].str.contains("Portakal", case=False)]
            daily_ant = portakal.groupby("date")["avg_price"].mean().reset_index()
            fig_fc.add_trace(go.Scatter(
                x=daily_ant["date"], y=daily_ant["avg_price"],
                mode="lines", name="Antalya Hal (Gerçek)",
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
                name=f"{horizon}g tahmin",
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
                          annotation_text=f"Maliyet: {breakeven:.1f} ₺")

        fig_fc.update_layout(
            title="Portakal Fiyat Tahminleri — Antalya Hal",
            height=450, hovermode="x unified",
            yaxis_title="₺/kg",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

    st.markdown("---")

    # ── Cost breakdown ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Maliyet Detayı (₺/kg)")
        costs = advice.get("costs", {})
        cost_items = {
            "hasat_iscilik": "Hasat İşçiliği",
            "nakliye_hale": "Nakliye (Finike→Antalya)",
            "komisyon_pct": "Hal Komisyonu (%)",
            "ambalaj": "Ambalaj",
            "ilac_gubre": "İlaç & Gübre",
            "sulama": "Sulama",
            "soguk_hava_gunluk": "Soğuk Hava (₺/kg/gün)",
        }
        cost_df = pd.DataFrame([
            {"Kalem": cost_items.get(k, k), "Tutar": f"{v:.2f}"}
            for k, v in costs.items()
        ])
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        st.markdown("*Maliyetleri düzenlemek için `src/models/farmer.py` → `DEFAULT_COSTS`*")

    with col_b:
        st.subheader("Soğuk Hava Senaryosu")
        cold_cost = costs.get("soguk_hava_gunluk", 0.15)

        storage_days = st.slider("Depolama Süresi (gün)", 0, 120, 30)
        storage_total = cold_cost * storage_days

        # Find forecast closest to selected days
        closest_horizon = min(forecasts.keys(), key=lambda h: abs(int(h) - storage_days)) if forecasts else None
        if closest_horizon:
            fc = forecasts[closest_horizon]
            expected = fc["price"]
            net_gain = expected - current - storage_total

            st.metric("Depolama Maliyeti", f"{storage_total:.1f} ₺/kg")
            st.metric("Beklenen Satış Fiyatı", f"{expected:.1f} ₺/kg")
            color = "normal" if net_gain > 0 else "inverse"
            st.metric("Net Kazanç/Kayıp", f"{net_gain:+.1f} ₺/kg",
                       delta="Karlı" if net_gain > 0 else "Zararlı")

    # ── Antalya vs Istanbul comparison ──
    if "antalya" in data:
        st.markdown("---")
        st.subheader("Antalya vs İstanbul Hal Fiyatları")

        ant = data["antalya"]
        portakal_ant = ant[ant["product"].str.contains("Portakal", case=False)]
        ant_daily = portakal_ant.groupby("date")["avg_price"].mean().reset_index()
        ant_daily = ant_daily.rename(columns={"avg_price": "Antalya"})

        ist_daily = prices[["date", "avg_price"]].copy()
        ist_daily = ist_daily.rename(columns={"avg_price": "İstanbul"})

        merged = ant_daily.merge(ist_daily, on="date", how="inner")
        merged["Fark (İst-Ant)"] = merged["İstanbul"] - merged["Antalya"]

        fig_comp = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=("Fiyat Karşılaştırması", "İstanbul-Antalya Farkı (₺/kg)"),
                                  row_heights=[0.6, 0.4])

        fig_comp.add_trace(go.Scatter(x=merged["date"], y=merged["Antalya"],
                                       name="Antalya Hal", line=dict(color="darkorange")), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=merged["date"], y=merged["İstanbul"],
                                       name="İstanbul Hal", line=dict(color="royalblue")), row=1, col=1)

        fig_comp.add_trace(go.Bar(x=merged["date"], y=merged["Fark (İst-Ant)"],
                                   marker_color=np.where(merged["Fark (İst-Ant)"] > 0, "green", "red"),
                                   name="Fark", opacity=0.6), row=2, col=1)

        fig_comp.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_comp, use_container_width=True)

        avg_spread = merged["Fark (İst-Ant)"].mean()
        st.info(f"Ortalama İstanbul-Antalya farkı: **{avg_spread:.1f} ₺/kg** (nakliye + komisyon + marj)")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Genel Bakış
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Genel Bakış":
    st.title("🍊 Portakal Fiyat Tahmini — Genel Bakış")

    # KPI Cards
    col1, col2, col3, col4, col5 = st.columns(5)

    latest = prices_filtered.iloc[-1]
    prev_30 = prices_filtered[prices_filtered["date"] <= latest["date"] - pd.Timedelta(days=30)]

    with col1:
        st.metric(
            "Güncel Fiyat",
            f"{latest['avg_price']:.1f} ₺/kg",
            delta=f"{latest['avg_price'] - prev_30.iloc[-1]['avg_price']:.1f} ₺" if not prev_30.empty else None,
        )
    with col2:
        st.metric("Min Fiyat", f"{latest['min_price']:.1f} ₺/kg")
    with col3:
        st.metric("Max Fiyat", f"{latest['max_price']:.1f} ₺/kg")
    with col4:
        spread = latest["max_price"] - latest["min_price"]
        st.metric("Spread", f"{spread:.1f} ₺")
    with col5:
        st.metric("Kayıt Sayısı", f"{len(prices_filtered):,}")

    st.markdown("---")

    # Main price chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portakal Hal Fiyatı (TL/kg)", "Günlük Spread (Max - Min)"),
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
            name="Min-Max Aralığı",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["avg_price"],
            mode="lines", line=dict(color="darkorange", width=1.5),
            name="Ortalama Fiyat",
        ),
        row=1, col=1,
    )

    # 30-day moving average
    prices_filtered["ma30"] = prices_filtered["avg_price"].rolling(30, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(
            x=prices_filtered["date"], y=prices_filtered["ma30"],
            mode="lines", line=dict(color="red", width=1, dash="dash"),
            name="30 Gün Ort.",
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
    fig.update_yaxes(title_text="TL/kg", row=1, col=1)
    fig.update_yaxes(title_text="TL", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Yıllık Ortalamalar")
        yearly = prices_filtered.copy()
        yearly["year"] = yearly["date"].dt.year
        yearly_stats = yearly.groupby("year")["avg_price"].agg(["mean", "min", "max", "std"]).round(2)
        yearly_stats.columns = ["Ortalama", "Min", "Max", "Std"]
        st.dataframe(yearly_stats, use_container_width=True)

    with col_b:
        st.subheader("Aylık Mevsimsellik")
        monthly = prices_filtered.copy()
        monthly["month"] = monthly["date"].dt.month
        month_names = {1:"Oca",2:"Şub",3:"Mar",4:"Nis",5:"May",6:"Haz",
                       7:"Tem",8:"Ağu",9:"Eyl",10:"Eki",11:"Kas",12:"Ara"}
        monthly_stats = monthly.groupby("month")["avg_price"].mean().round(2)
        monthly_stats.index = monthly_stats.index.map(month_names)
        fig_month = px.bar(
            x=monthly_stats.index, y=monthly_stats.values,
            labels={"x": "Ay", "y": "Ort. Fiyat (TL/kg)"},
            color=monthly_stats.values,
            color_continuous_scale="Oranges",
        )
        fig_month.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_month, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Fiyat Analizi
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Fiyat Analizi":
    st.title("📈 Fiyat Analizi")

    tab1, tab2, tab3 = st.tabs(["Trend & Momentum", "Volatilite", "YoY Karşılaştırma"])

    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Fiyat + Hareketli Ortalamalar", "7 Günlük Değişim (%)", "30 Günlük Değişim (%)"),
        )

        fig.add_trace(go.Scatter(x=prices_filtered["date"], y=prices_filtered["avg_price"],
                                  name="Fiyat", line=dict(color="darkorange")), row=1, col=1)

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
                                 subplot_titles=("30 Gün Volatilite (TL)", "Volatilite (%)"))
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
        selected_years = st.multiselect("Yıllar", years, default=years[-4:])

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
            height=500, xaxis_title="Yılın Günü", yaxis_title="TL/kg",
            title="Yıl Bazında Fiyat Karşılaştırması", hovermode="x unified",
        )
        st.plotly_chart(fig_yoy, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Hava & Çevre
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Hava & Çevre":
    st.title("🌤️ Hava Durumu & Çevre Faktörleri")

    if "weather" not in data:
        st.warning("Hava durumu verisi bulunamadı.")
        st.stop()

    weather = data["weather"]
    w_mask = (weather["date"].dt.date >= start_date) & (weather["date"].dt.date <= end_date)
    wf = weather[w_mask].copy()

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        frost_days = int(wf["frost"].sum()) if "frost" in wf.columns else 0
        st.metric("Don Günü", frost_days)
    with col2:
        avg_temp = wf["temp_mean"].mean()
        st.metric("Ort. Sıcaklık", f"{avg_temp:.1f}°C")
    with col3:
        total_precip = wf["precipitation"].sum()
        st.metric("Toplam Yağış", f"{total_precip:.0f} mm")
    with col4:
        max_wind = wf["wind_speed_max"].max() if "wind_speed_max" in wf.columns else 0
        st.metric("Max Rüzgar", f"{max_wind:.1f} km/h")

    st.markdown("---")

    tab1, tab2 = st.tabs(["Sıcaklık & Yağış", "Fiyat-Hava İlişkisi"])

    with tab1:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
            subplot_titles=("Sıcaklık (°C)", "Günlük Yağış (mm)", "Nem (%)"),
        )

        fig.add_trace(go.Scatter(x=wf["date"], y=wf["temp_max"], name="Max",
                                  line=dict(color="red", width=0.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=wf["date"], y=wf["temp_min"], name="Min",
                                  fill="tonexty", fillcolor="rgba(255,0,0,0.1)",
                                  line=dict(color="blue", width=0.5)), row=1, col=1)
        fig.add_shape(type="line", y0=0, y1=0, x0=wf["date"].min(), x1=wf["date"].max(),
                      line=dict(color="darkblue", dash="dash", width=1), row=1, col=1)

        fig.add_trace(go.Bar(x=wf["date"], y=wf["precipitation"],
                              marker_color="royalblue", name="Yağış", opacity=0.7), row=2, col=1)

        if "humidity" in wf.columns:
            fig.add_trace(go.Scatter(x=wf["date"], y=wf["humidity"],
                                      line=dict(color="teal", width=0.8), name="Nem"), row=3, col=1)

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
                labels={"temp_mean": "Ort. Sıcaklık (°C)", "avg_price": "Fiyat (TL/kg)", "frost": "Don"},
                title="Sıcaklık vs Fiyat",
                opacity=0.5,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_b:
            fig_scatter2 = px.scatter(
                merged, x="precipitation", y="avg_price",
                labels={"precipitation": "Yağış (mm)", "avg_price": "Fiyat (TL/kg)"},
                title="Yağış vs Fiyat",
                opacity=0.3,
                color_discrete_sequence=["royalblue"],
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Pazar & Politika
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Pazar & Politika":
    st.title("🌍 Pazar Dinamikleri & Politika Etkileri")

    tab1, tab2, tab3 = st.tabs(["Döviz & Uluslararası", "Politika Olayları", "Rekabet"])

    with tab1:
        col_a, col_b = st.columns(2)

        with col_a:
            if "fx" in data:
                fx = data["fx"]
                usd_col = "TRY_per_USD" if "TRY_per_USD" in fx.columns else "USD_TRY"
                fig_fx = px.line(fx, x="date", y=usd_col, title="USD/TRY Kuru",
                                 labels={"date": "", usd_col: "TL per USD"})
                fig_fx.update_traces(line_color="purple")
                fig_fx.update_layout(height=350)
                st.plotly_chart(fig_fx, use_container_width=True)

        with col_b:
            if "foreign" in data:
                foreign = data["foreign"]
                fig_fao = px.line(foreign, x="date", y="fao_fruit_index",
                                  title="FAO Meyve Fiyat Endeksi",
                                  labels={"date": "", "fao_fruit_index": "Endeks (2014-16=100)"})
                fig_fao.update_traces(line_color="darkgreen")
                fig_fao.update_layout(height=350)
                fig_fao.add_hline(y=100, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_fao, use_container_width=True)

        # Dual axis: Price vs USD/TRY
        if "fx" in data:
            fx = data["fx"]
            usd_col = "TRY_per_USD" if "TRY_per_USD" in fx.columns else "USD_TRY"

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            monthly_p = prices.set_index("date").resample("M")["avg_price"].mean().reset_index()
            monthly_fx = fx.set_index("date")[usd_col].resample("M").mean().reset_index()

            fig_dual.add_trace(
                go.Scatter(x=monthly_p["date"], y=monthly_p["avg_price"],
                           name="Portakal (TL/kg)", line=dict(color="darkorange", width=2)),
                secondary_y=False,
            )
            fig_dual.add_trace(
                go.Scatter(x=monthly_fx["date"], y=monthly_fx[usd_col],
                           name="USD/TRY", line=dict(color="purple", width=2)),
                secondary_y=True,
            )
            fig_dual.update_layout(title="Portakal Fiyatı vs Döviz Kuru (Aylık)", height=400,
                                    hovermode="x unified")
            fig_dual.update_yaxes(title_text="TL/kg", secondary_y=False)
            fig_dual.update_yaxes(title_text="USD/TRY", secondary_y=True)
            st.plotly_chart(fig_dual, use_container_width=True)

    with tab2:
        if "events" in data:
            events = data["events"]

            st.subheader("Politika ve Olay Zaman Çizelgesi")

            # Price chart with event markers
            fig_events = go.Figure()
            fig_events.add_trace(go.Scatter(
                x=prices["date"], y=prices["avg_price"],
                mode="lines", line=dict(color="darkorange", width=1),
                name="Fiyat", opacity=0.7,
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
                    hovertemplate=f"<b>{ev['description']}</b><br>Tarih: {ev['date'].strftime('%Y-%m-%d')}<br>"
                                 f"Tür: {ev['event_type']}<br>Etki: {ev['impact_direction']} ({ev['impact_magnitude']})<extra></extra>",
                ))

            fig_events.update_layout(height=500, title="Fiyat + Politika Olayları", hovermode="closest")
            st.plotly_chart(fig_events, use_container_width=True)

            # Event table
            st.subheader("Olay Listesi")
            display_events = events[["date", "event_type", "description", "impact_direction", "impact_magnitude"]].copy()
            display_events["date"] = display_events["date"].dt.strftime("%Y-%m-%d")
            display_events.columns = ["Tarih", "Tür", "Açıklama", "Etki Yönü", "Büyüklük"]
            st.dataframe(display_events, use_container_width=True, hide_index=True)

        # Policy impact score
        if "policy" in data:
            policy = data["policy"]
            fig_impact = go.Figure()
            pos_mask = policy["policy_impact_score"] > 0
            fig_impact.add_trace(go.Bar(
                x=policy.loc[pos_mask, "date"], y=policy.loc[pos_mask, "policy_impact_score"],
                marker_color="red", name="Fiyat Artırıcı", opacity=0.6,
            ))
            fig_impact.add_trace(go.Bar(
                x=policy.loc[~pos_mask, "date"], y=policy.loc[~pos_mask, "policy_impact_score"],
                marker_color="blue", name="Fiyat Düşürücü", opacity=0.6,
            ))
            fig_impact.update_layout(title="Politika Etki Skoru", height=300, hovermode="x unified",
                                      barmode="relative")
            st.plotly_chart(fig_impact, use_container_width=True)

    with tab3:
        if "foreign" in data:
            foreign = data["foreign"]

            col_a, col_b = st.columns(2)

            with col_a:
                if "eu_orange_price_eur_100kg" in foreign.columns:
                    fig_eu = px.line(foreign, x="date", y="eu_orange_price_eur_100kg",
                                     title="AB Portakal Fiyatı (EUR/100kg)",
                                     labels={"eu_orange_price_eur_100kg": "EUR/100kg"})
                    fig_eu.update_traces(line_color="royalblue")
                    fig_eu.update_layout(height=350)
                    st.plotly_chart(fig_eu, use_container_width=True)

            with col_b:
                if "competition_index" in foreign.columns:
                    comp = foreign.dropna(subset=["competition_index"])
                    fig_comp = px.bar(comp, x="date", y="competition_index",
                                      title="Rekabet Yoğunluğu Endeksi",
                                      labels={"competition_index": "Endeks"},
                                      color="active_competitors",
                                      color_continuous_scale="Reds")
                    fig_comp.update_layout(height=350)
                    st.plotly_chart(fig_comp, use_container_width=True)

            # Competitor production
            from src.data.foreign_markets import fetch_competitor_production
            prod = fetch_competitor_production()

            st.subheader("Rakip Ülke Üretim Karşılaştırması")
            year_select = st.selectbox("Yıl", sorted(prod["year"].unique(), reverse=True))
            yr = prod[prod["year"] == year_select].sort_values("production_kt", ascending=True)

            fig_prod = px.bar(yr, x="production_kt", y="country", orientation="h",
                               color="estimated_export_kt", color_continuous_scale="YlOrRd",
                               labels={"production_kt": "Üretim (bin ton)", "country": "",
                                        "estimated_export_kt": "İhracat (bin ton)"},
                               title=f"{year_select} Portakal Üretimi")
            fig_prod.update_layout(height=350)
            st.plotly_chart(fig_prod, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Model Sonuçları
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Model Sonuçları":
    st.title("🤖 Model Sonuçları")

    if "results" not in data:
        st.warning("Model sonuçları bulunamadı. Pipeline'ı çalıştırın.")
        st.stop()

    results = data["results"]

    # Best model highlight
    best = results.loc[results["MAE"].idxmin()]
    st.success(f"En iyi model: **{best['model']}** — MAE: {best['MAE']:.2f} TL/kg, "
               f"MAPE: {best['MAPE']*100:.1f}%, R²: {best['R2']:.3f} ({int(best['horizon_days'])} gün)")

    st.markdown("---")

    # Comparison charts
    col1, col2 = st.columns(2)

    with col1:
        fig_mae = px.bar(results, x="model", y="MAE", color="horizon_days",
                          barmode="group", title="MAE Karşılaştırması (TL/kg)",
                          color_continuous_scale="Viridis",
                          labels={"MAE": "MAE (TL/kg)", "model": "", "horizon_days": "Horizon"})
        fig_mae.update_layout(height=400)
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        fig_r2 = px.bar(results, x="model", y="R2", color="horizon_days",
                          barmode="group", title="R² Karşılaştırması",
                          color_continuous_scale="Viridis",
                          labels={"R2": "R²", "model": "", "horizon_days": "Horizon"})
        fig_r2.add_hline(y=0, line_dash="dash", line_color="red")
        fig_r2.update_layout(height=400)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Full results table
    st.subheader("Detaylı Sonuçlar")
    display_results = results.copy()
    display_results["MAPE"] = (display_results["MAPE"] * 100).round(1).astype(str) + "%"
    display_results["MAE"] = display_results["MAE"].round(2)
    display_results["RMSE"] = display_results["RMSE"].round(2)
    display_results["R2"] = display_results["R2"].round(3)
    display_results.columns = ["Model", "Horizon (gün)", "MAE", "MAPE", "RMSE", "R²"]
    st.dataframe(display_results, use_container_width=True, hide_index=True)

    # Feature importance (if we can compute it)
    st.markdown("---")
    st.subheader("Özellik Matrisi Özeti")

    if "features" in data:
        features = data["features"]
        numeric_cols = features.select_dtypes(include=[np.number]).columns

        horizon = st.selectbox("Tahmin Horizon", [30, 60, 90])
        target = f"target_{horizon}d"

        if target in features.columns:
            corr = features[numeric_cols].corrwith(features[target]).abs().sort_values(ascending=False)
            corr = corr.drop([c for c in corr.index if c.startswith("target_")])
            top_20 = corr.head(20)

            fig_fi = px.bar(x=top_20.values, y=top_20.index, orientation="h",
                             title=f"Top 20 Özellik — {horizon} Gün Hedefe Korelasyon",
                             labels={"x": "|Korelasyon|", "y": ""},
                             color=top_20.values, color_continuous_scale="Oranges")
            fig_fi.update_layout(height=500, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_fi, use_container_width=True)

        st.info(f"Toplam özellik sayısı: **{len(numeric_cols)}** | Satır: **{len(features):,}** | "
                f"Tarih: {features['date'].min().strftime('%Y-%m-%d')} — {features['date'].max().strftime('%Y-%m-%d')}")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Talep & Trendler
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Talep & Trendler":
    st.title("📊 Talep Sinyalleri & Google Trends")

    tab1, tab2 = st.tabs(["Google Trends", "Talep Faktörleri"])

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
                title="Google Trends — Portakal Arama İlgisi (Türkiye)",
                height=400, hovermode="x unified",
                yaxis_title="Arama İlgisi (0-100)",
            )
            st.plotly_chart(fig_trends, use_container_width=True)

            # Overlay with price
            fig_tp = make_subplots(specs=[[{"secondary_y": True}]])
            monthly_p = prices.set_index("date").resample("M")["avg_price"].mean().reset_index()
            if "trend_portakal_fiyat" in trends.columns:
                fig_tp.add_trace(go.Scatter(x=trends["date"], y=trends["trend_portakal_fiyat"],
                                             name="Trend: portakal fiyat", line=dict(color="blue")), secondary_y=False)
            fig_tp.add_trace(go.Scatter(x=monthly_p["date"], y=monthly_p["avg_price"],
                                         name="Fiyat (TL/kg)", line=dict(color="darkorange")), secondary_y=True)
            fig_tp.update_layout(title="Arama İlgisi vs Fiyat", height=350, hovermode="x unified")
            fig_tp.update_yaxes(title_text="Trend (0-100)", secondary_y=False)
            fig_tp.update_yaxes(title_text="TL/kg", secondary_y=True)
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.info("Trend verisi bulunamadı.")

    with tab2:
        if "demand" in data:
            demand = data["demand"]
            dm_filtered = demand[(demand["date"].dt.date >= start_date) & (demand["date"].dt.date <= end_date)]

            col1, col2 = st.columns(2)

            with col1:
                # Ramadan periods
                ramadan_data = dm_filtered[dm_filtered["ramadan_active"] == 1]
                st.subheader("Ramazan Dönemleri")
                fig_ram = go.Figure()
                fig_ram.add_trace(go.Scatter(
                    x=prices_filtered["date"], y=prices_filtered["avg_price"],
                    mode="lines", line=dict(color="darkorange"), name="Fiyat",
                ))
                # Shade Ramadan periods
                for _, row in dm_filtered[dm_filtered["ramadan_active"] == 1].groupby(
                    (dm_filtered["ramadan_active"] != dm_filtered["ramadan_active"].shift()).cumsum()
                ).agg({"date": ["min", "max"]}).iterrows():
                    fig_ram.add_vrect(x0=row[("date", "min")], x1=row[("date", "max")],
                                      fillcolor="green", opacity=0.1, line_width=0)
                fig_ram.update_layout(height=300, title="Fiyat + Ramazan Dönemleri (yeşil)")
                st.plotly_chart(fig_ram, use_container_width=True)

            with col2:
                # Input cost index
                st.subheader("Girdi Maliyet Endeksi")
                fig_input = px.line(dm_filtered, x="date", y="input_cost_index",
                                     title="Tarımsal Girdi Maliyet Endeksi (2015=100)")
                fig_input.update_traces(line_color="brown")
                fig_input.update_layout(height=300)
                st.plotly_chart(fig_input, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Turizm Yoğunluğu")
                fig_tour = px.line(dm_filtered, x="date", y="tourism_intensity",
                                    title="Antalya Turizm Yoğunluğu")
                fig_tour.update_traces(line_color="teal")
                fig_tour.update_layout(height=300)
                st.plotly_chart(fig_tour, use_container_width=True)

            with col4:
                st.subheader("TÜFE Endeksi")
                fig_cpi = px.line(dm_filtered, x="date", y="cpi_index",
                                   title="Tüketici Fiyat Endeksi (2007=100)")
                fig_cpi.update_traces(line_color="purple")
                fig_cpi.update_layout(height=300)
                st.plotly_chart(fig_cpi, use_container_width=True)
        else:
            st.info("Talep verisi bulunamadı.")


# ═════════════════════════════════════════════════════════════════════════════════
# PAGE: Tahminler & Uyarılar
# ═════════════════════════════════════════════════════════════════════════════════

elif page == "Tahminler & Uyarılar":
    st.title("🔮 Tahminler & Uyarı Sistemi")

    tab1, tab2, tab3 = st.tabs(["Fiyat Tahminleri", "Uyarılar", "SHAP Analizi"])

    with tab1:
        if "predictions" in data:
            preds = data["predictions"]
            current = prices.iloc[-1]["avg_price"]

            st.subheader(f"Güncel Fiyat: {current:.1f} ₺/kg")
            st.markdown("---")

            cols = st.columns(len(preds))
            for i, (_, row) in enumerate(preds.iterrows()):
                with cols[i]:
                    horizon = int(row["horizon_days"])
                    pred = row["prediction"]
                    change = pred - current
                    change_pct = (change / current) * 100

                    st.metric(
                        f"{horizon} Gün Tahmin",
                        f"{pred:.1f} ₺/kg",
                        delta=f"{change:+.1f} ₺ ({change_pct:+.1f}%)",
                    )

                    if "pred_lower" in row and "pred_upper" in row:
                        st.caption(f"Aralık: {row['pred_lower']:.1f} — {row['pred_upper']:.1f} ₺")

            # Prediction chart
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=prices.tail(90)["date"], y=prices.tail(90)["avg_price"],
                mode="lines", name="Gerçek Fiyat", line=dict(color="darkorange", width=2),
            ))

            for _, row in preds.iterrows():
                target_date = pd.Timestamp(row.get("target_date", ""))
                fig_pred.add_trace(go.Scatter(
                    x=[prices.iloc[-1]["date"], target_date],
                    y=[current, row["prediction"]],
                    mode="lines+markers",
                    name=f"{int(row['horizon_days'])}d tahmin",
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

            fig_pred.update_layout(title="Fiyat Tahmini", height=400, hovermode="x unified")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("Tahmin verisi bulunamadı. `python -m src.auto_refresh --predict` çalıştırın.")

    with tab2:
        if "alerts_text" in data:
            st.code(data["alerts_text"], language=None)
        else:
            # Generate live alerts
            st.subheader("Canlı Uyarı Kontrolü")
            if st.button("Uyarıları Kontrol Et"):
                from src.auto_refresh import run_alerts
                alerts = run_alerts()
                st.success(f"{len(alerts)} uyarı kontrol edildi.")

    with tab3:
        if "shap" in data:
            shap_df = data["shap"]
            top_n = st.slider("Top N Özellik", 10, 40, 20)
            top = shap_df.head(top_n)

            fig_shap = px.bar(
                top, x="shap_importance", y="feature", orientation="h",
                title=f"SHAP Feature Importance (Top {top_n})",
                labels={"shap_importance": "Ortalama |SHAP Değeri|", "feature": ""},
                color="shap_importance", color_continuous_scale="Oranges",
            )
            fig_shap.update_layout(height=max(400, top_n * 22), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_shap, use_container_width=True)

            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.info("SHAP verisi bulunamadı. `python -m src.pipeline --advanced` çalıştırın.")


# ─── Footer ──────────────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Veri Kaynakları:**\n"
    "- İBB Istanbul Hal\n"
    "- Open-Meteo\n"
    "- Frankfurter (FX)\n"
    "- FAO / Eurostat\n"
    "- USDA FAS"
)
st.sidebar.markdown(f"Son güncelleme: {prices['date'].max().strftime('%d.%m.%Y')}")
