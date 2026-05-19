"""Plain-English paragraph descriptions for every chart and table in the dashboard.

Centralized so a non-technical reader (e.g. an American capstone grader who is not
familiar with Turkish orange wholesale markets) can read one short paragraph below
every visualization and know what it shows, how to read it, and why it matters.

Used via the `explain(key)` helper which renders the paragraph as an italic
Streamlit markdown block directly under the chart.
"""
from __future__ import annotations

DESCRIPTIONS: dict[str, str] = {
    # ─── Farmer Panel ────────────────────────────────────────────────────────────
    "farmer_forecast": (
        "Predicted Antalya wholesale orange price (TRY per kilogram) over the next 7, 30, "
        "and 90 days. The dashed lines are model forecasts; the orange dot is the most "
        "recent actual price. Use this to compare today's price against where the model "
        "thinks the market is heading."
    ),
    "farmer_cost_breakdown": (
        "Per-kilogram cost of producing, harvesting, and selling the orange — including "
        "growing inputs, labor, transport, cold storage, and the wholesale-market "
        "commission. Sum is the farmer's break-even price; anything above this is profit."
    ),
    "farmer_cold_storage": (
        "Cold storage delays selling until prices typically rise later in the season. "
        "The number shown is the model's expected gain from storing now versus selling "
        "now, after subtracting storage cost. Positive means storage is worthwhile; "
        "negative means sell now."
    ),

    # ─── Overview ────────────────────────────────────────────────────────────────
    "overview_main_timeline": (
        "Long-run daily wholesale price of Finike-region oranges, smoothed by a "
        "30-day moving average so seasonal cycles are easier to read. Steep drops "
        "typically mark a glut at peak harvest; spikes mark frost, export shocks, "
        "or FX-driven cost surges."
    ),
    "overview_yearly_stats": (
        "Year-by-year averages of the daily wholesale price, the year's minimum and "
        "maximum, and the number of trading days covered. Useful to see how each year "
        "compares against the multi-year trend, especially the post-2021 inflation jump."
    ),
    "overview_monthly_seasonality": (
        "Average orange price by calendar month across all available years. Prices are "
        "lowest at peak harvest (December–April) and climb in summer when fresh supply "
        "thins and stored fruit dominates. This is the seasonal pattern the model has "
        "to beat to add real value."
    ),
    "overview_antalya_vs_istanbul": (
        "Side-by-side daily wholesale prices in the two Hal markets the dashboard "
        "tracks. Antalya is closer to the orchards; Istanbul is the largest consumption "
        "market. The gap between them reflects transport costs, regional demand, and "
        "arbitrage opportunities for traders."
    ),

    # ─── Price Analysis ─────────────────────────────────────────────────────────
    "price_with_ma": (
        "Daily wholesale price (thin orange line) with a 7-day and 30-day moving "
        "average overlaid. The moving averages strip out single-day noise to make the "
        "underlying trend visible. When the short MA crosses above the long MA, it "
        "often signals a shift toward higher prices."
    ),
    "price_volume_spread": (
        "Top panel: total daily volume (kilograms traded) at the wholesale market. "
        "Bottom panel: the daily price spread, computed as the maximum minus the "
        "minimum reported price on that trading day. A wider spread means more price "
        "uncertainty or quality variation between lots."
    ),
    "price_yoy_change": (
        "Same calendar day this year compared to the same day a year earlier, expressed "
        "as a percentage change. Positive bars mean prices are higher than last year; "
        "negative bars mean they are lower. Best read in context with the inflation "
        "rate of the same period."
    ),

    # ─── Weather & Environment ──────────────────────────────────────────────────
    "weather_subplots": (
        "Daily Finike weather over the last several years: temperature on top, "
        "precipitation in the middle, growing-degree-days on the bottom. The most "
        "price-relevant signals are sub-zero temperature events in December–February "
        "(frost risk) and unusually dry stretches during flowering."
    ),
    "weather_temp_vs_price": (
        "Each dot is a single day plotted by its average temperature (x) against its "
        "wholesale orange price (y). Look for clusters at low temperatures with high "
        "prices — that is the frost-shock signature the model uses to forecast frost-"
        "driven price spikes."
    ),
    "weather_rain_vs_price": (
        "Each dot is a single day plotted by its precipitation (x) against its "
        "wholesale orange price (y). Most days have light rain; large precipitation "
        "events can suppress harvest or damage fruit, both of which tend to lift prices "
        "in the following weeks."
    ),

    # ─── Market & Policy ────────────────────────────────────────────────────────
    "market_usd_try": (
        "Daily Turkish lira per US dollar exchange rate. A weaker lira (higher number) "
        "increases the cost of imported inputs (fertilizer, fuel, pesticides) and makes "
        "Turkish oranges cheaper for foreign buyers — both effects push wholesale "
        "prices up in TRY terms."
    ),
    "market_fao_fruit_index": (
        "FAO Fruit Price Index — a monthly global benchmark of fruit prices indexed "
        "to 2014–2016 = 100. Useful to separate Turkey-specific shocks from worldwide "
        "fruit-market moves. When Turkish orange prices diverge sharply from this "
        "index, the cause is usually domestic (weather, policy, or FX)."
    ),
    "market_price_vs_fx": (
        "Monthly average orange price (left axis, TRY/kg) plotted against the monthly "
        "average USD/TRY exchange rate (right axis). The two series usually move "
        "together during currency crises; divergences highlight months when domestic "
        "supply or demand shocks dominated."
    ),
    "market_event_timeline": (
        "Daily price line overlaid with markers for known policy and supply events "
        "(frost, sanctions, FX crises, regulations). Up-arrows mark events that "
        "historically lifted prices; down-arrows mark events that lowered them. Hover "
        "any marker for the event description."
    ),
    "market_event_list": (
        "Full list of the 36 hand-curated policy and market events the model treats as "
        "structured features. Each row shows the date, event type, English description, "
        "direction the event pushed prices, and the magnitude (Minor / Moderate / Major)."
    ),
    "market_news_feed": (
        "Most recent Turkish agriculture news articles classified by a DeepSeek "
        "large language model as relevant to orange supply, demand, or policy. "
        "Each row shows the publication date, the model's sentiment call, the "
        "mapped event type, a one-sentence English summary, and a link back to "
        "the source. Populated automatically by the daily refresh job; an empty "
        "section means no relevant articles fired in the latest run."
    ),
    "market_policy_impact_score": (
        "Daily synthesized score combining all recent policy events, weighted by their "
        "magnitude and decayed over time. Positive bars are days where recent events "
        "were net price-raising; negative bars are net price-lowering. The model uses "
        "this score as a feature."
    ),
    "market_eu_price": (
        "Average wholesale orange price across the European Union in euros per 100 kg, "
        "a competing-supply benchmark. Higher EU prices generally pull Turkish export "
        "volumes upward and tighten domestic supply, lifting Turkish wholesale prices."
    ),
    "market_competition_index": (
        "Composite measure of how much competing exporter countries (Egypt, Spain, "
        "Morocco, South Africa, etc.) are pushing supply into Turkey's export markets. "
        "Higher values mean Turkish exporters face stiffer competition for foreign "
        "buyers, indirectly softening Turkish wholesale prices."
    ),
    "market_competitor_production": (
        "Annual orange production in thousand metric tons for each competitor country, "
        "shaded by estimated export volume. Reading the bars top-to-bottom shows which "
        "countries dominate global supply each year; the export shading shows how much "
        "of that supply actually crosses borders."
    ),

    # ─── Model Results ──────────────────────────────────────────────────────────
    "model_mae_comparison": (
        "Mean Absolute Error (MAE) of each candidate model at each forecast horizon "
        "(7, 30, 90 days). MAE is reported in TRY/kg — it is the average distance "
        "between the model's predicted price and the actual price, so lower bars are "
        "better. Use this to pick the model with the smallest error at your horizon."
    ),
    "model_r2_comparison": (
        "R² (coefficient of determination) of each model at each horizon. R² runs from "
        "0 (the model is no better than predicting the average) to 1 (the model "
        "perfectly explains every day's price). Anything above 0.7 is strong for a "
        "real-world agricultural commodity."
    ),
    "model_detailed_results": (
        "Full numeric performance breakdown for each (model, horizon) pair: MAE, MAPE "
        "(percentage error), and R². MAPE is easier to compare across price regimes — "
        "a MAPE of 0.10 means the model's prediction is off by about 10% on average."
    ),
    "model_feature_importance": (
        "The top 20 features the best model relies on most, ranked by mean absolute "
        "SHAP value. SHAP measures each feature's contribution to a single prediction; "
        "averaging across all predictions gives a global importance score. Higher bars "
        "mean the model leans more heavily on that feature."
    ),

    # ─── Demand & Trends ────────────────────────────────────────────────────────
    "demand_google_trends": (
        "Weekly Google search interest for the term 'orange price' in Turkey, indexed "
        "0–100. Sharp spikes typically follow news coverage of price moves or major "
        "supply events; the model uses this as a demand-signal feature."
    ),
    "demand_trend_vs_price": (
        "Weekly search interest (right axis) overlaid against the orange wholesale "
        "price (left axis). When search interest leads price moves, it indicates "
        "consumer attention is driving demand; when it lags, attention is reacting to "
        "already-occurred price changes."
    ),
    "demand_ramadan": (
        "Highlighted dates of recent Ramadan holiday periods. Ramadan typically lifts "
        "fresh-fruit demand because oranges are a common iftar (fast-breaking) food, "
        "and the shifting Islamic calendar means Ramadan moves earlier by about 11 "
        "days each year relative to the orange harvest."
    ),
    "demand_input_cost": (
        "Index of farmer input costs over time (fertilizer, fuel, pesticides, labor). "
        "Rising input costs squeeze farmer margins and force higher wholesale prices "
        "in subsequent seasons; this index is one of the model's structural cost "
        "features."
    ),
    "demand_tourism": (
        "Tourism activity index for the Antalya region — proxy for local fresh-fruit "
        "demand from hotels, restaurants, and visitors. Higher tourism intensity "
        "during the summer months tends to keep prices supported despite lower "
        "overall consumption nationwide."
    ),
    "demand_cpi": (
        "Turkish Consumer Price Index — broad measure of nationwide inflation. "
        "Orange wholesale prices typically rise in nominal TRY along with general "
        "inflation, but model performance is judged against real-terms moves after "
        "removing this background pressure."
    ),

    # ─── Forecasts & Alerts ─────────────────────────────────────────────────────
    "forecast_predictions": (
        "Latest model predictions for the next 7, 14, 30, 60, and 90 days, plotted "
        "against recent actual prices. The shaded band (when shown) is the model's "
        "uncertainty range — wider bands mean less confidence about the exact "
        "outcome."
    ),
    "forecast_tracking": (
        "Side-by-side comparison of past predictions against the actual prices that "
        "later materialized. Each pair shows how close the model came on real, "
        "out-of-sample days — the test that matters more than any backtest number."
    ),
    "forecast_error": (
        "Time series of prediction error (predicted minus actual) for past forecasts. "
        "Errors clustered around zero mean the model is well calibrated; persistent "
        "positive or negative bias indicates the model is over- or underestimating "
        "and may need retraining."
    ),
    "forecast_pending": (
        "Forecasts the model has issued whose target date is still in the future — "
        "so we cannot yet score them against an actual price. Will be evaluated as "
        "their target dates roll past."
    ),
    "forecast_shap": (
        "For the most recent prediction, the features that pushed the model's output "
        "up or down the most. Each bar's length is the marginal effect of that "
        "feature on this specific prediction (in TRY/kg). This explains today's "
        "forecast specifically, not the model in general."
    ),
    "forecast_shap_table": (
        "Numeric companion to the SHAP chart above — the top contributing features "
        "with their current value and their TRY/kg contribution to today's forecast."
    ),

    # ─── Data Freshness Panel (top banner) ──────────────────────────────────────
    "freshness_table": (
        "Per-source check that the daily data pipeline is running. 'Fresh' means data "
        "is up to date within the last two days; 'Slightly stale' is 3–7 days; "
        "'Stale' is more than a week behind. Any red rows should be investigated "
        "before trusting that day's prediction."
    ),
}


def explain(key: str) -> None:
    """Render the description for `key` as an italic markdown paragraph.

    Called immediately below the matching chart or table. No-op (logs nothing) if
    the key is missing — a missing description is a soft bug, not a runtime failure.
    """
    import streamlit as st

    text = DESCRIPTIONS.get(key)
    if not text:
        return
    st.markdown(f"_{text}_")
