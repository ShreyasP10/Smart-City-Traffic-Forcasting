"""Smart City Real-Time Traffic Congestion Forecasting & Adaptation — Streamlit dashboard.

Hardened against:
- Stored XSS via html.escape on all interpolated user/prediction HTML.
- SSRF via validate_coordinates on custom latitude/longitude inputs.
- Unsafe joblib deserialization via safe_load_model with SHA-256 integrity verification.
- Feature vector drift via canonical schema.featurize_point.
- Cache file race conditions via isolated data/weather_cache.json with atomic replacement.
- Redundant API calls via @st.cache_data on route flow fetches.
- Division-by-zero in travel time and delay calculations.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from city_config import CITIES, get_city_now
from traffic_api import fetch_openweather, fetch_tomtom_flow, contribution_text, validate_coordinates
from data_generator import load_historical
from explain_shap import (
    compute_shap_explanation,
    plot_shap_waterfall_chart,
    plot_shap_factor_bars,
    plot_shap_driver_shares,
)
from schema import safe_load_model, featurize_point
from pipeline import predict_pipeline, classify

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib")
CHECKSUMS_FILE = os.path.join(os.path.dirname(__file__), "models", "checksums.json")

LEVEL_COLOR = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
LEVEL_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}

st.set_page_config(page_title="Smart City Traffic Forecaster", page_icon="🚦", layout="wide")


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        from train_model import main as train_main
        train_main()
    expected_sha = None
    if os.path.exists(CHECKSUMS_FILE):
        try:
            expected_sha = json.load(open(CHECKSUMS_FILE)).get(os.path.basename(MODEL_PATH))
        except Exception:
            pass
    return safe_load_model(MODEL_PATH, expected_sha)


@st.cache_data
def load_data():
    us = load_historical(indian_style=False)
    india = load_historical(indian_style=True)
    return us, india




def explain_prediction(pred: dict, weather: dict, flow: dict, bundle: dict) -> list[str]:
    f = pred["features"]
    out = []
    if f["is_peak"]:
        out.append((f"⏰ Peak hour ({f['hour']}:00) — strongest driver of {pred['final_level']} forecast", 3))
    elif f["is_night"]:
        out.append(("🌙 Off-peak night hours — pushes forecast toward Low", -2))
    else:
        out.append((f"🕐 Hour {f['hour']}:00 is off-peak — moderate effect", 0))
    if f["is_weekend"]:
        out.append(("📅 Weekend — typically 30–45% lower volume than weekday", -2))
    else:
        out.append(("📅 Weekday — recurrent commuter demand raises volume", 1))
    if f["is_holiday"]:
        out.append(("🎉 Holiday — ~40% lower volume expected", -2))
    rain = float(weather.get("rain_1h", 0))
    if rain > 2:
        out.append((f"🌧 Heavy rain ({rain} mm/h) — +15–25% congestion, slower speeds", 3))
    elif rain > 0:
        out.append((f"🌦 Light rain ({rain} mm/h) — +10% congestion", 1))
    else:
        out.append(("☀ Dry weather — no rain penalty", 0))
    ratio = float(flow.get("congestion_ratio", 0.7))
    if ratio < 0.5:
        out.append((f"🚗 Live speed ratio {ratio:.2f} — road already much slower than free-flow", 3))
    elif ratio < 0.75:
        out.append((f"🚗 Live speed ratio {ratio:.2f} — mild current slowdown", 1))
    else:
        out.append((f"🛣 Live speed ratio {ratio:.2f} — near free-flow, no uplift", -1))
    if f["clouds_all"] > 70:
        out.append((f"☁ Overcast ({f['clouds_all']}%) — small visibility effect", 1))
    out.sort(key=lambda x: -abs(x[1]))
    return [t for t, _ in out]


@st.cache_data(ttl=60)
def cached_fetch_flow(lat: float, lon: float, free_flow: float, key: str | None, hint: float):
    """Caches route flow fetches to avoid redundant API hammering on reruns."""
    return fetch_tomtom_flow(lat, lon, free_flow, key, congestion_hint=hint)


@st.cache_data(ttl=300, show_spinner=False)
def cached_fetch_weather(lat: float, lon: float, key: str | None) -> dict:
    """Caches live weather fetches for 5 minutes to avoid redundant API queries."""
    return fetch_openweather(lat, lon, key)


# ---------------- Sidebar ----------------
bundle = load_bundle()
us_df, india_df = load_data()

st.sidebar.title("🚦 Controls")
city = st.sidebar.selectbox("City (context)", list(CITIES.keys()), index=0)
routes = CITIES[city]["routes"]
route_name = st.sidebar.selectbox("Route / location", [r["name"] for r in routes], key=f"route_{city}")
route = next((r for r in routes if r["name"] == route_name), routes[0] if routes else None)
if route is None:
    route = {"name": route_name or "Default route", "lat": 19.076, "lon": 72.877, "free_flow": 50, "length_km": 5.0}

with st.sidebar.expander("➕ Custom location (optional)"):
    c_lat = st.number_input("Latitude", value=float(route["lat"]), min_value=-90.0, max_value=90.0, format="%.5f")
    c_lon = st.number_input("Longitude", value=float(route["lon"]), min_value=-180.0, max_value=180.0, format="%.5f")
    c_ff = st.number_input("Free-flow speed (km/h)", value=int(route["free_flow"]), min_value=10, max_value=120)
    use_custom = st.checkbox("Use custom location", value=False)
if use_custom:
    try:
        val_lat, val_lon = validate_coordinates(c_lat, c_lon)
        route = {"name": "Custom point", "lat": val_lat, "lon": val_lon, "free_flow": c_ff, "length_km": 5.0}
    except ValueError as _e:
        st.sidebar.error(f"Invalid coordinates: {_e}")

st.sidebar.subheader("⏱ Time")
use_now = st.checkbox("Use current time", value=True)
if use_now:
    at = get_city_now(city)
else:
    d = st.sidebar.date_input("Date", value=dt.date.today())
    h = st.sidebar.slider("Hour", 0, 23, 18)
    at = dt.datetime.combine(d, dt.time(h, 0))
horizon = st.sidebar.select_slider("Forecast horizon (min)", [0, 15, 30, 60], value=30)

st.sidebar.subheader("🔑 Live APIs (optional)")
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return os.environ.get(name, "")
tom_user = st.sidebar.text_input("TomTom KEY VALUE (optional)", value="", type="password",
                                help="Leave blank to use server environment / secrets.toml key.")
owm_user = st.sidebar.text_input("OpenWeather API key (optional)", value="", type="password",
                                help="Leave blank to use server environment / secrets.toml key. (Open-Meteo fallback active if blank)")
tom_key = tom_user.strip() or _secret("TOMTOM_API_KEY")
owm_key = owm_user.strip() or _secret("OPENWEATHER_API_KEY")
st.sidebar.caption("TomTom live ✅ · Weather: OpenWeather → Open-Meteo live fallback → simulation.")

# ---------------- Live data (isolated weather cache with atomic write & location keying) ----------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
WEATHER_CACHE_FILE = os.path.join(DATA_DIR, "weather_cache.json")

loc_key = f"{route['lat']:.3f},{route['lon']:.3f}"
all_cached_weather = {}
if os.path.exists(WEATHER_CACHE_FILE):
    try:
        with open(WEATHER_CACHE_FILE, "r") as _cf:
            all_cached_weather = json.load(_cf)
    except Exception:
        pass

try:
    weather = cached_fetch_weather(route["lat"], route["lon"], owm_key or None)
    if weather.get("ok", False):
        all_cached_weather[loc_key] = {
            "weather": weather,
            "cached_at": dt.datetime.now().isoformat(timespec="seconds")
        }
        try:
            tmp_cf = WEATHER_CACHE_FILE + ".tmp"
            with open(tmp_cf, "w") as _cf:
                json.dump(all_cached_weather, _cf)
            os.replace(tmp_cf, WEATHER_CACHE_FILE)
        except Exception:
            pass
    else:
        # Fallback to location-keyed cache (W6)
        loc_entry = all_cached_weather.get(loc_key)
        if loc_entry and loc_entry.get("weather"):
            cached_w = loc_entry["weather"].copy()
            cached_w["source"] = f"cached ({loc_entry.get('cached_at', 'earlier')})"
            st.sidebar.warning(f"Live weather API offline — using cached weather ({loc_entry.get('cached_at', 'earlier')}).")
            weather = cached_w
except Exception as _e:
    loc_entry = all_cached_weather.get(loc_key)
    if loc_entry and loc_entry.get("weather"):
        cached_w = loc_entry["weather"].copy()
        cached_w["source"] = f"cached ({loc_entry.get('cached_at', 'earlier')})"
        st.sidebar.warning(f"Live weather unavailable ({_e}) — using cached weather ({loc_entry.get('cached_at', 'earlier')}).")
        weather = cached_w
    else:
        weather = fetch_openweather(route["lat"], route["lon"], None)

# Congestion hint from historical mean
hist = india_df if city != "Minneapolis (Source / US)" else us_df
p95_scale = float(bundle.get("metrics", {}).get("p95_volume", 5500.0))
hour_mean = hist[hist["date_time"].dt.hour == at.hour]["traffic_volume"].mean()
hint = float(np.clip(hour_mean / p95_scale, 0.05, 0.95))
flow = cached_fetch_flow(route["lat"], route["lon"], route["free_flow"], tom_key or None, hint=hint)

# Safe travel time calculations (division-by-zero safeguarded)
length = max(float(route.get("length_km", 5.0)), 0.5)
ff_safe = max(float(flow.get("free_flow_speed") or 30.0), 10.0)
cur_safe = max(float(flow.get("current_speed") or 10.0), 5.0)
flow["travel_time_min"] = round(length / cur_safe * 60, 1)
flow["free_time_min"] = round(length / ff_safe * 60, 1)
flow["delay_min"] = round(max(flow["travel_time_min"] - flow["free_time_min"], 0.0), 1)

pred = predict_pipeline(at, weather, flow, bundle, minutes_ahead=0, city=city)
reasons = explain_prediction(pred, weather, flow, bundle)

# Future forecasts with exponential horizon decay
horizons = [15, 30, 60]
future_rows = []
for m in horizons:
    fp = predict_pipeline(at, weather, flow, bundle, minutes_ahead=m, city=city)
    future_rows.append({"in_min": m, "time": fp["target_dt"].strftime("%H:%M"), **fp})
future_df = pd.DataFrame(future_rows)

severe_now = pred["final_level"] == "Severe"
severe_soon = bool((future_df["final_level"] == "Severe").any())

# Compute SHAP explanation with pure feature attributions and transparent pipeline steps
shap_exp = compute_shap_explanation(
    at, weather, flow, bundle, horizon_min=horizon,
    pipeline_pred=pred if horizon == 0 else None, city=city
)

# ---------------- Header ----------------
st.title("🚦 Smart City Real-Time Traffic Congestion Forecasting & Adaptation")
st.caption(f"Base model trained on historical patterns ({bundle['metrics']['n_us']:,} rows) → " 
           f"calibrated on small India sample ({bundle['metrics']['n_india_calib']} rows) · "
           f"Thresholds Low<{bundle['thresholds'][0]:.0f} < Moderate<{bundle['thresholds'][1]:.0f} < "
           f"High<{bundle['thresholds'][2]:.0f} < Severe")

safe_city = html.escape(str(city))
safe_route = html.escape(str(route_name))

if severe_now or severe_soon:
    who = "NOW" if severe_now else f"in ~{future_df[future_df.final_level=='Severe'].iloc[0]['in_min']} min"
    st.error(f"🚨 CONGESTION ALERT — Severe congestion detected/predicted {who} on **{safe_route}, {safe_city}**. "
             f"Key driver: **{shap_exp['headline_sentence']}**. Consider delaying departure or choosing an alternate corridor.")
else:
    st.success(f"✅ No severe congestion — {safe_route} is **{pred['final_level']}** "
               f"({pred['final_vol']:,} veh/hr est.), delay +{flow['delay_min']} min. Driver: *{shap_exp['headline_sentence']}*")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📍 Overview", "🔮 Forecast (15/30/60)", "🗺 Heatmap", "📊 Patterns", "🔁 Adaptation", "⚖ Route compare"])

# ---------------- TAB 1 ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Live speed", f"{flow['current_speed']} km/h", f"free {flow['free_flow_speed']} km/h")
    c2.metric("Congestion ratio", f"{flow['congestion_ratio']:.2f}",
              "jammed" if flow["congestion_ratio"] < 0.5 else "flowing")
    c3.metric("Predicted volume", f"{pred['final_vol']:,} veh/hr", pred["final_level"])
    c4.metric("Delay", f"+{flow['delay_min']} min", f"trip {flow['travel_time_min']} min")

    if horizon == 0:
        hpred, hlabel = pred["final_vol"], pred["final_level"]
    else:
        hrow = future_df[future_df["in_min"] == horizon].iloc[0]
        hpred, hlabel = int(hrow["final_vol"]), str(hrow["final_level"])

    # XSS-Safe HTML Rendering
    safe_final_level = html.escape(str(pred['final_level']))
    safe_hlabel = html.escape(str(hlabel))
    color_now = LEVEL_COLOR.get(pred['final_level'], '#6b7280')
    color_h = LEVEL_COLOR.get(hlabel, '#6b7280')

    st.markdown(f"### Predicted congestion: "
                f"<span style='background:{color_now};color:white;"
                f"padding:4px 14px;border-radius:12px'>{safe_final_level}</span>"
                f" &nbsp;·&nbsp; T+{int(horizon)}min: "
                f"<span style='background:{color_h};color:white;"
                f"padding:4px 14px;border-radius:12px'>{safe_hlabel} ({hpred:,} veh/hr)</span>",
                unsafe_allow_html=True)

    st.write(f"**{at.strftime('%A, %d %b %Y %H:%M')}** · {safe_route}, {safe_city} · "
             f"Weather **{weather['weather_main']}**, {weather['temp']-273.15:.1f}°C, "
             f"rain {weather['rain_1h']} mm/h, clouds {weather['clouds_all']}%")
    st.caption(f"Data sources → traffic: `{flow['source']}` · weather: `{weather['source']}`")
    st.info("🧩 **How real-time data contributes:** " + contribution_text(pred["adapted_vol"],
                                                                          flow["congestion_ratio"], weather))

    # ---------------- Interpretability / SHAP Section ----------------
    target_time_str = (at + dt.timedelta(minutes=horizon)).strftime('%H:%M')

    if horizon == 0:
        st.subheader(f"🔍 Why {pred['final_level']} Congestion Now? (SHAP Explainability)")
    else:
        st.subheader(f"🔍 Why {hlabel} Congestion in {horizon} mins ({target_time_str})? (SHAP Explainability)")

    st.info(f"💡 **Key Prediction Drivers:** {shap_exp['headline_sentence']}.\n\n"
            f"*{shap_exp['summary_text']}*")

    if shap_exp.get("top_drivers"):
        m_cols = st.columns(min(3, len(shap_exp["top_drivers"])))
        for i, driver in enumerate(shap_exp["top_drivers"][:3]):
            with m_cols[i]:
                pct_val = driver.get("pct_push_up", 0) if driver["impact"] > 0 else driver.get("pct_pull_down", 0)
                sign = "Uplift" if driver["impact"] > 0 else "Reduction"
                st.metric(
                    label=driver["name"],
                    value=f"{pct_val:.0f}% contribution",
                    delta=f"{driver['impact']:+,.0f} veh/hr ({sign})",
                    delta_color="inverse" if driver["impact"] > 0 else "normal",
                )

    iv_tab1, iv_tab2, iv_tab3, iv_tab4 = st.tabs([
        "📊 Factor Contribution (%)",
        "🌊 Volume Build-Up Waterfall",
        "🍩 Uplift Share Breakdown",
        "🌐 Global Importance",
    ])

    with iv_tab1:
        st.plotly_chart(plot_shap_factor_bars(shap_exp), use_container_width=True, key="iv_factor_bars")
        st.caption("🔴 Positive impact increases congestion volume · 🟢 Negative impact eases traffic flow.")

    with iv_tab2:
        st.plotly_chart(plot_shap_waterfall_chart(shap_exp), use_container_width=True, key="iv_waterfall")
        st.caption(f"Traces the prediction from historical base average ({shap_exp['base_expected_value']:,.0f} veh/hr) to final forecasted volume.")

    with iv_tab3:
        st.plotly_chart(plot_shap_driver_shares(shap_exp), use_container_width=True, key="iv_driver_shares")
        st.caption("Proportional share of factors driving traffic above baseline.")

    with iv_tab4:
        imp = pd.DataFrame({"feature": list(bundle["importances"].keys()),
                            "importance": list(bundle["importances"].values())}).sort_values("importance")
        try:
            import plotly.express as px
            fig_glob = px.bar(imp, x="importance", y="feature", orientation="h",
                              title="Global Permutation Feature Importance (Overall Model)")
            st.plotly_chart(fig_glob, use_container_width=True)
        except Exception:
            st.bar_chart(imp.set_index("feature"))

    with st.expander("📝 Rule-Based Factor Breakdown"):
        for r in reasons:
            st.write("• " + r)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Short-term forecast")
    st.dataframe(future_df[["in_min", "time", "final_vol", "final_level", "adapted_vol", "base_vol"]]
                 .rename(columns={"in_min": "+min", "time": "at", "final_vol": "forecast_vol",
                                   "final_level": "level", "adapted_vol": "adapted_vol",
                                   "base_vol": "base_vol"}),
                 use_container_width=True)
    try:
        import plotly.express as px
        plot_df = pd.DataFrame({
            "horizon": ["now"] + [f"+{m}m" for m in horizons],
            "final": [pred["final_vol"]] + future_df["final_vol"].tolist(),
            "adapted": [pred["adapted_vol"]] + future_df["adapted_vol"].tolist(),
            "base": [pred["base_vol"]] + future_df["base_vol"].tolist(),
        })
        fig2 = px.line(plot_df, x="horizon", y=["final", "adapted", "base"],
                       markers=True, title="How congestion is expected to change")
        st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.line_chart(future_df.set_index("in_min")[["final_vol"]])
    st.caption("Forecast method: future time-features (hour/day) + persistent weather + "
               "live-speed fusion (65% adapted history + 35% live-implied volume).")

    st.divider()
    st.subheader("🔍 Explain Forecast Horizons (SHAP Values)")
    chosen_h = st.radio("Select forecast horizon to explain:", horizons, horizontal=True,
                        format_func=lambda m: f"T+{m} min ({future_df[future_df['in_min']==m].iloc[0]['time']})",
                        index=1, key="tab2_horizon_choice")
    h_row = future_df[future_df["in_min"] == chosen_h].iloc[0]
    h_shap = compute_shap_explanation(at, weather, flow, bundle, horizon_min=chosen_h, city=city)

    st.markdown(f"#### **{h_row['final_level']} Congestion** predicted in **{chosen_h} mins**: *{h_shap['headline_sentence']}*")
    st.info(h_shap["summary_text"])
    fc_col1, fc_col2 = st.columns([3, 2])
    with fc_col1:
        st.plotly_chart(plot_shap_factor_bars(h_shap), use_container_width=True, key="fc_factor_bars")
    with fc_col2:
        st.plotly_chart(plot_shap_driver_shares(h_shap), use_container_width=True, key="fc_driver_shares")

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader(f"Congestion heatmap — {safe_city}")
    rows = []
    for r in routes:
        w = weather
        fl = cached_fetch_flow(r["lat"], r["lon"], r["free_flow"], tom_key or None, hint)
        p = predict_pipeline(at, w, fl, bundle, city=city)
        rows.append({"route": r["name"], "lat": r["lat"], "lon": r["lon"],
                     "level": p["final_level"], "vol": p["final_vol"],
                     "speed": fl["current_speed"], "color": LEVEL_COLOR[p["final_level"]]})
    hmap = pd.DataFrame(rows)
    st.dataframe(hmap[["route", "level", "vol", "speed"]], use_container_width=True)
    try:
        import plotly.express as px
        figm = px.scatter_map(hmap, lat="lat", lon="lon", color="level",
                              color_discrete_map=LEVEL_COLOR, size="vol",
                              hover_name="route", hover_data={"vol": True, "speed": True, "lat": False, "lon": False},
                              zoom=11, height=520, title=f"Current & predicted congestion — {safe_city}")
        figm.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(figm, use_container_width=True)
    except Exception as e:
        st.warning(f"Interactive tiles unavailable ({e}); fallback map below.")
        st.map(hmap.rename(columns={"lat": "lat", "lon": "lon"}))
    st.caption("🟢 Low · 🟡 Moderate · 🟠 High · 🔴 Severe — circle size ∝ predicted volume.")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Historical patterns & trends (what the model learned)")
    st.caption("The base XGBoost model learned recurrent traffic behaviors from Interstate 94 historical data. "
               "Below is the empirical distribution of that training corpus alongside the simulated Indian corridor benchmark.")

    # Explicit dataset separation (W3, W4)
    hourly_us = us_df.copy()
    hourly_us["hour"] = hourly_us["date_time"].dt.hour
    hourly_us["dow"] = hourly_us["date_time"].dt.dayofweek

    is_indian = (city != "Minneapolis (Source / US)")
    if is_indian:
        pat_view = st.radio("Select dataset view:",
                            ["🇺🇸 Base Training Set (US Interstate-94)",
                             "🇮🇳 Target Benchmark (Synthetic Indian Corridor Shift)"],
                            horizontal=True)
        if "Synthetic" in pat_view:
            hourly = india_df.copy()
            hourly["hour"] = hourly["date_time"].dt.hour
            hourly["dow"] = hourly["date_time"].dt.dayofweek
            st.info("⚠️ **Simulated Indian Benchmark Note:** This synthetic dataset models Indian urban dynamics "
                    "(later evening rush hour, monsoon downpours) to validate cross-context Ridge transfer learning.")
        else:
            hourly = hourly_us
    else:
        hourly = hourly_us

    g_hour = hourly.groupby("hour")["traffic_volume"].mean()
    g_dow = hourly.groupby("dow")["traffic_volume"].mean()
    try:
        import plotly.express as px
        st.plotly_chart(px.line(x=g_hour.index, y=g_hour.values, markers=True,
                                labels={"x": "hour", "y": "avg veh/hr"},
                                title="Average volume by hour (peak-hour structure)"),
                        use_container_width=True)
        st.plotly_chart(px.bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                               y=[g_dow.get(i, 0) for i in range(7)],
                               labels={"x": "day", "y": "avg veh/hr"},
                               title="Weekday vs weekend behaviour"), use_container_width=True)
        piv = hourly.pivot_table(index="dow", columns="hour", values="traffic_volume", aggfunc="mean")
        st.plotly_chart(px.imshow(piv, labels=dict(x="hour", y="day (0=Mon)", color="veh/hr"),
                                  title="Heatmap: hour × day congestion pattern",
                                  aspect="auto"), use_container_width=True)
        wimp = hourly.groupby("weather_main")["traffic_volume"].mean().sort_values()
        st.plotly_chart(px.bar(x=wimp.index, y=wimp.values, labels={"x": "weather", "y": "avg veh/hr"},
                               title="Weather vs traffic volume"), use_container_width=True)
    except Exception:
        st.line_chart(g_hour)
        st.bar_chart(g_dow)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("AM peak (7–10h)", f"{g_hour.loc[7:10].mean():,.0f} veh/hr" if len(g_hour.loc[7:10]) else "N/A")
        st.metric("PM peak (17–21h)", f"{g_hour.loc[17:21].mean():,.0f} veh/hr" if len(g_hour.loc[17:21]) else "N/A")
    with c2:
        st.metric("Weekday avg", f"{g_dow.loc[0:4].mean():,.0f} veh/hr" if len(g_dow.loc[0:4]) else "N/A")
        st.metric("Weekend avg", f"{g_dow.loc[5:6].mean():,.0f} veh/hr" if len(g_dow.loc[5:6]) else "N/A")
    st.info("📌 **Pattern-based insight:** the model exploits the double rush-hour peaks, "
            "the weekday/weekend gap, holiday drops and rain penalties visible above.")

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("Cross-context adaptation: historical base → live corridors (transfer learning)")
    m = bundle["metrics"]
    st.write(f"**Base model** (XGBoost, chronological out-of-time test): "
             f"MAE **{m['base_mae']:.1f}**, RMSE **{m['base_rmse']:.1f}**.")
    st.write(f"**Without adaptation** on Indian corridor validation set: MAE **{m['calib_mae_unadapted']:.1f}** → "
             f"**with Ridge calibration: {m['calib_mae_adapted']:.1f}** "
             f"({(1-m['calib_mae_adapted']/max(m['calib_mae_unadapted'],1))*100:.1f}% held-out error drop).")
    st.caption(f"Evaluated on held-out 30% chronological validation split ({m.get('n_india_val', 101)} rows) — zero in-sample contamination.")
    comp = pd.DataFrame({
        "stage": ["Base", "Adapted (live corridors)", "Final (adapted + live fusion)"],
        "volume": [pred["base_vol"], pred["adapted_vol"], pred["final_vol"]],
        "level": [pred["base_level"], pred["adapted_level"], pred["final_level"]],
    })
    st.dataframe(comp, use_container_width=True)
    delta = pred["adapted_vol"] - pred["base_vol"]
    st.info(f"🔧 Calibration adjusted the US prediction by **{delta:+,} veh/hr** "
            f"({pred['base_level']} → {pred['adapted_level']}) using transfer Ridge stacking.")
    st.subheader("🇮🇳 Indian context calibration (live collector)")
    _csv = None
    for _p in (os.path.join(os.path.dirname(__file__), "data", "indian_realtime.csv"),
               os.path.join(os.path.dirname(__file__), "indian_realtime.csv"),
               "data/indian_realtime.csv", "indian_realtime.csv"):
        if os.path.exists(_p) and os.path.getsize(_p) > 0:
            _csv = _p
            break
    if _csv:
        try:
            ind = pd.read_csv(_csv)
            if "traffic_source" in ind.columns:
                real_ind = ind[ind["traffic_source"].str.startswith("live:", na=False)]
                sim_count = len(ind) - len(real_ind)
            else:
                real_ind = ind
                sim_count = 0
            st.write(f"Logged **{len(real_ind)}** genuine live sensor observations "
                     f"(omitted {sim_count} simulated fallback rows) across "
                     f"**{real_ind['location'].nunique() if len(real_ind)>0 else 0}** locations (`{_csv}`).")
            if len(real_ind) > 0:
                st.line_chart(real_ind, x="timestamp", y="congestion_ratio", color="location")
            else:
                st.info("Collector CSV contains simulated/initialization rows. Run collector.py with a live TomTom key to chart live sensor data.")
        except Exception as e:
            st.warning(f"Could not read collector CSV ({e}).")
    else:
        st.warning("No collector data yet — run `python collector.py` in a second terminal.")
    st.caption("Transfer learning Ridge calibration is trained on a synthetic Indian corridor benchmark to evaluate domain shift adaptation under controlled conditions. Ground-truth production validation uses genuine live TomTom sensor observations continuously gathered via collector.py.")

# ---------------- TAB 6 ----------------
with tab6:
    st.subheader("Route comparison — pick the faster corridor")
    col_a, col_b = st.columns(2)
    with col_a:
        a_name = st.selectbox("Route A", [r["name"] for r in routes], index=0, key=f"ra_{city}")
    with col_b:
        b_names = [r["name"] for r in routes]
        b_name = st.selectbox("Route B", b_names, index=min(1, len(b_names) - 1), key=f"rb_{city}")

    if a_name == b_name:
        st.warning("⚠️ Please select two different routes to compare corridors.")
    else:
        ra = next((r for r in routes if r["name"] == a_name), routes[0] if routes else None)
        rb = next((r for r in routes if r["name"] == b_name), (routes[1] if len(routes) > 1 else routes[0]) if routes else None)
        if not ra or not rb:
            st.info("Please select valid routes.")
        else:
            results = []
            for r in (ra, rb):
                fl = cached_fetch_flow(r["lat"], r["lon"], r["free_flow"], tom_key or None, hint)
                p = predict_pipeline(at, weather, fl, bundle, city=city)
                len_km = max(float(r.get("length_km", 5.0)), 0.5)
                cur_spd = max(float(fl.get("current_speed") or 10.0), 5.0)
                ff_spd = max(float(fl.get("free_flow_speed") or 30.0), 10.0)
                tmin = round(len_km / cur_spd * 60, 1)
                free_min = round(len_km / ff_spd * 60, 1)
                results.append({
                    "route": r["name"], "level": p["final_level"], "vol": p["final_vol"],
                    "speed": fl["current_speed"], "trip_min": tmin, "delay": round(max(tmin - free_min, 0.0), 1)
                })
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            best = res_df.loc[res_df["trip_min"].idxmin()]
            st.success(f"✅ Recommendation: **{html.escape(str(best['route']))}** — {best['trip_min']} min, "
                       f"{best['level']} congestion at {best['speed']} km/h.")
            try:
                import plotly.express as px
                st.plotly_chart(px.bar(res_df, x="route", y="trip_min", color="level",
                                       color_discrete_map=LEVEL_COLOR, title="Trip-time comparison"),
                                use_container_width=True)
            except Exception:
                st.bar_chart(res_df.set_index("route")["trip_min"])

st.divider()
st.caption("Hardened prototype: XGBoost + Ridge calibration · TomTom + OpenWeather / Open-Meteo · Canonical schema.")
