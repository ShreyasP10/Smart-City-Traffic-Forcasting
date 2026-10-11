"""Smart City Real-Time Traffic Congestion Forecasting & Adaptation — Streamlit dashboard.

Covers all PS requirements:
  current + predicted + 15/30/60-min forecast | historical patterns |
  factors influencing prediction | cross-context adaptation (US->India) |
  heatmap | alerts | route comparison | weather integration.
Runs offline (simulated live data) or live with TomTom + OpenWeather keys.
"""
from __future__ import annotations

import datetime as dt
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from city_config import CITIES
from traffic_api import fetch_openweather, fetch_tomtom_flow, contribution_text
from data_generator import load_historical
from explain_shap import (
    compute_shap_explanation,
    plot_shap_waterfall_chart,
    plot_shap_factor_bars,
    plot_shap_driver_shares,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib")

LEVEL_COLOR = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}
LEVEL_ORDER = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}

st.set_page_config(page_title="Smart City Traffic Forecaster", page_icon="🚦", layout="wide")


@st.cache_resource
def load_bundle():
    if not os.path.exists(MODEL_PATH):
        from train_model import main as train_main
        train_main()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    us = load_historical(indian_style=False)
    india = load_historical(indian_style=True)
    return us, india


def classify(v, thresholds):
    q25, q50, q75 = thresholds
    if v < q25:
        return "Low"
    if v < q50:
        return "Moderate"
    if v < q75:
        return "High"
    return "Severe"


def featurize(at: dt.datetime, weather: dict, bundle) -> np.ndarray:
    wc = bundle["weather_code"].get(weather.get("weather_main", "Clear"), 1)
    hour = at.hour
    dow = at.weekday()
    row = {
        "hour": hour, "day_of_week": dow, "month": at.month,
        "is_weekend": int(dow >= 5),
        "is_holiday": int((at.month, at.day) in {(1, 26), (8, 15), (10, 2), (12, 25), (5, 1), (1, 1), (7, 4)}),
        "temp": float(weather.get("temp", 300)),
        "rain_1h": float(weather.get("rain_1h", 0)),
        "snow_1h": 0.0,
        "clouds_all": int(weather.get("clouds_all", 20)),
        "weather_code": int(wc),
        "is_peak": int(hour in (7, 8, 9, 10, 17, 18, 19, 20, 21)),
        "is_night": int(hour < 5 or hour >= 23),
    }
    return np.array([[row[c] for c in bundle["features"]]]), row


def predict_pipeline(at, weather, flow, bundle):
    """Base -> adapted (transfer) -> live-fused final."""
    X, row = featurize(at, weather, bundle)
    base_vol = float(bundle["model"].predict(X)[0])
    cal_X = np.array([[base_vol, row["rain_1h"], row["hour"], row["is_peak"]]])
    adapted_vol = float(bundle["calibration"].predict(cal_X)[0])
    ratio = float(flow.get("congestion_ratio", 0.7))
    implied_vol = (1.0 - ratio) * 5500  # volume implied by live speed
    final_vol = 0.65 * adapted_vol + 0.35 * implied_vol
    th = bundle["thresholds"]
    return {
        "base_vol": int(base_vol), "base_level": classify(base_vol, th),
        "adapted_vol": int(adapted_vol), "adapted_level": classify(adapted_vol, th),
        "final_vol": int(final_vol), "final_level": classify(final_vol, th),
        "implied_vol": int(implied_vol), "features": row,
    }


def explain_prediction(pred, weather, flow, bundle) -> list[str]:
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
    # sort by absolute impact
    out.sort(key=lambda x: -abs(x[1]))
    return [t for t, _ in out]


# ---------------- Sidebar ----------------
bundle = load_bundle()
us_df, india_df = load_data()

st.sidebar.title("🚦 Controls")
city = st.sidebar.selectbox("City (context)", list(CITIES.keys()), index=0)
routes = CITIES[city]["routes"]
route_names = [r["name"] for r in routes]
route_name = st.sidebar.selectbox("Route / location", route_names)
route = next(r for r in routes if r["name"] == route_name)

with st.sidebar.expander("➕ Custom location (optional)"):
    c_lat = st.number_input("Latitude", value=float(route["lat"]), format="%.5f")
    c_lon = st.number_input("Longitude", value=float(route["lon"]), format="%.5f")
    c_ff = st.number_input("Free-flow speed (km/h)", value=int(route["free_flow"]), min_value=20, max_value=120)
    use_custom = st.checkbox("Use custom location", value=False)
if use_custom:
    route = {"name": "Custom point", "lat": c_lat, "lon": c_lon, "free_flow": c_ff, "length_km": 5.0}

st.sidebar.subheader("⏱ Time")
use_now = st.checkbox("Use current time", value=True)
if use_now:
    at = dt.datetime.now().replace(second=0, microsecond=0)
else:
    d = st.sidebar.date_input("Date", value=dt.date.today())
    h = st.sidebar.slider("Hour", 0, 23, 18)
    at = dt.datetime.combine(d, dt.time(h, 0))
horizon = st.sidebar.select_slider("Forecast horizon (min)", [0, 15, 30, 60], value=30)

st.sidebar.subheader("🔑 Live APIs (optional)")
def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))  # .streamlit/secrets.toml
    except Exception:
        return os.environ.get(name, "")
tom_default = _secret("TOMTOM_API_KEY")
owm_default = _secret("OPENWEATHER_API_KEY")
tom_key = st.sidebar.text_input("TomTom KEY VALUE", value=tom_default, type="password",
                                help="Pre-filled from .streamlit/secrets.toml. Paste the KEY VALUE (copy/eye icon), NOT the Key ID.")
owm_key = st.sidebar.text_input("OpenWeather API key", value=owm_default, type="password",
                                help="New keys need 30-120 min activation; Open-Meteo live fallback covers you meanwhile.")
st.sidebar.caption("TomTom live ✅ · Weather: OpenWeather → Open-Meteo live fallback → simulation. "
                   "Domain whitelist OFF is correct for localhost.")

# ---------------- Live data (cached fallback: demo never dies on network) ----------------
import json as _json
CACHE_FILE = os.path.join(os.path.dirname(__file__), "last_response.json")
try:
    weather = fetch_openweather(route["lat"], route["lon"], owm_key or None)
    if "live:" in str(weather.get("source", "")):
        try:
            with open(CACHE_FILE, "w") as _cf:
                _cf.write(_json.dumps({"weather": weather,
                                       "cached_at": dt.datetime.now().isoformat(timespec="seconds")}, default=str))
        except Exception:
            pass
except Exception as _e:
    try:
        _cached = _json.load(open(CACHE_FILE))
        weather = _cached.get("weather", _cached.get("live", {}).get("weather", {})) or _cached["live"]
        st.warning(f"Live weather unavailable ({_e}) — showing last cached data "
                   f"(cached at {_cached.get('cached_at', '?')}).")
    except Exception:
        weather = fetch_openweather(route["lat"], route["lon"], None)
# First pass congestion hint from historical mean for that hour
hist = india_df if city != "Minneapolis (Source / US)" else us_df
hour_mean = hist[hist["date_time"].dt.hour == at.hour]["traffic_volume"].mean()
hint = float(np.clip(hour_mean / 5500, 0.05, 0.95))
flow = fetch_tomtom_flow(route["lat"], route["lon"], route["free_flow"], tom_key or None,
                         congestion_hint=hint)
# adjust travel time
length = route.get("length_km", 5.0)
flow["travel_time_min"] = round(length / max(flow["current_speed"], 1) * 60, 1)
flow["free_time_min"] = round(length / max(flow["free_flow_speed"], 1) * 60, 1)
flow["delay_min"] = round(flow["travel_time_min"] - flow["free_time_min"], 1)

pred = predict_pipeline(at, weather, flow, bundle)
reasons = explain_prediction(pred, weather, flow, bundle)

# Future forecasts
horizons = [15, 30, 60]
future_rows = []
for m in horizons:
    fat = at + dt.timedelta(minutes=m)
    fp = predict_pipeline(fat, weather, flow, bundle)
    future_rows.append({"in_min": m, "time": fat.strftime("%H:%M"), **fp})
future_df = pd.DataFrame(future_rows)

severe_now = pred["final_level"] == "Severe"
severe_soon = bool((future_df["final_level"] == "Severe").any())

# ---------------- Header ----------------
st.title("🚦 Smart City Real-Time Traffic Congestion Forecasting & Adaptation")
st.caption(f"Base model trained on US history ({bundle['metrics']['n_us']:,} rows) → "
           f"calibrated on small India sample ({bundle['metrics']['n_india_calib']} rows) · "
           f"Thresholds Low<{bundle['thresholds'][0]:.0f} < Moderate<{bundle['thresholds'][1]:.0f} < "
           f"High<{bundle['thresholds'][2]:.0f} < Severe")

if severe_now or severe_soon:
    who = "NOW" if severe_now else f"in ~{future_df[future_df.final_level=='Severe'].iloc[0]['in_min']} min"
    st.error(f"🚨 CONGESTION ALERT — Severe congestion detected/predicted {who} on **{route_name}, {city}**. "
             f"Consider delaying departure or an alternate route. ({'; '.join(reasons[:2])})")
else:
    st.success(f"✅ No severe congestion — {route_name} is **{pred['final_level']}** "
               f"({pred['final_vol']:,} veh/hr est.), delay +{flow['delay_min']} min.")

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
    st.markdown(f"### Predicted congestion: "
                f"<span style='background:{LEVEL_COLOR[pred['final_level']]};color:white;"
                f"padding:4px 14px;border-radius:12px'>{pred['final_level']}</span>"
                f" &nbsp;·&nbsp; T+{horizon}min: "
                f"<span style='background:{LEVEL_COLOR[hlabel]};color:white;"
                f"padding:4px 14px;border-radius:12px'>{hlabel} ({hpred:,} veh/hr)</span>",
                unsafe_allow_html=True)
    st.write(f"**{at.strftime('%A, %d %b %Y %H:%M')}** · {route_name}, {city} · "
             f"Weather **{weather['weather_main']}**, {weather['temp']-273.15:.1f}°C, "
             f"rain {weather['rain_1h']} mm/h, clouds {weather['clouds_all']}%")
    st.caption(f"Data sources → traffic: `{flow['source']}` · weather: `{weather['source']}`")
    st.info("🧩 **How real-time data contributes:** " + contribution_text(pred["adapted_vol"],
                                                                          flow["congestion_ratio"], weather))
    # ---------------- Interpretability / SHAP Section ----------------
    shap_exp = compute_shap_explanation(at, weather, flow, bundle, horizon_min=horizon)
    target_time_str = (at + dt.timedelta(minutes=horizon)).strftime('%H:%M')

    if horizon == 0:
        st.subheader(f"🔍 Why {pred['final_level']} Congestion Now? (SHAP Explainability)")
    else:
        st.subheader(f"🔍 Why {hlabel} Congestion in {horizon} mins ({target_time_str})? (SHAP Explainability)")

    # Prominent callout with exact percentage contribution
    st.info(f"💡 **Key Prediction Drivers:** {shap_exp['headline_sentence']}.\n\n"
            f"*{shap_exp['summary_text']}*")

    # Display Top 3 Driver Metric Cards
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

    # Multi-tab visual interpretability
    iv_tab1, iv_tab2, iv_tab3, iv_tab4 = st.tabs([
        "📊 Factor Contribution (%)",
        "🌊 Volume Build-Up Waterfall",
        "🍩 Uplift Share Breakdown",
        "🌐 Global Importance",
    ])

    with iv_tab1:
        st.plotly_chart(plot_shap_factor_bars(shap_exp), use_container_width=True)
        st.caption("🔴 Positive impact increases congestion volume · 🟢 Negative impact eases traffic flow.")

    with iv_tab2:
        st.plotly_chart(plot_shap_waterfall_chart(shap_exp), use_container_width=True)
        st.caption(f"Traces the prediction from historical base average ({shap_exp['base_expected_value']:,.0f} veh/hr) to final forecasted volume.")

    with iv_tab3:
        st.plotly_chart(plot_shap_driver_shares(shap_exp), use_container_width=True)
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
        for lvl, col in LEVEL_COLOR.items():
            pass
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
    h_shap = compute_shap_explanation(at, weather, flow, bundle, horizon_min=chosen_h)

    st.markdown(f"#### **{h_row['final_level']} Congestion** predicted in **{chosen_h} mins**: *{h_shap['headline_sentence']}*")
    st.info(h_shap["summary_text"])
    fc_col1, fc_col2 = st.columns([3, 2])
    with fc_col1:
        st.plotly_chart(plot_shap_factor_bars(h_shap), use_container_width=True)
    with fc_col2:
        st.plotly_chart(plot_shap_driver_shares(h_shap), use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader(f"Congestion heatmap — {city}")
    rows = []
    for r in routes:
        w = weather  # same city weather for demo speed; per-route live speed simulated
        fl = fetch_tomtom_flow(r["lat"], r["lon"], r["free_flow"], tom_key or None, congestion_hint=hint)
        p = predict_pipeline(at, w, fl, bundle)
        rows.append({"route": r["name"], "lat": r["lat"], "lon": r["lon"],
                     "level": p["final_level"], "vol": p["final_vol"],
                     "speed": fl["current_speed"], "color": LEVEL_COLOR[p["final_level"]]})
    hmap = pd.DataFrame(rows)
    st.dataframe(hmap[["route", "level", "vol", "speed"]], use_container_width=True)
    # Map: plotly scatter_map (no token, open-street-map) + st.map fallback
    try:
        import plotly.express as px
        figm = px.scatter_map(hmap, lat="lat", lon="lon", color="level",
                              color_discrete_map=LEVEL_COLOR, size="vol",
                              hover_name="route", hover_data={"vol": True, "speed": True, "lat": False, "lon": False},
                              zoom=11, height=520, title=f"Current & predicted congestion — {city}")
        figm.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(figm, use_container_width=True)
    except Exception as e:
        st.warning(f"Interactive tiles unavailable ({e}); fallback map below.")
        st.map(hmap.rename(columns={"lat": "lat", "lon": "lon"}))
    st.caption("🟢 Low · 🟡 Moderate · 🟠 High · 🔴 Severe — circle size ∝ predicted volume.")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Historical patterns & trends (what the model learned)")
    hourly = hist.copy()
    hourly["hour"] = hourly["date_time"].dt.hour
    hourly["dow"] = hourly["date_time"].dt.dayofweek
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
        st.metric("AM peak (7–10h)", f"{g_hour.loc[7:10].mean():,.0f} veh/hr")
        st.metric("PM peak (17–21h)", f"{g_hour.loc[17:21].mean():,.0f} veh/hr")
    with c2:
        st.metric("Weekday avg", f"{g_dow.loc[0:4].mean():,.0f} veh/hr")
        st.metric("Weekend avg", f"{g_dow.loc[5:6].mean():,.0f} veh/hr")
    st.info("📌 **Pattern-based insight:** the model exploits the double rush-hour peaks, "
            "the weekday/weekend gap, holiday drops and rain penalties visible above. "
            "Indian calibration shifts peaks ~1h later and raises the overall scale.")

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("Cross-context adaptation: US base → Indian city (transfer learning)")
    m = bundle["metrics"]
    st.write(f"**Base model** (XGBoost, trained on US history): "
             f"MAE **{m['base_mae']:.1f}**, RMSE **{m['base_rmse']:.1f}**.")
    st.write(f"**Without adaptation** on India sample: MAE **{m['calib_mae_unadapted']:.1f}** → "
             f"**with Ridge calibration: {m['calib_mae_adapted']:.1f}** "
             f"({(1-m['calib_mae_adapted']/max(m['calib_mae_unadapted'],1))*100:.1f}% error drop).")
    comp = pd.DataFrame({
        "stage": ["Base (US)", "Adapted (India-calibrated)", "Final (adapted + live fusion)"],
        "volume": [pred["base_vol"], pred["adapted_vol"], pred["final_vol"]],
        "level": [pred["base_level"], pred["adapted_level"], pred["final_level"]],
    })
    st.dataframe(comp, use_container_width=True)
    delta = pred["adapted_vol"] - pred["base_vol"]
    st.info(f"🔧 Calibration adjusted the US prediction by **{delta:+,} veh/hr** "
            f"({pred['base_level']} → {pred['adapted_level']}) using live-calibrated Ridge stacking on "
            f"[base_pred, rain, hour, is_peak]. Live-speed fusion then moved it to "
            f"**{pred['final_vol']:,} ({pred['final_level']})**.")
    st.caption("Technique: two-stage fine-tuning / stacking — universal temporal patterns stay frozen in "
               "the base model; a tiny Ridge head corrects scale & monsoon/peak-shift bias for the new city. "
               "Needs only ~2 weeks of API polling to fit.")
    # Priority 3 — live collector proof (Hour 0 Track C): indian_realtime.csv
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
            st.write(f"Collected **{len(ind)}** real-time samples across "
                     f"**{ind['location'].nunique()}** locations (`{_csv}`).")
            st.line_chart(ind, x="timestamp", y="congestion_ratio", color="location")
            st.caption("Online calibration against live TomTom data: the Ridge head + live-speed fusion "
                       "(see adjustment above) is the adaptation mechanism — this growing CSV is its evidence, "
                       "even with limited rows.")
        except Exception as e:
            st.warning(f"Could not read collector CSV ({e}). Run `python collector.py` in terminal 2.")
    else:
        st.warning("No collector data yet — run `python collector.py` in a second terminal "
                   "(15-min cadence → ~20 rows/location by hour 5).")

# ---------------- TAB 6 ----------------
with tab6:
    st.subheader("Route comparison — pick the faster corridor")
    col_a, col_b = st.columns(2)
    with col_a:
        a_name = st.selectbox("Route A", route_names, index=0, key="ra")
    with col_b:
        b_name = st.selectbox("Route B", route_names, index=min(1, len(route_names)-1), key="rb")
    ra = next(r for r in routes if r["name"] == a_name)
    rb = next(r for r in routes if r["name"] == b_name)
    results = []
    for r in (ra, rb):
        fl = fetch_tomtom_flow(r["lat"], r["lon"], r["free_flow"], tom_key or None, congestion_hint=hint)
        p = predict_pipeline(at, weather, fl, bundle)
        length_r = r.get("length_km", 5.0)
        tmin = round(length_r / max(fl["current_speed"], 1) * 60, 1)
        results.append({"route": r["name"], "level": p["final_level"], "vol": p["final_vol"],
                        "speed": fl["current_speed"], "trip_min": tmin, "delay": round(tmin - length_r / max(fl["free_flow_speed"], 1) * 60, 1)})
    res_df = pd.DataFrame(results)
    st.dataframe(res_df, use_container_width=True)
    best = res_df.loc[res_df["trip_min"].idxmin()]
    st.success(f"✅ Recommendation: **{best['route']}** — {best['trip_min']} min, "
               f"{best['level']} congestion at {best['speed']} km/h. "
               f"Alternative is {res_df['trip_min'].max():.1f} min.")
    try:
        import plotly.express as px
        st.plotly_chart(px.bar(res_df, x="route", y="trip_min", color="level",
                               color_discrete_map=LEVEL_COLOR, title="Trip-time comparison"),
                        use_container_width=True)
    except Exception:
        st.bar_chart(res_df.set_index("route")["trip_min"])

st.divider()
st.caption("Prototype: XGBoost-class gradient boosting + Ridge calibration · "
           "TomTom Traffic Flow + OpenWeather One Call 3.0 (simulated fallback) · "
           "UCI Metro Interstate schema · Built for TE/BE Smart City PS.")
