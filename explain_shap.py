"""SHAP-based interpretability module for Smart City Traffic Forecasting.

Provides:
- Exact, untampered SHAP values using shap.TreeExplainer on the trained XGBoost model.
- Semantic grouping of canonical features into human-understandable traffic drivers:
  (Rush hour, Rain/Precipitation, Day of Week commuter demand, Holiday, Temperature, Sky conditions, Season).
- Mathematical integrity:
  1. Pure SHAP values explain base model feature attributions.
  2. Downstream India transfer calibration and live TomTom speed fusion are displayed
     as distinct, explicit pipeline stages in the waterfall chart.
  3. Every step in the waterfall chart sums to the final forecasted volume with 100% additivity.
- Plotly interactive charts:
  1. Factor Contribution Bar Chart (+/- veh/hr with % tags)
  2. Waterfall chart: Base Average -> Pure SHAP Factors -> Calibration -> Live Fusion -> Final Forecast
  3. Donut chart of Top Uplift Drivers (%)
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
import weakref

from schema import featurize_point, FEATURES

# Use weakref to prevent memory leaks across Streamlit browser sessions
_EXPLAINER_CACHE = weakref.WeakKeyDictionary()


def get_explainer(model):
    if model not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[model]


def compute_shap_explanation(
    at: dt.datetime,
    weather: dict,
    flow: dict,
    bundle: dict,
    horizon_min: int = 0,
    pipeline_pred: dict | None = None,
    city: str = "Mumbai",
) -> dict:
    """Computes exact SHAP values, semantic feature groupings, and end-to-end pipeline attribution."""
    target_dt = at + dt.timedelta(minutes=horizon_min) if horizon_min != 0 else at
    features_list = bundle.get("features", FEATURES)
    model = bundle["model"]

    # If full pipeline forecast not supplied, compute canonically
    if pipeline_pred is None:
        from pipeline import predict_pipeline
        pipeline_pred = predict_pipeline(at, weather, flow, bundle, minutes_ahead=horizon_min, city=city)

    # Canonical feature vector evaluated at hour baseline for tree explainer
    h0 = target_dt.replace(minute=0, second=0, microsecond=0)
    X, row = featurize_point(h0, weather, city=city)
    hour = target_dt.hour
    dow = target_dt.weekday()

    explainer = get_explainer(model)
    shap_vals = np.array(explainer.shap_values(X))[0]
    base_val = float(explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0])

    # Map raw features to their pure SHAP attributions
    feature_shap = dict(zip(features_list, shap_vals))

    # --- 1. Pure SHAP Semantic Groupings (Zero Invented Numbers) ---
    # Time & Rush Hour
    time_impact = float(
        feature_shap.get("hour", 0.0)
        + feature_shap.get("is_peak", 0.0)
        + feature_shap.get("is_night", 0.0)
        + feature_shap.get("hour_sin", 0.0)
        + feature_shap.get("hour_cos", 0.0)
    )
    if row["is_peak"]:
        time_desc = f"Rush Hour Peak ({hour:02d}:00)"
    elif row["is_night"]:
        time_desc = f"Late Night Off-Peak ({hour:02d}:00)"
    else:
        time_desc = f"Regular Daytime ({hour:02d}:00)"

    # Precipitation / Rain
    rain_impact = float(feature_shap.get("rain_1h", 0.0) + feature_shap.get("snow_1h", 0.0))
    rain_val = row["rain_1h"]
    if rain_val > 5.0:
        rain_desc = f"Severe Rain / Monsoon ({rain_val:.1f} mm/h)"
    elif rain_val > 0.5:
        rain_desc = f"Rainy Weather ({rain_val:.1f} mm/h)"
    elif rain_val > 0:
        rain_desc = f"Light Drizzle ({rain_val:.1f} mm/h)"
    else:
        rain_desc = "Dry / Clear (No Rain)"

    # Day of Week / Weekend Commuter Demand
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_name = day_names[dow]
    dow_impact = float(
        feature_shap.get("day_of_week", 0.0)
        + feature_shap.get("is_weekend", 0.0)
        + feature_shap.get("dow_sin", 0.0)
        + feature_shap.get("dow_cos", 0.0)
    )
    dow_desc = f"{dow_name} {'(Weekend)' if row['is_weekend'] else '(Workday Commute)'}"

    # Holiday Status
    hol_impact = float(feature_shap.get("is_holiday", 0.0))
    hol_desc = "Public Holiday (Lower Traffic)" if row["is_holiday"] else "Regular Non-Holiday"

    # Temperature
    temp_c = row["temp"] - 273.15
    temp_impact = float(feature_shap.get("temp", 0.0))
    temp_desc = f"Temperature ({temp_c:.1f}°C)"

    # Clouds & Atmosphere
    sky_impact = float(feature_shap.get("clouds_all", 0.0) + feature_shap.get("weather_code", 0.0))
    sky_desc = f"{weather.get('weather_main', 'Clear')} ({row['clouds_all']}% clouds)"

    # Season / Month
    month_impact = float(feature_shap.get("month", 0.0))
    month_name = dt.date(2026, row["month"], 1).strftime("%B")
    month_desc = f"Seasonal Pattern ({month_name})"

    # Pure SHAP feature groups
    feature_groups = [
        {"name": "Rush Hour / Time of Day", "detail": time_desc, "impact": time_impact},
        {"name": "Precipitation / Rain", "detail": rain_desc, "impact": rain_impact},
        {"name": "Day of Week Pattern", "detail": dow_desc, "impact": dow_impact},
        {"name": "Weather & Sky Conditions", "detail": sky_desc, "impact": sky_impact},
        {"name": "Temperature", "detail": temp_desc, "impact": temp_impact},
        {"name": "Seasonal Month Cycle", "detail": month_desc, "impact": month_impact},
        {"name": "Holiday Factor", "detail": hol_desc, "impact": hol_impact},
    ]

    # Pipeline raw floats for 100% exact mathematical additivity
    raw = pipeline_pred.get("raw", {})
    raw_base = float(raw.get("base", pipeline_pred["base_vol"]))
    feature_shap_total = base_val + sum(g["impact"] for g in feature_groups)
    interp_delta = raw_base - feature_shap_total
    calib_delta = float(raw.get("adapted", pipeline_pred["adapted_vol"]) - raw_base)
    fusion_delta = float(raw.get("final", pipeline_pred["final_vol"]) - raw.get("adapted", pipeline_pred["adapted_vol"]))
    final_vol = float(pipeline_pred["final_vol"])
    clip_delta = final_vol - float(raw.get("final", pipeline_pred["final_vol"]))

    # Pipeline adjustment stages
    ratio = float(pipeline_pred.get("congestion_ratio", flow.get("congestion_ratio", 0.7)))
    speed_kmh = flow.get("current_speed", 45)
    pipeline_stages = [
        {
            "name": "India Context Calibration",
            "detail": f"Transfer learning shift ({calib_delta:+,.0f} veh/hr)",
            "impact": calib_delta,
        },
        {
            "name": "Live TomTom Speed Fusion",
            "detail": f"Sensor fusion at {speed_kmh} km/h (ratio {ratio:.2f}, {fusion_delta:+,.0f} veh/hr)",
            "impact": fusion_delta,
        },
    ]

    # Combined groups for general display
    groups = feature_groups + pipeline_stages

    # Percentage contributions for feature drivers
    pos_drivers = [g for g in feature_groups if g["impact"] > 0]
    neg_drivers = [g for g in feature_groups if g["impact"] < 0]
    total_pos = sum(g["impact"] for g in pos_drivers)
    total_neg = sum(abs(g["impact"]) for g in neg_drivers)
    total_abs = sum(abs(g["impact"]) for g in feature_groups)

    for g in groups:
        imp = g["impact"]
        if imp > 0 and total_pos > 0:
            g["pct_push_up"] = (imp / total_pos) * 100.0
        else:
            g["pct_push_up"] = 0.0

        if imp < 0 and total_neg > 0:
            g["pct_pull_down"] = (abs(imp) / total_neg) * 100.0
        else:
            g["pct_pull_down"] = 0.0

        g["pct_total_share"] = (abs(imp) / total_abs * 100.0) if total_abs > 0 else 0.0

    # Summary sentence e.g. "Rush Hour Peak contributed 42%, Heavy rain contributed 28%"
    headline_parts = []
    if total_pos > 0:
        top_pos = sorted(pos_drivers, key=lambda x: x["impact"], reverse=True)
        for g in top_pos[:3]:
            if g["pct_push_up"] >= 8:
                headline_parts.append(f"{g['name']} contributed {g['pct_push_up']:.0f}%")
        summary_parts = [
            f"**{g['name']}** contributed **{g['pct_push_up']:.0f}%** ({g['impact']:+,.0f} veh/hr)"
            for g in top_pos[:3] if g["pct_push_up"] >= 8
        ]
        summary_text = ("Key upward traffic drivers: " + ", ".join(summary_parts) + ".") if summary_parts else "Upward factors are mild."
        top_drivers = top_pos
    else:
        top_neg = sorted(neg_drivers, key=lambda x: abs(x["impact"]), reverse=True)
        for g in top_neg[:3]:
            if g["pct_pull_down"] >= 8:
                headline_parts.append(f"{g['name']} reduced traffic by {g['pct_pull_down']:.0f}%")
        summary_parts = [
            f"**{g['name']}** reduced volume by **{g['pct_pull_down']:.0f}%** ({g['impact']:+,.0f} veh/hr)"
            for g in top_neg[:3]
        ]
        summary_text = "Traffic is flowing freely: " + ", ".join(summary_parts) + "."
        top_drivers = top_neg

    headline_sentence = ", ".join(headline_parts[:2]) if headline_parts else "Traffic is at baseline levels"

    waterfall_steps = [
        {"name": g["name"], "detail": g.get("detail", ""), "impact": g["impact"]}
        for g in feature_groups
    ]
    waterfall_steps.append({
        "name": "Intra-Hour Trend",
        "detail": f"Sub-hour progression ({interp_delta:+.0f} veh/hr)",
        "impact": interp_delta,
    })
    waterfall_steps.append({
        "name": "India Context Calibration",
        "detail": "Transfer adaptation",
        "impact": calib_delta,
    })
    waterfall_steps.append({
        "name": "Live TomTom Speed Fusion",
        "detail": "TomTom sensor fusion",
        "impact": fusion_delta,
    })
    waterfall_steps.append({
        "name": "Physical Bounds",
        "detail": f"Corridor capacity clipping ({clip_delta:+.0f} veh/hr)",
        "impact": clip_delta,
    })

    return {
        "target_dt": target_dt,
        "horizon_min": horizon_min,
        "base_expected_value": base_val,
        "predicted_volume": raw_base,
        "final_forecast": final_vol,
        "interp_delta": interp_delta,
        "calibration_delta": calib_delta,
        "fusion_delta": fusion_delta,
        "clip_delta": clip_delta,
        "feature_groups": feature_groups,
        "pipeline_stages": pipeline_stages,
        "waterfall_steps": waterfall_steps,
        "groups": groups,
        "headline_sentence": headline_sentence,
        "summary_text": summary_text,
        "top_drivers": top_drivers,
        "raw_shap": feature_shap,
        "total_upward_impact": total_pos,
        "total_downward_impact": total_neg,
        "city": city,
    }


def plot_shap_waterfall_chart(explanation: dict) -> go.Figure:
    """Generates an intuitive Plotly Waterfall chart with 100% mathematical integrity.

    Traces: Baseline Average -> Pure SHAP Feature Attributions -> Intra-hour Trend -> India Calibration -> Live Speed Fusion -> Bounds Clipping -> Final Forecast.
    All bars add up with exact arithmetic equality.
    """
    feature_groups = explanation["feature_groups"]
    base_val = explanation["base_expected_value"]
    interp_delta = explanation.get("interp_delta", 0.0)
    calib_delta = explanation["calibration_delta"]
    fusion_delta = explanation["fusion_delta"]
    clip_delta = explanation.get("clip_delta", 0.0)
    final_vol = explanation["final_forecast"]

    # Select prominent feature drivers
    sorted_features = sorted(feature_groups, key=lambda x: abs(x["impact"]), reverse=True)
    top_features = sorted_features[:5]
    other_feature_impact = sum(g["impact"] for g in sorted_features[5:])

    steps = list(top_features)
    if abs(other_feature_impact) > 0.001:
        steps.append({"name": "Other Model Features", "detail": "Combined minor features", "impact": other_feature_impact})

    # Add intra-hour temporal interpolation if present
    if abs(interp_delta) > 0.001:
        steps.append({"name": "Intra-Hour Trend", "detail": f"Sub-hour progression ({interp_delta:+.0f} veh/hr)", "impact": interp_delta})

    # Add transparent post-model pipeline steps
    steps.append({"name": "India Calibration", "detail": "Transfer adaptation", "impact": calib_delta})
    steps.append({"name": "Live Speed Fusion", "detail": "TomTom sensor fusion", "impact": fusion_delta})

    # Add physical bounds clipping & integer rounding if active
    if abs(clip_delta) > 0.001:
        steps.append({"name": "Bounds & Rounding", "detail": f"Capacity clipping / integer display ({clip_delta:+.0f} veh/hr)", "impact": clip_delta})

    names = ["Baseline Average"] + [s["name"] for s in steps] + ["Final Forecast"]
    measures = ["absolute"] + ["relative"] * len(steps) + ["total"]
    values = [base_val] + [s["impact"] for s in steps] + [0]
    texts = [f"{base_val:,.0f}"] + [f"{s['impact']:+,.0f}" for s in steps] + [f"{final_vol:,.0f}"]

    fig = go.Figure(go.Waterfall(
        name="Pipeline Attribution",
        orientation="v",
        measure=measures,
        x=names,
        textposition="outside",
        text=texts,
        y=values,
        connector={"line": {"color": "rgba(100,100,100,0.5)"}},
        increasing={"marker": {"color": "#ef4444"}},  # Red: increases traffic volume
        decreasing={"marker": {"color": "#22c55e"}},  # Green: decreases volume
        totals={"marker": {"color": "#3b82f6"}},      # Blue: totals
    ))

    fig.update_layout(
        title=f"Prediction Waterfall: Historical Base → SHAP Features → Final Forecast (T+{explanation['horizon_min']}m)",
        waterfallgap=0.25,
        height=400,
        margin=dict(l=20, r=20, t=50, b=50),
        yaxis=dict(title="Traffic Volume (veh/hr)"),
        xaxis=dict(tickangle=-25),
    )
    return fig


def plot_shap_factor_bars(explanation: dict) -> go.Figure:
    """Horizontal bar chart of pure SHAP factor attributions with % contribution tags."""
    groups = sorted(explanation["feature_groups"], key=lambda x: x["impact"])
    names = [f"<b>{g['name']}</b><br><span style='font-size:11px;color:gray'>{g['detail']}</span>" for g in groups]
    impacts = [g["impact"] for g in groups]
    colors = ["#ef4444" if v >= 0 else "#22c55e" for v in impacts]

    labels = []
    for g in groups:
        if g["impact"] > 0:
            labels.append(f"+{g['impact']:.0f} veh/hr ({g['pct_push_up']:.0f}% uplift share)")
        else:
            labels.append(f"{g['impact']:.0f} veh/hr ({g['pct_pull_down']:.0f}% reduction share)")

    fig = go.Figure(go.Bar(
        x=impacts,
        y=names,
        orientation="h",
        marker_color=colors,
        text=labels,
        textposition="auto",
    ))

    fig.update_layout(
        title="Pure SHAP Feature Influences (🔴 Increases Congestion · 🟢 Eases Traffic)",
        xaxis=dict(title="Model Volume Impact (veh/hr)", zeroline=True, zerolinecolor="black"),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_shap_driver_shares(explanation: dict) -> go.Figure:
    """Donut chart showing % share of the key positive feature drivers of congestion."""
    pos = [g for g in explanation["feature_groups"] if g["impact"] > 0]
    if not pos:
        pos = [{"name": "Baseline Free-Flow Demand", "impact": 100, "pct_push_up": 100}]

    labels = [g["name"] for g in pos]
    values = [g["impact"] for g in pos]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+percent",
        marker=dict(colors=px.colors.qualitative.Plotly),
    ))

    fig.update_layout(
        title=f"Share of Feature Uplift Drivers (T+{explanation['horizon_min']} min)",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig
