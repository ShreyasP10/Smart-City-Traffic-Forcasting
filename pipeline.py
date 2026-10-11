"""Unified Traffic Prediction Pipeline.

Centralizes:
- Canonical feature generation with city-specific holidays (via schema.py and city_config.py)
- Base model inference (XGBoost) with smooth sub-hour temporal interpolation
- Transfer learning calibration (Ridge)
- Real-time live speed ratio fusion with data-driven p95 scale and exponential horizon decay
- Exact raw float preservation for 100% mathematically additive SHAP waterfalls
- Congestion classification and physical bounds clipping

Ensures zero divergence between CLI (predict.py) and UI (app.py).
"""
from __future__ import annotations

import datetime as dt
import numpy as np

from schema import featurize_point

LEVELS = ["Low", "Moderate", "High", "Severe"]


def classify(vol: float, thresholds: tuple | list | np.ndarray) -> str:
    """Classifies a traffic volume into Low / Moderate / High / Severe."""
    q25, q50, q75 = thresholds[0], thresholds[1], thresholds[2]
    if vol < q25:
        return "Low"
    if vol < q50:
        return "Moderate"
    if vol < q75:
        return "High"
    return "Severe"


def classify_idx(idx: int) -> str:
    """Maps index (0..3) to level label."""
    return LEVELS[min(max(idx, 0), len(LEVELS) - 1)]


def predict_pipeline(
    at: dt.datetime,
    weather: dict,
    flow: dict,
    bundle: dict,
    minutes_ahead: int = 0,
    city: str = "Mumbai",
) -> dict:
    """End-to-end canonical prediction pipeline.

    1. Base model: evaluates canonical feature vector at target horizon time with
       smooth sub-hour linear interpolation between hour boundaries.
    2. India calibration: adapts US historical baseline to Indian corridor characteristics.
    3. Live speed fusion: fuses current live congestion ratio using the data-derived
       p95 capacity volume, decaying exponentially across the 60-minute horizon.
    4. Mathematical additivity: preserves exact unclipped floats in 'raw' and exposes
       'clip_delta' so waterfalls sum with 100% arithmetic integrity.
    """
    target_dt = at + dt.timedelta(minutes=minutes_ahead) if minutes_ahead != 0 else at

    # 1. Base model prediction with sub-hour temporal interpolation
    h0 = target_dt.replace(minute=0, second=0, microsecond=0)
    frac = target_dt.minute / 60.0
    X0, row0 = featurize_point(h0, weather, city=city)
    v0 = float(bundle["model"].predict(X0)[0])

    if frac > 0.0:
        h1 = h0 + dt.timedelta(hours=1)
        X1, _ = featurize_point(h1, weather, city=city)
        v1 = float(bundle["model"].predict(X1)[0])
        base_vol_raw = (1.0 - frac) * v0 + frac * v1
    else:
        base_vol_raw = v0

    # Featurize target_dt for metadata
    _, row = featurize_point(target_dt, weather, city=city)

    # 2. Transfer calibration (Ridge) on raw base volume
    calib = bundle.get("calibration")
    if calib is not None:
        cal_X = np.array([[base_vol_raw, row["rain_1h"], row["hour"], row["is_peak"]]])
        adapted_vol_raw = float(calib.predict(cal_X)[0])
    else:
        adapted_vol_raw = base_vol_raw

    # 3. Live speed ratio fusion with horizon decay
    # Use data-derived p95 capacity volume from training metrics
    p95 = float(bundle.get("metrics", {}).get("p95_volume", 5400.0))
    raw_ratio = float(flow.get("congestion_ratio", 0.7))
    ratio = float(np.clip(raw_ratio, 0.05, 1.0))
    implied_vol_raw = (1.0 - ratio) * p95

    # Exponential horizon decay: live sensor term smoothly fades over the hour
    w_live = 0.35 * float(np.exp(-max(minutes_ahead, 0) / 45.0))
    w_model = 1.0 - w_live
    final_vol_raw = w_model * adapted_vol_raw + w_live * implied_vol_raw

    # Physical volume bounds [50, 8500] veh/hr and integer rounding delta (W1)
    final_clipped = float(np.clip(final_vol_raw, 50.0, 8500.0))
    final_vol_int = int(round(final_clipped))
    clip_delta = float(final_vol_int) - final_vol_raw

    # Keep exact raw floats for mathematical additivity in explainers
    raw = {
        "base": base_vol_raw,
        "adapted": adapted_vol_raw,
        "implied": implied_vol_raw,
        "final": final_vol_raw,
        "v0": v0,
        "frac": frac,
    }

    th = bundle["thresholds"]
    return {
        "base_vol": int(round(base_vol_raw)),
        "base_level": classify(base_vol_raw, th),
        "adapted_vol": int(round(adapted_vol_raw)),
        "adapted_level": classify(adapted_vol_raw, th),
        "final_vol": final_vol_int,
        "final_level": classify(final_clipped, th),
        "implied_vol": int(round(implied_vol_raw)),
        "raw": raw,
        "clip_delta": clip_delta,
        "w_live": round(w_live, 4),
        "w_model": round(w_model, 4),
        "congestion_ratio": ratio,
        "p95_scale": p95,
        "features": row,
        "minutes_ahead": minutes_ahead,
        "target_dt": target_dt,
        "city": city,
    }
