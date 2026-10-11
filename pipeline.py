"""Unified Traffic Prediction Pipeline.

Centralizes:
- Canonical feature generation (via schema.py)
- Base model inference (XGBoost)
- Transfer learning calibration (Ridge)
- Real-time live speed ratio fusion with exponential horizon decay
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
) -> dict:
    """End-to-end canonical prediction pipeline.

    1. Base model: evaluates canonical feature vector at target horizon time.
    2. India calibration: adapts US historical baseline to Indian corridor characteristics.
    3. Live speed fusion: fuses current live congestion ratio with exponential horizon decay.
       At T+0 min, live sensor weight is 35%; at T+60 min, live weight smoothly decays to ~9%,
       allowing the machine learning temporal model to dominate forward horizons.
    4. Clipping: enforces physical corridor bounds [50, 8500] veh/hr.
    """
    target_dt = at + dt.timedelta(minutes=minutes_ahead) if minutes_ahead != 0 else at
    X, row = featurize_point(target_dt, weather)

    # 1. Base model prediction
    base_vol = float(bundle["model"].predict(X)[0])

    # 2. Transfer calibration (Ridge)
    calib = bundle.get("calibration")
    if calib is not None:
        cal_X = np.array([[base_vol, row["rain_1h"], row["hour"], row["is_peak"]]])
        adapted_vol = float(calib.predict(cal_X)[0])
    else:
        adapted_vol = base_vol

    # 3. Live speed ratio fusion with horizon decay
    # Safeguard and clip live congestion ratio to [0.05, 1.0]
    raw_ratio = float(flow.get("congestion_ratio", 0.7))
    ratio = float(np.clip(raw_ratio, 0.05, 1.0))
    implied_vol = (1.0 - ratio) * 5500.0

    # Exponential horizon decay for live sensor influence
    w_live = 0.35 * float(np.exp(-max(minutes_ahead, 0) / 45.0))
    w_model = 1.0 - w_live
    final_vol = w_model * adapted_vol + w_live * implied_vol

    # Physical volume bounds [50, 8500] veh/hr
    final_vol = float(np.clip(final_vol, 50.0, 8500.0))
    adapted_vol = float(np.clip(adapted_vol, 50.0, 8500.0))
    base_vol = float(np.clip(base_vol, 50.0, 8500.0))

    th = bundle["thresholds"]
    return {
        "base_vol": int(round(base_vol)),
        "base_level": classify(base_vol, th),
        "adapted_vol": int(round(adapted_vol)),
        "adapted_level": classify(adapted_vol, th),
        "final_vol": int(round(final_vol)),
        "final_level": classify(final_vol, th),
        "implied_vol": int(round(implied_vol)),
        "w_live": round(w_live, 4),
        "w_model": round(w_model, 4),
        "congestion_ratio": ratio,
        "features": row,
        "minutes_ahead": minutes_ahead,
        "target_dt": target_dt,
    }
