"""Train base + calibration (transfer-learning) models with chronological split.

Enforces zero temporal leakage and canonical schema from schema.py.
Outputs: models/traffic_model.joblib, models/metrics.json, models/checksums.json
"""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_XGB = False
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from schema import FEATURES, WEATHER_CODE, add_features, compute_sha256
from data_generator import load_historical

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    us = add_features(load_historical(indian_style=False))
    if "date_time" in us.columns:
        us = us.drop_duplicates(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)

    india_small = add_features(load_historical(indian_style=True)).iloc[:24 * 14].copy()

    X = us[FEATURES].to_numpy()
    y = us["traffic_volume"].to_numpy()

    # Strict chronological split: past 80% trains, future 20% tests (no temporal leakage)
    split_idx = int(len(X) * 0.8)
    Xtr, Xte = X[:split_idx], X[split_idx:]
    ytr, yte = y[:split_idx], y[split_idx:]

    if _HAS_XGB:
        base = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                            n_jobs=-1, random_state=42)
    else:
        base = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                             max_depth=None, random_state=42)
    base.fit(Xtr, ytr)
    pred = base.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(mean_squared_error(yte, pred) ** 0.5)

    # Data-driven congestion thresholds from training volumes (quartiles)
    q25, q50, q75 = [float(v) for v in np.percentile(ytr, [25, 50, 75])]
    thresholds = [q25, q50, q75]

    # ---- Stage 2: calibration on small India sample (transfer learning) ----
    # Enforce honest 70/30 chronological out-of-time split: fit on first 70%, evaluate strictly on held-out 30%
    cut = int(len(india_small) * 0.7)
    ind_tr, ind_val = india_small.iloc[:cut], india_small.iloc[cut:]

    base_tr = base.predict(ind_tr[FEATURES].to_numpy())
    cal_X_tr = np.column_stack([
        base_tr,
        ind_tr["rain_1h"].to_numpy(),
        ind_tr["hour"].to_numpy(),
        ind_tr["is_peak"].to_numpy(),
    ])
    calib = Ridge(alpha=1.0)
    calib.fit(cal_X_tr, ind_tr["traffic_volume"].to_numpy())

    # Honest out-of-time validation metrics on held-out 30%
    base_val = base.predict(ind_val[FEATURES].to_numpy())
    cal_X_val = np.column_stack([
        base_val,
        ind_val["rain_1h"].to_numpy(),
        ind_val["hour"].to_numpy(),
        ind_val["is_peak"].to_numpy(),
    ])
    adap_val = calib.predict(cal_X_val)
    mae_adap = float(mean_absolute_error(ind_val["traffic_volume"], adap_val))
    mae_unadap = float(mean_absolute_error(ind_val["traffic_volume"], base_val))
    p95_vol = float(np.percentile(ytr, 95))

    importances = None
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(base, Xte[:2000], yte[:2000], n_repeats=5, random_state=42)
        importances = {f: float(v) for f, v in zip(FEATURES, r.importances_mean)}
    except Exception:
        importances = {f: 1.0 / len(FEATURES) for f in FEATURES}

    bundle = {
        "model": base,
        "calibration": calib,
        "thresholds": thresholds,
        "features": FEATURES,
        "weather_code": WEATHER_CODE,
        "metrics": {
            "base_mae": mae, "base_rmse": rmse,
            "calib_mae_adapted": mae_adap, "calib_mae_unadapted": mae_unadap,
            "n_us": int(len(us)), "n_india_calib": int(len(india_small)),
            "n_india_train": int(len(ind_tr)), "n_india_val": int(len(ind_val)),
            "p95_volume": p95_vol,
        },
        "importances": importances,
    }

    joblib_path = os.path.join(MODEL_DIR, "traffic_model.joblib")
    joblib.dump(bundle, joblib_path)

    # Calculate and store model integrity SHA-256
    sha = compute_sha256(joblib_path)
    checksums_file = os.path.join(MODEL_DIR, "checksums.json")
    cur_checksums = {}
    if os.path.exists(checksums_file):
        with open(checksums_file, "r") as f:
            cur_checksums = json.load(f)
    cur_checksums["traffic_model.joblib"] = sha
    with open(checksums_file, "w") as f:
        json.dump(cur_checksums, f, indent=2)

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(bundle["metrics"] | {"thresholds": thresholds}, f, indent=2)

    print(f"Base Chronological MAE={mae:.1f} RMSE={rmse:.1f} thresholds={[round(t) for t in thresholds]}")
    print(f"Transfer: unadapted MAE={mae_unadap:.1f} -> adapted MAE={mae_adap:.1f}")
    print(f"Saved {joblib_path} (SHA-256: {sha[:12]}...)")


if __name__ == "__main__":
    main()
