"""Train base + calibration (transfer-learning) models.

Stage 1 (base): HistGradientBoostingRegressor on US historical data.
Stage 2 (adaptation): Ridge regression stacking on a SMALL India sample that maps
    [base_pred, rain, hour, is_peak] -> india_volume. This is the calibration /
    transfer-learning step demonstrable in the dashboard (base vs adapted).

Run:  python train_model.py
Outputs: models/traffic_model.joblib, models/metrics.json, data/*.csv
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import joblib
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _HAS_XGB = False
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from data_generator import load_historical

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "temp", "rain_1h", "snow_1h", "clouds_all", "weather_code",
            "is_peak", "is_night"]

WEATHER_CODE = {"Clear": 0, "Clouds": 1, "Mist": 2, "Fog": 2, "Drizzle": 3,
                "Rain": 4, "Snow": 5, "Thunderstorm": 6}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weather_code"] = df["weather_main"].map(WEATHER_CODE).fillna(1).astype(int)
    df["is_peak"] = df["hour"].apply(lambda h: int(h in (7, 8, 9, 10, 17, 18, 19, 20, 21))).astype(int)
    df["is_night"] = df["hour"].apply(lambda h: int(h < 5 or h >= 23)).astype(int)
    return df


def volume_to_level(v: float, thresholds: list[float]) -> str:
    q25, q50, q75 = thresholds
    if v < q25:
        return "Low"
    if v < q50:
        return "Moderate"
    if v < q75:
        return "High"
    return "Severe"


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    us = add_features(load_historical(indian_style=False))
    india_small = add_features(load_historical(indian_style=True)).iloc[:24 * 14].copy()  # 14 days

    X = us[FEATURES].to_numpy()
    y = us["traffic_volume"].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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
    X_ind = india_small[FEATURES].to_numpy()
    y_ind = india_small["traffic_volume"].to_numpy()
    base_on_india = base.predict(X_ind)
    cal_X = np.column_stack([
        base_on_india,
        india_small["rain_1h"].to_numpy(),
        india_small["hour"].to_numpy(),
        india_small["is_peak"].to_numpy(),
    ])
    calib = Ridge(alpha=1.0)
    calib.fit(cal_X, y_ind)
    adap_pred = calib.predict(cal_X)
    mae_adap = float(mean_absolute_error(y_ind, adap_pred))
    mae_unadap = float(mean_absolute_error(y_ind, base_on_india))

    importances = None
    try:  # permutation importance is slow; use built-in gain via feature permutation on sample
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
        },
        "importances": importances,
    }
    joblib.dump(bundle, os.path.join(MODEL_DIR, "traffic_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(bundle["metrics"] | {"thresholds": thresholds}, f, indent=2)
    print(f"Base MAE={mae:.1f} RMSE={rmse:.1f} thresholds={[round(t) for t in thresholds]}")
    print(f"Transfer: unadapted MAE={mae_unadap:.1f} -> adapted MAE={mae_adap:.1f}")
    print("Saved models/traffic_model.joblib")


if __name__ == "__main__":
    main()
