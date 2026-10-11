"""Historical Model Training with strict chronological temporal split (no data leakage).

Uses real UCI Metro Interstate dataset (48,204 rows).
Enforces canonical schema from schema.py and generates model integrity checksums.
"""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from schema import FEATURES, WEATHER_CODE, add_features, compute_sha256

CSV_CANDIDATES = [
    "Metro_Interstate_Traffic_Volume.csv",
    "data/Metro_Interstate_Traffic_Volume.csv",
    "data/Metro_Interstate_Traffic_Data.csv",
]
# INDIA-first override: TRAIN_CSV=data/synthetic_india_v2.csv trains the base
# on India-pattern data (peak-shifted, monsoon, denser mixed traffic).
CSV = os.environ.get("TRAIN_CSV") or next((c for c in CSV_CANDIDATES if os.path.exists(c)), CSV_CANDIDATES[0])

try:
    from xgboost import XGBRegressor
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor
    BACKEND = "sklearn-hgb"

print(f"Loading {CSV} ...")
df = pd.read_csv(CSV)
df["date_time"] = pd.to_datetime(df["date_time"])
# Deduplicate repeated timestamps and enforce strict chronological ordering
df = df.drop_duplicates(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)
df = add_features(df)

X = df[FEATURES]
y = df["traffic_volume"]

# Strict chronological split: past 80% trains, future 20% validates (ZERO temporal leakage)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

thresholds = np.percentile(y_train, [25, 50, 75])
p95 = float(np.percentile(y_train, 95))
print("Thresholds (Low/Moderate/High/Severe):", thresholds)
print(f"Training {BACKEND} chronologically: {len(X_train)} train rows -> {len(X_test)} future test rows ...")

# 1. Naive baseline benchmark: mean volume by (hour, day_of_week) on training data
train_bdf = pd.DataFrame({"hour": X_train["hour"], "dow": X_train["day_of_week"], "y": y_train})
grp_means = train_bdf.groupby(["hour", "dow"])["y"].mean().to_dict()
global_mean = float(y_train.mean())
base_preds = np.array([grp_means.get((h, d), global_mean) for h, d in zip(X_test["hour"], X_test["day_of_week"])])
baseline_mae = float(mean_absolute_error(y_test, base_preds))
baseline_rmse = float(np.sqrt(mean_squared_error(y_test, base_preds)))

if BACKEND == "xgboost":
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         n_jobs=-1, random_state=42)
else:
    model = XGBRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)

model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
error_drop = (1.0 - mae / baseline_mae) * 100.0

print(f"Historical (Hour × Day) Baseline MAE: {baseline_mae:.2f}  RMSE: {baseline_rmse:.2f}")
print(f"Chronological Out-of-Time MAE: {mae:.2f}  RMSE: {rmse:.2f}  ({error_drop:.1f}% reduction over naive baseline)")

out_path = "base_model.pkl"
joblib.dump({
    "model": model, "features": FEATURES, "thresholds": np.array(thresholds),
    "weather_map": WEATHER_CODE, "backend": BACKEND,
    "metrics": {
        "mae": float(mae), "rmse": rmse, "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse, "p95_volume": p95, "n": int(len(df))
    }
}, out_path)

sha = compute_sha256(out_path)
checksums_file = os.path.join("models", "checksums.json")
os.makedirs("models", exist_ok=True)
cur_checksums = {}
if os.path.exists(checksums_file):
    with open(checksums_file, "r") as f:
        cur_checksums = json.load(f)
cur_checksums["base_model.pkl"] = sha
with open(checksums_file, "w") as f:
    json.dump(cur_checksums, f, indent=2)

print(f"Saved {out_path} (SHA-256: {sha[:12]}...)")
