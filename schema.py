"""Canonical traffic forecasting schema and feature engineering.

Eliminates feature drift across training (train_base, train_model),
inference (predict.py), and dashboard explainability (app.py, explain_shap.py).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import joblib
import numpy as np
import pandas as pd

# Canonical feature list (16 features: base temporal + weather + cyclic trigonometric encodings)
FEATURES: list[str] = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "temp", "rain_1h", "snow_1h", "clouds_all", "weather_code",
    "is_peak", "is_night", "hour_sin", "hour_cos", "dow_sin", "dow_cos"
]

WEATHER_CODE: dict[str, int] = {
    "Clear": 0, "Clouds": 1, "Mist": 2, "Fog": 2, "Haze": 2, "Smoke": 2, "Sand": 2, "Dust": 2, "Ash": 2,
    "Drizzle": 3, "Rain": 4, "Squall": 4, "Snow": 5, "Thunderstorm": 6
}

CHECKSUMS_FILE = os.path.join(os.path.dirname(__file__), "models", "checksums.json")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all canonical features on a DataFrame containing date_time, weather_main, etc."""
    df = df.copy()
    if "date_time" in df.columns:
        dt_col = pd.to_datetime(df["date_time"])
        df["hour"] = dt_col.dt.hour
        df["day_of_week"] = dt_col.dt.dayofweek
        df["month"] = dt_col.dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        if "is_holiday" not in df.columns:
            df["is_holiday"] = ((df["holiday"] != "None") & df["holiday"].notna()).astype(int) if "holiday" in df.columns else 0

    if "weather_code" not in df.columns:
        df["weather_code"] = df["weather_main"].map(WEATHER_CODE).fillna(1).astype(int)

    # Smooth & interpretable peak indicators
    h = df["hour"].to_numpy()
    df["is_peak"] = np.isin(h, [7, 8, 9, 10, 17, 18, 19, 20, 21]).astype(int)
    df["is_night"] = ((h < 5) | (h >= 23)).astype(int)

    # Circular trigonometric representations (avoids hard step discontinuities in tree models)
    df["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    dow = df["day_of_week"].to_numpy()
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    for col in ["snow_1h", "rain_1h", "clouds_all"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


def featurize_point(at: dt.datetime, weather: dict, city: str = "Mumbai") -> tuple[np.ndarray, dict]:
    """Featurizes a single point in time and weather into canonical feature vector with city-specific holidays."""
    from city_config import is_holiday
    wc = WEATHER_CODE.get(weather.get("weather_main", "Clear"), 1)
    hour = at.hour
    dow = at.weekday()

    row = {
        "hour": hour,
        "day_of_week": dow,
        "month": at.month,
        "is_weekend": int(dow >= 5),
        # Demo override convention: weather dicts may carry _holiday_override=True
        # (set by api.py demo controls). Otherwise city-calendar lookup applies.
        "is_holiday": 1 if weather.get("_holiday_override") else int(is_holiday(at, city)),
        "temp": float(weather.get("temp", 300.0)),
        "rain_1h": float(weather.get("rain_1h", 0.0)),
        "snow_1h": 0.0,
        "clouds_all": int(weather.get("clouds_all", 20)),
        "weather_code": int(wc),
        "is_peak": int(hour in (7, 8, 9, 10, 17, 18, 19, 20, 21)),
        "is_night": int(hour < 5 or hour >= 23),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
    }
    vec = np.array([[row[f] for f in FEATURES]])
    return vec, row


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash for model verification."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def safe_load_model(path: str, expected_sha256: str | None = None) -> dict:
    """Loads a joblib model bundle and detects corrupted, modified, or swapped model files via SHA-256 verification."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    target_hash = os.environ.get("MODEL_SHA256") or expected_sha256
    if target_hash:
        actual_sha = compute_sha256(path)
        if not hmac.compare_digest(actual_sha, target_hash):
            raise ValueError(f"Integrity check failed for model file: {path} (expected {target_hash[:12]}..., got {actual_sha[:12]}...)")

    return joblib.load(path)
