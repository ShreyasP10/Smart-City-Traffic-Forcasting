"""Synthetic historical traffic generator + UCI loader.

Primary source: tries to load UCI Metro Interstate Traffic Volume CSV if present
at data/Metro_Interstate_Traffic_Data.csv. Otherwise generates a realistic
synthetic dataset with the same schema so the prototype runs offline.

Schema (UCI-compatible):
  holiday, temp, rain_1h, snow_1h, clouds_all, weather_main,
  date_time, traffic_volume
+ engineered: hour, day_of_week, month, is_weekend, is_holiday
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
UCI_PATH = os.path.join(DATA_DIR, "Metro_Interstate_Traffic_Data.csv")

WEATHERS = ["Clear", "Clouds", "Rain", "Drizzle", "Mist", "Fog", "Snow", "Thunderstorm"]

RNG = np.random.default_rng(42)


def _hourly_profile(hour: np.ndarray, peak_shift: int = 0) -> np.ndarray:
    """Two-peak gaussian rush-hour profile, normalised ~0..1."""
    h = (hour + peak_shift) % 24
    morning = np.exp(-0.5 * ((h - 8.2) / 1.4) ** 2)
    evening = np.exp(-0.5 * ((h - 17.6) / 1.8) ** 2)
    night = 0.12 + 0.05 * np.cos(2 * np.pi * (h - 3) / 24)
    return 0.25 * night + 0.85 * morning + 1.0 * evening


def generate_synthetic(
    n_hours: int = 24 * 365 * 2,  # 2 years hourly
    base: float = 2800,
    peak_shift: int = 0,
    indian_style: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({"date_time": idx})
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek  # Mon=0
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holidays: fixed list + random Sundays-as-holiday noise
    from city_config import INDIAN_HOLIDAYS, US_HOLIDAYS
    hol = INDIAN_HOLIDAYS if indian_style else US_HOLIDAYS
    df["is_holiday"] = df["date_time"].apply(lambda t: int((t.month, t.day) in hol)).astype(int)
    df["holiday"] = np.where(df["is_holiday"] == 1, "Holiday", "None")

    # Weather simulation with seasonal + monsoon character
    month = df["month"].to_numpy()
    if indian_style:
        # Hot summer, heavy monsoon Jun-Sep
        temp = 298 + 6 * np.sin(2 * np.pi * (month - 4) / 12) + rng.normal(0, 1.5, len(df))
        rain_prob = np.where(np.isin(month, [6, 7, 8, 9]), 0.35, 0.06)
    else:
        temp = 283 + 12 * np.sin(2 * np.pi * (month - 7) / 12) + rng.normal(0, 2.0, len(df))
        rain_prob = np.where(np.isin(month, [4, 5, 6]), 0.18, 0.08)
    is_rain = rng.random(len(df)) < rain_prob
    df["temp"] = temp
    df["rain_1h"] = np.where(is_rain, rng.exponential(2.5, len(df)).round(2), 0.0)
    is_snow = (~is_rain) & (rng.random(len(df)) < (0.06 if not indian_style else 0.0)) & (df["temp"] < 275).to_numpy()
    df["snow_1h"] = np.where(is_snow, rng.exponential(1.0, len(df)).round(2), 0.0)
    df["clouds_all"] = np.clip(
        (np.where(is_rain, 75, 20) + rng.normal(0, 15, len(df))).astype(int), 0, 100
    )
    df["weather_main"] = np.where(
        is_rain, rng.choice(["Rain", "Drizzle", "Thunderstorm"], len(df), p=[0.6, 0.25, 0.15]),
        np.where(df["clouds_all"] > 60, "Clouds",
                 np.where(df["clouds_all"] > 30, "Mist", "Clear")),
    )
    # Traffic volume
    prof = _hourly_profile(df["hour"].to_numpy(), peak_shift=peak_shift)
    weekday_boost = np.where(df["is_weekend"] == 1, 0.55, 1.0)
    holiday_damp = np.where(df["is_holiday"] == 1, 0.55, 1.0)
    rain_damp = np.where(df["rain_1h"] > 0, 0.92, 1.0)  # fewer discretionary trips in rain
    snow_damp = np.where(df["snow_1h"] > 0, 0.75, 1.0)
    seasonal = 1 + 0.08 * np.sin(2 * np.pi * (month - 6) / 12)
    noise = rng.normal(1.0, 0.12, len(df))
    vol = base * prof * weekday_boost * holiday_damp * rain_damp * snow_damp * seasonal * noise
    if indian_style:
        vol = vol * 1.35  # denser mixed traffic
    df["traffic_volume"] = np.clip(vol.astype(int), 50, 7500)
    return df


def load_historical(indian_style: bool = False) -> pd.DataFrame:
    """Load UCI CSV if available else synthetic. Caches synthetic to data/."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(UCI_PATH) and not indian_style:
        df = pd.read_csv(UCI_PATH)
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["hour"] = df["date_time"].dt.hour
        df["day_of_week"] = df["date_time"].dt.dayofweek
        df["month"] = df["date_time"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_holiday"] = (df["holiday"] != "None").astype(int)
        return df.sort_values("date_time").reset_index(drop=True)
    fname = "synthetic_india.csv" if indian_style else "synthetic_us.csv"
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath, parse_dates=["date_time"])
        return df
    df = generate_synthetic(indian_style=indian_style, seed=7 if indian_style else 42,
                            peak_shift=1 if indian_style else 0)
    df.to_csv(fpath, index=False)
    return df


if __name__ == "__main__":
    us = load_historical(indian_style=False)
    india = load_historical(indian_style=True)
    print("US:", us.shape, us["traffic_volume"].describe().to_dict())
    print("India:", india.shape, india["traffic_volume"].describe().to_dict())
