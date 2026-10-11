"""Real-Time API + Prediction Pipeline.

Hardened:
- Safe model integrity loading with SHA-256 verification (prevents arbitrary code execution).
- Unified canonical schema (zero feature drift).
- Robust tomllib secrets loading.
- Subsystem-isolated cache (data/predict_cache.json) avoiding file race conditions.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

try:
    import tomllib
except ImportError:
    import tomli as tomllib

sys.path.insert(0, os.path.dirname(__file__))
from traffic_api import fetch_openweather, fetch_tomtom_flow, validate_coordinates
from schema import safe_load_model, featurize_point, FEATURES
from pipeline import predict_pipeline, classify

CHECKSUMS_FILE = os.path.join(os.path.dirname(__file__), "models", "checksums.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "base_model.pkl")

# Verify SHA-256 integrity hash if checksums record exists
expected_hash = None
if os.path.exists(CHECKSUMS_FILE):
    try:
        with open(CHECKSUMS_FILE, "r") as f:
            checksums = json.load(f)
            fname = os.path.basename(MODEL_PATH)
            expected_hash = checksums.get(fname)
    except Exception:
        pass

bundle = safe_load_model(MODEL_PATH, expected_hash)
MODEL = bundle["model"]
CALIB = bundle.get("calibration")
THRESH = bundle["thresholds"]


def _load_keys() -> tuple[str, str]:
    base = os.path.dirname(__file__)
    sec_file = os.path.join(base, ".streamlit", "secrets.toml")
    file_keys = {}
    if os.path.exists(sec_file):
        try:
            with open(sec_file, "rb") as f:
                file_keys = tomllib.load(f)
        except Exception:
            pass

    tt = os.environ.get("TOMTOM_KEY") or os.environ.get("TOMTOM_API_KEY") or file_keys.get("TOMTOM_KEY") or file_keys.get("TOMTOM_API_KEY") or ""
    ow = os.environ.get("OPENWEATHER_KEY") or os.environ.get("OPENWEATHER_API_KEY") or file_keys.get("OPENWEATHER_KEY") or file_keys.get("OPENWEATHER_API_KEY") or ""
    return str(tt).strip(), str(ow).strip()


TT_KEY, OW_KEY = _load_keys()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_FILE = os.path.join(DATA_DIR, "predict_cache.json")


def _save_cache(lat: float, lon: float, live: dict) -> None:
    try:
        tmp_file = CACHE_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump({"lat": lat, "lon": lon, "live": live,
                       "cached_at": datetime.now().isoformat(timespec="seconds")}, f, default=str)
        os.replace(tmp_file, CACHE_FILE)
    except Exception:
        pass


def _load_cache() -> dict | None:
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def fetch_live(lat: float, lon: float) -> dict:
    lat_f, lon_f = validate_coordinates(lat, lon)
    flow = fetch_tomtom_flow(lat_f, lon_f, 50, TT_KEY or None, 0.4)
    w = fetch_openweather(lat_f, lon_f, OW_KEY or None)
    temp_k = float(w["temp"]) if float(w["temp"]) > 200 else float(w["temp"]) + 273.15
    return {
        "current_speed": flow["current_speed"],
        "free_flow_speed": flow["free_flow_speed"],
        "congestion_ratio": flow["congestion_ratio"],
        "temp": temp_k,
        "rain_1h": float(w.get("rain_1h", 0.0)),
        "weather": w.get("weather_main", "Clear"),
        "clouds": int(w.get("clouds_all", 20)),
        "traffic_source": flow["source"],
        "weather_source": w["source"],
    }


def predict_now(lat: float, lon: float, offset_min: int = 0) -> dict:
    lat_f, lon_f = validate_coordinates(lat, lon)
    try:
        live = fetch_live(lat_f, lon_f)
        if "live:" in str(live.get("traffic_source", "")) or "live:" in str(live.get("weather_source", "")):
            _save_cache(lat_f, lon_f, live)
    except Exception as e:
        cached = _load_cache()
        if cached and cached.get("live"):
            live = cached["live"]
        else:
            raise

    weather_dict = {
        "temp": live["temp"],
        "rain_1h": live["rain_1h"],
        "clouds_all": live["clouds"],
        "weather_main": live["weather"],
    }
    flow_dict = {
        "current_speed": live["current_speed"],
        "free_flow_speed": live["free_flow_speed"],
        "congestion_ratio": live["congestion_ratio"],
    }

    # Canonical prediction pipeline (identical to app.py)
    pipe_out = predict_pipeline(datetime.now(), weather_dict, flow_dict, bundle, minutes_ahead=offset_min)

    # Compute SHAP explanation with pure feature attributions and transparent pipeline steps
    why_text = ""
    try:
        from explain_shap import compute_shap_explanation
        shap_info = compute_shap_explanation(
            datetime.now(), weather_dict, flow_dict, bundle,
            horizon_min=offset_min, pipeline_pred=pipe_out
        )
        why_text = shap_info["headline_sentence"]
    except Exception as _e:
        why_text = f"SHAP explanation pending ({_e})"

    return {
        "live": live,
        "raw_prediction": pipe_out["base_vol"],
        "adapted_prediction": pipe_out["adapted_vol"],
        "final_prediction": pipe_out["final_vol"],
        "congestion_class": pipe_out["final_level"],
        "horizon_min": offset_min,
        "why": why_text,
    }


if __name__ == "__main__":
    lat, lon = 19.0653, 72.8686  # Mumbai BKC
    print(f"Keys: TomTom {'found' if TT_KEY else 'MISSING'} | OpenWeather {'found' if OW_KEY else 'MISSING (Open-Meteo fallback)'}")
    for offset in [0, 15, 30, 60]:
        r = predict_now(lat, lon, offset)
        print(f"T+{offset:02d}min -> {r['congestion_class']:8s} "
              f"(final={r['final_prediction']:.0f}, base={r['raw_prediction']:.0f}) "
              f"| Why: {r['why']}")
