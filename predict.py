# predict.py — Hour 1:30-2:15, Real-Time API + Prediction Pipeline.
# Spec-faithful structure (classify/fetch_live/build_features/predict_now + 4 horizons),
# hardened: keys from env -> .streamlit/secrets.toml -> .env (no dotenv needed),
# live via traffic_api (TomTom live + Open-Meteo fallback when OWM 401s),
# temp normalised to Kelvin (UCI training scale), TomTom errors logged not crashed.
import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from traffic_api import fetch_openweather, fetch_tomtom_flow

bundle = joblib.load(os.path.join(os.path.dirname(__file__), "base_model.pkl"))
MODEL, FEATURES = bundle["model"], bundle["features"]
THRESH, WEATHER_MAP = bundle["thresholds"], bundle["weather_map"]


def _load_keys():
    def kv(path):
        out = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        out[k.strip()] = v.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
        return out

    base = os.path.dirname(__file__)
    fk = {}
    fk.update(kv(os.path.join(base, ".streamlit", "secrets.toml")))
    fk.update(kv(os.path.join(base, ".env")))
    tt = os.environ.get("TOMTOM_KEY") or os.environ.get("TOMTOM_API_KEY") or fk.get("TOMTOM_KEY") or fk.get("TOMTOM_API_KEY") or ""
    ow = os.environ.get("OPENWEATHER_KEY") or os.environ.get("OPENWEATHER_API_KEY") or fk.get("OPENWEATHER_KEY") or fk.get("OPENWEATHER_API_KEY") or ""
    return tt.strip(), ow.strip()


TT_KEY, OW_KEY = _load_keys()

CACHE_FILE = os.path.join(os.path.dirname(__file__), "last_response.json")


def _save_cache(lat, lon, live) -> None:
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({"lat": lat, "lon": lon, "live": live,
                       "cached_at": datetime.now().isoformat(timespec="seconds")}, f, default=str)
    except Exception:
        pass


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def classify(vol):
    if vol <= THRESH[0]:
        return "Low"
    if vol <= THRESH[1]:
        return "Moderate"
    if vol <= THRESH[2]:
        return "High"
    return "Severe"


def fetch_live(lat, lon):
    flow = fetch_tomtom_flow(lat, lon, 50, TT_KEY or None, 0.4)
    if "tomtom-error" in flow.get("source", "") or "Key ID" in flow.get("source", ""):
        print(f"  [tomtom-note] {flow['source']}", flush=True)
    w = fetch_openweather(lat, lon, OW_KEY or None)
    temp_k = float(w["temp"]) if float(w["temp"]) > 200 else float(w["temp"]) + 273.15
    return {
        "current_speed": flow["current_speed"],
        "free_flow_speed": flow["free_flow_speed"],
        "congestion_ratio": flow["congestion_ratio"],
        "temp": temp_k,  # Kelvin, matches UCI training
        "rain_1h": float(w.get("rain_1h", 0)),
        "weather": w.get("weather_main", "Clear"),
        "clouds": int(w.get("clouds_all", 20)),
        "traffic_source": flow["source"],
        "weather_source": w["source"],
    }


def build_features(live, target_dt=None):
    dt = target_dt or datetime.now()
    return np.array([[
        dt.hour, dt.weekday(), dt.month,
        int(dt.weekday() >= 5), 0,
        live["temp"], live["rain_1h"], 0,
        live["clouds"], WEATHER_MAP.get(live["weather"], 0)
    ]])


def predict_now(lat, lon, offset_min=0):
    try:
        live = fetch_live(lat, lon)
        # Cache last successful LIVE response (demo-proofing: network timeouts)
        if "live:" in str(live.get("traffic_source", "")) or "live:" in str(live.get("weather_source", "")):
            _save_cache(lat, lon, live)
    except Exception as e:
        print(f"  [api-error] {e} — trying cache {CACHE_FILE}", flush=True)
        cached = _load_cache()
        if cached and cached.get("live"):
            print(f"  [cache] using response cached at {cached.get('cached_at')}", flush=True)
            live = cached["live"]
        else:
            raise
    target = datetime.now() + timedelta(minutes=offset_min)
    X = build_features(live, target)
    raw_vol = float(MODEL.predict(X)[0])
    # Calibration factor from congestion ratio (Indian context adjustment, per spec)
    adjusted = raw_vol * (1 + (1 - live["congestion_ratio"])) * 0.5 + raw_vol * 0.5
    # Compute SHAP explanation for the prediction
    why_text = ""
    try:
        from explain_shap import compute_shap_explanation
        model_bundle = joblib.load(os.path.join(os.path.dirname(__file__), "models", "traffic_model.joblib"))
        flow_stub = {"current_speed": live["current_speed"], "free_flow_speed": live["free_flow_speed"], "congestion_ratio": live["congestion_ratio"]}
        weather_stub = {"temp": live["temp"], "rain_1h": live["rain_1h"], "clouds_all": live["clouds"], "weather_main": live["weather"]}
        shap_info = compute_shap_explanation(datetime.now(), weather_stub, flow_stub, model_bundle, horizon_min=offset_min)
        why_text = shap_info["headline_sentence"]
    except Exception as _e:
        why_text = f"SHAP pending ({_e})"

    return {
        "live": live,
        "raw_prediction": raw_vol,
        "adjusted_prediction": adjusted,
        "congestion_class": classify(adjusted),
        "horizon_min": offset_min,
        "why": why_text,
    }


if __name__ == "__main__":
    lat, lon = 19.0653, 72.8686  # Mumbai BKC
    print(f"Keys: TomTom {'found' if TT_KEY else 'MISSING'} | OpenWeather {'found' if OW_KEY else 'MISSING (Open-Meteo fallback)'}")
    for offset in [0, 15, 30, 60]:
        r = predict_now(lat, lon, offset)
        print(f"T+{offset:02d}min -> {r['congestion_class']:8s} "
              f"(raw={r['raw_prediction']:.0f}, adj={r['adjusted_prediction']:.0f}) "
              f"| Why: {r['why']}")
