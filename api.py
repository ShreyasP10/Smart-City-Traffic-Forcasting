"""FastAPI backend for the React UI. Reuses the canonical pipeline verbatim.

Run:  uvicorn api:app --port 8000
React dev server proxies /api -> localhost:8000 (see frontend/vite.config.js).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import joblib  # noqa: F401 (kept for safe_load_model internals)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from city_config import CITIES
from data_generator import load_historical
from pipeline import predict_pipeline
from schema import safe_load_model
from traffic_api import fetch_openweather, fetch_tomtom_flow

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "models", "traffic_model.joblib")
CHECKSUMS_FILE = os.path.join(BASE, "models", "checksums.json")
LEVEL_COLOR = {"Low": "#22c55e", "Moderate": "#eab308", "High": "#f97316", "Severe": "#ef4444"}

app = FastAPI(title="Smart City Traffic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    p = os.path.join(BASE, ".streamlit", "secrets.toml")
    try:
        import tomllib
        with open(p, "rb") as f:
            return str(tomllib.load(f).get(name, ""))
    except Exception:
        return ""


TT_KEY = _secret("TOMTOM_API_KEY") or _secret("TOMTOM_KEY")
OW_KEY = _secret("OPENWEATHER_API_KEY") or _secret("OPENWEATHER_KEY")

_RUNTIME = {"tomtom": "", "openweather": ""}


def tt_key() -> str:
    return _RUNTIME["tomtom"] or TT_KEY


def ow_key() -> str:
    return _RUNTIME["openweather"] or OW_KEY

_bundle = None
_data = None


def bundle():
    global _bundle
    if _bundle is None:
        sha = None
        try:
            sha = json.load(open(CHECKSUMS_FILE)).get("traffic_model.joblib")
        except Exception:
            pass
        _bundle = safe_load_model(MODEL_PATH, sha)
    return _bundle


def data():
    global _data
    if _data is None:
        _data = (load_historical(indian_style=False), load_historical(indian_style=True))
    return _data


def _route(city: str, route: str) -> dict:
    if city not in CITIES:
        raise HTTPException(404, f"unknown city {city}")
    for r in CITIES[city]["routes"]:
        if r["name"] == route:
            return r
    raise HTTPException(404, f"unknown route {route}")


def _live(city: str, route: dict, at: dt.datetime, rain_override: bool = False,
          holiday_override: bool = False):
    b = bundle()
    weather = fetch_openweather(route["lat"], route["lon"], ow_key() or None)
    if rain_override:
        weather["rain_1h"] = max(float(weather.get("rain_1h", 0)), 3.5)
        if weather.get("weather_main") == "Clear":
            weather["weather_main"] = "Rain"
    if holiday_override:
        weather["_holiday_override"] = True
    _, india = data()
    hist = india if city != "Minneapolis (Source / US)" else data()[0]
    try:
        hm = float(hist[hist["date_time"].dt.hour == at.hour]["traffic_volume"].mean())
    except Exception:
        hm = 1500.0
    import numpy as np
    hint = float(np.clip(hm / 5500, 0.05, 0.95))
    flow = fetch_tomtom_flow(route["lat"], route["lon"], route["free_flow"], tt_key() or None, hint)
    length = float(route.get("length_km", 5.0))
    flow["travel_time_min"] = round(length / max(flow["current_speed"], 1) * 60, 1)
    flow["free_time_min"] = round(length / max(flow["free_flow_speed"], 1) * 60, 1)
    flow["delay_min"] = round(flow["travel_time_min"] - flow["free_time_min"], 1)
    return weather, flow


def _clean(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dt.datetime):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = _clean(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [x.isoformat() if isinstance(x, dt.datetime) else x for x in v]
        else:
            try:
                import numpy as _np
                if isinstance(v, _np.generic):
                    v = v.item()
            except Exception:
                pass
            out[k] = v
    return out


def _shap(at, weather, flow, b, horizon=0, pipe=None):
    try:
        from explain_shap import compute_shap_explanation
        s = compute_shap_explanation(at, weather, flow, b, horizon_min=horizon,
                                     pipeline_pred=pipe if horizon == 0 else None)
        return {"headline": s.get("headline_sentence", ""),
                "summary": s.get("summary_text", ""),
                "drivers": [{"name": d.get("name"), "impact": float(d.get("impact", 0)),
                             "pct": float(d.get("pct_push_up", 0) if d.get("impact", 0) > 0
                                         else d.get("pct_pull_down", 0))}
                            for d in s.get("top_drivers", [])[:6]]}
    except Exception as e:
        return {"headline": "", "summary": f"SHAP unavailable ({e})", "drivers": []}


@app.get("/api/health")
def health():
    return {"ok": True, "tomtom": bool(tt_key()), "openweather": bool(ow_key())}


@app.post("/api/keys")
def set_keys(payload: dict):
    if payload.get("tomtom") is not None:
        _RUNTIME["tomtom"] = str(payload["tomtom"]).strip()
    if payload.get("openweather") is not None:
        _RUNTIME["openweather"] = str(payload["openweather"]).strip()
    return {"ok": True, "tomtom": bool(tt_key()), "openweather": bool(ow_key())}


def _custom_route(lat: float, lon: float) -> dict:
    return {"name": f"Custom ({lat:.4f}, {lon:.4f})", "lat": float(lat), "lon": float(lon),
            "free_flow": 50, "length_km": 5.0}


@app.get("/api/cities")
def cities():
    return {"cities": [
        {"name": c, "center": list(CITIES[c]["center"]),
         "routes": [{"name": r["name"], "lat": r["lat"], "lon": r["lon"],
                     "free_flow": r["free_flow"], "length_km": r.get("length_km", 5.0)}
                    for r in CITIES[c]["routes"]]}
        for c in CITIES]}


@app.get("/api/thresholds")
def thresholds():
    b = bundle()
    return {"thresholds": [float(x) for x in b["thresholds"]], "metrics": b.get("metrics", {})}


@app.get("/api/overview")
def overview(city: str = "Mumbai", route: str = "Andheri → BKC", hour: int | None = None,
             lat: float | None = None, lon: float | None = None,
             rain: bool = False, holiday: bool = False):
    b = bundle()
    r = _custom_route(lat, lon) if lat is not None and lon is not None else _route(city, route)
    at = dt.datetime.now().replace(second=0, microsecond=0)
    if hour is not None:
        at = at.replace(hour=int(hour) % 24, minute=0)
    weather, flow = _live(city, r, at, rain_override=rain, holiday_override=holiday)
    p = _clean(predict_pipeline(at, weather, flow, b, city=city.split(" ")[0]))
    return {"at": at.isoformat(), "city": city, "route": r["name"],
            "weather": weather, "flow": flow, "pred": p,
            "colors": LEVEL_COLOR, "shap": _shap(at, weather, flow, b, 0, None)}


@app.get("/api/forecast")
def forecast(city: str = "Mumbai", route: str = "Andheri → BKC",
             lat: float | None = None, lon: float | None = None,
             rain: bool = False, holiday: bool = False):
    b = bundle()
    r = _custom_route(lat, lon) if lat is not None and lon is not None else _route(city, route)
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather, flow = _live(city, r, at, rain_override=rain, holiday_override=holiday)
    rows = []
    for m in [0, 15, 30, 60]:
        p = _clean(predict_pipeline(at, weather, flow, b, minutes_ahead=m, city=city.split(" ")[0]))
        rows.append({"in_min": m, "time": (at + dt.timedelta(minutes=m)).strftime("%H:%M"), **p})
    return {"at": at.isoformat(), "city": city, "route": r["name"],
            "weather": weather, "flow": flow, "rows": rows}


@app.get("/api/heatmap")
def heatmap(city: str = "Mumbai"):
    b = bundle()
    if city not in CITIES:
        raise HTTPException(404, "unknown city")
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather, _ = _live(city, CITIES[city]["routes"][0], at)
    out = []
    for r in CITIES[city]["routes"]:
        fl = fetch_tomtom_flow(r["lat"], r["lon"], r["free_flow"], tt_key() or None, 0.4)
        p = predict_pipeline(at, weather, fl, b, city=city.split(" ")[0])
        out.append({"route": r["name"], "lat": r["lat"], "lon": r["lon"],
                    "level": p["final_level"], "vol": p["final_vol"],
                    "speed": fl["current_speed"], "source": fl["source"]})
    return {"city": city, "weather": weather, "points": out, "colors": LEVEL_COLOR}


@app.get("/api/patterns")
def patterns():
    us, india = data()
    h = india.copy()
    h["hour"] = h["date_time"].dt.hour
    h["dow"] = h["date_time"].dt.dayofweek
    gh = h.groupby("hour")["traffic_volume"].mean()
    gd = h.groupby("dow")["traffic_volume"].mean()
    piv = h.pivot_table(index="dow", columns="hour", values="traffic_volume", aggfunc="mean")
    w = h.groupby("weather_main")["traffic_volume"].mean().sort_values()
    return {
        "hourly": [{"hour": int(i), "vol": round(float(v))} for i, v in gh.items()],
        "weekday": [{"dow": int(i), "vol": round(float(v))} for i, v in gd.items()],
        "heatmap": {"dows": [int(i) for i in piv.index], "hours": [int(c) for c in piv.columns],
                    "values": [[round(float(x)) for x in row] for row in piv.values.tolist()]},
        "weather": [{"w": str(i), "vol": round(float(v))} for i, v in w.items()],
        "peaks": {"am": round(float(gh.loc[7:10].mean())), "pm": round(float(gh.loc[17:21].mean())),
                  "weekday": round(float(gd.loc[0:4].mean())), "weekend": round(float(gd.loc[5:6].mean()))},
    }


@app.get("/api/adaptation")
def adaptation(city: str = "Mumbai", route: str = "Andheri → BKC"):
    b = bundle()
    m = b.get("metrics", {})
    r = _route(city, route)
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather, flow = _live(city, r, at)
    p = _clean(predict_pipeline(at, weather, flow, b, city=city.split(" ")[0]))
    coll = {"rows": 0, "locations": 0, "points": []}
    for _p in (os.path.join(BASE, "data", "indian_realtime.csv"),
               os.path.join(BASE, "indian_realtime.csv")):
        if os.path.exists(_p) and os.path.getsize(_p) > 0:
            try:
                import pandas as pd
                ind = pd.read_csv(_p)
                live_rows = ind[ind["traffic_source"].str.startswith("live:", na=False)] \
                    if "traffic_source" in ind.columns else ind
                coll = {"rows": int(len(live_rows)),
                        "locations": int(live_rows["location"].nunique()) if len(live_rows) else 0,
                        "points": [{"t": str(t), "loc": str(l), "ratio": float(x)}
                                   for t, l, x in zip(live_rows["timestamp"], live_rows["location"],
                                                      live_rows["congestion_ratio"])][-120:]}
            except Exception:
                pass
            break
    return {"metrics": m, "thresholds": [float(x) for x in b["thresholds"]],
            "stages": [{"stage": "Base (US)", "volume": p["base_vol"], "level": p["base_level"]},
                       {"stage": "Adapted (India)", "volume": p["adapted_vol"], "level": p["adapted_level"]},
                       {"stage": "Final (live fusion)", "volume": p["final_vol"], "level": p["final_level"]}],
            "delta": p["adapted_vol"] - p["base_vol"], "collector": coll}


@app.get("/api/compare")
def compare(city: str = "Mumbai", a: str = "Andheri → BKC", b: str = "Dadar → Sion"):
    bb = bundle()
    at = dt.datetime.now().replace(second=0, microsecond=0)
    ra, rb = _route(city, a), _route(city, b)
    weather, _ = _live(city, ra, at)
    res = []
    for r in (ra, rb):
        fl = fetch_tomtom_flow(r["lat"], r["lon"], r["free_flow"], tt_key() or None, 0.4)
        p = predict_pipeline(at, weather, fl, bb, city=city.split(" ")[0])
        tmin = round(float(r.get("length_km", 5.0)) / max(fl["current_speed"], 1) * 60, 1)
        res.append({"route": r["name"], "level": p["final_level"], "vol": p["final_vol"],
                    "speed": fl["current_speed"], "trip_min": tmin})
    best = min(res, key=lambda x: x["trip_min"])
    return {"city": city, "results": res, "best": best["route"], "colors": LEVEL_COLOR}


@app.get("/api/whatif")
def whatif(hour: int = 18, dow: int = 2, month: int = 10, weekend: int = 0, holiday: int = 0,
           temp_c: float = 30.0, rain: float = 0.0, snow: float = 0.0, clouds: int = 20,
           weather: str = "Clear"):
    """Interactive ML simulator: arbitrary scenario -> base prediction + level + importances."""
    import numpy as np
    from pipeline import classify as _classify
    from schema import FEATURES, WEATHER_CODE
    b = bundle()
    at = dt.datetime(2026, max(1, min(12, month)), 15, max(0, min(23, hour)), 0)
    at = at.replace(year=2026)
    # weekday override: shift date to matching weekday (approx via day offset)
    at = at + dt.timedelta(days=(max(0, min(6, dow)) - at.weekday()) % 7)
    w = {"temp": float(temp_c) + 273.15, "rain_1h": float(rain), "snow_1h": float(snow),
         "clouds_all": int(clouds), "weather_main": weather}
    from schema import featurize_point
    X, row = featurize_point(at, w)
    row["is_weekend"] = int(weekend)
    row["is_holiday"] = int(holiday)
    X = np.array([[row[f] for f in FEATURES]])
    vol = float(b["model"].predict(X)[0])
    imp = b.get("importances", {})
    order = sorted(imp, key=lambda k: -imp[k]) if imp else list(FEATURES)
    return {"volume": int(round(vol)), "level": _classify(vol, b["thresholds"]),
            "features": {k: row[k] for k in FEATURES},
            "importances": [{"feature": k, "importance": float(imp.get(k, 0))} for k in order],
            "prescription": ("Shift departure ~2h earlier (saves ~35%)" if vol > float(b["thresholds"][1])
                             else f"Waiting out rain saves ~{min(25, rain * 4):.0f}%" if rain > 3
                             else "Conditions optimal — depart now")}


@app.get("/api/diagnostics")
def diagnostics():
    b = bundle()
    m = b.get("metrics", {})
    imp = b.get("importances", {})
    calib = b.get("calibration")
    try:
        coef = [float(x) for x in calib.coef_] if calib is not None and hasattr(calib, "coef_") else []
        intercept = float(calib.intercept_) if calib is not None and hasattr(calib, "intercept_") else 0.0
    except Exception:
        coef, intercept = [], 0.0
    return {"metrics": m, "thresholds": [float(x) for x in b["thresholds"]],
            "importances": [{"feature": k, "importance": float(v)}
                            for k, v in sorted(imp.items(), key=lambda kv: -kv[1])],
            "calibration": {"coef": coef, "intercept": intercept,
                            "inputs": ["base_pred", "rain_1h", "hour", "is_peak"]},
            "backend": getattr(b.get("model"), "__class__", type("?", (), {})).__name__}


@app.get("/api/insight")
def insight(city: str = "Mumbai", route: str = "Andheri → BKC"):
    """Deterministic advisory text (Gemini-style template fallback, no key needed)."""
    b = bundle()
    r = _route(city, route)
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather, flow = _live(city, r, at)
    rows = [_clean(predict_pipeline(at, weather, flow, b, minutes_ahead=m, city=city.split(" ")[0]))
            for m in [15, 30, 60]]
    lvl, pct = rows[0]["final_level"], int(round(rows[0].get("congestion_ratio", 0.5) * 100))
    rain = float(weather.get("rain_1h", 0))
    rain_txt = f" and {rain}mm rain is adding ~{min(25, rain * 4):.0f}% pressure" if rain > 0 else ""
    nxt = next((x for x in rows if x["final_level"] in ("High", "Severe")), None)
    fwd = (f"expecting {nxt['final_level']} conditions within {nxt['minutes_ahead']} min" if nxt
           else "with flow expected to hold steady")
    text = (f"Traffic in {route}, {city} is presently {lvl} at {flow['current_speed']} km/h "
            f"(free-flow {flow['free_flow_speed']}), driven by peak-hour demand{rain_txt}, {fwd}. "
            f"Commuters should allow an extra 15–25 min or use alternate corridors.")
    return {"text": text, "source": "template", "level": lvl}


@app.get("/api/geocode")
def geocode(q: str = Query(..., min_length=2)):
    import requests
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"format": "json", "q": q, "limit": 5, "addressdetails": 1},
                         headers={"Accept": "application/json", "User-Agent": "traffic-forecaster/1.0"},
                         timeout=8)
        r.raise_for_status()
        out = []
        for g in r.json():
            disp = g.get("display_name", "")
            out.append({"name": ", ".join(disp.split(",")[:2]), "lat": float(g["lat"]),
                        "lon": float(g["lon"]),
                        "country": (g.get("address") or {}).get("country", "")})
        return {"results": out}
    except Exception as e:
        raise HTTPException(502, f"geocode failed: {e}")


@app.get("/api/tiles")
def tiles():
    if not tt_key():
        return {"url": None, "live": False}
    return {"url": "https://api.tomtom.com/traffic/map/4/tile/flow/relative0/{z}/{x}/{y}.png?key=" + tt_key(),
            "live": True}
