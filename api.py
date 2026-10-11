"""FastAPI backend for the React UI. Reuses the canonical pipeline verbatim.

Run:  uvicorn api:app --port 8000
React dev server proxies /api -> localhost:8000 (see frontend/vite.config.js).
"""
from __future__ import annotations

import datetime as dt
import json
import os

import joblib  # noqa: F401 (kept for safe_load_model internals)
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
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
_bundle_meta = {"custom": False, "trained_at": None, "source": "models/traffic_model.joblib"}


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
    city_data = CITIES.get(city) or CITIES.get("Mumbai") or list(CITIES.values())[0]
    for r in city_data["routes"]:
        if r["name"] == route:
            return r
    return city_data["routes"][0]


def _live(city: str, route: dict, at: dt.datetime, rain_override: bool = False,
          rain_mm: float = 0.0, holiday_override: bool = False):
    b = bundle()
    weather = fetch_openweather(route["lat"], route["lon"], ow_key() or None)
    if rain_override or rain_mm > 0:
        weather["rain_1h"] = max(float(weather.get("rain_1h", 0)), float(rain_mm) if rain_mm > 0 else 3.5)
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


def _causes(at, weather: dict, flow: dict, pred: dict, lat: float, lon: float) -> dict:
    """Why is there traffic here? Ranked causes: live incidents > weather >
    demand patterns > current bottleneck state. Always returns at least one entry."""
    from traffic_api import fetch_tomtom_incidents
    causes = []
    f = pred.get("features", {}) if isinstance(pred, dict) else {}
    level = pred.get("final_level", "?") if isinstance(pred, dict) else "?"
    # 1. Live reported incidents (needs TomTom key; silent skip otherwise)
    try:
        inc = fetch_tomtom_incidents(lat, lon, tt_key() or None)
        for g in (inc.get("incidents") or [])[:3]:
            delay_m = round(g.get("delay_s", 0) / 60, 1)
            causes.append({
                "kind": "incident", "severity": 3 if delay_m >= 5 else 2,
                "title": f"{g.get('category_name', 'Incident')} ~{g.get('distance_km', '?')} km away",
                "detail": (g.get("description") or "Reported on live traffic feed.")
                          + (f" Adding ~{delay_m} min." if delay_m > 0 else ""),
            })
        inc_source = inc.get("source", "")
    except Exception:
        inc_source = "unavailable"
    # 2. Weather as a cause
    rain = float(weather.get("rain_1h", 0) or 0)
    wmain = str(weather.get("weather_main", "Clear"))
    if rain >= 5 or wmain in ("Thunderstorm", "Snow"):
        causes.append({"kind": "weather", "severity": 2.5,
                       "title": f"Severe weather: {wmain} ({rain} mm/h)",
                       "detail": "Cuts grip and visibility; model adds 15–25% congestion pressure."})
    elif rain > 0.5 or wmain in ("Rain", "Drizzle", "Fog"):
        causes.append({"kind": "weather", "severity": 1.5,
                       "title": f"{wmain} ({rain} mm/h)",
                       "detail": "Wet-road friction slows flow ~10%."})
    # 3. Demand patterns (recurrent causes)
    if f.get("is_holiday"):
        causes.append({"kind": "demand", "severity": 1.5,
                       "title": "Holiday travel pattern",
                       "detail": "Lighter commute but event/leisure trips reshape peaks."})
    elif f.get("is_peak") and not f.get("is_weekend"):
        causes.append({"kind": "demand", "severity": 2,
                       "title": f"Peak-hour commuter demand ({f.get('hour')}:00 weekday)",
                       "detail": "Recurrent office commute surge — the strongest historical driver."})
    elif f.get("is_weekend"):
        causes.append({"kind": "demand", "severity": 1,
                       "title": "Weekend movement pattern",
                       "detail": "Shopping/leisure corridors replace commuter peaks (~40% lower)."})
    # 4. Current bottleneck state from live speed
    ratio = float(flow.get("congestion_ratio", 1) or 1)
    if ratio < 0.5:
        causes.append({"kind": "bottleneck", "severity": 2.5,
                       "title": f"Active bottleneck — {round(ratio * 100)}% of free-flow speed",
                       "detail": "Live TomTom speeds confirm standing queues right now."})
    elif ratio < 0.75:
        causes.append({"kind": "bottleneck", "severity": 1.5,
                       "title": f"Building pressure — {round(ratio * 100)}% of free-flow",
                       "detail": "Mild live slowdown on top of the historical pattern."})
    if level == "Low" and not causes:
        causes.append({"kind": "flow", "severity": 0,
                       "title": "Flowing freely — no active cause",
                       "detail": "Speeds near free-flow and no incidents, weather, or peak pressure."})
    causes.sort(key=lambda c: -c["severity"])
    return {"level": level, "causes": causes, "incidents_source": inc_source}


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


def _at_with(hour: int | None = None, dow: int | None = None) -> dt.datetime:
    """Target datetime: upcoming `dow` (Mon=0..Sun=6) at `hour`, else now."""
    at = dt.datetime.now().replace(second=0, microsecond=0)
    if dow is not None:
        at = at + dt.timedelta(days=(int(dow) % 7 - at.weekday()) % 7)
    if hour is not None:
        at = at.replace(hour=int(hour) % 24, minute=0)
    return at


@app.get("/api/overview")
def overview(city: str = "Mumbai", route: str = "Andheri → BKC", hour: int | None = None,
             dow: int | None = None,
             lat: float | None = None, lon: float | None = None,
             rain: bool = False, rain_mm: float = 0.0, holiday: bool = False):
    b = bundle()
    r = _custom_route(lat, lon) if lat is not None and lon is not None else _route(city, route)
    at = _at_with(hour, dow)
    weather, flow = _live(city, r, at, rain_override=rain, rain_mm=rain_mm, holiday_override=holiday)
    p = _clean(predict_pipeline(at, weather, flow, b, city=city.split(" ")[0]))
    causes = _causes(at, weather, flow, p, r["lat"], r["lon"])
    return {"at": at.isoformat(), "city": city, "route": r["name"],
            "weather": weather, "flow": flow, "pred": p, "causes": causes,
            "colors": LEVEL_COLOR, "shap": _shap(at, weather, flow, b, 0, None)}


@app.get("/api/forecast")
def forecast(city: str = "Mumbai", route: str = "Andheri → BKC",
             dow: int | None = None,
             lat: float | None = None, lon: float | None = None,
             rain: bool = False, rain_mm: float = 0.0, holiday: bool = False):
    b = bundle()
    r = _custom_route(lat, lon) if lat is not None and lon is not None else _route(city, route)
    at = _at_with(None, dow)
    weather, flow = _live(city, r, at, rain_override=rain, rain_mm=rain_mm, holiday_override=holiday)
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
    gwd = h[h["dow"] < 5].groupby("hour")["traffic_volume"].mean()
    gwe = h[h["dow"] >= 5].groupby("hour")["traffic_volume"].mean()
    piv = h.pivot_table(index="dow", columns="hour", values="traffic_volume", aggfunc="mean")
    w = h.groupby("weather_main")["traffic_volume"].mean().sort_values()
    return {
        "hourly": [{"hour": int(i), "vol": round(float(v))} for i, v in gh.items()],
        "hourly_weekday": [{"hour": int(i), "vol": round(float(v))} for i, v in gwd.items()],
        "hourly_weekend": [{"hour": int(i), "vol": round(float(v))} for i, v in gwe.items()],
        "weekday": [{"dow": int(i), "vol": round(float(v))} for i, v in gd.items()],
        "heatmap": {"dows": [int(i) for i in piv.index], "hours": [int(c) for c in piv.columns],
                    "values": [[round(float(x)) for x in row] for row in piv.values.tolist()]},
        "weather": [{"w": str(i), "vol": round(float(v))} for i, v in w.items()],
        "peaks": {"am": round(float(gh.loc[7:10].mean())), "pm": round(float(gh.loc[17:21].mean())),
                  "weekday": round(float(gd.loc[0:4].mean())), "weekend": round(float(gd.loc[5:6].mean()))},
    }


@app.get("/api/adaptation")
def adaptation(city: str = "Mumbai", route: str = "Andheri → BKC",
               lat: float | None = None, lon: float | None = None):
    b = bundle()
    m = b.get("metrics", {})
    r = _custom_route(lat, lon) if lat is not None and lon is not None else _route(city, route)
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
    src_label = "India" if m.get("base_source", "us-uci").startswith("india") else "US"
    return {"metrics": m, "thresholds": [float(x) for x in b["thresholds"]],
            "base_source": m.get("base_source", "us-uci"),
            "stages": [{"stage": f"Base ({src_label})", "volume": p["base_vol"], "level": p["base_level"]},
                       {"stage": "Adapted (live corridors)", "volume": p["adapted_vol"], "level": p["adapted_level"]},
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
    # Staged GBM-ensemble inference: base -> Ridge-adapted (marginal deltas for waterfall)
    adapted = vol
    calib = b.get("calibration")
    if calib is not None:
        try:
            adapted = float(calib.predict(np.array([[vol, row["rain_1h"], row["hour"], row["is_peak"]]]))[0])
        except Exception:
            pass
    th = b["thresholds"]
    imp = b.get("importances", {})
    order = sorted(imp, key=lambda k: -imp[k]) if imp else list(FEATURES)
    return {"volume": int(round(vol)), "level": _classify(vol, th),
            "adapted_volume": int(round(adapted)), "adapted_level": _classify(adapted, th),
            "calibration_delta": int(round(adapted - vol)),
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


@app.get("/api/model")
def model_info():
    b = bundle()
    m = b.get("metrics", {})
    return {"backend": type(b.get("model")).__name__, "metrics": m,
            "thresholds": [float(x) for x in b["thresholds"]],
            "custom": _bundle_meta["custom"], "trained_at": _bundle_meta["trained_at"],
            "source": _bundle_meta["source"]}


@app.post("/api/retrain")
async def retrain(file: UploadFile = File(...), num_trees: int = 60, max_depth: int = 6, lr: float = 0.05):
    """CSV upload retrain (UCI schema) -> swaps the live bundle in memory. Training may take ~1 min."""
    import io
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from schema import FEATURES, WEATHER_CODE, add_features
    from data_generator import load_historical

    if file is None or not hasattr(file, "read"):
        raise HTTPException(400, "upload a CSV file as 'file'")
    raw = await file.read()
    if len(raw) < 100:
        raise HTTPException(400, "CSV too small")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"unreadable CSV: {e}")
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {"volume": "traffic_volume", "count": "traffic_volume", "flow": "traffic_volume",
              "datetime": "date_time", "timestamp": "date_time",
              "temperature": "temp", "rain": "rain_1h", "snow": "snow_1h", "clouds": "clouds_all"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for need in ("traffic_volume", "date_time"):
        if need not in df.columns:
            raise HTTPException(400, f"CSV needs '{need}' column (got {list(df.columns)})")
    for col, default in (("holiday", "None"), ("temp", 290.0), ("rain_1h", 0.0),
                         ("snow_1h", 0.0), ("clouds_all", 20), ("weather_main", "Clear")):
        if col not in df.columns:
            df[col] = default
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    df = df.dropna(subset=["date_time", "traffic_volume"]).drop_duplicates("date_time")
    df = df.sort_values("date_time").reset_index(drop=True)
    if len(df) < 500:
        raise HTTPException(400, f"need >=500 rows, got {len(df)}")
    df = add_features(df)
    X, y = df[FEATURES], df["traffic_volume"]
    k = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:k], X.iloc[k:], y.iloc[:k], y.iloc[k:]
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=max(10, min(150, num_trees)),
                             max_depth=max(2, min(8, max_depth)),
                             learning_rate=float(lr), n_jobs=-1, random_state=42)
        backend = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=max(10, min(150, num_trees)), random_state=42)
        backend = "sklearn-hgb"
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = float(mean_absolute_error(yte, pred))
    rmse = float(mean_squared_error(yte, pred) ** 0.5)
    th = [float(v) for v in np.percentile(ytr, [25, 50, 75])]
    p95 = float(np.percentile(ytr, 95))
    # Refit India calibration head on synthetic corridor sample (same convention as train_model.py)
    ind = load_historical(indian_style=True).iloc[:336].copy()
    ind["date_time"] = pd.to_datetime(ind["date_time"])
    ind = add_features(ind)
    base_ind = model.predict(ind[FEATURES].to_numpy())
    cal = Ridge(alpha=1.0)
    cal.fit(np.column_stack([base_ind, ind["rain_1h"].to_numpy(),
                             ind["hour"].to_numpy(), ind["is_peak"].to_numpy()]),
            ind["traffic_volume"].to_numpy())
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xte.iloc[:500].to_numpy(), yte.iloc[:500].to_numpy(),
                                   n_repeats=3, random_state=42)
        imp = {f: float(v) for f, v in zip(FEATURES, r.importances_mean)}
    except Exception:
        imp = {f: 1.0 / len(FEATURES) for f in FEATURES}
    global _bundle
    _bundle = {"model": model, "calibration": cal, "thresholds": th, "features": FEATURES,
               "weather_code": WEATHER_CODE,
               "metrics": {"base_mae": mae, "base_rmse": rmse, "p95_volume": p95,
                           "n_us": int(len(df)), "n_india_calib": 336,
                           "calib_mae_adapted": float(mean_absolute_error(
                               ind["traffic_volume"].to_numpy(), cal.predict(np.column_stack(
                                   [base_ind, ind["rain_1h"].to_numpy(), ind["hour"].to_numpy(),
                                    ind["is_peak"].to_numpy()])))),
                           "calib_mae_unadapted": float(mean_absolute_error(
                               ind["traffic_volume"].to_numpy(), base_ind))},
               "importances": imp}
    _bundle_meta.update({"custom": True, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
                         "source": f"upload:{file.filename}"})
    return {"ok": True, "backend": backend, "metrics": _bundle["metrics"], "thresholds": th,
            "custom": True}


@app.get("/api/corridor")
def corridor(olat: float, olon: float, dlat: float, dlon: float, alts: int = 2):
    """Free source->destination routing with selectable path options.
    TomTom live routes (maxAlternatives) if key, else synthesized fastest/shortest options.
    Midpoint congestion prediction + advice shared across options (same endpoints)."""
    import math
    from traffic_api import fetch_osrm_route, fetch_tomtom_routes
    b = bundle()
    options, live = [], False
    if tt_key():
        try:
            for i, rt in enumerate(fetch_tomtom_routes((olat, olon), (dlat, dlon), tt_key(), alts)):
                rt["id"] = i
                rt["label"] = ["Fastest", "Alternative 1", "Alternative 2"][min(i, 2)]
                options.append(rt)
            live = bool(options)
        except Exception:
            options = []
    if not options:
        # No TomTom key: free OSRM road geometry (fastest) + synthesized shorter option
        geom, dist, mins = [], 0.0, 0.0
        try:
            o = fetch_osrm_route((olat, olon), (dlat, dlon))
            if o.get("ok"):
                geom, dist, mins = o["geometry"], o["distance_km"], o["live_min"]
        except Exception:
            pass
        if not geom:
            R = 6371.0
            p1, p2 = math.radians(olat), math.radians(dlat)
            hh = (math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2)
                  * math.sin(math.radians(dlon - olon) / 2) ** 2)
            dist = 2 * R * math.asin(math.sqrt(hh))
            mins = dist / 30 * 60
            geom = [[olat, olon], [dlat, dlon]]
        free_min = dist / 50 * 60
        options = [
            {"id": 0, "label": "Fastest", "dist_km": round(dist, 2),
             "live_min": round(mins, 1), "free_min": round(free_min, 1),
             "delay_min": round(mins - free_min, 1), "geometry": geom,
             "source": "live:osrm-road-geometry"},
            {"id": 1, "label": "Shortest", "dist_km": round(dist * 0.92, 2),
             "live_min": round(dist * 0.92 / 26 * 60, 1), "free_min": round(dist * 0.92 / 45 * 60, 1),
             "delay_min": round(dist * 0.92 / 26 * 60 - dist * 0.92 / 45 * 60, 1),
             "geometry": None, "source": "estimated"},
        ]
    route_info = options[0]
    mlat, mlon = (olat + dlat) / 2, (olon + dlon) / 2
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather = fetch_openweather(mlat, mlon, ow_key() or None)
    flow = fetch_tomtom_flow(mlat, mlon, 50, tt_key() or None, 0.4)
    p = _clean(predict_pipeline(at, weather, flow, b, city="Mumbai"))
    lvl = p["final_level"]
    advice = ("Leave ~2h later or split the trip — severe mid-corridor load." if lvl == "Severe"
              else "Expect slow patches — allow +15 min buffer." if lvl == "High"
              else "Corridor flowing — good time to depart.")
    return {"origin": [olat, olon], "dest": [dlat, dlon], "mid": [mlat, mlon],
            "route": route_info, "options": options, "live_routing": live, "pred": p,
            "causes": _causes(at, weather, flow, p, mlat, mlon),
            "weather": weather, "flow": flow, "advice": advice, "colors": LEVEL_COLOR}


GRID_OFFSETS = [
    (-0.018, -0.018, "North-West Arterial"), (-0.018, 0.0, "North Radial Ring"),
    (-0.018, 0.018, "North-East Expressway"), (0.0, -0.018, "West Central Avenue"),
    (0.0, 0.0, "City Center Hub"), (0.0, 0.018, "East Business Corridor"),
    (0.018, -0.018, "South-West Terminal"), (0.018, 0.0, "South Coastal Link"),
    (0.018, 0.018, "South-East Bypass"),
]


@app.get("/api/grid")
def grid(city: str = "Mumbai", horizon: int = 0):
    """9-point telemetry grid around a city center for the given forecast horizon."""
    b = bundle()
    if city not in CITIES:
        raise HTTPException(404, "unknown city")
    clat, clon = CITIES[city]["center"]
    at = dt.datetime.now().replace(second=0, microsecond=0)
    weather, _ = _live(city, {"lat": clat, "lon": clon, "free_flow": 50, "length_km": 5.0}, at)
    out = []
    for i, (dla, dlo, name) in enumerate(GRID_OFFSETS):
        plat, plon = clat + dla, clon + dlo
        fl = fetch_tomtom_flow(plat, plon, 50 + (i % 3) * 5, tt_key() or None, 0.4)
        p = predict_pipeline(at, weather, fl, b, minutes_ahead=max(0, min(60, horizon)),
                             city=city.split(" ")[0])
        out.append({"name": name, "lat": plat, "lon": plon, "level": p["final_level"],
                    "vol": p["final_vol"], "speed": fl["current_speed"],
                    "free": fl["free_flow_speed"], "source": fl["source"]})
    return {"city": city, "horizon": horizon, "points": out, "colors": LEVEL_COLOR}


@app.get("/api/routes")
def city_routes(city: str = "Mumbai", horizon: int = 0):
    """Every corridor as a road-aligned route: OSRM geometry + live prediction +
    per-route stats (level, volume, speed, trip time, distance)."""
    from concurrent.futures import ThreadPoolExecutor
    from traffic_api import fetch_osrm_route
    b = bundle()
    if city not in CITIES:
        raise HTTPException(404, "unknown city")
    at = dt.datetime.now().replace(second=0, microsecond=0)
    c0 = CITIES[city]["routes"][0]
    weather, _ = _live(city, c0, at)

    def one(r: dict) -> dict:
        import math
        span = float(r.get("length_km", 5.0)) / 111.0  # km -> degrees (approx)
        dlat, dlon = r["lat"] + span * 0.6, r["lon"] + span * 0.8
        geom, dist, src = None, float(r.get("length_km", 5.0)), "estimated"
        try:
            o = fetch_osrm_route((r["lat"], r["lon"]), (dlat, dlon))
            if o.get("ok") and o.get("geometry"):
                geom, dist, src = o["geometry"], o["distance_km"], o["source"]
        except Exception:
            pass
        if not geom:
            geom = [[r["lat"], r["lon"]], [dlat, dlon]]
        fl = fetch_tomtom_flow(r["lat"], r["lon"], r["free_flow"], tt_key() or None, 0.4)
        p = predict_pipeline(at, weather, fl, b, minutes_ahead=max(0, min(60, horizon)),
                             city=city.split(" ")[0])
        tmin = round(dist / max(fl["current_speed"], 1) * 60, 1)
        return {"route": r["name"], "level": p["final_level"], "vol": p["final_vol"],
                "speed": fl["current_speed"], "free": fl["free_flow_speed"],
                "dist_km": round(dist, 2), "trip_min": tmin,
                "delay_min": round(tmin - dist / max(fl["free_flow_speed"], 1) * 60, 1),
                "geometry": geom, "geo_source": src, "flow_source": fl["source"]}

    with ThreadPoolExecutor(max_workers=5) as ex:
        out = list(ex.map(one, CITIES[city]["routes"]))
    return {"city": city, "horizon": horizon, "routes": out, "colors": LEVEL_COLOR}
