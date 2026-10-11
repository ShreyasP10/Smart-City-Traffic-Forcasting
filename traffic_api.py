"""Real-time traffic + weather integration.

Hardened against:
- Secret credential leakage in exception strings and URL logs (_redact).
- SSRF via validate_coordinates and params dicts in requests.
- Division-by-zero via physical lower bounds.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import requests

_WEATHER_MAIN_TO_CODE = {"Clear": 0, "Clouds": 1, "Mist": 2, "Fog": 2, "Drizzle": 3,
                         "Rain": 4, "Snow": 5, "Thunderstorm": 6, "Haze": 2}

_rng = np.random.default_rng(11)


def _redact(e: Exception, *secrets: str) -> str:
    """Sanitizes exception messages to prevent leaking API keys through URL query strings."""
    s = f"{type(e).__name__}: {e}"
    for k in secrets:
        if k and len(k) > 4:
            s = s.replace(k, "***")
    return s[:200]


def validate_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """Validates and bounds geographic coordinates to prevent injection / SSRF."""
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        raise ValueError("Latitude and Longitude must be valid floats")
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        raise ValueError("Latitude must be between -90 and 90, Longitude between -180 and 180")
    return lat_f, lon_f


def _open_meteo(lat: float, lon: float) -> dict | None:
    """Free no-key live weather (fallback when OpenWeather key inactive)."""
    try:
        lat_f, lon_f = validate_coordinates(lat, lon)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat_f,
            "longitude": lon_f,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,cloud_cover,wind_speed_10m",
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        cur = r.json().get("current", {})
        code = int(cur.get("weather_code", 0))
        if code in (71, 73, 75, 77, 85, 86):
            wmain = "Snow"
        elif code in (95, 96, 99):
            wmain = "Thunderstorm"
        elif code in (51, 53, 55, 56, 57):
            wmain = "Drizzle"
        elif code in (61, 63, 65, 66, 67, 80, 81, 82):
            wmain = "Rain"
        elif code in (45, 48):
            wmain = "Fog"
        elif code in (1, 2, 3) or code > 3:
            wmain = "Clouds" if code >= 2 else "Clear"
        else:
            wmain = "Clear"
        return {
            "ok": True,
            "temp": float(cur.get("temperature_2m", 30.0)) + 273.15,
            "rain_1h": float(cur.get("precipitation", 0.0)),
            "clouds_all": int(cur.get("cloud_cover", 20)),
            "weather_main": wmain,
            "humidity": float(cur.get("relative_humidity_2m", 60)),
            "wind": float(cur.get("wind_speed_10m", 3.0)),
            "source": "live:open-meteo (no-key fallback)",
        }
    except Exception:
        return None


def fetch_openweather(lat: float, lon: float, api_key: str | None) -> dict:
    """Return {ok, temp, rain_1h, clouds_all, weather_main, humidity, wind, source}.

    Chain: One Call 3.0 -> 2.5/weather -> Open-Meteo (no key) -> simulation.
    """
    try:
        lat_f, lon_f = validate_coordinates(lat, lon)
    except ValueError as e:
        return {
            "ok": False,
            "temp": 300.0, "rain_1h": 0.0, "clouds_all": 20, "weather_main": "Clear",
            "humidity": 50.0, "wind": 3.0, "source": f"simulated (invalid-coordinates: {_redact(e)})"
        }

    api_key = (api_key or "").strip()
    err = "no-key"
    if api_key:
        for label, url, params in [
            ("onecall3", "https://api.openweathermap.org/data/3.0/onecall",
             {"lat": lat_f, "lon": lon_f, "appid": api_key, "units": "metric", "exclude": "minutely,daily,alerts"}),
            ("2.5", "https://api.openweathermap.org/data/2.5/weather",
             {"lat": lat_f, "lon": lon_f, "appid": api_key, "units": "metric"}),
        ]:
            try:
                r = requests.get(url, params=params, timeout=8)
                r.raise_for_status()
                j = r.json()
                if label == "onecall3":
                    cur = j.get("current", {})
                    wmain = cur.get("weather", [{}])[0].get("main", "Clear")
                    return {
                        "ok": True,
                        "temp": float(cur.get("temp", 30.0)) + 273.15,
                        "rain_1h": float(cur.get("rain", {}).get("1h", 0.0)),
                        "clouds_all": int(cur.get("clouds", 20)),
                        "weather_main": wmain if isinstance(wmain, str) else "Clear",
                        "humidity": float(cur.get("humidity", 60)),
                        "wind": float(cur.get("wind_speed", 3.0)),
                        "source": "live:openweather-onecall3",
                    }
                else:
                    wmain = j.get("weather", [{}])[0].get("main", "Clear")
                    return {
                        "ok": True,
                        "temp": float(j.get("main", {}).get("temp", 30.0)) + 273.15,
                        "rain_1h": float((j.get("rain") or {}).get("1h", 0.0)),
                        "clouds_all": int(j.get("clouds", {}).get("all", 20)),
                        "weather_main": wmain if isinstance(wmain, str) else "Clear",
                        "humidity": float(j.get("main", {}).get("humidity", 60)),
                        "wind": float(j.get("wind", {}).get("speed", 3.0)),
                        "source": "live:openweather-2.5",
                    }
            except Exception as e:
                err = f"{label}: {_redact(e, api_key)}"
                continue

    # Free live fallback
    om = _open_meteo(lat_f, lon_f)
    if om:
        return om

    # Seasonal simulation fallback
    now = dt.datetime.now()
    monsoon = now.month in (6, 7, 8, 9)
    rain = float(_rng.exponential(2.0)) if _rng.random() < (0.3 if monsoon else 0.08) else 0.0
    clouds = int(np.clip(80 if rain > 0 else 25 + _rng.normal(0, 12), 0, 100))
    return {
        "ok": False,
        "temp": float(300 + _rng.normal(0, 2)),
        "rain_1h": round(rain, 2),
        "clouds_all": clouds,
        "weather_main": "Rain" if rain > 0.5 else ("Clouds" if clouds > 60 else "Clear"),
        "humidity": float(70 if rain > 0 else 45),
        "wind": float(round(abs(_rng.normal(3, 1.5)), 1)),
        "source": f"simulated ({err})",
    }


def fetch_tomtom_flow(lat: float, lon: float, free_flow: float,
                      api_key: str | None, congestion_hint: float = 0.4) -> dict:
    """Live speeds via Traffic API Flow Segment Data with sanitized requests and redacted errors."""
    try:
        lat_f, lon_f = validate_coordinates(lat, lon)
    except ValueError as e:
        ff = max(float(free_flow or 45.0), 10.0)
        return {
            "ok": False,
            "current_speed": ff * 0.7, "free_flow_speed": ff,
            "congestion_ratio": 0.7, "travel_time_min": None,
            "source": f"simulated (invalid-coordinates: {_redact(e)})"
        }

    free_flow_safe = max(float(free_flow or 45.0), 10.0)
    api_key = (api_key or "").strip()

    if api_key and len(api_key) < 20 and "-" in api_key:
        cur = float(np.clip(free_flow_safe * (1 - congestion_hint), 5.0, free_flow_safe))
        return {
            "ok": False,
            "current_speed": round(cur, 1),
            "free_flow_speed": round(free_flow_safe, 1),
            "congestion_ratio": round(cur / free_flow_safe, 3),
            "travel_time_min": None,
            "source": "simulated (you pasted Key ID eceb3134-... paste the KEY VALUE, not ID)",
        }

    if api_key:
        try:
            url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
            params = {
                "point": f"{lat_f},{lon_f}",
                "unit": "KMPH",
                "openLr": "false",
                "key": api_key,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            seg = r.json().get("flowSegmentData", {})
            cur = float(seg.get("currentSpeed", free_flow_safe * (1 - congestion_hint)))
            ff = float(seg.get("freeFlowSpeed", free_flow_safe))
            ff_safe = max(ff, 10.0)
            cur_safe = max(cur, 5.0)
            ratio = cur_safe / ff_safe
            return {
                "ok": True,
                "current_speed": round(cur_safe, 1),
                "free_flow_speed": round(ff_safe, 1),
                "congestion_ratio": round(ratio, 3),
                "travel_time_min": None,
                "source": "live:tomtom-traffic-api",
            }
        except Exception as e:
            err = f"tomtom-error: {_redact(e, api_key)}"
    else:
        err = "no-key"

    cur = float(np.clip(free_flow_safe * (1 - congestion_hint * _rng.uniform(0.85, 1.15)), 5.0, free_flow_safe))
    return {
        "ok": False,
        "current_speed": round(cur, 1),
        "free_flow_speed": round(free_flow_safe, 1),
        "congestion_ratio": round(cur / free_flow_safe, 3),
        "travel_time_min": None,
        "source": f"simulated ({err})",
    }


def fetch_tomtom_route(origin: tuple[float, float], dest: tuple[float, float],
                       api_key: str | None) -> dict:
    """Live travel time with traffic via Routing API with sanitized coordinates and redacted errors."""
    try:
        o_lat, o_lon = validate_coordinates(origin[0], origin[1])
        d_lat, d_lon = validate_coordinates(dest[0], dest[1])
    except ValueError as e:
        return {"ok": False, "live_min": None, "free_min": None, "delay_min": None,
                "distance_km": None, "source": f"simulated (invalid-coordinates: {_redact(e)})"}

    api_key = (api_key or "").strip()
    if api_key and len(api_key) >= 20:
        try:
            url = f"https://api.tomtom.com/routing/1/calculateRoute/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
            params = {
                "routeType": "fastest",
                "traffic": "true",
                "travelMode": "car",
                "key": api_key,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            summ = r.json()["routes"][0]["summary"]
            live = summ["travelTimeInSeconds"] / 60
            free = summ.get("noTrafficTravelTimeInSeconds", summ["travelTimeInSeconds"]) / 60
            return {
                "ok": True,
                "live_min": round(live, 1),
                "free_min": round(free, 1),
                "delay_min": round(live - free, 1),
                "distance_km": round(summ.get("lengthInMeters", 0) / 1000, 2),
                "source": "live:tomtom-routing-api",
            }
        except Exception as e:
            err = f"routing-error: {_redact(e, api_key)}"
    else:
        err = "no-key"

    return {"ok": False, "live_min": None, "free_min": None, "delay_min": None,
            "distance_km": None, "source": f"simulated ({err})"}


def _leg_geometry(route: dict, cap: int = 300) -> list:
    """Road-following polyline from TomTom legs[].points[] -> [[lat, lon], ...]."""
    pts: list = []
    try:
        for leg in route.get("legs", []):
            for p in leg.get("points", []):
                pts.append([float(p["latitude"]), float(p["longitude"])])
    except Exception:
        return []
    if len(pts) > cap:
        step = len(pts) / cap
        pts = [pts[int(i * step)] for i in range(cap)]
    return pts


def fetch_osrm_route(origin: tuple[float, float], dest: tuple[float, float]) -> dict:
    """Free road geometry via OSRM demo server (no key). Returns geometry + summary."""
    try:
        o_lat, o_lon = validate_coordinates(origin[0], origin[1])
        d_lat, d_lon = validate_coordinates(dest[0], dest[1])
    except ValueError as e:
        return {"ok": False, "source": f"simulated (invalid-coordinates: {_redact(e)})"}
    try:
        url = (f"https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
               "?overview=full&geometries=geojson")
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        rt = r.json()["routes"][0]
        geom = [[c[1], c[0]] for c in rt["geometry"]["coordinates"]]
        dist = rt["distance"] / 1000
        mins = rt["duration"] / 60
        return {"ok": True, "geometry": geom, "distance_km": round(dist, 2),
                "live_min": round(mins, 1), "source": "live:osrm-road-geometry"}
    except Exception as e:
        return {"ok": False, "source": f"simulated (osrm-error: {_redact(e)})"}


def fetch_tomtom_routes(origin: tuple[float, float], dest: tuple[float, float],
                        api_key: str | None, max_alts: int = 2) -> list[dict]:
    """Live alternative routes with road geometry (TomTom maxAlternatives)."""
    try:
        o_lat, o_lon = validate_coordinates(origin[0], origin[1])
        d_lat, d_lon = validate_coordinates(dest[0], dest[1])
    except ValueError:
        return []
    api_key = (api_key or "").strip()
    if not api_key or len(api_key) < 20:
        return []
    try:
        url = f"https://api.tomtom.com/routing/1/calculateRoute/{o_lat},{o_lon}:{d_lat},{d_lon}/json"
        params = {"routeType": "fastest", "traffic": "true", "travelMode": "car",
                  "maxAlternatives": max(0, min(2, int(max_alts))), "key": api_key}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        out = []
        for rt in r.json().get("routes", [])[:3]:
            s = rt.get("summary", {})
            lv = s.get("travelTimeInSeconds", 0) / 60
            fr = s.get("noTrafficTravelTimeInSeconds", s.get("travelTimeInSeconds", 0)) / 60
            out.append({
                "live_min": round(lv, 1), "free_min": round(fr, 1),
                "delay_min": round(lv - fr, 1),
                "distance_km": round(s.get("lengthInMeters", 0) / 1000, 2),
                "geometry": _leg_geometry(rt),
                "source": "live:tomtom-routing-api",
            })
        return out
    except Exception:
        return []


def fetch_tomtom_incidents(lat: float, lon: float, api_key: str | None,
                           radius_deg: float = 0.05) -> dict:
    """Nearby incidents for alerts banner with bounded bbox coordinates."""
    try:
        lat_f, lon_f = validate_coordinates(lat, lon)
    except ValueError as e:
        return {"ok": False, "count": 0, "incidents": [], "source": f"simulated (invalid-coordinates: {_redact(e)})"}

    api_key = (api_key or "").strip()
    if api_key and len(api_key) >= 20:
        try:
            bbox = f"{lon_f - radius_deg},{lat_f - radius_deg},{lon_f + radius_deg},{lat_f + radius_deg}"
            url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
            params = {
                "bbox": bbox,
                "fields": "{incidents{type,properties{iconCategory,delay}}}",
                "key": api_key,
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            inc = r.json().get("incidents", [])
            return {"ok": True, "count": len(inc), "incidents": inc[:5], "source": "live:tomtom-incidents"}
        except Exception as e:
            return {"ok": False, "count": 0, "incidents": [], "source": f"simulated (incidents-error: {_redact(e, api_key)})"}

    return {"ok": False, "count": 0, "incidents": [], "source": "simulated (no-key)"}


def contribution_text(volume_pred: int, congestion_ratio: float, weather: dict) -> str:
    parts = [f"Historical pattern predicts ~{volume_pred:,} veh/hr for this time slot."]
    if congestion_ratio < 0.5:
        parts.append(f"Live traffic shows heavy slowdown (speed ratio {congestion_ratio:.2f}), pushing the forecast toward Severe.")
    elif congestion_ratio < 0.75:
        parts.append(f"Live traffic shows moderate slowdown (ratio {congestion_ratio:.2f}), nudging the forecast up one level.")
    else:
        parts.append(f"Live traffic is near free-flow (ratio {congestion_ratio:.2f}), so the historical forecast stands.")
    if weather.get("rain_1h", 0) > 0.5:
        parts.append(f"Rain ({weather['rain_1h']} mm/h) is factored into local road friction.")
    return " ".join(parts)
