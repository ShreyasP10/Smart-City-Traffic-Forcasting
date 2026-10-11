"""Real-time traffic + weather integration.

TomTom (your plan enables all of these):
  - Traffic API — Flow Segment Data (live speeds):
      https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
      ?point={lat},{lon}&unit=KMPH&openLr=false&key={KEY}
  - Routing API — live travel time with traffic (best for Indian-city demo):
      https://api.tomtom.com/routing/1/calculateRoute/{lat},{lon}:{lat2},{lon2}/json
      ?routeType=fastest&traffic=true&travelMode=car&key={KEY}
  - Traffic Incidents (optional alerts):
      https://api.tomtom.com/traffic/services/5/incidentDetails
      ?bbox={w},{s},{e},{n}&fields={incidents{type,geometry{coordinates},properties{iconCategory,delay,magnitudeOfDelay}}}&key={KEY}
  - OpenWeather One Call 3.0:
      https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={KEY}&units=metric

IMPORTANT: the dashboard needs the actual API KEY VALUE (long string you copy
with the copy/eye icon), NOT the Key ID (eceb3134-...). Domain whitelist Off
is correct for localhost testing.

Without keys the module falls back to calibrated SIMULATION so the demo never
breaks. Every response flags `source: live|simulated`.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import requests

_WEATHER_MAIN_TO_CODE = {"Clear": 0, "Clouds": 1, "Mist": 2, "Fog": 2, "Drizzle": 3,
                         "Rain": 4, "Snow": 5, "Thunderstorm": 6}

_rng = np.random.default_rng(11)


def _open_meteo(lat: float, lon: float) -> dict | None:
    """Free no-key live weather (fallback when OpenWeather key inactive)."""
    try:
        url = ("https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,"
               "precipitation,weather_code,cloud_cover,wind_speed_10m&timezone=auto")
        r = requests.get(url, timeout=10)
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
        elif code >= 3:
            wmain = "Clouds"
        elif code == 2:
            wmain = "Mist"
        else:
            wmain = "Clear"
        return {
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
    """Return {temp, rain_1h, clouds_all, weather_main, humidity, wind, source}.

    Chain: One Call 3.0 -> 2.5/weather -> Open-Meteo (no key) -> simulation.
    New OWM keys often return 401 for ~30-120 min until activated.
    """
    api_key = (api_key or "").strip()
    if api_key:
        for label, url in [
            ("onecall3", "https://api.openweathermap.org/data/3.0/onecall"
                         f"?lat={lat}&lon={lon}&appid={api_key}&units=metric&exclude=minutely,daily,alerts"),
            ("2.5", f"https://api.openweathermap.org/data/2.5/weather"
                    f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"),
        ]:
            try:
                r = requests.get(url, timeout=8)
                r.raise_for_status()
                j = r.json()
                if label == "onecall3":
                    cur = j.get("current", {})
                    wmain = cur.get("weather", [{}])[0].get("main", "Clear")
                    return {
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
                        "temp": float(j.get("main", {}).get("temp", 30.0)) + 273.15,
                        "rain_1h": float((j.get("rain") or {}).get("1h", 0.0)),
                        "clouds_all": int(j.get("clouds", {}).get("all", 20)),
                        "weather_main": wmain if isinstance(wmain, str) else "Clear",
                        "humidity": float(j.get("main", {}).get("humidity", 60)),
                        "wind": float(j.get("wind", {}).get("speed", 3.0)),
                        "source": "live:openweather-2.5",
                    }
            except Exception as e:
                err = f"{label}: {e}"
                continue
    else:
        err = "no-key"
    # Free live fallback — keeps Weather Integration brownie point alive
    om = _open_meteo(lat, lon)
    if om:
        return om
    # ---- simulated weather: season-aware ----
    now = dt.datetime.now()
    monsoon = now.month in (6, 7, 8, 9)
    rain = float(_rng.exponential(2.0)) if _rng.random() < (0.3 if monsoon else 0.08) else 0.0
    clouds = int(np.clip(80 if rain > 0 else 25 + _rng.normal(0, 12), 0, 100))
    return {
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
    """Live speeds via Traffic API Flow Segment Data. Falls back to simulation.

    Return {current_speed, free_flow_speed, congestion_ratio, travel_time_min, source}.
    congestion_hint in [0,1]: expected congestion from the ML model (0=free, 1=jam).
    """
    api_key = (api_key or "").strip()
    # Guard: users often paste the Key ID (uuid) instead of the key value
    if api_key and len(api_key) < 20 and "-" in api_key:
        return {
            "current_speed": round(float(np.clip(free_flow * (1 - congestion_hint), 5, free_flow)), 1),
            "free_flow_speed": round(float(free_flow), 1),
            "congestion_ratio": round(float(np.clip(1 - congestion_hint, 0.05, 1)), 3),
            "travel_time_min": None,
            "source": "simulated (you pasted Key ID eceb3134-… — paste the KEY VALUE, not the ID)",
        }
    if api_key:
        try:
            url = ("https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
                   f"?point={lat}%2C{lon}&unit=KMPH&openLr=false&key={api_key}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            seg = r.json().get("flowSegmentData", {})
            cur = float(seg.get("currentSpeed", free_flow * (1 - congestion_hint)))
            ff = float(seg.get("freeFlowSpeed", free_flow))
            ratio = cur / max(ff, 1)
            return {
                "current_speed": round(cur, 1),
                "free_flow_speed": round(ff, 1),
                "congestion_ratio": round(ratio, 3),
                "travel_time_min": None,
                "source": "live:tomtom-traffic-api",
            }
        except Exception as e:
            err = f"tomtom-error: {e}"
        else:
            err = ""
    else:
        err = "no-key"
    cur = float(np.clip(free_flow * (1 - congestion_hint * _rng.uniform(0.85, 1.15)), 5, free_flow))
    return {
        "current_speed": round(cur, 1),
        "free_flow_speed": round(float(free_flow), 1),
        "congestion_ratio": round(cur / max(free_flow, 1), 3),
        "travel_time_min": None,
        "source": f"simulated ({err})",
    }


def fetch_tomtom_route(origin: tuple[float, float], dest: tuple[float, float],
                        api_key: str | None) -> dict:
    """Live travel time with traffic via Routing API (your plan includes it).

    Returns {live_min, free_min, delay_min, distance_km, source}.
    Falls back to simulated if no/invalid key.
    """
    api_key = (api_key or "").strip()
    if api_key and len(api_key) >= 20:
        try:
            o = f"{origin[0]},{origin[1]}"
            d = f"{dest[0]},{dest[1]}"
            url = (f"https://api.tomtom.com/routing/1/calculateRoute/{o}:{d}/json"
                   f"?routeType=fastest&traffic=true&travelMode=car&key={api_key}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            summ = r.json()["routes"][0]["summary"]
            live = summ["travelTimeInSeconds"] / 60
            free = summ.get("noTrafficTravelTimeInSeconds", summ["travelTimeInSeconds"]) / 60
            return {
                "live_min": round(live, 1),
                "free_min": round(free, 1),
                "delay_min": round(live - free, 1),
                "distance_km": round(summ.get("lengthInMeters", 0) / 1000, 2),
                "source": "live:tomtom-routing-api",
            }
        except Exception as e:
            err = f"routing-error: {e}"
        else:
            err = ""
    else:
        err = "no-key"
    return {"live_min": None, "free_min": None, "delay_min": None,
            "distance_km": None, "source": f"simulated ({err})"}


def fetch_tomtom_incidents(lat: float, lon: float, api_key: str | None,
                           radius_deg: float = 0.05) -> dict:
    """Optional: nearby incidents for the alerts banner (Traffic Incidents API)."""
    api_key = (api_key or "").strip()
    if api_key and len(api_key) >= 20:
        try:
            bbox = f"{lon-radius_deg},{lat-radius_deg},{lon+radius_deg},{lat+radius_deg}"
            url = ("https://api.tomtom.com/traffic/services/5/incidentDetails"
                   f"?bbox={bbox}&fields=%7Bincidents%7Btype%2Cproperties%7BiconCategory%2Cdelay%7D%7D%7D"
                   f"&key={api_key}")
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            inc = r.json().get("incidents", [])
            return {"count": len(inc), "incidents": inc[:5], "source": "live:tomtom-incidents"}
        except Exception as e:
            return {"count": 0, "incidents": [], "source": f"simulated (incidents-error: {e})"}
    return {"count": 0, "incidents": [], "source": "simulated (no-key)"}


def contribution_text(volume_pred: int, congestion_ratio: float, weather: dict) -> str:
    parts = [f"Historical pattern predicts ~{volume_pred:,} veh/hr for this time slot."]
    if congestion_ratio < 0.5:
        parts.append(f"Live traffic shows heavy slowdown (speed ratio {congestion_ratio:.2f}), pushing the forecast toward Severe.")
    elif congestion_ratio < 0.75:
        parts.append(f"Live traffic shows moderate slowdown (ratio {congestion_ratio:.2f}), nudging the forecast up one level.")
    else:
        parts.append(f"Live traffic is near free-flow (ratio {congestion_ratio:.2f}), so the historical forecast stands.")
    if weather.get("rain_1h", 0) > 0.5:
        parts.append(f"Rain ({weather['rain_1h']} mm/h) typically adds 10–20% congestion in Indian cities — factored in.")
    return " ".join(parts)
