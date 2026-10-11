"""Track C — Indian real-time data collector (TomTom + weather, every 15 min).

Run in a SECOND terminal and leave it running:
    python collector.py                  # every 900s (15 min) forever
    python collector.py --once           # single round (3 rows) — for testing
    python collector.py --interval 120 --limit 5   # fast demo mode

Keys resolved in order: env (TOMTOM_API_KEY/TOMTOM_KEY) -> .streamlit/secrets.toml -> .env
Uses traffic_api.fetch_* so weather falls back to live Open-Meteo when your
OpenWeather key returns 401 (new keys need 30-120 min activation).
Output: data/indian_realtime.csv (appended, header written once).
By hour 5 at 15-min cadence: ~20 rows/location — enough to demo calibration.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from traffic_api import fetch_openweather, fetch_tomtom_flow

LOCATIONS = {
    "Mumbai_BKC": (19.0653, 72.8686),
    "Bengaluru_SilkBoard": (12.9172, 77.6229),
    "Delhi_Connaught": (28.6315, 77.2167),
}
FREE_FLOW = {"Mumbai_BKC": 50, "Bengaluru_SilkBoard": 45, "Delhi_Connaught": 55}
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "indian_realtime.csv")
FIELDNAMES = ["timestamp", "location", "lat", "lon", "current_speed", "free_flow_speed",
              "congestion_ratio", "temp_c", "humidity", "weather", "rain_1h",
              "traffic_source", "weather_source"]


try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_keys() -> tuple[str, str]:
    base = os.path.dirname(__file__)
    file_keys = {}
    sec_file = os.path.join(base, ".streamlit", "secrets.toml")
    if os.path.exists(sec_file):
        try:
            with open(sec_file, "rb") as f:
                file_keys.update(tomllib.load(f))
        except Exception:
            pass
    env_file = os.path.join(base, ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        file_keys[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass

    tt = (os.environ.get("TOMTOM_API_KEY") or os.environ.get("TOMTOM_KEY")
          or file_keys.get("TOMTOM_API_KEY") or file_keys.get("TOMTOM_KEY") or "")
    ow = (os.environ.get("OPENWEATHER_API_KEY") or os.environ.get("OPENWEATHER_KEY")
          or file_keys.get("OPENWEATHER_API_KEY") or file_keys.get("OPENWEATHER_KEY") or "")
    return str(tt).strip(), str(ow).strip()


def collect_once(tt_key: str, ow_key: str) -> list[dict]:
    rows = []
    for name, (lat, lon) in LOCATIONS.items():
        try:
            flow = fetch_tomtom_flow(lat, lon, FREE_FLOW[name], tt_key or None, 0.4)
            w = fetch_openweather(lat, lon, ow_key or None)
            rows.append({
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "location": name, "lat": lat, "lon": lon,
                "current_speed": flow["current_speed"],
                "free_flow_speed": flow["free_flow_speed"],
                "congestion_ratio": flow["congestion_ratio"],
                "temp_c": round(w["temp"] - 273.15, 1),
                "humidity": w.get("humidity", ""),
                "weather": w.get("weather_main", ""),
                "rain_1h": w.get("rain_1h", 0),
                "traffic_source": flow["source"],
                "weather_source": w["source"],
            })
            print(f"[{rows[-1]['timestamp']}] {name}: {flow['current_speed']} km/h "
                  f"ratio={flow['congestion_ratio']:.2f} {w['weather_main']} "
                  f"({flow['source']} / {w['source']})", flush=True)
        except Exception as e:
            print(f"Error {name}: {e}", flush=True)
    return rows


def append_rows(rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    exists = os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in FIELDNAMES})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tt, ow = load_keys()
    print(f"TomTom key: {'found' if tt else 'MISSING'} | "
          f"OpenWeather key: {'found' if ow else 'MISSING (Open-Meteo live fallback will be used)'}")
    print(f"CSV: {CSV_PATH} | interval={a.interval}s")
    n = 0
    while True:
        rows = collect_once(tt, ow)
        if rows:
            append_rows(rows)
            print(f"  -> appended {len(rows)} rows (total rounds: {n + 1})", flush=True)
        n += 1
        if a.once or (a.limit and n >= a.limit):
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
