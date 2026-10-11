"""Backend smoke check — ascii-safe output for cp1252 consoles."""
import json
import requests

b = "http://localhost:8000"
out = {}
out["health"] = requests.get(b + "/api/health", timeout=10).json()
o = requests.get(b + "/api/overview", timeout=90).json()
out["overview"] = {"level": o["pred"]["final_level"], "vol": o["pred"]["final_vol"],
                   "flow": o["flow"]["source"], "wx": o["weather"]["source"]}
orain = requests.get(b + "/api/overview", params={"rain": "true", "holiday": "true"}, timeout=90).json()
out["override"] = {"rain": orain["weather"]["rain_1h"],
                   "holiday_flag": orain["weather"].get("_holiday_override", False),
                   "level": orain["pred"]["final_level"]}
f = requests.get(b + "/api/forecast", timeout=90).json()
out["forecast"] = [(r["in_min"], r["final_level"], r["final_vol"]) for r in f["rows"]]
d = requests.get(b + "/api/diagnostics", timeout=60).json()
out["diag_backend"] = d["backend"]
out["diag_top_feature"] = d["importances"][0]
w = requests.get(b + "/api/whatif", params={"hour": 8, "rain": 9}, timeout=60).json()
out["whatif_rush_rain"] = {"vol": w["volume"], "level": w["level"], "rx": w["prescription"]}
with open("check.json", "w") as fh:
    json.dump(out, fh, default=str)
print("saved check.json")
