"""Sanity check verification with safe model loading and canonical schema."""
import datetime as dt
import os
import json
from schema import safe_load_model, featurize_point

CHECKSUMS_FILE = os.path.join("models", "checksums.json")
expected_sha = None
if os.path.exists(CHECKSUMS_FILE):
    try:
        expected_sha = json.load(open(CHECKSUMS_FILE)).get("base_model.pkl")
    except Exception:
        pass

bundle = safe_load_model("base_model.pkl", expected_sha)
model, thresh = bundle["model"], bundle["thresholds"]


def classify(vol):
    if vol <= thresh[0]:
        return "Low"
    if vol <= thresh[1]:
        return "Moderate"
    if vol <= thresh[2]:
        return "High"
    return "Severe"


# Test a rush-hour weekday (Monday 8:00 AM)
sample, _ = featurize_point(dt.datetime(2026, 10, 12, 8, 0), {"weather_main": "Clear", "temp": 290.0, "clouds_all": 40})
pred_val = float(model.predict(sample)[0])
label = classify(pred_val)
print(f"Predicted: {pred_val:.2f} -> {label}")
assert label == "Severe", f"Expected Severe, got {label}"
print("Sanity check passed [OK]")
