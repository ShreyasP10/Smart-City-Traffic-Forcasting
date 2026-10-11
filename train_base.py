# train_base.py — Hour 0:30-1:15, Historical Model Training (Track C plan).
# Uses REAL UCI Metro Interstate data (Metro_Interstate_Traffic_Volume.csv).
# Prefers XGBRegressor; falls back to sklearn HistGradientBoostingRegressor
# (same gradient-boosting family) if xgboost is not installed.
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

CSV_CANDIDATES = [
    "Metro_Interstate_Traffic_Volume.csv",
    "data/Metro_Interstate_Traffic_Volume.csv",
    "data/Metro_Interstate_Traffic_Data.csv",
]
CSV = next((c for c in CSV_CANDIDATES if os.path.exists(c)), CSV_CANDIDATES[0])

try:
    from xgboost import XGBRegressor
    BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor as XGBRegressor  # fallback
    BACKEND = "sklearn-hgb"

print(f"Loading {CSV} ...")
df = pd.read_csv(CSV)
df["date_time"] = pd.to_datetime(df["date_time"])
df["hour"] = df["date_time"].dt.hour
df["day_of_week"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_holiday"] = (df["holiday"] != "None").astype(int)

df["weather_main"] = df["weather_main"].fillna("Clear")
weather_map = {w: i for i, w in enumerate(sorted(df["weather_main"].unique()))}
df["weather_code"] = df["weather_main"].map(weather_map)

FEATURES = ["hour", "day_of_week", "month", "is_weekend", "is_holiday",
            "temp", "rain_1h", "snow_1h", "clouds_all", "weather_code"]

X = df[FEATURES]
y = df["traffic_volume"]

thresholds = np.percentile(y, [25, 50, 75])
print("Thresholds (Low/Moderate/High/Severe):", thresholds)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if BACKEND == "xgboost":
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                         n_jobs=-1, random_state=42)
else:
    model = XGBRegressor(max_iter=300, learning_rate=0.05, max_depth=6, random_state=42)
print(f"Training backend={BACKEND} on {len(X_train)} rows ...")
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  backend={BACKEND}")

joblib.dump({"model": model, "features": FEATURES, "thresholds": np.array(thresholds),
             "weather_map": weather_map, "backend": BACKEND,
             "metrics": {"mae": float(mae), "rmse": rmse, "n": int(len(df))}},
            "base_model.pkl")
print("Saved base_model.pkl")
