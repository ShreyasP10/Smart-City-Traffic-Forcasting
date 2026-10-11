# Smart City Real-Time Traffic Congestion Forecasting & Adaptation

TE/BE prototype for the Smart City PS: an ML system that **learns traffic patterns from
historical data** (time, day, weather, holidays), **fuses live TomTom + weather APIs**,
**adapts US-trained patterns to Indian cities** via calibration (transfer learning), and
serves everything through an **interpretable 6-tab Streamlit dashboard** with
Low / Moderate / High / Severe congestion levels.

Last updated: 2026-10-11 ~11:20 — detailed rewrite. Collector at 12 rows / 3 locs and growing.

---

## 1. Quickstart (Windows, two terminals)

Terminal 1 — train + verify + dashboard:
```powershell
pip install -r requirements.txt
python train_base.py        # real UCI (48,204 rows) -> base_model.pkl | XGB MAE 252.5, RMSE 436.9
python train_model.py       # base + Ridge calibration -> models/traffic_model.joblib | transfer 2220 -> 595
python sanity_check.py      # expect: Predicted: 5649.82 -> Severe
python predict.py           # expect: 4 live lines T+0/15/30/60 (currently Moderate, Sat off-peak)
python test_tomtom.py <KEY># optional key check: flow + routing + incidents, all live
streamlit run app.py        # dashboard http://localhost:8501 (keys pre-filled)
```

Terminal 2 — leave running for the full 5 hours:
```powershell
python collector.py           # every 15 min -> data/indian_realtime.csv (~20 rows/location by hour 5)
python collector.py --once    # single round (3 rows), for testing
```

No keys? The app still runs fully offline in calibrated simulation mode
(every response is tagged `source: live:*|simulated`, so judges see what is real).

## 2. API keys (current live status)

| API | Status | Detail |
|---|---|---|
| TomTom Traffic Flow | ✅ live | `live:tomtom-traffic-api` — Andheri 22–32 km/h verified |
| TomTom Routing (traffic) | ✅ live | `live:tomtom-routing-api` — Andheri→BKC 24.7 min / 11.57 km |
| TomTom Incidents | ✅ live | `live:tomtom-incidents` — 17 near BKC at test time |
| OpenWeather 2.5 | ✅ live | `live:openweather-2.5` (key activated ~10:50; was 401 before) |
| OpenWeather One Call 3.0 | ⚠️ needs subscription | chain tries 3.0 → 2.5 → Open-Meteo → simulation |
| Open-Meteo (no key) | ✅ live fallback | keeps weather live even if OWM lapses |

- In TomTom portal copy the **KEY VALUE** (eye/copy icon), NOT the Key ID (`eceb3134-…`).
  `python test_tomtom.py` rejects a pasted Key ID and tells you so.
- Keys live in `.streamlit/secrets.toml` (pre-fills the sidebar). `Domain whitelist: Off`
  is correct for localhost. **Do NOT commit that file publicly.**

## 3. How each PS requirement is met

| # | PS requirement | Implementation | Evidence |
|---|---|---|---|
| 1 | Pattern learning (time/day/weather/holidays/volume) | `train_base.py`: 10 features (hour, dow, month, weekend, holiday, temp, rain, snow, clouds, weather_code) on real UCI | MAE 252.5 (~8% of mean 3259); Patterns tab curves |
| 2 | Forecast current + near-future (15/30/60) | `predict.py:predict_now(lat,lon,offset)` + Forecast tab table + line | 4 live lines on every run |
| 3 | Real-time traffic API in the pipeline | `traffic_api.fetch_tomtom_flow()` (flowSegmentData) + Routing for trip times; fusion `0.65*adapted + 0.35*live-implied` | `live:*` badges; contribution note in Overview |
| 4 | Context adaptation to Indian city | Ridge stacking head `[base_pred, rain, hour, is_peak]` → India scale; Adaptation tab base-vs-adapted + 🇮🇳 live-CSV section | 2220 → 595 MAE (73% drop) |
| 5 | Interpretability | `explain_shap.py`: SHAP TreeExplainer, % contributions (e.g. "Heavy rain 40%, Rush hour 35%"), Plotly Waterfall + Factor Influence + Donut charts in Overview & Forecast | Overview + Forecast "Why this prediction?" |
| 6 | Interactive dashboard | `app.py`: 6 tabs, city/route/custom-point + time + horizon controls | serves :8501/:8502 verified |
| 7 | Classification Low/Moderate/High/Severe | Data-driven quartiles [1192, 3371, 4932] from training volumes | badges + map colors everywhere |
| Brownie | Traffic heatmap | Heatmap tab: `px.scatter_map` (OSM, no token), size ∝ volume | green→red corridors |
| Brownie | Short-term forecasting visualised | Forecast tab line (final/adapted/base) + horizon slider + T+horizon badge | varies at hour boundaries |
| Brownie | Cross-context prediction | Adaptation tab side-by-side base/adapted/final + delta | e.g. +delta veh/hr shown live |
| Brownie | Transfer/calibration | Two-stage stacking, needs only ~2 weeks of polling | metrics + method caption |
| Brownie | Weather integration | `fetch_openweather()` chain; rain/clouds as features + SHAP-visible | live temp/rain in Overview |
| Brownie | Congestion alerts | Red banner when Severe now or in any horizon, with causes | trigger via 18:00 preset |
| Brownie | Route comparison | Route-compare tab: A vs B metrics + ✅ recommendation + bar | trip-min comparison |
| Brownie | Pattern insights | Patterns tab: hour curve, weekday bar, hour×day heatmap, weather bar, peak metrics | AM/PM peak numbers |

## 4. Architecture & data flow

```
data/Metro_Interstate_Traffic_Volume.csv (UCI, 48,204 rows, 2012-2018)
        │  train_base.py (XGBRegressor n=300/d=6/lr=0.05)
        ▼
base_model.pkl {model, features[10], thresholds[1193,3380,4933], weather_map, MAE 252.5}
        │  train_model.py (+ Ridge on 336-row India sample)
        ▼
models/traffic_model.joblib {model, calibration, thresholds, importances, metrics}
        │  app.py / predict.py
        ├─ live: TomTom flow + routing + incidents ─┐
        ├─ live: OWM 2.5 → Open-Meteo → sim ────────┤ fusion
        ▼                                          ▼
final = 0.65 * Ridge(base_pred, rain, hour, peak) + 0.35 * (1 - speed_ratio) * 5500
        → quartile bin → Low/Moderate/High/Severe → dashboard + alerts
        → collector.py (15 min, 3 cities) → data/indian_realtime.csv → Adaptation proof
        → last_response.json (last live snapshot; demo fallback on network failure)
```

Feature list (10): hour, day_of_week, month, is_weekend, is_holiday, temp (Kelvin),
rain_1h, snow_1h, clouds_all, weather_code (+ is_peak/is_night in dashboard bundle).
Calibration head (4): base_pred, rain_1h, hour, is_peak.

## 5. Honest performance audit (2026-10-11 ~11:15)

- **Base model: good ✅.** MAE 252.5 / mean 3259 ≈ 8%; sanity 8am-Mon → Severe, 3am-Sun → Low.
- **Live integration: working ✅, lightly stressed.** All sources live; but all 12 collector
  ratios read 1.00 (Saturday-morning free-flow) — the congested path is exercised via the
  18:00 preset, not yet observed live.
- **Adaptation: directionally proven, absolutely modest ⚠️.** 73% relative drop, but adapted
  MAE 595 vs India-sample mean ~1100 (~54%). Calibrator trains on *synthetic* India data and
  no ground-truth Indian volumes exist (speed→volume is heuristic). Fix: keep the collector
  through weekday peaks for ratio variance, then refit Ridge on real collected speeds.

## 6. Files (what each does)

- `app.py` — 6-tab dashboard (Overview / Forecast / Heatmap / Patterns / Adaptation / Route compare),
  horizon slider, T+horizon badge, full SHAP suite (waterfall, factor influence, donut), `last_response.json` weather fallback.
- `explain_shap.py` — SHAP TreeExplainer module: semantic factor grouping (Rush hour, Rain, Commute, Road slowdown),
  exact percentage contributions ("Heavy rain 40%, Rush hour 35%"), and Plotly waterfall, factor bars, and uplift donut charts.
- `predict.py` — spec pipeline: `classify/fetch_live/build_features/predict_now`, horizons 0/15/30/60,
  Kelvin normalisation, SHAP "Why" explanations printed, live caching to `last_response.json`.
- `train_base.py` — Hour 0:30–1:30 training (XGB preferred, sklearn-HGB fallback) → `base_model.pkl`.
- `train_model.py` — base + Ridge calibration bundle → `models/traffic_model.joblib` + `metrics.json`.
- `collector.py` — Track C background collector, keys env→secrets→.env, `--once/--interval/--limit`.
- `traffic_api.py` — TomTom flow/routing/incidents + OWM chain + Open-Meteo + simulation; all tagged.
- `data_generator.py` — UCI loader (both filenames) + synthetic US/India fallback generator.
- `city_config.py` — Mumbai (5 routes), Bengaluru (4), Delhi NCR (4), Pune (3), Minneapolis source.
- `sanity_check.py` / `test_tomtom.py` — one-command checkpoints.
- `codebase.md` — Complete single-file codebase bundle containing all project code and configs.
- `base_model.pkl` / `models/traffic_model.joblib` — trained artifacts (don't delete).
- `data/` — UCI CSVs (both names), `synthetic_*.csv`, `indian_realtime.csv` (12 rows, growing).
- `last_response.json` — last live snapshot (regenerated on live calls; demo safety net).
- `.streamlit/secrets.toml` — local keys only, never commit.
- `timeline.md` — timestamped build log, updated every step.
- `issues_log.md` — 19 solved issues + verification log + open risks (judges' Q&A shield).

## 7. Dashboard tour (2-minute demo)

1. **Overview:** Mumbai → Andheri–BKC. Point at live vs free speed, ratio, `live:*` sources,
   prediction badge + T+horizon badge, fusion note ("how real-time contributes"), factor bullets.
2. **Forecast:** 0/15/30/60 table + line; move the horizon slider, watch the badge follow.
3. **Heatmap:** all city corridors green→red, size ∝ volume (OSM tiles, no token).
4. **Patterns:** double rush-hour peaks, weekday/weekend gap, hour×day heatmap, weather effect.
5. **Adaptation:** 2220→595 transfer numbers, base/adapted/final table, 🇮🇳 collector count + chart.
6. **Route compare:** A vs B → ✅ recommendation with trip minutes.
7. **Alert:** set hour 18:00 → Severe banner with causes + alternate-route advice.
Fallbacks to narrate: keys empty → simulation; network dead → `last_response.json` cache banner.

## 8. Operations & troubleshooting (from `issues_log.md`)

- Key pasted but "simulated (you pasted Key ID…)" → you pasted the ID; copy the VALUE.
- OWM 401 → new key / no One Call subscription; 2.5 + Open-Meteo cover you automatically.
- Flat horizons mid-hour → normal (same-hour features); cross an hour boundary or use 18:00.
- `indian_realtime.csv` missing/empty warning → start `python collector.py` in terminal 2.
- Port busy (`:8501` taken) → `streamlit run app.py --server.port 8502`.
- `use_container_width` / ScriptRunContext warnings → cosmetic, ignore.
