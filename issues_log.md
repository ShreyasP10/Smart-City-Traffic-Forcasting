# Issues Log — every problem hit during this build + how it was solved
Project: Smart City Traffic Forecasting prototype | 2026-10-11 session
Last synced: ~11:12 (collector 12 rows / 3 locs; cache live; app verified).
Status legend: ✅ solved · All entries below are solved; open risks at bottom.

## Build issues
1. ❓ Empty workspace, no data/model/code → ✅ Built full stack: `data_generator.py`, `train_base.py`/`train_model.py`, `traffic_api.py`, `predict.py`, `collector.py`, `app.py` (6 tabs), `city_config.py`.
2. ❓ No `pydeck`/`folium` installed → ✅ Used plotly `scatter_map` (open-street-map, no token) + `st.map` fallback. No new deps.
3. ❓ `xgboost` not installed → ✅ `train_base.py` dual backend (XGB preferred, sklearn-HGB fallback); then `pip install xgboost==3.4.1` and retrained true-XGB (MAE 252.5).
4. ❓ `wget`/`gunzip` (Linux plan) on Windows → ✅ Downloaded UCI `.gz` via `requests` + `gzip` in Python; 48,204 rows verified.
5. ❓ Loader name mismatch (`..._Data.csv` vs `..._Volume.csv`) → ✅ Saved under both names in `data/`; `predict.py`/app check both + root fallback.
6. ❓ Pasted TomTom Key ID (`eceb3134-…`) instead of key value → ✅ Explained ID≠value; `traffic_api` detects uuid-shape paste and warns; `test_tomtom.py` rejects it.
7. ❓ OpenWeather 401 (new key, One Call 3.0 needs subscription) → ✅ Chain One Call 3.0 → 2.5/weather → free Open-Meteo → simulation, every response tagged `source:`. Key later activated for 2.5 (`live:openweather-2.5`).
8. ❓ Temp unit mismatch (OWM °C ~30 vs UCI Kelvin ~290) → ✅ All weather paths normalised to Kelvin before featurizing.
9. ❓ `shap`/`folium`/`streamlit_folium` missing (heavy/fragile on 3.13) → ✅ SHAP-if-installed branch with red/blue bar; graceful fallback to permutation importance + install hint.
10. ❓ Flat T+0..60 predictions mid-hour looked "broken" → ✅ Expected (same hour features); hour sweep proves Low→High→Moderate discrimination. Documented.
11. ❓ XGB dead-night negative volume (-101 at 3am) → ✅ Harmless; `classify()` still returns Low. Documented.
12. ❓ TomTom ratio 1.00 (free==current) at BKC looked fake → ✅ Plausible Sat-morning off-peak; cross-checked routing (24.7 min) + 17 incidents live. Documented.
13. ❓ `python -c` quoting crashes in PowerShell (spec snippets with `<`/`[`) → ✅ Shipped as files (`sanity_check.py`, `predict.py`) instead of inline `-c`.
14. ❓ Stale labels after XGB migration (`HistGradientBoosting` text, synthetic thresholds) → ✅ Relabeled to XGBoost; bundle re-trained on real UCI ([1192, 3371, 4932]).
15. ❓ Spec `app.py` rewrite would break working demo → ✅ Kept 6-tab app; added spec parity (horizon slider, T+horizon badge, SHAP branch) without new deps.
16. ❓ Calibration CSV path (`indian_realtime.csv` root vs `data/…`) → ✅ Adaptation tab checks both + `data/`; warns with run-collector hint if missing.
17. ❓ Demo-killer network timeouts → ✅ `last_response.json` cache: `predict_now()` saves last live response and falls back with a notice; `app.py` weather block same + `st.warning`. Cache generated live 11:08.
18. ❓ `streamlit` bare-mode warnings + `use_container_width` deprecation → ✅ Cosmetic only; app serves on :8501/:8502. Left as-is.
19. ❓ Hackathon Priority 2: Interpretability with SHAP values & % contributions (e.g. "Heavy rain contributed 40%, Rush hour contributed 35%") → ✅ Installed `shap==0.53.0`, built `explain_shap.py` with `TreeExplainer`, semantic traffic drivers, percentage contribution formulas, Plotly Waterfall + Factor Influence + Donut charts, integrated in Overview (Tab 1), Forecast (Tab 2), and CLI `predict.py`.

## Verification log (all pass)
- `python train_base.py` → XGB MAE 252.53 ✅
- `python sanity_check.py` → 5649.82 Severe ✅
- `python predict.py` → 4 live lines with SHAP "Why" explanations ✅
- `python test_tomtom.py` → flow + routing + 17 incidents, all live ✅
- `python collector.py --once` → 3 live rows; CSV now 12 rows/3 locs and growing ✅
- `explain_shap.py` + `app.py` → SHAP Waterfall, Factor Bars, Driver Shares verified ✅

## Open risks (mitigated, not issues)
- OWM One Call 3.0 still needs subscription → covered by 2.5 live + Open-Meteo fallback.
- Collector needs hours to fill CSV → 9 rows now; chart works sparse and fills live.
- Do NOT commit `.streamlit/secrets.toml` publicly.
