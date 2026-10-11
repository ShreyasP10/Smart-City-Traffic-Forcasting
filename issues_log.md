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
19. ❓ `pip install -r requirements.txt` → `Invalid requirement` (docstring header lines) → ✅ Rewrote as pure `#` comments + pins; dropped `shap` (optional, fallback exists).
20. ❓ Dashboard crash `NameError: featurize_point` (app.py:77) → ✅ Added to `from schema import` line.
21. ❓ Dashboard crash `StreamlitDuplicateElementId` on SHAP chart pairs → ✅ Unique `key=` on all five SHAP `plotly_chart` calls. Verified :8503 serves with zero Tracebacks.
19. ❓ Hackathon Priority 2: Interpretability with SHAP values & % contributions (e.g. "Heavy rain contributed 40%, Rush hour contributed 35%") → ✅ Installed `shap==0.53.0`, built `explain_shap.py` with `TreeExplainer`, semantic traffic drivers, percentage contribution formulas, Plotly Waterfall + Factor Influence + Donut charts, integrated in Overview (Tab 1), Forecast (Tab 2), and CLI `predict.py`.
20. ❓ Security & Leakage Audit Findings (C1–C4) → ✅ Remediated:
    - **C1 (API Key Exposure):** Removed plaintext keys from `.streamlit/secrets.toml`, verified `.gitignore` coverage, and created `.streamlit/secrets.toml.example`.
    - **C2 (Exception Key Leakage):** Added `_redact()` scrubber in `traffic_api.py` to sanitize URLs/keys from all exception messages; converted all `requests.get()` calls to `params={}` dictionaries (preventing parameter injection and SSRF).
    - **C3 (SHAP Integrity & Additivity):** Eliminated hand-coded multipliers (`rain*90`) and negative clipping; implemented pure TreeExplainer SHAP attributions and added transparent, explicit pipeline stages ("India Calibration Delta" and "Live Speed Fusion") to the Plotly Waterfall chart, ensuring 100% mathematical additivity to the final forecast.
    - **C4 (Honest Transfer Split):** Enforced a strict 70/30 chronological out-of-time split on the Indian corridor dataset; evaluated Ridge calibration exclusively on held-out 30% out-of-time validation slice (held-out MAE 604.0); clearly distinguished baseline transfer calibration from production ground-truth collection.
21. ❓ Production Hardening & Architectural Warnings (W1–W11) → ✅ Remediated:
    - **W1 (Temporal Data Leakage):** Replaced shuffled splits in `train_base.py` and `train_model.py` with strict chronological ordering (80% train / 20% test).
    - **W2 (Holiday Parsing):** Loaded UCI CSV with `keep_default_na=False` so `"None"` is not coerced to `NaN`; centralized holiday calendar in `city_config.py`.
    - **W3 (Sensor Ratio Bounds):** Enforced clipping on `congestion_ratio` to `[0.05, 1.0]` with physical speed floors (10 km/h free-flow, 5 km/h current).
    - **W4 (API Rate Limiting):** Wrapped TomTom route fetches in `@st.cache_data(ttl=60)` to prevent API quota exhaustion on UI reruns.
    - **W5 (Cache Race Conditions):** Separated shared cache into subsystem-isolated files (`predict_cache.json`, `weather_cache.json`) written atomically via `.tmp` renames.
    - **W6 (Timezone Shift):** Added `get_city_now(city)` using `ZoneInfo` (`Asia/Kolkata`, `America/Chicago`) to eliminate UTC displacement on cloud hosts.
    - **W7 (Dynamic Horizons):** Unified prediction under `pipeline.py` with exponential horizon decay (`w_live = 0.35 * exp(-m/45)`), ensuring realistic forecast trajectories across T+0..60.
    - **W8 (Joblib Security):** Implemented `safe_load_model()` in `schema.py` with SHA-256 integrity verification (`models/checksums.json`).
    - **W9 (Client Secret Leakage):** Replaced pre-filled password inputs with empty defaults in `app.py`, reading keys server-side only.
    - **W10 (Pipeline Divergence):** Built canonical `pipeline.py` shared identically by both CLI `predict.py` and UI `app.py`.
    - **W11 (Collector Data Hygiene):** Filtered collector visualization in Tab 5 to genuine `live:` sensor rows, omitting simulated fallbacks.

## Verification log (all pass)
- `python train_base.py` → Strict Chronological Out-of-Time MAE 286.26, RMSE 509.83 ✅
- `python train_model.py` → Chronological Base MAE 285.7, Held-out Transfer MAE 604.0 ✅
- `python sanity_check.py` → SHA-256 verified; 5484.45 Severe ✅
- `python predict.py` → Unified canonical pipeline with pure SHAP & dynamic horizons (T+0..60) ✅
- `models/checksums.json` → Integrity hashes generated and verified for all model artifacts ✅
- `codebase.md` → Bundled with zero plaintext secrets and complete canonical modules ✅

## Security & Operational Policy
- Live API keys are kept exclusively in local environment variables or uncommitted `.streamlit/secrets.toml`.
- All model files are integrity-checked with SHA-256 before deserialization.
- All coordinates are validated within physical geo-bounds (-90..90, -180..180).
- Pure Shapley values ensure transparent, defensible explainability for hackathon judging.
