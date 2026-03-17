# Flight Delay Demo - Task Checklist

## Phase 0 - Environment + Data Acquisition
- [x] Create `.venv` and install dependencies from `requirements.txt`
- [x] Validate imports for all required packages
- [x] Configure Kaggle API access (ensure `kaggle.json` credentials and permissions)
- [x] Run credential check command to confirm API access
- [x] Download dataset files into `data/raw/`
- [x] Record local metadata note: dataset URL, download date/time, expected files
- [x] Extract archive if needed
- [x] Verify 2023 source file exists and can be loaded
- [x] Record row/column preview and schema snapshot

## Phase 1 - Notebook 01 Scaffold + lakeFS Connectivity
- [x] Create `01_data_prep.ipynb` skeleton with sections 1.1–1.4
- [x] Add configuration cell (lakeFS endpoint/credentials, repo name, paths, sample size cap, random seed)
- [x] Initialize lakeFS client and verify repository connection
- [x] Write failing unit tests in `tests/test_helpers.py` (parquet round-trip, commit wrapper, branch creation, chart save)
- [x] Create `helpers.py` with lakeFS helpers (read/write parquet, commit wrapper, branch creation) and chart helpers (save figure with consistent style, `dpi=150`)
- [x] Run unit tests — confirm they pass
- [x] Write integration tests in `tests/test_lakefs_integration.py` (create temp repo, write/read parquet, create branch, commit, verify log, cleanup)
- [x] Run integration tests — confirm they pass against live lakeFS

## Phase 2 - Bronze Ingestion + EDA in Notebook 01
- [x] Write failing tests for filtering and sampling logic (year filtering, stratified sampling, row cap)
- [x] Load and filter raw data for 2023
- [x] Create delay target precursor (`arrival_delay > 15`) and apply stratified sampling if needed
- [x] Run tests — confirm filtering and sampling tests pass
- [x] Write bronze parquet to `bronze/flights_2023.parquet` on `main`
- [x] Commit bronze state in lakeFS: `Ingest raw 2023 flight data (bronze layer)`
- [x] Generate EDA statistics (total rows/columns, class balance, missing-value summary)
- [x] Generate EDA charts: delay histogram + 15-min threshold line
- [x] Generate EDA charts: delay by airline bar chart
- [x] Generate EDA charts: top delayed routes horizontal bar chart
- [x] Save all chart images to `outputs/charts/` at `dpi=150`

## Phase 3 - Silver Cleaning in Notebook 01
- [ ] Write failing tests for cleaning rules and target creation (drop canceled, remove missing arrival delay, remove negative air time, `is_delayed` computation)
- [ ] Create `silver` branch from `main`
- [ ] Load bronze data from lakeFS
- [ ] Drop canceled flights
- [ ] Remove rows with missing arrival delay
- [ ] Handle remaining missing values (drop/imputation policy)
- [ ] Remove erroneous rows (e.g., negative air time)
- [ ] Normalize data types (date/time/categorical)
- [ ] Create target column: `is_delayed = 1 if arrival_delay > 15 else 0`
- [ ] Run tests — confirm cleaning and target tests pass
- [ ] Write `silver/flights_2023_clean.parquet` and commit: `Clean flight data, create delay target (silver layer)`
- [ ] Print cleaning audit summary (rows removed by rule + final shape)

## Phase 4 - Notebook 02 Scaffold + Shared Modeling Utilities
- [ ] Create `02_experiments.ipynb` skeleton with sections 2.1–2.4
- [ ] Connect to lakeFS and load silver dataset
- [ ] Write failing tests for modeling utilities (train/test split stratification, metrics keys/values, metrics JSON round-trip, predictions parquet round-trip)
- [ ] Add modeling utilities to `helpers.py` (deterministic split, XGBoost factory, metrics function, plotting functions, metrics/predictions save/load)
- [ ] Run tests — confirm modeling utility tests pass

## Phase 5 - Experiment A (Time Features)
- [ ] Write failing tests for time-based feature engineering (`hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday_period`, `time_of_day_bucket`)
- [ ] Create branch `experiment-time-features` from `silver`
- [ ] Engineer time-based features
- [ ] Run tests — confirm time feature tests pass
- [ ] Apply label encoding for required categoricals
- [ ] Build feature set and split data (time features + `airline`, `origin`, `distance`)
- [ ] Write `gold/features_time.parquet` and commit: `Gold layer: time-based features`
- [ ] Train XGBoost and evaluate model
- [ ] Generate visuals: confusion matrix, PR curve, top-15 feature importance
- [ ] Write `gold/metrics_time.json` and `gold/predictions_time.parquet`
- [ ] Save charts to `outputs/charts/`
- [ ] Commit: `Train XGBoost on time-based features, save metrics`

## Phase 6 - Experiment B (Route Features + Frequency Encoding)
- [ ] Write failing tests for route feature engineering (frequency encoding, delay-rate computation from train split only, fallback for unseen categories)
- [ ] Create branch `experiment-route-features` from `silver`
- [ ] Engineer route/geography features (`airline`, `origin`, `destination`, `route`, `distance`, `distance_bucket`)
- [ ] Create leakage-safe historical rate features (`origin_delay_rate`, `airline_delay_rate`, `route_delay_rate` from train split only)
- [ ] Apply frequency encoding for high-cardinality categoricals
- [ ] Run tests — confirm route feature tests pass
- [ ] Write `gold/features_route.parquet` and commit: `Gold layer: route-based features`
- [ ] Train XGBoost and evaluate model (same hyperparameters as Experiment A)
- [ ] Generate visuals: confusion matrix, PR curve, top-15 feature importance
- [ ] Write `gold/metrics_route.json` and `gold/predictions_route.parquet`
- [ ] Save charts to `outputs/charts/`
- [ ] Commit: `Train XGBoost on route-based features, save metrics`

## Phase 7 - Comparison, Winner Selection, Merge to Main
- [ ] Load metrics (`gold/metrics_*.json`) and predictions (`gold/predictions_*.parquet`) from both branches
- [ ] Build side-by-side comparison table
- [ ] Generate overlay PR curve (both experiments on one chart) and save to `outputs/charts/`
- [ ] Apply winner decision rule (primary: F1, tiebreaker: AUC-PR)
- [ ] Merge winner branch into `main` with commit: `Merge winning experiment ([name]) to main`
- [ ] Print final commit lineage output (bronze -> silver -> experiments -> merge)

## Phase 8 - Validation + Demo Readiness
- [ ] Restart kernels and run both notebooks end-to-end from clean state
- [ ] Confirm no errors and runtime is under 5 minutes
- [ ] Verify all expected PNGs exist in `outputs/charts/` and are legible at slide resolution
- [ ] Cross-check charts against spec's visualization summary table
- [ ] Verify lakeFS commit history: bronze -> silver -> experiments -> merge
- [ ] Confirm winning experiment is merged to `main`
- [ ] Run full test suite (unit + integration + e2e) as regression check

## Acceptance Checklist
- [ ] Dataset downloaded via Kaggle API and validated
- [ ] Both notebooks execute end-to-end without errors
- [ ] lakeFS repository shows clear commit history: bronze -> silver -> experiments -> merge
- [ ] All 5 visualization types render cleanly at slide-quality resolution in `outputs/charts/`
- [ ] Two experiments produce meaningfully different results (different top features, different metrics)
- [ ] Total execution time is under 5 minutes
- [ ] All unit, integration, and e2e tests pass
- [ ] `helpers.py` is the single source of shared logic across both notebooks
- [ ] Git history on `wip/build-demo` has commits at each phase boundary
