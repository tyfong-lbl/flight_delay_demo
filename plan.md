# Flight Delay Demo Project Plan

## 1) Scope and Decisions (Approved)

- **Objective:** Build a conference-ready, screenshot-driven demo showing lakeFS data versioning/branching with JupyterHub and OpenCode using 2023 flight data.
- **Execution mode:** Notebooks for slides (not live demo), optimized for end-to-end reliability in ~1 day.
- **Data acquisition path:** **Kaggle API download on this instance** (primary path).
- **Modeling approach:** XGBoost binary classifier (`is_delayed = arrival_delay > 15`) with fixed hyperparameters.
- **Encoding strategy (approved):**
  - Experiment A (time-based): label encoding
  - Experiment B (route-based): frequency encoding for high-cardinality categoricals + engineered delay-rate features
- **lakeFS strategy:** `main (bronze)` -> `silver` -> `experiment-time-features` and `experiment-route-features` -> merge winner to `main`.

---

## 2) Technology Choices

- **Runtime:** Python 3.11 in local `.venv`
- **Core libs:** `lakefs`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`
- **Notebook environment:** JupyterHub/JupyterLab
- **Data format:** Parquet for bronze/silver/gold artifacts
- **Artifact format:** JSON for metrics in gold layer
- **Versioning/lineage:** lakeFS Python SDK only (for consistency in narrative)
- **Reproducibility controls:** `random_state=42`, fixed split strategy, deterministic sampling seed

---

## 3) Artifact Paths

### lakeFS Repository (`flight-delay-demo`)

```
bronze/
└── flights_2023.parquet          # Raw ingested data (main branch)
silver/
└── flights_2023_clean.parquet    # Cleaned data with target column (silver branch)
gold/
├── features_time.parquet         # Time-based feature set (experiment-time-features branch)
├── features_route.parquet        # Route-based feature set (experiment-route-features branch)
├── metrics_time.json             # Experiment A metrics + prediction arrays
├── metrics_route.json            # Experiment B metrics + prediction arrays
├── predictions_time.parquet      # y_true + y_scores for PR curve overlay
└── predictions_route.parquet     # y_true + y_scores for PR curve overlay
```

### Local Project

```
data/raw/                         # Raw CSV downloads (gitignored)
outputs/charts/                   # Saved chart PNGs for slide embedding
```

All charts are saved to `outputs/charts/` at `dpi=150` with consistent seaborn styling.

---

## 4) Delivery Principles

- Keep each step small enough for safe execution and debugging in notebooks.
- Commit meaningful state transitions in lakeFS at each medallion/experiment boundary.
- Prefer low-risk implementation over model complexity.
- Ensure every chart/table is presentation-ready and reproducible.

---

## 5) Git Version Control Strategy

This project has two versioning systems: **lakeFS** for data artifacts and **git** for code/notebooks.

- **Branch:** Create a `wip/build-demo` branch from `main` before starting work.
- **Commit points:** Git commit at the end of each phase, and after any non-trivial code change within a phase.
- **Commit content:** Notebooks, `requirements.txt`, shared modules, chart outputs, and any supporting scripts.
- **Exclusions:** `data/raw/` (large CSVs) should be in `.gitignore`. lakeFS holds the data artifacts.
- **Merge:** Squash-merge `wip/build-demo` into `main` when the demo is complete and validated.

---

## 6) Phased Execution Blueprint (Right-Sized Steps)

## Phase 0 - Environment + Data Acquisition

### Goal
Establish a reproducible local environment and pull the dataset onto this instance using Kaggle API.

### Steps
1. **Create Python environment**
   - Create `.venv` and install dependencies from `requirements.txt`.
   - Validate imports for all required packages.

2. **Configure Kaggle API access**
   - Ensure `kaggle.json` credentials are present in expected location and permissions are correct.
   - Run a credential check command to confirm API access.
   - If Kaggle auth fails, fall back to manual download and placement in `data/raw/`.

3. **Download dataset to local storage**
   - Pull dataset files into `data/raw/`.
   - Keep a local metadata note: dataset URL, download date/time, and expected files.

4. **Unpack and validate raw files**
   - Extract archive if needed.
   - Verify the 2023 source file exists and can be loaded.
   - Record row/column preview and schema snapshot for traceability.

### Exit Criteria
- Environment is functional; dataset exists locally and is readable.
- Team can point to a clear raw source path on this instance.

---

## Phase 1 - Notebook 01 Scaffold + lakeFS Connectivity

### Goal
Create `01_data_prep.ipynb` structure and verify stable lakeFS connectivity.

### Steps
1. **Create notebook skeleton**
   - Add sections 1.1-1.4 with markdown cell headers matching spec.

2. **Add configuration cell**
   - Centralize constants: lakeFS endpoint/credentials env vars, repo name, local raw file path, sample size cap, random seed.

3. **Initialize lakeFS client and repository checks**
   - Connect via SDK.
   - Create or connect to `flight-delay-demo`.
   - Print minimal connection/repo verification output.

4. **Write failing unit tests for `helpers.py` lakeFS and chart functions**
   - Create `tests/test_helpers.py` with tests for: parquet round-trip (write → read), commit wrapper return value, branch creation, and chart save (file exists, correct path/DPI).
   - Run tests — confirm they fail (no implementation yet).

5. **Create shared `helpers.py` module**
   - Create `helpers.py` in the project root, imported by both notebooks.
   - lakeFS helpers: read/write parquet to lakeFS, commit wrapper, branch creation.
   - Chart helpers: save figure to `outputs/charts/` with consistent style and `dpi=150`.
   - Run unit tests — confirm they pass.

6. **Write and run integration tests for lakeFS connectivity**
   - Create `tests/test_lakefs_integration.py` with tests for: create temp repo, write/read parquet, create branch, commit, verify log, cleanup.
   - Run integration tests — confirm they pass against live lakeFS.

### Exit Criteria
- Notebook opens cleanly and lakeFS repo access is confirmed from notebook runtime.
- `helpers.py` is importable from notebook runtime.
- All unit and integration tests pass.

---

## Phase 2 - Bronze Ingestion + EDA in Notebook 01

### Goal
Ingest 2023 data to bronze, commit to `main`, and produce core EDA visuals/statistics.

### Steps
1. **Write failing tests for filtering and sampling logic**
   - Add tests to `tests/test_helpers.py` for: year filtering produces only 2023 rows, stratified sampling preserves class balance within tolerance, sampling respects row cap.
   - Run tests — confirm they fail.

2. **Load and filter raw data for 2023**
   - Load source data.
   - Restrict to 2023 if source includes multiple years.

3. **Create delay target precursor and sampling path**
   - Build temporary delay flag based on `arrival_delay > 15`.
   - If row count exceeds cap, apply stratified sampling preserving delay class balance.
   - Run tests — confirm filtering and sampling tests now pass.

4. **Write bronze parquet artifact**
   - Convert to parquet and write to `bronze/flights_2023.parquet` on `main`.

5. **Commit bronze state in lakeFS**
   - Commit with message: `Ingest raw 2023 flight data (bronze layer)`.

6. **Generate EDA statistics**
   - Total rows/columns, class balance, missing-value summary.

7. **Generate EDA charts and save images**
   - Delay histogram + 15-min threshold line
   - Delay by airline bar chart
   - Top delayed routes horizontal bar chart
   - Save images at presentation quality (`dpi=150`) with consistent style.

### Exit Criteria
- Bronze parquet is committed on `main`.
- Required EDA stats/charts render and are saved for slides.
- Filtering and sampling tests pass.

---

## Phase 3 - Silver Cleaning in Notebook 01

### Goal
Create `silver` branch, clean data, create target, and commit silver artifact.

### Steps
1. **Write failing tests for each cleaning rule and target creation**
   - Add tests to `tests/test_helpers.py` for: drop canceled flights removes correct rows, missing arrival delay rows removed, negative air time rows removed, target column `is_delayed` computed correctly for known inputs.
   - Run tests — confirm they fail.

2. **Create `silver` from `main`**
   - Branch creation + verification output.

3. **Load bronze data from branch-aware path**
   - Read bronze dataset from lakeFS.

4. **Apply cleaning rules incrementally**
   - Drop canceled flights.
   - Remove rows with missing arrival delay.
   - Handle remaining missing values via explicit drop/imputation policy.
   - Remove obviously erroneous rows (e.g., negative air time).
   - Normalize data types (date/time/categorical where needed).

5. **Create final target column**
   - `is_delayed = 1 if arrival_delay > 15 else 0`.
   - Run tests — confirm cleaning and target tests now pass.

6. **Write silver artifact and commit**
   - Save `silver/flights_2023_clean.parquet`.
   - Commit message: `Clean flight data, create delay target (silver layer)`.

7. **Print cleaning audit summary**
   - Rows removed by rule category + final shape.

### Exit Criteria
- Silver branch exists with cleaned parquet and commit history for cleaning transformations.
- All cleaning rule and target creation tests pass.

---

## Phase 4 - Notebook 02 Scaffold + Shared Modeling Utilities

### Goal
Create `02_experiments.ipynb` structure and shared utilities for fair experiment comparison.

### Steps
1. **Create notebook skeleton**
   - Add sections 2.1-2.4 matching spec.

2. **Connect to lakeFS and load silver dataset**
   - Validate branch read from `silver`.

3. **Write failing tests for modeling utilities**
   - Add tests to `tests/test_helpers.py` for:
     - Train/test split preserves stratification and uses correct random state.
     - Metrics function returns expected keys (accuracy, precision, recall, F1, AUC-PR) and correct values for known y_true/y_pred.
     - Metrics JSON round-trip (save → load) produces identical dict.
     - Predictions parquet round-trip (save → load) produces identical DataFrame.
   - Run tests — confirm they fail.

4. **Add modeling utilities to `helpers.py`**
   - Deterministic train/test split (`stratify`, `random_state=42`).
   - XGBoost model factory with fixed hyperparameters.
   - Metrics function: accuracy, precision, recall, F1, AUC-PR.
   - Plotting functions: confusion matrix, PR curve, feature importance (all saving to `outputs/charts/`).
   - Metrics + predictions save/load: write metrics JSON and `predictions_*.parquet` (y_true, y_scores) to gold paths for cross-experiment comparison in Phase 7.
   - Run tests — confirm modeling utility tests now pass.

### Exit Criteria
- Experiment notebook has reusable, consistent training/evaluation primitives via `helpers.py`.
- All modeling utility tests pass.

---

## Phase 5 - Experiment A (Time Features)

### Goal
Run baseline time-feature experiment on dedicated branch and persist outputs.

### Steps
1. **Write failing tests for time-based feature engineering**
   - Add tests to `tests/test_helpers.py` for: `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday_period`, `time_of_day_bucket` produce correct values for known input rows.
   - Run tests — confirm they fail.

2. **Create branch `experiment-time-features` from `silver`.**

3. **Engineer time-based features**
   - `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday_period`, `time_of_day_bucket`.
   - Run tests — confirm time feature tests now pass.

4. **Apply encoding strategy for A**
   - Label encoding for required categoricals.

5. **Build feature set and split data**
   - Include time features + `airline`, `origin` (top airports), `distance`.

6. **Persist gold feature artifact and commit**
   - Write `gold/features_time.parquet`.
   - Commit message: `Gold layer: time-based features`.

7. **Train/evaluate model and generate visuals**
   - Confusion matrix, PR curve, top-15 feature importance.
   - Print required metrics.

8. **Persist metrics, predictions, and commit**
   - Write `gold/metrics_time.json` (scalar metrics).
   - Write `gold/predictions_time.parquet` (y_true + y_scores for cross-experiment PR curve overlay in Phase 7).
   - Save charts to `outputs/charts/`.
   - Commit message: `Train XGBoost on time-based features, save metrics`.

### Exit Criteria
- Time experiment branch has feature artifact, metrics artifact, prediction artifact, and evaluation visuals.
- All time feature engineering tests pass.

---

## Phase 6 - Experiment B (Route Features + Frequency Encoding)

### Goal
Run route-heavy experiment with approved encoding strategy and persist outputs.

### Steps
1. **Write failing tests for route feature engineering**
   - Add tests to `tests/test_helpers.py` for:
     - Frequency encoding produces correct counts/frequencies for known input.
     - Delay-rate computation from training split only (no leakage into test split).
     - Fallback for unseen categories in test split returns expected default value.
   - Run tests — confirm they fail.

2. **Create branch `experiment-route-features` from `silver`.**

3. **Engineer route/geography features**
   - `airline`, `origin`, `destination`, `route`, `distance`, `distance_bucket`.

4. **Create leakage-safe historical rate features**
   - Compute `origin_delay_rate`, `airline_delay_rate`, `route_delay_rate` from training split only.
   - Apply to test split with fallback for unseen categories.

5. **Apply encoding strategy for B**
   - Frequency encoding for high-cardinality categoricals (`origin`, `destination`, `route`, and optionally `airline`).
   - Run tests — confirm route feature tests now pass.

6. **Persist gold feature artifact and commit**
   - Write `gold/features_route.parquet`.
   - Commit message: `Gold layer: route-based features`.

7. **Train/evaluate model and generate visuals**
   - Same model hyperparameters and evaluation outputs as Experiment A.

8. **Persist metrics, predictions, and commit**
   - Write `gold/metrics_route.json` (scalar metrics).
   - Write `gold/predictions_route.parquet` (y_true + y_scores for cross-experiment PR curve overlay in Phase 7).
   - Save charts to `outputs/charts/`.
   - Commit message: `Train XGBoost on route-based features, save metrics`.

### Exit Criteria
- Route experiment branch has complete artifacts, prediction artifact, and comparable metrics/plots.
- All route feature engineering tests pass.

---

## Phase 7 - Comparison, Winner Selection, Merge to Main

### Goal
Compare both experiments fairly, select winner, merge back to `main`, and show lineage.

### Steps
1. **Load metrics and predictions from both branches**
   - Load `gold/metrics_*.json` for the comparison table.
   - Load `gold/predictions_*.parquet` (y_true + y_scores) for PR curve overlay.
   - Build side-by-side comparison table.

2. **Generate overlay PR curve**
   - Plot both PR curves on one chart using the loaded prediction arrays.
   - Save to `outputs/charts/`.

3. **Apply winner decision rule**
   - Primary: F1
   - Tiebreaker: AUC-PR

4. **Merge winner branch into `main`**
   - Perform merge and commit with message: `Merge winning experiment ([experiment name]) to main`.

5. **Print final commit lineage output**
   - Show bronze -> silver -> experiments -> merge narrative.

### Exit Criteria
- Winner is merged to `main` and full lineage is demonstrable in notebook output.

---

## Phase 8 - Validation + Demo Readiness

### Goal
Verify end-to-end execution and confirm all automated assets are complete.

### Steps
1. **Run both notebooks end-to-end from clean state**
   - Restart kernels and run all cells sequentially.
   - Confirm no errors and runtime is under 5 minutes.

2. **Verify chart outputs**
   - Confirm all expected PNGs exist in `outputs/charts/` and are legible at slide resolution.
   - Cross-check against the spec's visualization summary table.

3. **Verify lakeFS state**
   - Confirm commit history shows: bronze → silver → experiments → merge.
   - Confirm winning experiment is merged to `main`.

### Exit Criteria
- Both notebooks execute cleanly from a cold start.
- All chart assets and lakeFS state match spec requirements.

### Manual Follow-Up (outside this plan)
- Capture notebook output screenshots for slides.
- Capture OpenCode workflow screenshots for the AI-assisted development narrative.
- Assemble slide deck from screenshot assets.

---

## 7) Testing Strategy

Tests live in `tests/` and use `pytest`. Follow TDD: write failing test first, then implement.

### Unit Tests (`tests/test_helpers.py`)
Test the pure functions in `helpers.py`:
- **lakeFS helpers:** parquet round-trip (write → read) produces identical DataFrame; commit wrapper returns expected ref.
- **Cleaning logic:** dropping cancelled flights, removing missing arrival delays, filtering negative air time — each rule tested individually with small synthetic DataFrames.
- **Feature engineering:** time-based feature derivation (hour_of_day, is_weekend, is_holiday_period, etc.) produces correct values for known inputs.
- **Route feature engineering:** frequency encoding, delay-rate computation from training split, fallback for unseen categories.
- **Metrics function:** returns expected keys (accuracy, precision, recall, F1, AUC-PR) given known y_true/y_pred inputs.

### Integration Tests (`tests/test_lakefs_integration.py`)
Require a live lakeFS connection:
- Create a temporary test repository, write/read parquet, create branch, commit, verify commit log, delete test repo.
- Validate branch-aware reads return correct data after commits to different branches.

### End-to-End Test (`tests/test_e2e.py`)
- Execute both notebooks programmatically via `nbconvert` or `papermill`.
- Assert: no cell errors, expected output files exist in `outputs/charts/`, lakeFS repo has expected branches and commit history.

### When Tests Run
TDD cycles are embedded directly in the phase steps above. Each phase that adds logic to `helpers.py` starts by writing failing tests, then implements until green.
- **Unit tests:** Written and run within Phases 1–6 as the first step before each implementation.
- **Integration tests:** Written and run within Phase 1, step 6.
- **E2E test:** Run during Phase 8 validation.
- **Full suite:** Run at Phase 8 as a regression check before declaring the demo complete.

---

## 8) Risks and Mitigations

- **Kaggle auth/setup risk:** Validate credentials first; maintain manual upload fallback.
- **Runtime risk on large data:** Enforce stratified sampling cap early.
- **Feature leakage risk in Experiment B:** Compute all delay-rate encodings from train split only.
- **Branch confusion risk:** Centralize branch/repo constants and print active branch before writes.
- **Slide quality risk:** Save plots with fixed style and DPI in dedicated output paths.

---

## 9) Acceptance Checklist

Covers all spec success criteria plus plan-specific validations.

- [ ] Dataset downloaded to this instance via Kaggle API and validated
- [ ] Both notebooks execute end-to-end without errors (spec)
- [ ] lakeFS repository shows clear commit history: bronze → silver → experiments → merge (spec)
- [ ] All 5 visualization types render cleanly at slide-quality resolution in `outputs/charts/` (spec)
- [ ] The two experiments produce meaningfully different results (different top features, different metrics) (spec)
- [ ] Total execution time is under 5 minutes (spec)
- [ ] All unit, integration, and e2e tests pass
- [ ] `helpers.py` is the single source of shared logic across both notebooks
- [ ] git history on `wip/build-demo` has commits at each phase boundary
