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

## 3) Delivery Principles

- Keep each step small enough for safe execution and debugging in notebooks.
- Commit meaningful state transitions in lakeFS at each medallion/experiment boundary.
- Prefer low-risk implementation over model complexity.
- Ensure every chart/table is presentation-ready and reproducible.

---

## 4) Phased Execution Blueprint (Right-Sized Steps)

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

3. **Download dataset to local storage**
   - Pull dataset files into `data/raw/`.
   - Keep a local metadata note: dataset URL, download date/time, and expected files.

4. **Unpack and validate raw files**
   - Extract archive if needed.
   - Verify the 2023 source file exists and can be loaded.
   - Record row/column preview and schema snapshot for traceability.

5. **Define fallback procedure (document only)**
   - Document manual upload fallback if Kaggle auth fails.
   - Do not execute fallback unless blocked.

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

4. **Add helper utilities**
   - Small reusable helpers for loading/saving parquet, lakeFS object writes, and commit wrappers.

### Exit Criteria
- Notebook opens cleanly and lakeFS repo access is confirmed from notebook runtime.

---

## Phase 2 - Bronze Ingestion + EDA in Notebook 01

### Goal
Ingest 2023 data to bronze, commit to `main`, and produce core EDA visuals/statistics.

### Steps
1. **Load and filter raw data for 2023**
   - Load source data.
   - Restrict to 2023 if source includes multiple years.

2. **Create delay target precursor and sampling path**
   - Build temporary delay flag based on `arrival_delay > 15`.
   - If row count exceeds cap, apply stratified sampling preserving delay class balance.

3. **Write bronze parquet artifact**
   - Convert to parquet and write to `bronze/flights_2023.parquet` on `main`.

4. **Commit bronze state in lakeFS**
   - Commit with message: `Ingest raw 2023 flight data (bronze layer)`.

5. **Generate EDA statistics**
   - Total rows/columns, class balance, missing-value summary.

6. **Generate EDA charts and save images**
   - Delay histogram + 15-min threshold line
   - Delay by airline bar chart
   - Top delayed routes horizontal bar chart
   - Save images at presentation quality (`dpi=150`) with consistent style.

### Exit Criteria
- Bronze parquet is committed on `main`.
- Required EDA stats/charts render and are saved for slides.

---

## Phase 3 - Silver Cleaning in Notebook 01

### Goal
Create `silver` branch, clean data, create target, and commit silver artifact.

### Steps
1. **Create `silver` from `main`**
   - Branch creation + verification output.

2. **Load bronze data from branch-aware path**
   - Read bronze dataset from lakeFS.

3. **Apply cleaning rules incrementally**
   - Drop canceled flights.
   - Remove rows with missing arrival delay.
   - Handle remaining missing values via explicit drop/imputation policy.
   - Remove obviously erroneous rows (e.g., negative air time).
   - Normalize data types (date/time/categorical where needed).

4. **Create final target column**
   - `is_delayed = 1 if arrival_delay > 15 else 0`.

5. **Write silver artifact and commit**
   - Save `silver/flights_2023_clean.parquet`.
   - Commit message: `Clean flight data, create delay target (silver layer)`.

6. **Print cleaning audit summary**
   - Rows removed by rule category + final shape.

### Exit Criteria
- Silver branch exists with cleaned parquet and commit history for cleaning transformations.

---

## Phase 4 - Notebook 02 Scaffold + Shared Modeling Utilities

### Goal
Create `02_experiments.ipynb` structure and shared utilities for fair experiment comparison.

### Steps
1. **Create notebook skeleton**
   - Add sections 2.1-2.4 matching spec.

2. **Connect to lakeFS and load silver dataset**
   - Validate branch read from `silver`.

3. **Add shared modeling utilities**
   - Deterministic train/test split (`stratify`, `random_state=42`).
   - Common XGBoost model factory with fixed hyperparameters.
   - Common metrics function: accuracy, precision, recall, F1, AUC-PR.
   - Common plotting functions: confusion matrix, PR curve, feature importance.

4. **Add artifact save/load helpers**
   - Save feature parquet + metrics JSON to branch-local gold paths.

### Exit Criteria
- Experiment notebook has reusable, consistent training/evaluation primitives.

---

## Phase 5 - Experiment A (Time Features)

### Goal
Run baseline time-feature experiment on dedicated branch and persist outputs.

### Steps
1. **Create branch `experiment-time-features` from `silver`.**
2. **Engineer time-based features**
   - `hour_of_day`, `day_of_week`, `month`, `is_weekend`, `is_holiday_period`, `time_of_day_bucket`.

3. **Apply encoding strategy for A**
   - Label encoding for required categoricals.

4. **Build feature set and split data**
   - Include time features + `airline`, `origin` (top airports), `distance`.

5. **Persist gold feature artifact and commit**
   - Write `gold/features_time.parquet`.
   - Commit message: `Gold layer: time-based features`.

6. **Train/evaluate model and generate visuals**
   - Confusion matrix, PR curve, top-15 feature importance.
   - Print required metrics.

7. **Persist metrics and commit**
   - Write `gold/metrics_time.json`.
   - Commit message: `Train XGBoost on time-based features, save metrics`.

### Exit Criteria
- Time experiment branch has feature artifact, metrics artifact, and evaluation visuals.

---

## Phase 6 - Experiment B (Route Features + Frequency Encoding)

### Goal
Run route-heavy experiment with approved encoding strategy and persist outputs.

### Steps
1. **Create branch `experiment-route-features` from `silver`.**

2. **Engineer route/geography features**
   - `airline`, `origin`, `destination`, `route`, `distance`, `distance_bucket`.

3. **Create leakage-safe historical rate features**
   - Compute `origin_delay_rate`, `airline_delay_rate`, `route_delay_rate` from training split only.
   - Apply to test split with fallback for unseen categories.

4. **Apply encoding strategy for B**
   - Frequency encoding for high-cardinality categoricals (`origin`, `destination`, `route`, and optionally `airline`).

5. **Persist gold feature artifact and commit**
   - Write `gold/features_route.parquet`.
   - Commit message: `Gold layer: route-based features`.

6. **Train/evaluate model and generate visuals**
   - Same model hyperparameters and evaluation outputs as Experiment A.

7. **Persist metrics and commit**
   - Write `gold/metrics_route.json`.
   - Commit message: `Train XGBoost on route-based features, save metrics`.

### Exit Criteria
- Route experiment branch has complete artifacts and comparable metrics/plots.

---

## Phase 7 - Comparison, Winner Selection, Merge to Main

### Goal
Compare both experiments fairly, select winner, merge back to `main`, and show lineage.

### Steps
1. **Load metrics from both branches**
   - Build side-by-side comparison table.

2. **Generate overlay PR curve**
   - Plot both curves in one chart for direct comparison.

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

## Phase 8 - Slide Asset Packaging + Demo Readiness

### Goal
Prepare final screenshot assets and verify the narrative flow for presentation.

### Steps
1. **Collect chart exports**
   - Ensure required plots/tables are available and legible.

2. **Capture notebook output screenshots**
   - lakeFS commits/branching, experiment outputs, comparison table/plot.

3. **Capture OpenCode workflow screenshots**
   - Examples of assistant-generated code in terminal/notebook workflow.

4. **Perform final execution sanity pass**
   - Validate both notebooks run end-to-end without manual patching.
   - Confirm runtime is within target window.

### Exit Criteria
- All slide assets are ready; demo story is coherent and reproducible.

---

## 5) Suggested Ticket Breakdown Template (for Engineering Lead)

- One ticket per phase subsection (typically 1-2 hours each).
- Each ticket should include:
  - Scope (exact notebook cells/files touched)
  - Definition of done (artifact path + expected output)
  - Validation command/cell execution notes
  - Risks and rollback notes (especially for branch/merge steps)

---

## 6) Risks and Mitigations

- **Kaggle auth/setup risk:** Validate credentials first; maintain manual upload fallback.
- **Runtime risk on large data:** Enforce stratified sampling cap early.
- **Feature leakage risk in Experiment B:** Compute all delay-rate encodings from train split only.
- **Branch confusion risk:** Centralize branch/repo constants and print active branch before writes.
- **Slide quality risk:** Save plots with fixed style and DPI in dedicated output paths.

---

## 7) Acceptance Checklist

- [ ] Dataset downloaded to this instance via Kaggle API and validated
- [ ] Notebook 01 runs bronze -> silver end-to-end with required EDA visuals
- [ ] Notebook 02 runs both experiments with consistent evaluation outputs
- [ ] Winner selection rule applied and merged to `main`
- [ ] lakeFS history clearly demonstrates medallion + experiment branching narrative
- [ ] Slide assets (plots/tables/screenshots) are ready and legible
- [ ] End-to-end runtime and stability are acceptable for demo workflow
