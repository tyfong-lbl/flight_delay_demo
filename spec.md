# Flight Delay Prediction Demo: lakeFS + JupyterHub + OpenCode

## Overview

A conference/meetup demo showcasing **lakeFS data versioning and branching** integrated with **JupyterHub** for a machine learning workflow. The demo uses the [Flight Delay and Cancellation Dataset (2019-2023)](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?resource=download) to train an XGBoost binary classifier predicting whether a flight will be delayed (>15 minutes).

**Format:** Slides built from notebook screenshots (not a live demo).
**Key narrative angle:** Using **OpenCode** (AI coding assistant) to write the notebook code, demonstrating AI-assisted data science workflows.
**Timeline:** ~1 day to build and execute.

---

## Goals & Key Messages

1. **Data versioning with lakeFS** — Show how datasets are versioned with commits, enabling reproducibility.
2. **Branching for experimentation** — Use lakeFS branches to run competing feature engineering experiments on the same clean dataset, compare results, and merge the winner.
3. **Medallion architecture on lakeFS** — Demonstrate bronze → silver → gold data layers using lakeFS branches.
4. **AI-assisted development** — Highlight OpenCode writing the notebook code in real time.

---

## Dataset

- **Source:** [Kaggle — Flight Delay and Cancellation Dataset 2019-2023](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023?resource=download)
- **Scope:** Use a **single year (2023)** to keep the dataset manageable for a 1-day build. If 2023 alone is still very large (millions of rows), take a **random stratified sample of ~500k rows** to ensure fast iteration during development and snappy notebook execution for screenshots.
- **Prediction target:** Binary classification — **delayed (1) vs. not delayed (0)**, where "delayed" is defined as arrival delay > 15 minutes.

---

## Architecture

### Infrastructure

| Component     | Details                                          |
|---------------|--------------------------------------------------|
| JupyterHub    | Containerized (Docker), running on local/remote  |
| lakeFS        | Remote instance (separate machine)               |
| lakeFS access | Python SDK (`lakefs`)                            |
| Storage       | Whatever backend the remote lakeFS uses (S3-compatible, local, etc.) |

### Python Dependencies

Make a .venv for the repo with the following requirements.txt:

```
lakefs>=0.5.0
xgboost>=2.0
pandas>=2.0
numpy
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

---

## lakeFS Branch Strategy

```
main (bronze)
│   Raw CSV ingested, committed as bronze layer
│
└── silver (branched from main)
    │   Cleaned & transformed data committed here
    │
    ├── experiment-time-features (branched from silver)
    │   Gold layer: time-based feature engineering
    │   XGBoost model trained & evaluated
    │
    └── experiment-route-features (branched from silver)
        Gold layer: route-based feature engineering
        XGBoost model trained & evaluated
```

**Final step:** Compare experiment results, merge the winning branch back to main.

### lakeFS Repository Setup

- **Repository name:** `flight-delay-demo`
- **Default branch:** `main`
- All lakeFS operations use the **Python SDK** (`import lakefs`)

---

## Notebook 1: Data Ingestion & Cleaning (Bronze → Silver)

**Filename:** `01_data_prep.ipynb`

### Section 1.1 — Setup & Configuration

- Import libraries (`lakefs`, `pandas`, `numpy`)
- Configure lakeFS client connection (endpoint, credentials)
- Create the `flight-delay-demo` repository (or connect to existing)
- Verify connection

### Section 1.2 — Bronze Layer (main branch)

- Load raw CSV file(s) for 2023
- If dataset is too large, perform stratified sampling (~500k rows) preserving the delay/no-delay class distribution
- Upload raw data to lakeFS `main` branch at path `bronze/flights_2023.parquet` (convert CSV → Parquet for efficiency)
- **Commit** to main with message: `"Ingest raw 2023 flight data (bronze layer)"`

### Section 1.3 — EDA (Exploratory Data Analysis)

Produce the following visualizations (for slide screenshots):

1. **Delay distribution histogram** — Distribution of arrival delay minutes, with a vertical line at the 15-minute threshold showing the binary classification split
2. **Delay by airline bar chart** — Average delay or delay rate by airline carrier
3. **Delay by route** — Top 20 most-delayed routes (origin-destination pairs), displayed as a horizontal bar chart

Include basic dataset statistics:
- Total rows, columns
- Class balance (% delayed vs. not delayed)
- Missing value summary

### Section 1.4 — Silver Layer (silver branch)

- **Create branch** `silver` from `main`
- Perform data cleaning:
  - Drop cancelled flights (we're predicting delays, not cancellations)
  - Handle missing values (drop rows with missing arrival delay; impute or drop other missing values as appropriate)
  - Remove obviously erroneous records (e.g., negative air time)
  - Convert data types as needed (dates, categoricals)
  - Create the binary target column: `is_delayed = 1 if arrival_delay > 15 else 0`
- Save cleaned data to `silver/flights_2023_clean.parquet`
- **Commit** to silver with message: `"Clean flight data, create delay target (silver layer)"`
- Print a summary of rows removed and final dataset shape

---

## Notebook 2: Experiments & Comparison (Gold + Modeling)

**Filename:** `02_experiments.ipynb`

### Section 2.1 — Setup

- Import libraries (`lakefs`, `pandas`, `xgboost`, `sklearn`, `matplotlib`, `seaborn`)
- Connect to lakeFS, point to `flight-delay-demo` repo
- Load cleaned data from `silver` branch

### Section 2.2 — Experiment A: Time-Based Features

- **Create branch** `experiment-time-features` from `silver`
- Feature engineering — derive time-based features:
  - `hour_of_day` — scheduled departure hour
  - `day_of_week` — Monday=0, Sunday=6
  - `month` — month of year
  - `is_weekend` — binary flag
  - `is_holiday_period` — binary flag for known US holiday periods (Thanksgiving week, Christmas week, July 4th week, etc.)
  - `time_of_day_bucket` — morning/afternoon/evening/night
- Select features for model: time-based features + `airline` (encoded) + `origin` (encoded, top airports) + `distance`
- Apply label/ordinal encoding for categoricals (keep it simple)
- Train/test split (80/20, stratified)
- Save gold-layer feature set to `gold/features_time.parquet`
- **Commit** to branch: `"Gold layer: time-based features"`
- Train XGBoost classifier:
  ```python
  xgb.XGBClassifier(
      n_estimators=200,
      max_depth=6,
      learning_rate=0.1,
      eval_metric='logloss',
      use_label_encoder=False,
      random_state=42
  )
  ```
- Evaluate and produce:
  - **Confusion matrix heatmap**
  - **Precision-Recall curve**
  - **Feature importance bar chart** (top 15 features)
  - Print accuracy, precision, recall, F1 score
- Save model metrics to `gold/metrics_time.json`
- **Commit** to branch: `"Train XGBoost on time-based features, save metrics"`

### Section 2.3 — Experiment B: Route-Based Features

- **Create branch** `experiment-route-features` from `silver`
- Feature engineering — derive route/geography-based features:
  - `airline` (encoded)
  - `origin` (encoded)
  - `destination` (encoded)
  - `route` — origin-destination pair (encoded, top N routes)
  - `distance`
  - `distance_bucket` — short/medium/long haul
  - `origin_delay_rate` — historical average delay rate for the origin airport (computed from the training set)
  - `airline_delay_rate` — historical average delay rate for the airline (computed from the training set)
  - `route_delay_rate` — historical average delay rate for the route (computed from the training set)
- Train/test split (80/20, stratified, same random_state=42)
- Save gold-layer feature set to `gold/features_route.parquet`
- **Commit** to branch: `"Gold layer: route-based features"`
- Train XGBoost classifier (same hyperparameters as Experiment A for fair comparison)
- Evaluate and produce:
  - **Confusion matrix heatmap**
  - **Precision-Recall curve**
  - **Feature importance bar chart** (top 15 features)
  - Print accuracy, precision, recall, F1 score
- Save model metrics to `gold/metrics_route.json`
- **Commit** to branch: `"Train XGBoost on route-based features, save metrics"`

### Section 2.4 — Comparison & Merge

- Load metrics from both experiment branches
- Side-by-side comparison table (accuracy, precision, recall, F1, AUC-PR)
- Overlay **Precision-Recall curves** from both experiments on a single chart
- Declare the winner based on F1 score (or AUC-PR)
- **Merge** the winning branch back to `main`
- **Commit** summary: `"Merge winning experiment ([experiment name]) to main"`
- Print final lakeFS commit log showing the full history: bronze → silver → experiments → merge

---

## Visualization Summary

| Chart | Notebook | Section | Purpose |
|-------|----------|---------|---------|
| Delay distribution histogram | 01 | 1.3 | EDA — show class distribution |
| Delay by airline bar chart | 01 | 1.3 | EDA — airline performance |
| Delay by route bar chart | 01 | 1.3 | EDA — route performance |
| Confusion matrix heatmap (x2) | 02 | 2.2, 2.3 | Model evaluation per experiment |
| Precision-Recall curve (x2 + overlay) | 02 | 2.2, 2.3, 2.4 | Model evaluation + comparison |
| Feature importance bar chart (x2) | 02 | 2.2, 2.3 | Feature analysis per experiment |
| Metrics comparison table | 02 | 2.4 | Final experiment comparison |

### Visualization Style

- Use **seaborn** with `set_style("whitegrid")` for clean, presentation-ready charts
- Use a consistent color palette across all charts (e.g., `sns.color_palette("deep")`)
- Titles and labels should be large enough to read on slides (fontsize 14+ for titles, 12+ for labels)
- Save all charts as high-resolution PNGs (`dpi=150`) for slide embedding

---

## File Structure in lakeFS Repository

```
flight-delay-demo/
├── bronze/
│   └── flights_2023.parquet          # Raw ingested data
├── silver/
│   └── flights_2023_clean.parquet    # Cleaned data with target column
└── gold/
    ├── features_time.parquet         # Time-based feature set (Experiment A)
    ├── features_route.parquet        # Route-based feature set (Experiment B)
    ├── metrics_time.json             # Experiment A model metrics
    └── metrics_route.json            # Experiment B model metrics
```

---

## Suggested Slide Outline

| # | Slide Title | Content Source | Notes |
|---|------------|---------------|-------|
| 1 | Title Slide | — | "Flight Delay Prediction with lakeFS, JupyterHub & OpenCode" |
| 2 | The Problem | — | "Can we predict flight delays? What tools make this easier?" |
| 3 | Tech Stack Overview | Diagram | lakeFS + JupyterHub + OpenCode logos/architecture diagram |
| 4 | The Dataset | Notebook 01 §1.2 | Dataset description, row counts, source attribution |
| 5 | EDA: Delay Distribution | Notebook 01 §1.3 | Screenshot of delay histogram |
| 6 | EDA: Delays by Airline | Notebook 01 §1.3 | Screenshot of airline bar chart |
| 7 | EDA: Delays by Route | Notebook 01 §1.3 | Screenshot of route bar chart |
| 8 | Medallion Architecture | Diagram | Bronze → Silver → Gold with lakeFS branches diagram |
| 9 | Data Versioning with lakeFS | Notebook 01 §1.2, §1.4 | Screenshot of lakeFS commits (Python SDK output) |
| 10 | Data Cleaning (Silver Layer) | Notebook 01 §1.4 | Key cleaning steps, before/after row counts |
| 11 | Branching for Experiments | Diagram | Branch diagram showing silver → two experiment branches |
| 12 | Experiment A: Time Features | Notebook 02 §2.2 | Feature list + code screenshot (highlight OpenCode writing it) |
| 13 | Experiment A: Results | Notebook 02 §2.2 | Confusion matrix + feature importance screenshots |
| 14 | Experiment B: Route Features | Notebook 02 §2.3 | Feature list + code screenshot (highlight OpenCode writing it) |
| 15 | Experiment B: Results | Notebook 02 §2.3 | Confusion matrix + feature importance screenshots |
| 16 | Head-to-Head Comparison | Notebook 02 §2.4 | Metrics table + overlaid PR curves |
| 17 | Merge the Winner | Notebook 02 §2.4 | lakeFS merge output + commit log screenshot |
| 18 | OpenCode in Action | Screenshots | Show OpenCode writing notebook code in the terminal |
| 19 | Key Takeaways | — | Bullet points: versioning, branching, reproducibility, AI-assisted dev |
| 20 | Q&A | — | Contact info, links to repo/resources |

---

## Implementation Notes

### Keeping It Simple (1-Day Build)

- **No hyperparameter tuning** — use fixed, reasonable XGBoost defaults for both experiments. The goal is to show the lakeFS workflow, not optimize the model.
- **No deep learning** — XGBoost only, minimal dependencies.
- **No MLflow/experiment tracking tools** — lakeFS branches serve as the experiment tracking mechanism.
- **No pipeline orchestration** — just sequential notebook cells.
- **Encoding:** Use simple label encoding or pandas `.cat.codes` for categorical features. Avoid complex preprocessing pipelines.
- **Data size:** If 2023 data exceeds ~500k rows, sample it down. The model doesn't need to be production-quality; it needs to be demo-quality.

### lakeFS Connection

```python
import lakefs

# Configure connection
client = lakefs.Client(
    host="http://<lakefs-host>:8000",
    username="<access_key_id>",
    password="<secret_access_key>"
)

# Create or get repo
repo = lakefs.Repository("flight-delay-demo", client=client)
```

Adjust host/credentials to match the remote lakeFS instance.

### OpenCode Workflow (for slides)

When building the notebooks, use OpenCode to:
1. Generate the lakeFS setup and connection code
2. Write the data cleaning / transformation logic
3. Write the feature engineering code for both experiments
4. Generate the XGBoost training and evaluation code
5. Create the visualization code

Capture screenshots of OpenCode generating code for inclusion in the slide deck.

---

## Success Criteria

- [ ] Both notebooks execute end-to-end without errors
- [ ] lakeFS repository shows clear commit history: bronze → silver → experiments → merge
- [ ] All 5 visualization types render cleanly at slide-quality resolution
- [ ] The two experiments produce meaningfully different results (different top features, different metrics)
- [ ] Total execution time is under 5 minutes (for a smooth screenshot workflow)
- [ ] All code was written/assisted by OpenCode (for the narrative)
