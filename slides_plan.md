# Slide Deck Storyboard: CBorg Studio — AI + lakeFS for Data Science

## Act 1: The Problem & The Toolkit (3 slides)

### Slide 1 — Title

- "Flight Delay Prediction with CBorg Studio, lakeFS & AI"
- Subtitle: "From spec to trained model in one session"

### Slide 2 — The Challenge

- "Can we predict whether a flight will be delayed >15 minutes?"
- 454k flights, 2023 US domestic data, XGBoost binary classifier
- Real question for this talk: how do AI + data versioning change the data science workflow?

### Slide 3 — The Toolkit

- CBorg Studio = JupyterHub + OpenCode (AI coding agent) + MCP tool integrations
- lakeFS = Git-like branching and commits for *data*
- Diagram: JupyterHub with OpenCode panel, lakeFS as data backend, MCP connectors (search, journal, GitHub)
- Callout: multiple models available (GPT-5.3 Codex, Claude Opus 4) — pick the right one for the task

---

## Act 2: AI-Driven Planning (3 slides)

### Slide 4 — "Start with a Spec, Let AI Plan"

- Invoke `/plan` — AI reads `spec.md`, proposes tech choices, asks clarifying questions
- **Screenshot `09.27.27`**: OpenCode reading spec.md, proposing Python 3.11, Parquet, XGBoost, lakeFS SDK. Note the MCP panel showing connected tools.
- Key message: AI plans the project *before* writing any code

### Slide 5 — "Interactive Decision-Making"

- AI presents encoding strategy tradeoffs with structured pros/cons
- User selects: "1 please"
- **Screenshot `09.34.23`**: The encoding discussion — label vs. frequency vs. target encoding, AI recommendation, numbered choices
- Key message: AI collaborates, it doesn't just execute blindly

### Slide 6 — "Todo-Driven Execution with Memory"

- Plan becomes a checklist in `todo.md`, tied to GitHub issues
- `/do-todo` picks the next item, plans it, implements with TDD, checks it off
- Before starting, AI consults its private journal (project notes + user-global notes) for relevant context from earlier work
- **Screenshot `10.57.38`**: `/do-todo` running — AI reading todo.md + plan.md, then consulting two journal entries via `private-journal` MCP calls (one project-level, one user-level). Shows the AI building context before acting.
- **Screenshot `10.32.37`**: Near-complete todo list with `[✓]` items, AI writing learnings back to its journal via `private-journal_process_thoughts` calls — recording project notes and technical insights for future sessions

---

## Act 3: Building the Pipeline with lakeFS (4 slides)

### Slide 7 — "Medallion Architecture on lakeFS Branches"

- Diagram of the lakeFS branch strategy:
  ```
  main (bronze) --> silver --> experiment-time-features
                           --> experiment-route-features
                           --> merge winner back to main
  ```
- Each layer = a lakeFS branch + commit. Versioned, reproducible, auditable.
- **Suggested screenshot**: lakeFS UI showing the branch list for `flight-delay-demo`

### Slide 8 — "Bronze & Silver: Ingest and Clean"

- Bronze: 463k raw rows committed to `main` as `bronze/flights_2023.parquet`
- Silver: branch from main, apply cleaning rules, create binary target
  - Dropped 7,809 cancelled flights, 1,262 missing arrival delays
  - Created `is_delayed = arrival_delay > 15`
  - Final: 454,413 clean rows
- lakeFS commits capture each state transition with descriptive messages
- **Suggested screenshot**: lakeFS UI showing commit log on `main` or `silver` branch — the commit messages tell the data story

### Slide 9 — "EDA: What Does the Data Look Like?"

- Class balance: 78% on-time, 22% delayed
- Frontier (F9), JetBlue (B6), Spirit (NK) have highest average delays
- DEN-ABE route averages 1,000+ minutes delay
- **Screenshot `13.45.21`**: Average Arrival Delay by Airline chart
- **Screenshot `13.45.29`**: Top 20 Most Delayed Routes chart
- **Suggested screenshot**: Delay distribution histogram from notebook 01 output (with the 15-min threshold line)

### Slide 10 — "Temporal Train/Test Split"

- Train: Jan–Jun 2023 (334k rows), Test: Jul–Aug 2023 (120k rows)
- More realistic than random split — mirrors how a model would be deployed
- Deviation from spec, recommended by AI during planning: avoids future-data leakage
- Train delay rate: 21.5%, Test delay rate: 24.5% (slight seasonal shift — exactly the kind of thing a temporal split reveals)

---

## Act 4: Branching for Experiments (4 slides)

### Slide 11 — "Experiment A: Time-Based Features"

- Branch `experiment-time-features` from `silver`
- Features: hour_of_day, day_of_week, month, is_weekend, is_holiday_period, time_of_day_bucket + airline, origin, distance
- Label encoding for categoricals
- Results: **F1 = 0.371, AUC-PR = 0.495**, Accuracy = 0.770
- **Suggested screenshot**: Notebook output showing time features table + metrics printout
- **Suggested screenshot**: Confusion matrix and/or feature importance chart from notebook 02

### Slide 12 — "Experiment B: Route-Based Features"

- Branch `experiment-route-features` from `silver`
- Frequency encoding for high-cardinality categoricals (348 origins, 6,088 routes)
- Leakage-safe delay rates computed from training split only (with global-rate fallback for unseen categories)
- Results: **F1 = 0.030, AUC-PR = 0.310**, Accuracy = 0.752
- The model barely predicts delays (recall = 1.6%) — route features alone aren't enough
- **Suggested screenshot**: Notebook output showing route feature matrix + metrics

### Slide 13 — "Head-to-Head: A Clear Winner"

- Comparison table from notebook output:

  | Metric    | Exp A (Time) | Exp B (Route) | Winner   |
  |-----------|-------------|---------------|----------|
  | accuracy  | 0.7700      | 0.7522        | A (Time) |
  | auc_pr    | 0.4949      | 0.3101        | A (Time) |
  | f1        | 0.3708      | 0.0302        | A (Time) |
  | precision | 0.5636      | 0.3790        | A (Time) |
  | recall    | 0.2762      | 0.0157        | A (Time) |

- Overlay PR curve shows Time features dominating
- Winner: Experiment A across every metric
- **Suggested screenshot**: The comparison table output + overlay PR curve from notebook 02 section 2.4

### Slide 14 — "Merge the Winner, Preserve the Lineage"

- `merge_branch("experiment-time-features" -> "main")`
- Final commit log tells the full data story:
  ```
  ac1f7b57  Repository created
  ce974476  Ingest raw 2023 flight data (bronze layer)
  54d9600d  Clean flight data, create delay target (silver layer)
  6a93a745  Gold layer: time-based features
  4bc1c7e4  Train XGBoost on time-based features, save metrics
  89c1b6e8  Merge winning experiment (Time Features) to main
  ```
- Losing experiment branch is preserved — you can always go back and inspect it
- **Suggested screenshot**: lakeFS UI showing `main` branch commit history after merge

---

## Act 5: AI Capabilities Highlighted (3 slides)

### Slide 15 — "AI That Debugs Transparently"

- lakeFS server was unreachable — AI reported exactly what it tried and what it didn't
- Three methods attempted (curl healthcheck, Python SDK calls, integration test reachability check)
- Clearly stated: "I lack the information to diagnose further without your input"
- **Screenshot `11.50.40`**: The structured debugging report with bold headers, numbered attempts, "What I did NOT try" section
- Key message: no hallucination, no silent failures

### Slide 16 — "AI That Writes Tests First (TDD)"

- Every phase: write failing test, implement until green, commit
- Subagent sessions handle tasks autonomously, report back with diffs
- **Screenshot `13.14.50`**: Subagent view showing todo.md diff (marking Phase 2 test item complete), GitHub issue creation, TDD red-phase verification
- Key message: AI follows engineering discipline, not just code generation

### Slide 17 — "AI That Learns and Remembers"

- Private journal (MCP tool) records project notes and technical insights after each task
- Journal is consulted at the start of each new task for relevant context
- Read/write cycle: consult past entries before acting, record new learnings after completing
- **Screenshot `10.57.38`** (detail crop): The `private-journal_list_recent_entries` and `private-journal_read_journal_entry` calls — AI reading its own notes before starting work
- **Screenshot `10.32.37`** (detail crop): The `private-journal_process_thoughts` calls — AI recording what it learned (project notes, technical insights, user context) after completing a task
- Key message: continuity across sessions, not a blank slate every time

---

## Act 6: Wrap-Up (2 slides)

### Slide 18 — Key Takeaways

1. **lakeFS branches replace ad-hoc experiment tracking** — every data state is a versioned, addressable snapshot
2. **AI doesn't just generate code** — it plans, asks questions, tracks progress, writes tests, and debugs transparently
3. **Medallion architecture maps cleanly to lakeFS** — bronze/silver/gold as branch+commit boundaries
4. **MCP integrations extend AI capabilities** — search, journal, GitHub are tools the AI uses autonomously
5. **Multiple models, one interface** — pick GPT-5.3 Codex for speed or Claude Opus 4 for reasoning

### Slide 19 — Q&A

- Contact info, links to lakeFS docs, CBorg Studio

---

## Screenshot Inventory

### Already captured (in `/Users/TYFong/Documents/cborg_studio_demo/`)

| File | Content | Used in slide(s) |
|------|---------|-------------------|
| `Screenshot 2026-03-17 at 09.27.27.png` | `/plan` skill — AI reading spec, proposing tech choices | 4 |
| `Screenshot 2026-03-17 at 09.34.23.png` | Encoding tradeoff discussion, user selects option 1 | 5 |
| `Screenshot 2026-03-17 at 10.23.40.png` | `/do-todo` with Todo sidebar (backup for slide 6) | — (reserve) |
| `Screenshot 2026-03-17 at 10.32.37.png` | Completed todo list + journal write calls | 6, 17 |
| `Screenshot 2026-03-17 at 10.57.38.png` | `/do-todo` with journal read calls | 6, 17 |
| `Screenshot 2026-03-17 at 11.50.40.png` | lakeFS debugging — transparent failure report | 15 |
| `Screenshot 2026-03-17 at 13.14.50.png` | Subagent session — TDD, GitHub issue, todo diff | 16 |
| `Screenshot 2026-03-17 at 13.45.21.png` | EDA chart: Average Arrival Delay by Airline | 9 |
| `Screenshot 2026-03-17 at 13.45.29.png` | EDA chart: Top 20 Most Delayed Routes | 9 |

### Still need to capture

| Slide(s) | What to capture | Where to find it |
|----------|----------------|------------------|
| 7 | lakeFS UI: branch list for `flight-delay-demo` | lakeFS web UI |
| 8, 14 | lakeFS UI: commit history on `main` branch (showing bronze/silver/gold/merge commits) | lakeFS web UI |
| 8 | lakeFS UI: object browser showing `bronze/` and `silver/` paths | lakeFS web UI |
| 9 | Delay distribution histogram (with 15-min threshold line) | Notebook 01 cell output (section 1.3) |
| 10 | Silver cleaning audit summary (rows removed by rule + final shape) | Notebook 01 cell output (section 1.4) |
| 11 | Experiment A: time features table + metrics printout | Notebook 02 cell outputs (section 2.2) |
| 11 | Experiment A: confusion matrix and/or feature importance chart | Notebook 02 cell outputs (section 2.2) |
| 12 | Experiment B: route feature matrix + metrics printout | Notebook 02 cell outputs (section 2.3) |
| 13 | Comparison table + overlay PR curve | Notebook 02 cell outputs (section 2.4) |
| 14 | Final commit lineage output | Notebook 02 cell output (section 2.4, last cell) |

---
---

# Condensed Slide Deck (10 slides)

Condensed version of the 19-slide storyboard above. Merges acts 2–5 and drops detail that can be covered verbally.

## Slide 1 — Title

- "Flight Delay Prediction with CBorg Studio, lakeFS & AI"
- Subtitle: "From spec to trained model in one session"

---

## Slide 2 — The Challenge & The Toolkit

- "Can we predict whether a flight will be delayed >15 minutes?"
- 454k flights, 2023 US domestic data, XGBoost binary classifier
- Real question: how do AI + data versioning change the data science workflow?
- CBorg Studio = JupyterHub + OpenCode (AI coding agent) + MCP tool integrations
- lakeFS = Git-like branching and commits for *data*
- Diagram: JupyterHub with OpenCode panel, lakeFS as data backend, MCP connectors (search, journal, GitHub)
- Callout: multiple models available (GPT-5.3 Codex, Claude Opus 4) — pick the right one for the task

---

## Slide 3 — "Start with a Spec, Let AI Plan"

- Invoke `/plan` — AI reads `spec.md`, proposes tech choices, asks clarifying questions
- AI presents encoding strategy tradeoffs with structured pros/cons; user selects: "1 please"
- Key message: AI plans the project *before* writing any code, and collaborates rather than executing blindly
- **Screenshot `09.27.27`**: OpenCode reading spec.md, proposing Python 3.11, Parquet, XGBoost, lakeFS SDK
- **Screenshot `09.34.23`**: The encoding discussion — label vs. frequency vs. target encoding, AI recommendation, numbered choices

---

## Slide 4 — "Todo-Driven Execution with Memory"

- Plan becomes a checklist in `todo.md`, tied to GitHub issues
- `/do-todo` picks the next item, plans it, implements with TDD, checks it off
- Before starting, AI consults its private journal for relevant context from earlier work
- After completing, AI records learnings (project notes, technical insights) back to journal
- Key message: continuity across sessions, not a blank slate every time
- **Screenshot `10.57.38`**: `/do-todo` running — AI reading todo.md + plan.md, then consulting journal entries via `private-journal` MCP calls
- **Screenshot `10.32.37`**: Near-complete todo list with `[✓]` items, AI writing learnings back to journal

---

## Slide 5 — "Medallion Architecture on lakeFS Branches"

- Diagram of the lakeFS branch strategy:
  ```
  main (bronze) --> silver --> experiment-time-features
                            --> experiment-route-features
                            --> merge winner back to main
  ```
- Each layer = a lakeFS branch + commit. Versioned, reproducible, auditable.
- Bronze: 463k raw rows → Silver: cleaned to 454k, binary target `is_delayed`
- Temporal train/test split (Jan–Jun train, Jul–Aug test) — recommended by AI to avoid data leakage
- **Suggested screenshot**: lakeFS UI showing the branch list for `flight-delay-demo`

---

## Slide 6 — "EDA: What Does the Data Look Like?"

- Class balance: 78% on-time, 22% delayed
- Frontier (F9), JetBlue (B6), Spirit (NK) have highest average delays
- DEN-ABE route averages 1,000+ minutes delay
- Train delay rate: 21.5%, Test delay rate: 24.5% (seasonal shift the temporal split reveals)
- **Screenshot `13.45.21`**: Average Arrival Delay by Airline chart
- **Screenshot `13.45.29`**: Top 20 Most Delayed Routes chart
- **Suggested screenshot**: Delay distribution histogram from notebook 01 (with 15-min threshold line)

---

## Slide 7 — "Branching for Experiments: Head-to-Head"

- **Experiment A** (branch `experiment-time-features`): hour_of_day, day_of_week, month, is_weekend, time_of_day_bucket + airline, origin, distance. Label encoding.
- **Experiment B** (branch `experiment-route-features`): frequency encoding for high-cardinality categoricals (348 origins, 6,088 routes). Leakage-safe delay rates from training split only.
- Comparison table:

  | Metric    | Exp A (Time) | Exp B (Route) | Winner   |
  |-----------|-------------|---------------|----------|
  | f1        | 0.371       | 0.030         | A (Time) |
  | auc_pr    | 0.495       | 0.310         | A (Time) |
  | recall    | 0.276       | 0.016         | A (Time) |

- Winner merged to `main`; losing branch preserved for inspection
- Final commit log on `main` tells the full data story (bronze → silver → gold → merge)
- **Suggested screenshot**: Comparison table + overlay PR curve from notebook 02 section 2.4
- **Suggested screenshot**: lakeFS UI showing `main` branch commit history after merge

---

## Slide 8 — "AI That Debugs and Tests with Discipline"

- **Transparent debugging**: lakeFS server unreachable — AI reported exactly what it tried (curl, SDK, integration test), what it didn't try, and where it got stuck. No hallucination, no silent failures.
- **TDD discipline**: every phase follows write-failing-test → implement → green → commit. Subagent sessions handle tasks autonomously.
- **Screenshot `11.50.40`**: Structured debugging report with bold headers, numbered attempts, "What I did NOT try" section
- **Screenshot `13.14.50`**: Subagent view showing todo.md diff, GitHub issue creation, TDD red-phase verification

---

## Slide 9 — Key Takeaways

1. **lakeFS branches replace ad-hoc experiment tracking** — every data state is a versioned, addressable snapshot
2. **AI doesn't just generate code** — it plans, asks questions, tracks progress, writes tests, and debugs transparently
3. **Medallion architecture maps cleanly to lakeFS** — bronze/silver/gold as branch+commit boundaries
4. **MCP integrations extend AI capabilities** — search, journal, GitHub are tools the AI uses autonomously
5. **Multiple models, one interface** — pick GPT-5.3 Codex for speed or Claude Opus 4 for reasoning

---

## Slide 10 — Q&A

- Contact info, links to lakeFS docs, CBorg Studio

---

### Condensed Screenshot Inventory

| Slide | Screenshots | Source |
|-------|------------|--------|
| 3 | `09.27.27`, `09.34.23` | Already captured |
| 4 | `10.57.38`, `10.32.37` | Already captured |
| 5 | lakeFS branch list UI | **Need to capture** |
| 6 | `13.45.21`, `13.45.29`, delay histogram | First two captured; histogram **needs capture** |
| 7 | Comparison table + PR curve, lakeFS commit history | **Need to capture** |
| 8 | `11.50.40`, `13.14.50` | Already captured |
