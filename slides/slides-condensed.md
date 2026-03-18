---
theme: seriph
title: "Flight Delay Prediction with CBorg Studio, lakeFS & AI"
info: 
  From spec to trained model in one session.
  Conference talk on AI-driven data science workflows with data versioning.
  Tim Fong and Andrew Schmeder, Lawrence Berkeley National Laboratory, California USA
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Flight Delay Prediction with CBorg Studio, lakeFS & AI

From spec to trained model in one session

<div class="abs-br m-6 flex gap-2">
  <span class="text-sm opacity-50">LBNL / CBorg Studio</span>
</div>

<!--
Welcome. Today I'll walk through a demo project: predicting US domestic flight delays.
The interesting part isn't the model — it's how AI and data versioning change the workflow.
-->

---
transition: fade-out
---

# The Challenge & The Toolkit

**Can we predict whether a flight will be delayed >15 minutes?**

- 454k flights, 2023 US domestic data, XGBoost binary classifier
- Real question for this talk: **how do AI + data versioning change the data science workflow?**

<div class="grid grid-cols-2 gap-8 mt-4">
<div>

### CBorg Studio
- JupyterHub + **OpenCode** (AI coding agent)
- MCP tool integrations (search, journal, GitHub)
- Multiple models: Open weight (on prem), Commercial Cloud Providers (AWS/GCP/OpenAI/Grok)  
- Models provided to users via a middleware layer (LiteLLM),  
</div>
<div>

### lakeFS
- **Git-like branching and commits for data**
- Every data state is versioned, addressable, auditable
- Branch per experiment, merge the winner

</div>
</div>

<!-- TODO: Replace with architecture diagram showing JupyterHub + OpenCode + lakeFS + MCP connectors -->

<!--
CBorg Studio gives us a JupyterHub environment with an embedded AI coding agent.
lakeFS gives us Git for data — branches, commits, merges, all for Parquet files.
-->

---
layout: two-cols-header
---

# Start with a Spec, Let AI Plan

AI reads the project spec, proposes technology choices, and asks clarifying questions — *before writing any code*.

::left::

<img src="/screenshots/plan-spec.png" class="h-55 rounded shadow" />

<p class="text-xs opacity-60 mt-1">AI reading spec.md, proposing Python 3.11, Parquet, XGBoost, lakeFS SDK</p>

::right::

<img src="/screenshots/encoding-tradeoff.png" class="h-55 rounded shadow" />

<p class="text-xs opacity-60 mt-1">AI presents encoding tradeoffs; user selects "1 please"</p>

<!--
The AI doesn't just execute blindly. It reads the spec, proposes a tech stack, then asks structured
questions about tradeoffs — like which encoding strategy to use — and waits for your decision.
-->

---
layout: two-cols-header
---

# Todo-Driven Execution with Memory

Plan becomes a checklist. AI picks the next task, implements with TDD, checks it off — and **remembers across sessions**.

::left::

<img src="/screenshots/do-todo-journal-read.png" class="h-55 rounded shadow" />

<p class="text-xs opacity-60 mt-1">AI consults its journal before starting work</p>

::right::

<img src="/screenshots/todo-complete-journal-write.png" class="h-55 rounded shadow" />

<p class="text-xs opacity-60 mt-1">After completing tasks, AI records learnings back to journal</p>

<!--
The AI has a private journal — an MCP tool. Before each task it reads past entries for context.
After completing work, it writes back what it learned. This gives it continuity across sessions.
It's not a blank slate every time.
-->

---

# Medallion Architecture on lakeFS Branches

Each data layer is a **lakeFS branch + commit** — versioned, reproducible, auditable.

```mermaid
gitGraph
  commit id: "repo created"
  commit id: "bronze: raw 463k rows"
  branch silver
  commit id: "silver: cleaned 454k rows"
  branch experiment-time-features
  commit id: "gold: time features"
  commit id: "train XGBoost"
  checkout silver
  branch experiment-route-features
  commit id: "gold: route features"
  commit id: "train XGBoost "
  checkout main
  merge experiment-time-features id: "merge winner"
```

<div class="mt-2 text-sm">

- **Bronze** (main): 463k raw rows as Parquet
- **Silver** (branch): cleaned to 454k, binary target `is_delayed = arrival_delay > 15`
- Temporal train/test split: Jan–Jun train, Jul–Aug test — *recommended by AI* to avoid data leakage

</div>

<!-- TODO: Add lakeFS UI screenshot showing branch list for flight-delay-demo -->

<!--
We map the medallion architecture directly onto lakeFS branches.
Bronze is raw data on main. Silver branches off, cleans, creates the target variable.
Then each experiment gets its own branch from silver. The winner merges back to main.
The AI actually recommended the temporal split — it deviated from the spec to avoid future-data leakage.
-->

---

# EDA: What Does the Data Look Like?

<div class="grid grid-cols-2 gap-4">

<div>
<img src="/screenshots/eda-airline-delay.png" class="h-52 rounded shadow" />
<p class="text-xs opacity-60 mt-1">Frontier, JetBlue, Spirit lead in average delays</p>
</div>

<div>
<img src="/screenshots/eda-top-routes.png" class="h-52 rounded shadow" />
<p class="text-xs opacity-60 mt-1">DEN-ABE route averages 1,000+ min delay</p>
</div>

</div>

<div class="mt-4 grid grid-cols-3 gap-4 text-center">

<div class="bg-blue-50 dark:bg-blue-900 rounded p-3">
  <div class="text-2xl font-bold">78%</div>
  <div class="text-sm">On-time</div>
</div>

<div class="bg-orange-50 dark:bg-orange-900 rounded p-3">
  <div class="text-2xl font-bold">22%</div>
  <div class="text-sm">Delayed</div>
</div>

<div class="bg-gray-50 dark:bg-gray-800 rounded p-3">
  <div class="text-sm">Train: 21.5% delayed</div>
  <div class="text-sm">Test: 24.5% delayed</div>
  <div class="text-xs opacity-60 mt-1">Seasonal shift the temporal split reveals</div>
</div>

</div>

<!-- TODO: Add delay distribution histogram with 15-min threshold line (from notebook 01 section 1.3) -->

<!--
Key EDA findings: the dataset is imbalanced — 78% on-time, 22% delayed.
Certain airlines and routes are far worse than average.
The temporal split shows a seasonal shift in delay rates between train and test sets,
which is exactly the kind of thing you'd miss with a random split.
-->

---

# Branching for Experiments: Head-to-Head

<div class="grid grid-cols-2 gap-6">
<div>

### Experiment A: Time Features
Branch `experiment-time-features`

- hour, day_of_week, month, is_weekend, time_of_day_bucket
- \+ airline, origin, distance
- Label encoding

</div>
<div>

### Experiment B: Route Features
Branch `experiment-route-features`

- Frequency encoding for 348 origins, 6,088 routes
- Leakage-safe delay rates (training split only)
- Global-rate fallback for unseen categories

</div>
</div>

<div class="mt-4">

| Metric | Exp A (Time) | Exp B (Route) | Winner |
|--------|:-----------:|:------------:|:------:|
| F1     | **0.371**   | 0.030        | A      |
| AUC-PR | **0.495**   | 0.310        | A      |
| Recall | **0.276**   | 0.016        | A      |

</div>

<div class="mt-2 text-sm">

Winner merged to `main`. Losing branch preserved — you can always go back and inspect it.

</div>

<!-- TODO: Add overlay PR curve from notebook 02 section 2.4 -->
<!-- TODO: Add lakeFS UI screenshot showing main branch commit history after merge -->

<!--
Each experiment gets its own branch from silver. Experiment A uses time-based features,
Experiment B uses route-based features with frequency encoding.
Time features win decisively across every metric. Route features alone produce near-zero recall.
We merge the winner back to main — the losing branch stays around for inspection.
-->

---
layout: two-cols-header
---

# AI That Debugs and Tests with Discipline

::left::

### Transparent Debugging

- lakeFS server unreachable
- AI reported **exactly** what it tried: curl, SDK, integration test
- Clearly stated what it did **not** try
- No hallucination, no silent failures

<img src="/screenshots/debug-report.png" class="h-40 rounded shadow mt-2" />

::right::

### TDD Discipline

- Every phase: write failing test, implement, green, commit
- Subagent sessions handle tasks autonomously
- Reports back with diffs and status

<img src="/screenshots/tdd-subagent.png" class="h-40 rounded shadow mt-2" />

<!--
Two things that set this apart from typical AI code generation.
First: when something breaks, the AI doesn't hallucinate a fix. It reports structured debugging output
showing exactly what it tried, what worked, what didn't, and where it got stuck.
Second: it follows TDD discipline. Write a failing test first, implement until green, commit.
-->

---

# Key Takeaways

<div class="grid grid-cols-1 gap-4 mt-6">

<v-clicks>

<div class="flex items-start gap-3">
  <div class="text-2xl">1.</div>
  <div><strong>lakeFS branches replace ad-hoc experiment tracking</strong> — every data state is a versioned, addressable snapshot</div>
</div>

<div class="flex items-start gap-3">
  <div class="text-2xl">2.</div>
  <div><strong>AI doesn't just generate code</strong> — it plans, asks questions, tracks progress, writes tests, and debugs transparently</div>
</div>

<div class="flex items-start gap-3">
  <div class="text-2xl">3.</div>
  <div><strong>Medallion architecture maps cleanly to lakeFS</strong> — bronze / silver / gold as branch + commit boundaries</div>
</div>

<div class="flex items-start gap-3">
  <div class="text-2xl">4.</div>
  <div><strong>MCP integrations extend AI capabilities</strong> — search, journal, GitHub are tools the AI uses autonomously</div>
</div>

<div class="flex items-start gap-3">
  <div class="text-2xl">5.</div>
  <div><strong>Multiple models, one interface</strong> — pick GPT-5.3 Codex for speed or Claude Opus 4 for reasoning</div>
</div>

</v-clicks>

</div>

<!--
Five things to take away.
lakeFS gives you Git for data — no more ad-hoc experiment tracking.
The AI agent is more than a code generator — it plans, collaborates, tests, and debugs.
Medallion architecture maps naturally onto lakeFS branches.
MCP integrations let the AI use external tools autonomously.
And you can pick the right model for the job.
-->

---
layout: center
class: text-center
---

# Questions?

<div class="mt-8 text-lg">

CBorg Studio &middot; lakeFS &middot; OpenCode

</div>

<div class="mt-4 text-sm opacity-60">

<!-- TODO: Add contact info, links to lakeFS docs, CBorg Studio -->

</div>

<!--
Happy to take questions.
-->
