# Slide Deck Storyboard v2: CBorg Studio — Platform-Focused

## Narrative Arc
Problem (why integration matters) → Platform (what CBorg Studio is) → Components (CBORG API, lakeFS, OpenCode) → Demo (5 slides showing the platform in action) → Experimental capabilities → Takeaways

## Constraints
- AI agent is "OpenCode" throughout
- VS Code: mention it exists via JupyterHub launcher but don't feature it (not demo-ready)
- Skip prompt logging/introspection
- TODO placeholders for missing screenshots
- Speaker notes on every slide

## Assets Available
Screenshots (all in `slides/public/screenshots/`):
- `plan-spec.png` — AI reading spec, proposing tech stack
- `encoding-tradeoff.png` — encoding strategy tradeoff discussion
- `do-todo-journal-read.png` — AI consulting journal before work
- `todo-complete-journal-write.png` — AI recording learnings after work
- `debug-report.png` — transparent debugging report
- `tdd-subagent.png` — TDD subagent session
- `eda-airline-delay.png` — average delay by airline chart
- `eda-top-routes.png` — top 20 most delayed routes chart

Images (in `slides/public/images/`):
- `cborg-logo.png` — 500x500 CBORG logo

---

## Act 1: Context and Motivation (slides 1–3)

### Slide 1 — Title

- "CBorg Studio: An AI-Ready Data Science Environment for Scientific Programming"
- Authors: Tim Fong and Andrew Schmeder
- ScienceIT Department, Lawrence Berkeley National Laboratory, California USA
- CBORG logo (large, `h-48`)
- Speaker notes: Welcome, set expectation — this is about a platform, we'll show a real project built with it.

### Slide 2 — The Problem

- Scientists need compute, AI models, data versioning, and coding tools — but they arrive as disconnected pieces
- Each tool requires its own setup: API keys, environment config, model selection, data management
- What if you could log in and have an AI-ready environment with everything integrated?
- No screenshot — text/visual slide
- Speaker notes: Frame from the scientist's perspective. They're not DevOps engineers. Integration is the gap.

### Slide 3 — CBorg Studio Overview

- "What You Get When You Log In"
- Compute: JupyterLab (VS Code available via launcher)
- AI dev: OpenCode (AI coding agent), preloaded prompts/workflows, framework for new agents
- Data: lakeFS for version control — branches, commits, merges for datasets
- CBORG API: LiteLLM gateway (budget enforcement) → on-prem open-weight + commercial cloud models
- Zero-friction: LDAP auth provisions ephemeral API key with budget controls
- TODO: architecture diagram
- Speaker notes: This is the front door. JupyterHub launches, everything is pre-configured. The API key is already there.

---

## Act 2: Platform Components (slides 4–6)

### Slide 4 — CBORG API: One Key, Many Models

- LiteLLM gateway — single endpoint, many backend models
- On-prem open-weight models (Llama etc) — data stays at the lab
- Commercial cloud  — for tasks needing frontier capabilities
- LDAP-provisioned keys with per-user budget controls
- Key injected at login — users don't manage credentials
- TODO: diagram showing LDAP → key → LiteLLM → model backends
- Speaker notes: Zero-friction model access. Log in, call any model. Budget controls prevent runaway costs. On-prem option matters for data sensitivity or researchers with severe budget constraints.

### Slide 5 — lakeFS: Git for Data

- Branches, commits, merges — but for datasets, not source code
- S3-compatible storage backends (on-prem and cloud)
- Every data state is versioned, addressable, auditable
- Branch per experiment, merge the winner, keep losers for inspection
- Integrated into CBorg Studio — accessible from notebooks via Python SDK
- Visual: Mermaid gitGraph diagram (branch structure)
- TODO: lakeFS UI branch list screenshot
- Speaker notes: Data gets the same version control discipline as code. Branches let you experiment without corrupting shared state.

### Slide 6 — OpenCode: The AI Coding Agent

- AI coding agent embedded in the environment
- Plans projects from specs before writing code
- Presents tradeoffs, waits for user decisions
- Follows TDD discipline — failing tests first, then implementation
- MCP tool integrations: search, journal, GitHub
- Preloaded prompts/workflows: `/plan`, `/do-todo`, `/brainstorm`
- Prompts based on Superpowers repo: https://github.com/obra/superpowers
- Incorporates search via a Duck Duck Go MCP server
- Learning integration via a forked version of https://github.com/DrCatHicks/learning-opportunities
- Framework for introducing new coding agents as they emerge
- Screenshot: `plan-spec.png` (AI reading spec, proposing tech stack)
- Speaker notes: OpenCode is not just a code generator. It plans, collaborates, tests, debugs. MCP tools extend its capabilities. The preloaded prompts encode best practices.

---

## Act 3: Demo — Flight Delay Project (slides 7–11)

Each slide frames a demo artifact as evidence of a platform capability.

### Slide 7 — Demo: AI Plans, Then Asks Before Acting

- Project: predict US domestic flight delays (454k flights, 2023 data, XGBoost)
- AI reads `spec.md`, proposes tech stack, identifies risks
- At design decisions (encoding strategy), it presents structured tradeoffs and waits
- Layout: two-cols-header
- Screenshots: `plan-spec.png` (left), `encoding-tradeoff.png` (right)
- Speaker notes: This entire project was built inside CBorg Studio. The AI reads the spec, plans, and collaborates on decisions. Note: `plan-spec.png` appears here and on slide 6. If that feels redundant, remove it from slide 6.

### Slide 8 — Demo: Persistent Memory Across Sessions

- Plan becomes `todo.md` checklist tied to GitHub issues
- `/do-todo` picks next task, implements with TDD, checks it off
- Before starting: AI reads its private journal for context from earlier sessions
- private journal is an MCP server, https://github.com/obra/private-journal-mcp/tree/main
- After completing: AI records learnings — project notes, technical insights, user preferences
- Layout: two-cols-header
- Screenshots: `do-todo-journal-read.png` (left), `todo-complete-journal-write.png` (right)
- Speaker notes: This is "agentic memory." The AI has a private journal (MCP tool). It reads before acting, writes after completing. Not a blank slate every time.

### Slide 9 — Demo: Data Pipeline on lakeFS

- Medallion architecture on lakeFS branches
- Bronze (main): 463k raw rows as Parquet
- Silver (branch): cleaned to 454k rows, binary target `is_delayed`
- Gold (experiment branches): feature engineering + model training
- AI recommended temporal train/test split to avoid future-data leakage — deviating from spec
- Visual: Mermaid gitGraph + EDA charts as small insets
- Screenshots: `eda-airline-delay.png`, `eda-top-routes.png` (small)
- Speaker notes: This is lakeFS in action. Medallion layers map to branches. The AI itself recommended the temporal split. 78% on-time, 22% delayed.

### Slide 10 — Demo: Branch, Experiment, Compare, Merge

- Experiment A (time features): F1 = 0.371, AUC-PR = 0.495
- Experiment B (route features): F1 = 0.030, AUC-PR = 0.310
- Compact 3-row comparison table
- Winner merged to `main` — full lineage in commit log
- Losing branch preserved for inspection
- TODO: lakeFS UI commit history screenshot
- Speaker notes: Each experiment gets its own branch, its own model. Comparison by loading metrics from both. Winner merges back. The commit log tells the full data story. Entire pipeline was AI-driven.

### Slide 11 — Demo: Transparent Debugging and TDD

- lakeFS server unreachable — AI reported exactly what it tried and didn't try
- "I lack the information to diagnose further" — no hallucination
- Every phase follows TDD: failing test → implement → green → commit
- Subagent sessions handle tasks autonomously
- Layout: two-cols-header
- Screenshots: `debug-report.png` (left), `tdd-subagent.png` (right)
- Speaker notes: Two behaviors that distinguish this from typical AI code generation: transparent debugging (no hallucinated fixes) and TDD discipline.

---

## Act 4: Experimental Capabilities (slides 12–13)

### Slide 12 — Experimental AI-First Capabilities

- Agentic memory: journal MCP tool gives AI continuity across sessions (shown in slide 8)
- Data version control: lakeFS branches as experiment isolation and lineage (shown in slides 9–10)
- Agentic workflow orchestration: todo-driven execution with subagent delegation (shown in slides 8, 11)
- Next: agent-to-agent collaboration, cross-project memory, richer introspection
- No screenshot needed — summary/forward-looking slide
- Speaker notes: These capabilities are working today, as demonstrated. What's next: richer cross-project memory, agent collaboration.

### Slide 13 — The Integrated Stack

- User logs in via JupyterHub → LDAP provisions CBORG API key
- JupyterLab launches with OpenCode, lakeFS, MCP tools pre-configured
- Everything demonstrated was built inside this environment — no external setup
- TODO: architecture diagram (same as slide 3, or more detailed)
- Speaker notes: Bring it back to the platform message. Everything we showed happened inside CBorg Studio. The user didn't install anything. Integration eliminates friction.

---

## Act 5: Wrap-Up (slides 14–15)

### Slide 14 — Key Takeaways

- Progressive reveal (v-clicks):
  1. Zero-friction AI access: LDAP login gives you an API key, a coding agent, connected tools — no setup
  2. Data version control changes experimentation: lakeFS branches isolate experiments; commit log is the audit trail
  3. AI agents follow engineering discipline: planning, TDD, transparent debugging, persistent memory
  4. Integration is the product: individual tools exist elsewhere — the value is having them work together out of the box
  5. Experimental capabilities are shipping: agentic memory, data version control, workflow orchestration — all working today
- Speaker notes: Five takeaways. Zero friction. Data versioning. AI discipline. Integration. Working today.

### Slide 15 — Questions?

- CBORG logo (centered, `h-20`)
- "CBorg Studio · lakeFS · OpenCode"
- TODO: contact info, links
- Speaker notes: Thank audience, open for questions.

---

## Screenshot Inventory

### Available (in `slides/public/screenshots/`)

| File | Content | Used in slide(s) |
|------|---------|-------------------|
| `plan-spec.png` | AI reading spec, proposing tech choices | 6, 7 |
| `encoding-tradeoff.png` | Encoding tradeoff discussion | 7 |
| `do-todo-journal-read.png` | AI consulting journal before work | 8 |
| `todo-complete-journal-write.png` | AI recording learnings after work | 8 |
| `debug-report.png` | Transparent debugging report | 11 |
| `tdd-subagent.png` | TDD subagent session | 11 |
| `eda-airline-delay.png` | Average delay by airline chart | 9 |
| `eda-top-routes.png` | Top 20 most delayed routes chart | 9 |

### Still need to capture

| Slide(s) | What to capture | Where to find it |
|----------|----------------|------------------|
| 3, 13 | Architecture diagram (JupyterHub + OpenCode + lakeFS + CBORG API) | Create or screenshot |
| 4 | CBORG API diagram (LDAP → key → LiteLLM → model backends) | Create |
| 5 | lakeFS UI: branch list for `flight-delay-demo` | lakeFS web UI |
| 10 | lakeFS UI: commit history on `main` after merge | lakeFS web UI |
