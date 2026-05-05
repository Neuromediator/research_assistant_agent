---
title: Research Assistant Agent
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
---

<!--
The block above is YAML frontmatter consumed by Hugging Face Spaces to
configure the deployed Space (title, SDK, entry file). GitHub renders it
as the leading horizontal-rule + key:value lines below — visually OK,
ignored semantically. Don't remove it: the deploy workflow pushes this
file verbatim to the Space, where the frontmatter IS load-bearing.
-->

# Research Assistant Agent

A web research agent that turns a topic + depth into a markdown report with **inline citations grounded in sources the agent actually fetched and read**. Built as a 4-agent CrewAI Flow on Anthropic Claude, deployed to Hugging Face Spaces.

![CI](https://github.com/Neuromediator/research_assistant_agent/actions/workflows/ci.yml/badge.svg)

> **Live demo:** [huggingface.co/spaces/Neuromediator/research-assistant-agent](https://huggingface.co/spaces/Neuromediator/research-assistant-agent)

---

## What it does

You give it:

- a **topic** (free-form text, ≤500 chars)
- a **depth** — `quick` (≈300 words, ~8 sources), `standard` (≈800 words, ~15 sources), or `deep` (≈2000 words, ~25 sources)

It produces a markdown report where **every factual claim carries a footnote pointing at a real URL the agent actually fetched and excerpted**. No phantom citations.

Behind the scenes, four specialist agents run sequentially with a bounded one-shot retry:

```
planner  →  searcher  →  critic  →  [retry once if needed]  →  writer
(Sonnet)    (Haiku)      (Sonnet)                              (Sonnet)
```

Each agent has a narrow, testable job. The split is what makes hallucinated citations hard: search/fetch is separate from claim/cite, and the critic is paid to push back when an excerpt doesn't actually support its claim.

---

## Quick start (local)

You need Python 3.10–3.13, [uv](https://docs.astral.sh/uv/), and the API keys listed in `.env.example`.

```bash
# 1. Clone and install
git clone https://github.com/Neuromediator/research_assistant_agent.git
cd research_assistant_agent
uv sync

# 2. Configure secrets
cp .env.example .env
# edit .env — fill in ANTHROPIC_API_KEY, SERPER_API_KEY, LANGFUSE_* (optional)

# 3a. Run the Gradio UI (recommended)
uv run python app.py
# → opens http://localhost:7860

# 3b. Or run from the CLI
uv run run_crew "OpenTelemetry context propagation in Python" quick
# → prints the markdown report; saves it under outputs/
```

Approximate cost per run (Anthropic + Serper combined): **~$0.05–0.10** for `quick`, **~$0.30** for `standard`, **~$0.40+** for `deep`. **Configure a hard monthly spend cap on console.anthropic.com before deploying anywhere public.**

---

## Architecture

```
                 ┌──────────────────────────────┐
                 │  Gradio UI / CLI             │  ResearchInput {topic, depth}
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │  ResearchFlow (CrewAI Flow)  │  one trace per request
                 └──────────────┬───────────────┘
                                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ planner │ ──▶ │searcher │ ──▶ │ critic  │ ──▶ │ writer  │ ──▶ Report
   │ Sonnet  │     │ Haiku + │     │ Sonnet  │     │ Sonnet  │
   └─────────┘     │ tools   │     └────┬────┘     └─────────┘
                   └─────────┘          │
                                        │ ok=false (max once)
                                        ▼
                                  ┌─────────┐
                                  │ retry-  │
                                  │ search  │
                                  └─────────┘
```

- **planner (Sonnet 4.6)** — decomposes the topic into N concrete sub-questions.
- **searcher (Haiku 4.5)** — runs Serper searches, fetches articles via the custom `CleanArticleExtractor` (trafilatura + SQLite cache), pulls a ≤800-char excerpt per claim. Lower-cost model because most of its tokens are tool I/O, not reasoning.
- **critic (Sonnet 4.6)** — checks each claim against its excerpt; flags weak claims and missing sub-questions.
- **writer (Sonnet 4.6)** — composes the final markdown with footnote citations; never adds claims absent from the findings.

The verifier loop is bounded — exactly **one retry** allowed, so the agent can't run away on cost. Retry budgeting lives in plain Python state, not an LLM-driven loop.

Structured outputs (Pydantic models) are passed between every stage instead of free-form strings — bad LLM output fails fast at the deserialization boundary instead of corrupting downstream stages silently.

---

## Tech stack

| Concern         | Choice                                             |
| --------------- | -------------------------------------------------- |
| Agent framework | CrewAI 1.14.4 (`Crew` + `Flow`)                    |
| LLM provider    | Anthropic — Sonnet 4.6 + Haiku 4.5                 |
| Web search      | Serper (Google)                                    |
| Article fetch   | `trafilatura` (strips nav/ads/comments)            |
| UI              | Gradio                                             |
| Deployment      | Hugging Face Spaces (public, free CPU)             |
| Observability   | Langfuse v3 + OpenInference (CrewAI + Anthropic)   |
| Cache           | SQLite, in `.cache/articles.db`                    |
| Tests           | pytest (mocked Anthropic + mocked Serper, no real APIs in CI) |
| Lint / format   | ruff                                               |
| Python          | ≥3.10, <3.14                                       |
| Package mgr     | uv                                                 |
| CI / CD         | GitHub Actions → push to HF Space on `main`        |

---

## Development

```bash
uv run pytest -q              # 24 tests, all mocked, ~3 s
uv run ruff check .           # lint
uv run ruff format .          # format

uv run run_crew "<topic>" quick   # smoke-test a single run
```

Tests never hit real APIs — they mock the LLM via CrewAI's stubbing and Serper at the HTTP layer. CI runs lint + format check + tests on every push and PR (see `.github/workflows/ci.yml`).

---

## Observability

If you set `LANGFUSE_*` in `.env`, every run produces one trace at [cloud.langfuse.com](https://cloud.langfuse.com) showing:

- the four stage spans (`flow.plan`, `flow.search`, `flow.critique`, `flow.write`) with Input / Output summaries
- nested Crew → Agent → Tool spans (Serper, Article Extractor)
- LLM generation spans for the planner / critic / writer (model + tokens + cost)

There is one **known gap**: the searcher's Haiku LLM calls don't yet appear in traces because they go through the Anthropic beta tool-use endpoint, which the OpenInference 0.1.x instrumentor doesn't wrap. Searcher spend is still summed into the `[run summary]` line printed at the end of each run.

Without Langfuse keys the app runs fine — tracing is silently disabled.

---

## Deploying to Hugging Face Spaces

`app.py` at the repo root is the HF Spaces entry point. Steps:

1. Create a Space at huggingface.co, SDK = Gradio.
2. Add **Space Secrets**: `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, optionally `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`.
3. Add the **GitHub Secrets** `HF_TOKEN`, `HF_USERNAME`, `HF_SPACE_NAME` to enable auto-deploy via `.github/workflows/deploy.yml`.
4. Push to `main`. CI runs first (lint + tests); on green, the deploy job mirrors the repo to your Space, including a freshly-generated `requirements.txt`.

---

## Repo layout

```
src/research_assistant_agent/
├── flow.py              # ResearchFlow — the verifier loop
├── crew.py              # Agent / Task / Crew factories (@CrewBase)
├── models.py            # Pydantic structured-output contracts
├── ui.py                # Gradio Blocks
├── observability.py     # Langfuse + OpenInference bootstrap
├── main.py              # CLI entry (`uv run run_crew`)
├── tools/
│   └── article_extractor.py
└── config/
    ├── agents.yaml      # role / goal / backstory per agent
    └── tasks.yaml       # task descriptions + expected outputs

app.py                   # HF Spaces entry; thin wrapper around ui.py
tests/                   # mocked unit + integration tests
.github/workflows/
├── ci.yml               # lint + format + pytest on every push / PR
└── deploy.yml           # mirror to HF Space on push to main
```

---

## Status

This is a personal portfolio project — first end-to-end agent build for the author. v1 ships the core loop, observability, and a public demo. Open polish items:

- Wrap the Anthropic beta endpoint to fully observe the searcher's LLM calls.
- Reduce per-run token cost further.
- Optional: source ingestion beyond HTML (PDF, arXiv, YouTube transcripts).
