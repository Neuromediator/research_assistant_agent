# CLAUDE.md

Guidance for Claude Code sessions working in this repo.

**Read [SPEC.md](./SPEC.md) before making architectural decisions.** It captures the design choices already locked in via interview — don't re-litigate them without asking the user.

> **🔒 Never read `.env`.** It contains live API keys (Anthropic, Serper as for now). Reading it exposes the values in this conversation's transcript. If you need to know which api keys are inside this file, ask user about it.

---

## Project summary

A web research agent that turns a topic + depth into a markdown report with **inline citations grounded in sources the agent actually fetched and read**. Built as a 4-agent CrewAI Flow (planner → searcher → critic → writer) with a one-shot verifier retry. Web search via Serper, article extraction via a custom `trafilatura`-backed tool. Gradio UI deployed to Hugging Face Spaces. First-time AI-agent project for the user — favor clarity over cleverness.

## Goals

1. **Ship a real, deployed agent end-to-end** — not a notebook demo. Code on GitHub, app live on HF Spaces, CI/CD pushing on merge to `main`.
2. **Build mental model of multi-agent systems** — concretely understand when to split work across agents, how CrewAI Flow differs from Crew, and how structured outputs flow between stages.
3. **Practice grounded generation** — make hallucinated citations impossible by separating "search/fetch" from "claim/cite" and adding a critic that checks claims against fetched text.
4. **Learn the surrounding craft** — custom tools, observability (Langfuse), cost discipline (mixed Sonnet/Haiku), test discipline (mocked integration, no real APIs in CI).
5. **Produce a portfolio-quality artifact** — a public Space link the user can show off, backed by a clean repo a stranger can read in 10 minutes.

## Success criteria

The project is "done" when **all** of these are true:

- [ ] Live HF Space accepts a topic + depth and returns a cited markdown report
- [ ] Every factual claim in the output links to a source URL the agent actually fetched (verifiable by checking the report's footnotes against `findings`)
- [ ] One full `standard`-depth run finishes in < 90s and costs < $0.20 in API spend
- [ ] `pytest -q` green; `ruff check` clean; CI/CD auto-deploys on push to `main`
- [ ] Anthropic console has a hard monthly spend cap configured
- [ ] `README.md` lets a stranger run the project locally and understand the architecture without reading the code
- [ ] User can explain — in their own words — why the architecture is split into 4 agents and what each one's success criterion is

## Tech stack

- Python >=3.10,<3.14, managed with `uv`
- CrewAI 1.14.4 (`Crew` + `Flow`)
- Anthropic: Sonnet 4.6 (planner/critic/writer), Haiku 4.5 (searcher)
- Gradio for UI, HF Spaces for deploy
- `trafilatura` for article extraction, SQLite for fetch cache
- `pytest` for tests, `ruff` for lint+format
- Langfuse free tier for tracing

## Common commands

```bash
uv sync                     # install deps
uv run crewai run           # run the crew locally (CLI, default inputs)
uv run python app.py        # run the Gradio UI locally
uv run pytest -q            # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
```

## Repo layout

- `src/research_assistant_agent/flow.py` — the Flow orchestrating the verifier loop
- `src/research_assistant_agent/crew.py` — `@CrewBase` factory (agents + tasks)
- `src/research_assistant_agent/config/agents.yaml` — agent prompts (role/goal/backstory)
- `src/research_assistant_agent/config/tasks.yaml` — task descriptions and expected outputs
- `src/research_assistant_agent/tools/article_extractor.py` — the one custom tool
- `src/research_assistant_agent/models.py` — Pydantic models for structured outputs
- `src/research_assistant_agent/ui.py` — Gradio Blocks
- `app.py` (repo root) — HF Spaces entry, thin wrapper around `ui.py`
- `tests/` — unit + mocked-integration tests, no real API calls

## CrewAI references (`.claude/skills/`)

The repo ships with a set of skills under `.claude/skills/` that document CrewAI patterns and APIs:

- `getting-started/` — project scaffolding, `Crew` vs `Flow` vs `Agent.kickoff()`, YAML configuration, when to use what
- `design-agent/` — agent role/goal/backstory design, LLM selection, tool assignment, knowledge sources, custom tools
- `design-task/` — task descriptions, expected outputs, structured outputs (`output_pydantic`), dependencies via `context`, guardrails
- `ask-docs/` — query the official CrewAI docs for anything the above skills don't cover
- `find-skills/` — discover other skills

**When to use them:** Before writing or modifying agents, tasks, tools, or the Flow, consult the relevant skill rather than guessing from training-data memory of CrewAI. Versions and APIs change; the skills reflect 1.14.4.

## Conventions

- **Configure agents in YAML, not Python.** `agents.yaml` and `tasks.yaml` are the source of truth for prompts; `crew.py` just wires them up. If you need conditional logic, that belongs in the Flow.
- **Structured outputs everywhere.** Every task returns a Pydantic model from `models.py` via `output_pydantic=`. Don't pass freeform strings between stages.
- **No real API calls in tests.** Mock Anthropic via `crewai`'s LLM stubbing; mock Serper at the HTTP layer. Tests must run offline in <10s.
- **Article cache is dev-only convenience, not a correctness mechanism.** Don't rely on cache contents for tests; clear it (`./.cache/articles.db`) freely.
- **Cost discipline.** When debugging, prefer running individual stages in isolation over re-running the full Flow. Use the Langfuse trace to see where tokens went.

# Learning Mode
When implementing features, explain the WHY behind architectural decisions.
When using CrewAI constructs ( for example Agents, Tasks, Crews, Tools), 
reference which CrewAI concept is being used and why.
Don't silently fix errors — explain what went wrong first.

## Gotchas

- HF Spaces requires `app.py` at the repo root and either `requirements.txt` or `pyproject.toml`. Keep `app.py` thin — real logic lives under `src/`.
- The `SerperDevTool` from `crewai-tools` reads `SERPER_API_KEY` from env; don't pass it explicitly.
- `trafilatura` extraction quality varies by site. Treat empty extraction as a fetch failure, not as "the page has no content".
- The verifier retry budget is exactly 1 — see SPEC.md §2.2. Don't introduce unbounded loops.

## What "done" looks like for a feature

1. Code changes pass `ruff check` and `ruff format --check`
2. `pytest -q` is green
3. A local `python app.py` run produces a sensible report end-to-end
4. The change is reflected in SPEC.md if it changes architecture or contracts
