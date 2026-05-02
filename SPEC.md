# Research Assistant Agent — Spec

> Project goal: an end-to-end, deployed AI research agent built with CrewAI. First-time AI-agent project for the author; aims for "real, demonstrable, and learnable" rather than novel.

---

## 1. What it does

Given a research **topic** and a **depth** (quick / standard / deep), the agent:

1. Plans subtopics to investigate
2. Searches the open web (Google via Serper)
3. Fetches and reads each source's full article text
4. Self-critiques to catch unsupported claims and gaps
5. Re-searches once if the critic finds problems
6. Writes a markdown report with **inline citations**, every claim grounded in a source the agent actually fetched and read

The output is a markdown report rendered in a Gradio UI, downloadable as a `.md` file, and also saved to `outputs/`.

**Non-goals (v1):**

- Multi-turn conversation / memory across runs
- PDF / arXiv / YouTube / Reddit ingestion
- Multilingual sources
- User accounts, billing, multi-tenant

---

## 2. Architecture

### 2.1 Crew — 4 agents

| Agent      | Model        | Job                                                                   | Tools                          |
| ---------- | ------------ | --------------------------------------------------------------------- | ------------------------------ |
| `planner`  | Sonnet 4.6   | Decompose topic into 3–N concrete subtopics / search queries          | none (LLM only)                |
| `searcher` | Haiku 4.5    | Run Serper queries, pick top URLs, fetch full text via custom tool    | SerperDevTool, ArticleExtractor |
| `critic`   | Sonnet 4.6   | Verify every claim against fetched sources; flag unsupported/missing  | none (LLM only, reads context) |
| `writer`   | Sonnet 4.6   | Compose final markdown report with inline citations and Sources list  | none (LLM only)                |

**Why this split?** The split makes each agent's prompt and success criterion narrow and testable. Haiku for the searcher because most of its tokens go into tool I/O — reasoning isn't the bottleneck. Sonnet for the others because planning, critique, and writing benefit from reasoning quality.

### 2.2 Process — CrewAI Flow with bounded retry

CrewAI Flow (not a plain sequential Crew) — needed because the verifier loop is a real conditional branch:

```
START
  │
  ▼
[planner]
  │  → subtopics: list[str]
  ▼
[searcher]
  │  → findings: list[{claim, url, excerpt}]
  ▼
[critic]
  │  → {ok: bool, missing_claims: list[str], weak_claims: list[str]}
  │
  ├─ ok=true ─────────────┐
  │                       ▼
  └─ ok=false → [searcher (retry, scoped to gaps)]
                          │
                          ▼
                       [writer]
                          │
                          ▼
                        END → markdown report
```

**Retry budget: exactly 1.** Bounded so the agent can't spin or run away on cost.

### 2.3 Tools

**Built-in:** `SerperDevTool` from `crewai-tools`. Configured per-depth:

- quick: top 3 results / subtopic
- standard: top 6 / subtopic
- deep: top 12 / subtopic

**Custom: `CleanArticleExtractor`**

- Input: URL
- Output: clean main-article markdown text + metadata (title, byline, published date if available)
- Backed by [`trafilatura`](https://trafilatura.readthedocs.io/) for main-content extraction (strips nav, ads, comments)
- Local SQLite cache (`./.cache/articles.db`) keyed on URL — saves re-fetches during dev iteration and on retries within the same session
- Failure modes: dead URL → return `{"error": "fetch_failed", "url": ...}` so the searcher can drop and move on
- Timeout: 15s per fetch
- User-Agent: identifying the project (politeness)

### 2.4 Inputs

| Field   | Type   | Values                            | Default    |
| ------- | ------ | --------------------------------- | ---------- |
| `topic` | str    | non-empty, ≤500 chars              | required   |
| `depth` | enum   | `quick` \| `standard` \| `deep`   | `standard` |

**Depth → behavior:**

| depth      | sources / subtopic | total sources cap | report length target |
| ---------- | ------------------ | ----------------- | -------------------- |
| `quick`    | 3                  | 8                 | ~300 words           |
| `standard` | 6                  | 15                | ~800 words           |
| `deep`     | 12                 | 25                | ~2000 words          |

### 2.5 Outputs

- **Gradio UI:** rendered markdown + a "Download report" button
- **Disk:** `outputs/{ISO-timestamp}_{slug}.md`
- **Report format:** see [Appendix A](#appendix-a--report-format)

---

## 3. Tech stack

| Concern         | Choice                                                |
| --------------- | ----------------------------------------------------- |
| Agent framework | CrewAI 1.14.4 (Flow + Crew)                           |
| LLM provider    | Anthropic (Claude Sonnet 4.6 + Haiku 4.5)             |
| Web search      | Serper (Google)                                       |
| Article fetch   | `trafilatura`                                         |
| UI              | Gradio                                                |
| Deployment      | Hugging Face Spaces (public, free CPU)                |
| Observability   | CrewAI `verbose=True` + Langfuse (free tier)          |
| Cache           | SQLite (file-based, in `./.cache/`)                   |
| Tests           | pytest, with mocked Anthropic + mocked Serper         |
| Lint/format     | ruff (single tool: `ruff check` + `ruff format`)      |
| CI/CD           | GitHub Actions → push to HF Space on `main`           |
| Python          | >=3.10, <3.14                                         |
| Package mgr     | uv                                                    |

---

## 4. Project structure (target)

```
research_assistant_agent/
├── .github/workflows/
│   ├── ci.yml               # ruff + pytest on PRs and main
│   └── deploy.yml           # push to HF Space on main
├── src/research_assistant_agent/
│   ├── __init__.py
│   ├── main.py              # CLI entry: `crewai run` (reads topic from CLI args)
│   ├── ui.py                # Gradio Blocks definition
│   ├── flow.py              # ResearchFlow (CrewAI Flow with verifier loop)
│   ├── crew.py              # Crew + agent factories (read from config/)
│   ├── models.py            # Pydantic models: Subtopic, Finding, CritiqueResult, Report
│   ├── config/
│   │   ├── agents.yaml      # 4 agents
│   │   └── tasks.yaml       # 4 tasks (one per agent stage)
│   └── tools/
│       ├── __init__.py
│       └── article_extractor.py
├── tests/
│   ├── test_article_extractor.py
│   ├── test_flow_mocked.py
│   └── conftest.py          # fixtures: mock anthropic, mock serper
├── outputs/                 # gitignored, created at runtime
├── .cache/                  # gitignored, SQLite article cache
├── app.py                   # HF Spaces entry — thin wrapper around src/.../ui.py
├── .env.example             # template; real .env stays gitignored
├── .gitignore
├── pyproject.toml
├── requirements.txt         # generated from pyproject for HF Spaces
├── README.md                # user-facing
├── SPEC.md                  # this file
└── CLAUDE.md                # guidance for Claude Code sessions
```

The top-level `app.py` is required by HF Spaces convention; it just imports and launches the Gradio Blocks defined in `src/research_assistant_agent/ui.py`.

---

## 5. Data contracts

Pydantic models in `models.py` (used as `output_pydantic` on tasks for structured outputs):

```python
class Subtopic(BaseModel):
    question: str
    rationale: str          # why the planner chose this

class Finding(BaseModel):
    claim: str              # one factual statement
    source_url: str
    source_title: str
    excerpt: str            # ≤500 chars from the source supporting the claim
    fetched_at: datetime

class CritiqueResult(BaseModel):
    ok: bool
    weak_claims: list[str]      # claims with shaky evidence
    missing_claims: list[str]   # gaps the report should address
    notes: str

class Report(BaseModel):
    topic: str
    depth: Literal["quick", "standard", "deep"]
    markdown: str           # the rendered report
    sources: list[str]      # deduped URLs used
    generated_at: datetime
```

---

## 6. Failure modes & guardrails

| Mode                              | Handling                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Serper returns 0 results          | Searcher reports `no_sources`; writer produces a "Could not find sources on X" stub  |
| Article fetch fails (timeout/404) | Drop the URL, continue with remaining; if <2 succeed, mark subtopic incomplete       |
| Critic loops forever              | Hard cap: retry budget = 1. After retry, writer composes whatever exists.            |
| Anthropic rate limit              | Exponential backoff (built into anthropic SDK); fail fast after 3 retries            |
| Anthropic spend runaway           | Hard cap set in console.anthropic.com (recommend $20/mo while learning)              |
| Prompt injection from web pages   | Article extractor strips HTML; agents told to treat fetched content as data, not instructions |
| Empty / abusive input             | Gradio input validation: 1 ≤ len(topic) ≤ 500, depth must be enum value             |

---

## 7. Testing strategy

**Unit (`tests/test_article_extractor.py`):**

- Fixture HTML pages → assert clean text extraction
- Cache hit returns without network call (mock `requests`)
- Bad URL returns error dict, doesn't raise

**Integration (`tests/test_flow_mocked.py`):**

- Full Flow run with mocked Anthropic (deterministic stub responses) and mocked Serper
- Asserts: writer output references at least one URL from the searcher's findings
- Asserts: critic-flagged retry actually triggers searcher again
- Run time target: <10s, cost: $0

**No real-API tests in CI** — would couple test reliability to provider uptime and burn money.

---

## 8. CI/CD

### 8.1 `ci.yml` — runs on every PR and push

1. Setup Python 3.12 + uv
2. `uv sync`
3. `ruff check .`
4. `ruff format --check .`
5. `pytest -q`

### 8.2 `deploy.yml` — runs on push to `main` only

1. Setup Python 3.12 + uv
2. Generate `requirements.txt` from `pyproject.toml`
3. `git push https://USER:HF_TOKEN@huggingface.co/spaces/{user}/{space} main`

**GitHub Secrets required:** `HF_TOKEN`, `HF_USERNAME`, `HF_SPACE_NAME`.
**HF Space Secrets required:** `ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.

---

## 9. Cost & latency budget

Per query (standard depth, no retry):

| Stage     | Tokens (est.)        | Cost (est.) |
| --------- | -------------------- | ----------- |
| Planner   | 1k in / 0.5k out     | ~$0.01      |
| Searcher  | 5k in / 2k out       | ~$0.02 (Haiku) |
| Critic    | 8k in / 1k out       | ~$0.04      |
| Writer    | 8k in / 2k out       | ~$0.05      |
| Serper    | 6 queries × $0.001   | ~$0.006     |
| **Total** |                      | **~$0.13**  |

With one retry: ~$0.18. Quick depth: ~$0.05. Deep depth: ~$0.40.

Latency target: < 90s for `standard`, < 180s for `deep`.

Anthropic console hard cap: **$20/mo** recommended while learning.

---

## 10. Observability

- **CrewAI `verbose=True`** — agent thoughts and tool calls printed to stdout (HF Spaces logs)
- **Langfuse free tier** — distributed tracing UI; one span per agent run, nested tool calls, token + cost tracking. Initialized via env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`).
- **Per-run cost summary** — printed at end of each run (sum across agents, from CrewAI's usage_metrics)

---

## 11. Security & privacy

- API keys in `.env` locally (gitignored), HF Secrets in production. Never logged.
- The Anthropic key currently in `.env` is in this conversation's transcript — **rotate before deploying**.
- HF Space is public. Mitigation: Anthropic spend cap. Acceptable risk for portfolio piece.
- Fetched web content is treated as data, not instructions. Agents are prompted to ignore meta-instructions found inside source pages.
- No PII collection; no analytics; topic queries logged to Langfuse only (private).

---

## 12. Out of scope / explicit v2

- Multi-turn chat
- Source ingestion beyond HTML pages (PDF, YouTube, Reddit, arXiv)
- User-supplied document upload
- Caching final reports (only article fetches are cached)
- Multilingual research
- Auth beyond Gradio's `auth=` kwarg

---

## 13. Build sequence (order I'll implement)

1. Clean up template (remove demo `topic="AI LLMs"`, knowledge folder, default agents/tasks)
2. Update `pyproject.toml` (add `gradio`, `trafilatura`, `langfuse`, `pytest`, `ruff`)
3. Build the `CleanArticleExtractor` tool + its unit tests (proves the foundations)
4. Define Pydantic models in `models.py`
5. Wire 4 agents in `agents.yaml`, 4 tasks in `tasks.yaml`, factory in `crew.py`
6. Wire the Flow with verifier loop in `flow.py`
7. Build Gradio UI in `ui.py` + root `app.py`
8. Integration test with mocks
9. Local end-to-end sanity run
10. Set up `.github/workflows/ci.yml`
11. Push to GitHub
12. Create HF Space, set up Secrets, set up `deploy.yml`
13. Push to `main` → live demo

---

## Appendix A — Report format

Markdown structure the writer agent must produce:

```markdown
# {Topic}

*Generated {date} • Depth: {depth} • {N} sources*

## TL;DR

{2–3 sentence summary}

## {Subtopic 1}

{Prose with inline citations like [^1] [^2]}

## {Subtopic 2}

...

## Sources

[^1]: [{title}]({url}) — {publisher, date if known}
[^2]: ...
```

Every factual claim in the prose must carry a footnote referencing a real source from `findings`. The critic agent's job is to enforce this.

---

## Appendix B — Open questions deferred to implementation

- Exact prompt wording per agent (will iterate during build)
- Whether to use `output_file` on the writer task vs writing markdown manually after the Flow returns (depends on how cleanly CrewAI Flow integrates with task `output_file`)
- Langfuse self-hosted vs cloud free tier (default to cloud free tier; switch if rate-limited)
- Whether `requirements.txt` is auto-generated by CI or committed (start with: committed, regenerated as part of `deploy.yml`)
