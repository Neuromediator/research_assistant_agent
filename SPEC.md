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

> **Contract — full body stays out of LLM prompts.** The extractor's clean text is large by design (a long-form article can be 30–80k characters / 10–25k tokens). It must be cached and excerpted, never passed verbatim to an LLM. The searcher reasons over `Finding.excerpt` (capped per `models.py`), and the agents.yaml prompts must explicitly forbid quoting the body wholesale into reasoning. Violating this is what produced the v1 600k-prompt-token regression — see §6.

- Input: URL
- Output: clean main-article markdown text + metadata (title, byline, published date if available)
- Backed by [`trafilatura`](https://trafilatura.readthedocs.io/) for main-content extraction (strips nav, ads, comments)
- Local SQLite cache (`./.cache/articles.db`) keyed on URL — saves re-fetches during dev iteration and on retries within the same session
- Failure modes: dead URL or unparseable page → return a string starting with `ERROR:` (e.g., `ERROR: Could not fetch <url> (request failed: Timeout)`). The searcher's prompt is told to drop any URL whose extractor output begins with `ERROR:` and try the next one. Why a string prefix instead of a dict? CrewAI tools must return strings — the output is fed verbatim into the LLM context, so an in-band sentinel is the simplest control signal and avoids JSON parsing inside the prompt.
- Failures are NOT cached — only successful fetches are written to SQLite. A transient DNS/timeout error must not poison a retry within the same session (the verifier loop in §2.2 may re-issue the same URL).
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

- **Gradio UI:**
  - Per-stage status line (e.g. "🔎 Searching the web for 3 subtopics…") + Gradio progress bar — updated between Flow stages.
  - Rendered markdown report once the writer finishes.
  - "Download .md" button (hidden until the report is ready).
  - **Stop** button — soft cancel: interrupts the run at the next stage boundary. An LLM call already in flight still completes (and bills) — the Anthropic SDK doesn't expose mid-call cancellation. Resets status to "🛑 Research stopped. Try again." and clears report/download.
- **Disk:** `outputs/{ISO-timestamp}_{slug}.md`
- **Report format:** see [Appendix A](#appendix-a--report-format)

> **Implementation note.** The UI's click handler is a generator that walks the same 4 stages as `ResearchFlow` by hand (plan → search → critique → [retry] → write), `yield`-ing between them so Gradio can update the progress bar and honour the Stop button's `cancels=` wiring. `ResearchFlow` itself remains the source of truth for orchestration semantics and is what the CLI / `crewai run` path uses; the duplication is the price of cooperating with Gradio's progress + cancel primitives, which can't observe the inside of a monolithic `flow.kickoff()` call.

---

## 3. Tech stack

| Concern         | Choice                                                |
| --------------- | ----------------------------------------------------- |
| Agent framework | CrewAI 1.14.4 (Flow + Crew)                           |
| LLM provider    | Anthropic (Claude Sonnet 4.6 + Haiku 4.5)             |
| Web search      | Serper (Google)                                       |
| Article fetch   | `trafilatura`                                         |
| UI              | Gradio 6.x                                            |
| Deployment      | Hugging Face Spaces (public, free CPU)                |
| Observability   | CrewAI `verbose=True` + Langfuse 4.x (free tier)      |
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
| **Stage exceeds 2× §9 token budget** | **Treat as a regression, not a runtime error.** The fix is in prompts/contracts (e.g. searcher passing full article bodies instead of excerpts), not retries. Diagnose via Langfuse trace before touching anything else. |
| Serper returns 0 results          | Searcher reports `no_sources`; writer produces a "Could not find sources on X" stub  |
| Article fetch fails (timeout/404) | Drop the URL, continue with remaining; if <2 succeed, mark subtopic incomplete       |
| Critic loops forever              | Hard cap: retry budget = 1. After retry, writer composes whatever exists.            |
| Anthropic rate limit              | Exponential backoff (built into anthropic SDK); fail fast after 3 retries            |
| Anthropic spend runaway           | Hard cap set in console.anthropic.com (recommend $20/mo while learning) — **must be configured before any public deploy.** |
| Prompt injection from web pages   | Article extractor strips HTML; agents told to treat fetched content as data, not instructions |
| Empty / abusive input             | Gradio input validation: 1 ≤ len(topic) ≤ 500, depth must be enum value             |

### Known regressions / open issues

- **v1 600k prompt-token regression on `quick`** — observed during step 12 sanity run: a `quick`-depth run consumed ~618k prompt tokens against §9's 6k estimate (≈100×). Root cause hypothesis: the searcher feeds full extracted article bodies into LLM context across multiple agent steps. Fix lives in `tasks.yaml` / `agents.yaml` — searcher must work from `Finding.excerpt` only. **Tracked as build-sequence step 14 below; must be resolved before HF deploy.**

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

## 9. Cost & latency contract

> **This is a contract, not an estimate.** A run that exceeds **2×** any of the totals below is a regression — diagnose via Langfuse trace before merging or deploying. CI does not enforce this (no real APIs in CI by §7), so the discipline lives in code review and build-sequence step 14.

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

- **CrewAI `verbose=True`** — agent thoughts and tool calls printed to stdout (HF Spaces logs). **Active.**
- **Langfuse free tier (v3+)** — distributed tracing UI; auto-traced via three OpenTelemetry-aware instrumentors:
    - `openinference-instrumentation-crewai` — Crew / Agent / Task / Tool spans (structural picture).
    - `openinference-instrumentation-anthropic` (pinned `<1.0` to match CrewAI 1.14.4's `anthropic==0.73.x` pin) — wraps `anthropic.Anthropic.messages.create`, capturing the planner / critic / writer LLM calls as `generation`-typed spans (model + tokens + cost).
    - A root `research_trace` context manager opened around each `flow.kickoff()` so every nested span lands in **one** trace per request, with `topic` / `depth` on the root.

  The Flow stage methods (`plan`, `search`, `critique`, `retry_search`, `write`) also carry an `@observe` decorator + an explicit `log_stage(input_=…, output=…)` call so each appears as a labelled parent span with readable Input / Output (not `{}`). Initialized via env vars (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`); region defaults to `https://cloud.langfuse.com` (EU). Bootstrap lives in `observability.py` and is called once per process from `main.py` and `app.py`; absent keys disable tracing silently. **Active.**

  **Known gap (v2 polish, not blocking deploy):** the searcher uses `client.beta.messages.create` (the Anthropic beta tool-use endpoint) when reasoning over its Serper + CleanArticleExtractor tools. OpenInference's anthropic instrumentor 0.1.x does NOT wrap the beta endpoint, so the searcher's Haiku LLM calls never appear as `generation` spans in Langfuse — its tool calls are visible, but per-step token / cost detail is missing. Aggregate searcher cost is still captured: `CrewOutput.token_usage` is summed in stdout's `[run summary]` line at end of run. Fix path: write a beta-aware wrapper, or wait for a newer instrumentor compatible with CrewAI's pinned anthropic SDK.
- **Per-run cost summary** — printed to stderr at the end of each run, aggregated from `CrewOutput.token_usage` across all stages (planner/searcher/critic/[retry]/writer). Visible in the same terminal as the report; complements Langfuse's per-trace cost view. **Active.**

---

## 11. Security & privacy

- API keys in `.env` locally (gitignored), HF Secrets in production. Never logged.
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

Steps 1–12 are complete. Steps 13+ reflect the post-§9-regression priority reset: **fix tracing & cost before public deploy**.

1. ✅ Clean up template (remove demo `topic="AI LLMs"`, knowledge folder, default agents/tasks)
2. ✅ Update `pyproject.toml` (add `gradio`, `trafilatura`, `langfuse`, `pytest`, `ruff`)
3. ✅ Build the `CleanArticleExtractor` tool + its unit tests (proves the foundations)
4. ✅ Define Pydantic models in `models.py`
5. ✅ Wire 4 agents in `agents.yaml`, 4 tasks in `tasks.yaml`, factory in `crew.py`
6. ✅ Wire the Flow with verifier loop in `flow.py`
7. ✅ Build Gradio UI in `ui.py` + root `app.py`
8. ✅ Integration test with mocks
9. ✅ Local end-to-end sanity run
10. ✅ Set up `.github/workflows/ci.yml`
11. ✅ Push to GitHub
12. ✅ Wire Langfuse tracing (OpenInference CrewAI instrumentor + `@observe` on Flow stages + `research_trace` root wrapper). One trace per request lands in cloud.langfuse.com; bootstrap centralized in `observability.py`; safe to clone without Langfuse credentials.
13. **Make tracing actionable.** Capture explicit `input` / `output` on each Flow stage's `@observe` (so a stage span isn't `{}`); add Anthropic Sonnet 4.6 + Haiku 4.5 to Project Settings → Models in Langfuse so cost rolls up to the trace root. Verify on a fresh quick run: trace root shows non-zero `Total cost`, and clicking `flow.search` shows readable input/output.
14. **Token-budget audit & fix.** Use the now-actionable trace from step 13 to find which stage burned the ~600k prompt tokens observed during step 12. Fix at the contract layer (`tasks.yaml` and/or `agents.yaml`): searcher reasons over `Finding.excerpt`, not raw article bodies; critic / writer get the same excerpt-only contract. Re-run quick + standard; both must land within 2× of §9 budgets before continuing.
15. **README.md** — short, complete: what the project does, how to run locally (`uv sync` + `uv run run_crew "<topic>" quick` + `uv run python app.py`), architecture diagram or short prose pointing at SPEC for depth, env-var checklist (`.env.example`), CI badge. Stranger-readable in 10 minutes.
16. **Anthropic spend cap** — set in console.anthropic.com (`$20/mo` while learning). Confirmed before step 17.
17. **Create HF Space**, add Space Secrets (`ANTHROPIC_API_KEY`, `SERPER_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`), write `.github/workflows/deploy.yml`. Generate `requirements.txt` from `pyproject.toml` either in CI or as a committed file.
18. **Push to `main` → live demo.** Final verification: open the public Space, run a query, confirm the trace appears in Langfuse with cost and the report cites real fetched URLs.

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
