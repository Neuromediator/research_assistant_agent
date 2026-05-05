"""Gradio UI for the research assistant.

WHY a separate `ui.py` rather than putting Gradio in `app.py`?

HF Spaces requires an `app.py` at the repo root, but per CLAUDE.md the real
logic lives under `src/`. Keeping the Blocks here lets us:

  - Import `demo` from tests / scripts without going through Spaces' entry path.
  - Keep `app.py` to ~3 lines (a launcher), so the deploy surface stays trivial.

WHY validate via `ResearchInput` instead of trusting Gradio's components?

Gradio's `Dropdown` constrains the depth choice client-side, but a determined
caller (or another module that imports `run_research` directly) can still pass
garbage. `ResearchInput` is the Pydantic model already used by `main.py`; using
it here means there's exactly one place that defines what valid input is —
defense in depth, no duplication. The `ValidationError → gr.Error` translation
gives the user a clean message instead of a stack trace.

WHY a generator handler that drives stages directly, instead of `flow.kickoff()`?

Two UX features the user asked for — a real progress bar and a working stop
button — both require boundary points where Gradio can either update the UI or
abort the run. `flow.kickoff()` is one monolithic blocking call; from inside it
there's no way to surface "stage X of N" to Gradio, and Gradio's `cancels=…`
can only interrupt a generator at a `yield` statement, not in the middle of a
synchronous function. So this handler walks the same stages as `flow.py`
(plan → search → critique → [retry] → write) by hand, calling each per-stage
Crew via the shared `ResearchAssistantAgent` factory and yielding between them.
The Flow itself is still the source of truth for orchestration semantics
(retry budget, branch labels) — this UI just rebuilds the same graph in a
shape that cooperates with Gradio's progress + cancel primitives.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import gradio as gr
from pydantic import ValidationError

from research_assistant_agent.crew import ResearchAssistantAgent
from research_assistant_agent.flow import (
    DEPTH_PROFILE,
    MAX_RETRIES,
    expect_pydantic,
    format_initial_brief,
    format_retry_brief,
)
from research_assistant_agent.models import (
    CritiqueResult,
    Depth,
    Findings,
    Report,
    ResearchInput,
    Subtopics,
)
from research_assistant_agent.observability import format_token_summary, research_trace

OUTPUTS_DIR = Path("outputs")
DEPTH_CHOICES: list[Depth] = ["quick", "standard", "deep"]


# ---- Helpers -----------------------------------------------------------


def _slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe lowercase slug for the saved-report filename.

    NFKD-normalize then ASCII-strip so non-Latin scripts don't end up in
    filenames (some filesystems mishandle them in CI/CD pipelines). If the
    input slugifies to nothing (e.g., all CJK), fall back to "report" so we
    never produce an empty stem.
    """
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug:
        return "report"
    return slug[:max_len].rstrip("-") or "report"


def _save_report_to_disk(markdown: str, topic: str) -> Path:
    """Write the report to `outputs/{ISO-timestamp}_{slug}.md` and return its path.

    Uses `-` instead of `:` in the timestamp because `:` is illegal in Windows
    filenames — and HF Spaces builds run on Linux but contributors on Windows
    should still be able to clone and run locally.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = OUTPUTS_DIR / f"{timestamp}_{_slugify(topic)}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


# ---- Stop handler ------------------------------------------------------


def on_stop() -> tuple[str, str, dict]:
    """Click handler for the Stop button. Resets the UI to a clean idle state.

    `cancels=[run_event]` (wired in `build_demo`) tells Gradio to interrupt
    the generator at its next `yield`. That alone leaves whatever the last
    in-flight stage wrote on screen — a stale "🔎 Searching…" line next to
    no report, with the user unsure whether the run is still going. So
    Stop ALSO has its own click handler whose only job is to overwrite the
    three outputs with explicit "stopped" state. Gradio runs both wirings
    on the same click.
    """
    return (
        "🛑 Research stopped. Try again.",
        "",
        gr.update(value=None, visible=False),
    )


# ---- Submit handler ----------------------------------------------------


def run_research(
    topic: str,
    depth: str,
    progress: gr.Progress = gr.Progress(),  # noqa: B008 — Gradio idiom: a fresh Progress instance is injected per-call by the framework based on the annotation; the call here is just the marker.
) -> Iterator[tuple[str, str, dict]]:
    """Generator click handler: validate, drive each Flow stage, yield progress.

    Yields a 3-tuple matched to the three `outputs=` components:

      1. A status line (`gr.Markdown`) shown above the report — the most
         visible signal that *something is happening* during the 1-2 minute
         run. `gr.Progress()` alone is too easy to miss (Gradio renders it
         either inside the button or as a small floating overlay), so we
         keep an explicit human-readable status front-and-center.
      2. The report markdown (empty until the writer finishes).
      3. A `gr.update(...)` payload for the download widget (hidden until
         the report is saved to disk).

    Between yields, Gradio is allowed to interrupt the run via the Stop
    button's `cancels=` wiring. That gives users a soft-cancel — within a
    stage we can't interrupt a paid LLM call, but they won't waste a writer
    call after, e.g., changing their mind during the search phase.
    """
    try:
        request = ResearchInput(topic=(topic or "").strip(), depth=cast(Depth, depth))
    except ValidationError as e:
        first = e.errors()[0]
        field = ".".join(str(p) for p in first["loc"]) or "input"
        raise gr.Error(f"Invalid {field}: {first['msg']}") from None

    factory = ResearchAssistantAgent()
    profile = DEPTH_PROFILE[request.depth]
    # Per-request token accumulator. Mirrors `ResearchFlow._token_usages` —
    # the UI rebuilds the Flow's stage graph by hand, so it also rebuilds
    # this telemetry path by hand. Logged once at the end of the run.
    token_usages: list[object] = []

    # `research_trace` opens ONE root Langfuse span around the whole run so
    # every stage / Crew / tool / LLM call lands in a single trace with
    # aggregate cost. The `with` block stays open across `yield` calls
    # because Python keeps the generator's frame (and its with-stack) alive
    # until the generator is exhausted — so OTel context propagation
    # survives Gradio's progress yields. No-op if observability is off.
    with research_trace(topic=request.topic, depth=request.depth):
        # Stage 1 — plan. Initial yield clears any previous report from a prior
        # run so users don't see stale content while the new one is in flight.
        msg = "⏳ Planning subtopics…"
        progress(0.05, desc=msg)
        yield msg, "", gr.update(visible=False)

        plan_result = factory.planning_crew().kickoff(
            inputs={
                "topic": request.topic,
                "depth": request.depth,
                "subtopic_count": profile["subtopic_count"],
            }
        )
        token_usages.append(plan_result.token_usage)
        subtopics = expect_pydantic(plan_result.pydantic, Subtopics, plan_result.raw)

        # Stage 2 — initial search.
        msg = f"🔎 Searching the web for {len(subtopics.items)} subtopics… (~30-60s)"
        progress(0.25, desc=msg)
        yield msg, "", gr.update(visible=False)

        search_result = factory.search_crew().kickoff(
            inputs={
                "search_brief": format_initial_brief(subtopics),
                "sources_per_subtopic": profile["sources_per_subtopic"],
                "total_sources_min": profile["total_min"],
                "total_sources_max": profile["total_max"],
            }
        )
        token_usages.append(search_result.token_usage)
        findings = expect_pydantic(search_result.pydantic, Findings, search_result.raw)

        # Stage 3 — critique.
        msg = f"🧐 Reviewing {len(findings.items)} findings for grounding…"
        progress(0.55, desc=msg)
        yield msg, "", gr.update(visible=False)

        critique_result = factory.critique_crew().kickoff(
            inputs={
                "topic": request.topic,
                "subtopics_json": subtopics.model_dump_json(),
                "findings_json": findings.model_dump_json(),
            }
        )
        token_usages.append(critique_result.token_usage)
        critique = expect_pydantic(critique_result.pydantic, CritiqueResult, critique_result.raw)

        # Stage 3b — bounded retry. Same MAX_RETRIES contract as flow.py: once.
        if not critique.ok and MAX_RETRIES > 0:
            msg = (
                f"♻️ Critic flagged {len(critique.missing_claims)} gaps; "
                "re-searching for additional sources…"
            )
            progress(0.7, desc=msg)
            yield msg, "", gr.update(visible=False)

            retry_result = factory.search_crew().kickoff(
                inputs={
                    "search_brief": format_retry_brief(critique),
                    "sources_per_subtopic": profile["sources_per_subtopic"],
                    "total_sources_min": profile["total_min"],
                    "total_sources_max": profile["total_max"],
                }
            )
            token_usages.append(retry_result.token_usage)
            retry_findings = expect_pydantic(retry_result.pydantic, Findings, retry_result.raw)
            # Append, don't replace — original findings are still valid evidence.
            findings = Findings(items=[*findings.items, *retry_findings.items])

        # Stage 4 — write.
        msg = f"✍️ Writing the report from {len(findings.items)} findings…"
        progress(0.9, desc=msg)
        yield msg, "", gr.update(visible=False)

        writer_result = factory.writing_crew().kickoff(
            inputs={
                "topic": request.topic,
                "depth": request.depth,
                "findings_json": findings.model_dump_json(),
                "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        )
        token_usages.append(writer_result.token_usage)
        report = expect_pydantic(writer_result.pydantic, Report, writer_result.raw)

        # Persist to disk and surface the download.
        progress(1.0, desc="Done — saving report to disk…")
        saved = _save_report_to_disk(report.markdown, request.topic)
        # Per-run cost summary (SPEC §10). Goes to stderr / Spaces logs — never
        # to the user-visible status line, since it's an operator concern.
        print(f"[run summary] {format_token_summary(token_usages)}", file=sys.stderr)
        # Empty status on the final yield — the report itself is now visible, and
        # a stale "writing…" line beside a finished report would be misleading.
        yield "✅ Done.", report.markdown, gr.update(value=str(saved), visible=True)


# ---- Blocks ------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Construct the Gradio Blocks. Factored into a function so tests / scripts
    can rebuild a fresh instance without re-importing the module."""
    # NOTE: `theme=` was moved from `Blocks(...)` to `.launch(...)` in Gradio
    # 6.0 — see `app.py` for where the theme is now set.
    with gr.Blocks(title="Research Assistant") as demo:
        gr.Markdown(
            "# Research Assistant\n"
            "Give me a topic — I'll search the web and produce a cited markdown report."
        )
        with gr.Row():
            topic_in = gr.Textbox(
                label="Topic",
                placeholder="e.g. AI agent frameworks in 2026",
                max_lines=2,
                scale=4,
            )
            depth_in = gr.Dropdown(
                choices=cast(list, DEPTH_CHOICES),
                value="standard",
                label="Depth",
                scale=1,
            )
        with gr.Row():
            # `variant="primary"` makes the Run button the visually dominant
            # one; the Stop button stays the secondary "stop" red-tinted shape
            # so users can scan for it without reading the labels.
            run_btn = gr.Button("Run research", variant="primary", scale=4)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        # Visible status line — updated on every `yield`. Lives BETWEEN the
        # button row and the report so it's the first thing the user's eye
        # lands on while waiting. `gr.Progress` (the bar inside the button)
        # is a nice complement but not a substitute — users miss it.
        status_out = gr.Markdown(value="")

        gr.Markdown("### Report")
        report_out = gr.Markdown()
        download_out = gr.File(label="Download .md", visible=False)

        run_event = run_btn.click(
            fn=run_research,
            inputs=[topic_in, depth_in],
            outputs=[status_out, report_out, download_out],
            # One run at a time — the Flow makes paid LLM calls; we don't
            # want a single user with a fast trigger finger to fan out 5
            # parallel requests by accident.
            concurrency_limit=1,
        )
        # Stop has TWO wirings on the same click:
        #   1. `cancels=[run_event]` — interrupts the generator at its next
        #      `yield`. Soft cancel: an LLM call already in flight still
        #      finishes (and bills), but no further stage will start.
        #   2. `fn=on_stop` — overwrites status / report / download with an
        #      explicit "stopped" state, so the user sees a clear ack of
        #      their click instead of a frozen "🔎 Searching…" line.
        stop_btn.click(
            fn=on_stop,
            inputs=None,
            outputs=[status_out, report_out, download_out],
            cancels=[run_event],
        )
    return demo


# Module-level singleton: HF Spaces' app.py imports this and calls .launch().
# Tests that need a fresh instance can call `build_demo()` directly.
demo = build_demo()
