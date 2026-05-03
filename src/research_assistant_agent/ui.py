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
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import gradio as gr
from pydantic import ValidationError

from research_assistant_agent.flow import ResearchFlow
from research_assistant_agent.models import Depth, ResearchInput

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


# ---- Submit handler ----------------------------------------------------


def run_research(topic: str, depth: str) -> tuple[str, dict]:
    """Click handler: validate, run the Flow, save the report, return outputs.

    Returns a 2-tuple matched to the two `outputs=` components:

      1. The markdown string for the `gr.Markdown` view.
      2. A `gr.update(...)` payload that flips the download widget visible
         and points it at the freshly saved file.
    """
    try:
        request = ResearchInput(topic=(topic or "").strip(), depth=cast(Depth, depth))
    except ValidationError as e:
        # First-error-wins keeps the message short and actionable for the user.
        first = e.errors()[0]
        field = ".".join(str(p) for p in first["loc"]) or "input"
        raise gr.Error(f"Invalid {field}: {first['msg']}") from None

    flow = ResearchFlow()
    flow.kickoff(inputs=request.model_dump())

    report = flow.state.report
    if report is None:
        # Flow ran but produced no report — surface as user-visible error,
        # not a silent empty Markdown panel.
        raise gr.Error("Flow finished without producing a report — see server logs.")

    saved = _save_report_to_disk(report.markdown, request.topic)
    return report.markdown, gr.update(value=str(saved), visible=True)


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
        run_btn = gr.Button("Run research", variant="primary")

        gr.Markdown("### Report")
        report_out = gr.Markdown()
        download_out = gr.File(label="Download .md", visible=False)

        run_btn.click(
            fn=run_research,
            inputs=[topic_in, depth_in],
            outputs=[report_out, download_out],
            # One run at a time — the Flow makes paid LLM calls; we don't
            # want a single user with a fast trigger finger to fan out 5
            # parallel requests by accident.
            concurrency_limit=1,
        )
    return demo


# Module-level singleton: HF Spaces' app.py imports this and calls .launch().
# Tests that need a fresh instance can call `build_demo()` directly.
demo = build_demo()
