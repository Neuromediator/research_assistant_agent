"""Local CLI entry point. Invoked via `uv run crewai run` or `uv run run_crew`.

Reads two args from argv (both optional):
    1. topic  — the research topic (default: "AI agent frameworks in 2026")
    2. depth  — quick | standard | deep (default: standard)

The Gradio UI in `ui.py` is the primary interface; this CLI exists for
local smoke testing and for the `crewai run` convention. Real logic lives
in `flow.py`.
"""

from __future__ import annotations

import sys
from typing import cast

from dotenv import load_dotenv

from research_assistant_agent.flow import ResearchFlow
from research_assistant_agent.models import Depth, ResearchInput
from research_assistant_agent.observability import (
    format_token_summary,
    research_trace,
    setup_observability,
)

# Load .env so ANTHROPIC_API_KEY / SERPER_API_KEY / LANGFUSE_* are visible
# to the SDKs before we instantiate the Flow. On HF Spaces there is no
# `.env` file — secrets arrive as real env vars — so this call is a no-op
# there.
load_dotenv()
# Wire up Langfuse + CrewAI auto-instrumentation once per process. Idempotent;
# silently disables itself when the LANGFUSE_* keys are absent.
setup_observability()


def run() -> None:
    """Run the research Flow once and print the resulting markdown."""
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI agent frameworks in 2026"
    depth_arg = sys.argv[2] if len(sys.argv) > 2 else "standard"

    if depth_arg not in ("quick", "standard", "deep"):
        raise SystemExit(f"Invalid depth {depth_arg!r}; expected quick|standard|deep")

    request = ResearchInput(topic=topic, depth=cast(Depth, depth_arg))
    flow = ResearchFlow()
    # `research_trace` opens a single root Langfuse span around the whole
    # kickoff so every nested stage / Crew / tool / LLM call lands in ONE
    # trace with aggregate cost. No-op if Langfuse is disabled.
    with research_trace(topic=request.topic, depth=request.depth):
        flow.kickoff(inputs=request.model_dump())

    if flow.state.report is None:
        raise SystemExit("Flow finished without producing a report — check logs.")
    print(flow.state.report.markdown)
    # Per-run cost summary (SPEC §10). Goes to stderr so piping the report
    # into a file leaves the markdown clean.
    print(f"\n[run summary] {format_token_summary(flow._token_usages)}", file=sys.stderr)
