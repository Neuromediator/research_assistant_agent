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

from research_assistant_agent.flow import ResearchFlow
from research_assistant_agent.models import Depth, ResearchInput


def run() -> None:
    """Run the research Flow once and print the resulting markdown."""
    topic = sys.argv[1] if len(sys.argv) > 1 else "AI agent frameworks in 2026"
    depth_arg = sys.argv[2] if len(sys.argv) > 2 else "standard"

    if depth_arg not in ("quick", "standard", "deep"):
        raise SystemExit(f"Invalid depth {depth_arg!r}; expected quick|standard|deep")

    request = ResearchInput(topic=topic, depth=cast(Depth, depth_arg))
    flow = ResearchFlow()
    flow.kickoff(inputs=request.model_dump())

    if flow.state.report is None:
        raise SystemExit("Flow finished without producing a report — check logs.")
    print(flow.state.report.markdown)
