"""Hugging Face Spaces entry point.

HF Spaces convention: a file named `app.py` at the repo root is auto-detected
and run with `python app.py`. We keep this file deliberately thin — the actual
Gradio Blocks live in `src/research_assistant_agent/ui.py` so they're testable
and importable without going through the Spaces launch path.

WHY the sys.path insertion below: locally, `uv sync` installs this project as
an editable package (per `pyproject.toml`), so `research_assistant_agent` is
importable from anywhere. On HF Spaces only `requirements.txt` is processed —
the project itself is NOT pip-installed (the deploy workflow uses
`uv export --no-emit-project` to keep it out of requirements.txt; HF's build
runs `pip install` before the source tree is copied to /app, so installing
the project there would fail anyway). Putting `src/` on the path here lets
`from research_assistant_agent.X import Y` resolve without an install. The
local layout is unaffected because uv's editable install takes precedence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Load `.env` BEFORE importing `ui` (which transitively imports `crew.py`,
# which instantiates `SerperDevTool` — the tool reads `SERPER_API_KEY` from
# `os.environ` at construction time on some versions). On HF Spaces there's
# no `.env` file; secrets arrive as real env vars and this is a no-op.
load_dotenv()

# Wire Langfuse + CrewAI tracing once per process. Must come AFTER load_dotenv
# (so the keys are visible) and BEFORE importing `ui` (which pulls in the
# Crew factories — instrumenting after they exist is fine, but doing it here
# keeps the order obvious).
from research_assistant_agent.observability import setup_observability  # noqa: E402

setup_observability()

from research_assistant_agent.ui import demo  # noqa: E402

if __name__ == "__main__":
    # `theme=` lives on `launch()` rather than the `Blocks(...)` constructor as
    # of Gradio 6.0. See ui.py for the matching note.
    demo.launch(theme=gr.themes.Soft())
