"""Hugging Face Spaces entry point.

HF Spaces convention: a file named `app.py` at the repo root is auto-detected
and run with `python app.py`. We keep this file deliberately thin — the actual
Gradio Blocks live in `src/research_assistant_agent/ui.py` so they're testable
and importable without going through the Spaces launch path.
"""

import gradio as gr
from dotenv import load_dotenv

# Load `.env` BEFORE importing `ui` (which transitively imports `crew.py`,
# which instantiates `SerperDevTool` — the tool reads `SERPER_API_KEY` from
# `os.environ` at construction time on some versions). On HF Spaces there's
# no `.env` file; secrets arrive as real env vars and this is a no-op.
load_dotenv()

from research_assistant_agent.ui import demo  # noqa: E402

if __name__ == "__main__":
    # `theme=` lives on `launch()` rather than the `Blocks(...)` constructor as
    # of Gradio 6.0. See ui.py for the matching note.
    demo.launch(theme=gr.themes.Soft())
