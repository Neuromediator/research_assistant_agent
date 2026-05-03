"""Unit tests for the pure helpers in ui.py.

We don't test the Gradio Blocks themselves — UI tests are slow and brittle,
and the click handler is covered transitively by the Flow integration tests.
The helpers (`_slugify`, `_save_report_to_disk`), on the other hand, are pure
functions with edge cases worth pinning down (Unicode, oversize input,
filesystem layout).
"""

from __future__ import annotations

from research_assistant_agent.ui import _save_report_to_disk, _slugify


def test_slugify_basic() -> None:
    assert _slugify("Hello World") == "hello-world"


def test_slugify_strips_punctuation_and_collapses_dashes() -> None:
    assert _slugify("AI agent frameworks in 2026!?") == "ai-agent-frameworks-in-2026"


def test_slugify_non_ascii_falls_back_to_report() -> None:
    # Cyrillic strips entirely under NFKD+ASCII; we must never produce an
    # empty filename stem.
    assert _slugify("Привет") == "report"


def test_slugify_truncates_to_max_len() -> None:
    out = _slugify("a" * 500, max_len=60)
    assert len(out) <= 60
    assert out == "a" * 60


def test_save_report_to_disk_writes_under_outputs(tmp_path, monkeypatch) -> None:
    # `_save_report_to_disk` writes relative to CWD's `outputs/`; chdir into
    # tmp_path so the test doesn't pollute the repo's real outputs dir.
    monkeypatch.chdir(tmp_path)
    path = _save_report_to_disk("# Hello world\n\nbody.\n", "My Topic")
    assert path.exists()
    # The function returns a relative path (`outputs/...md`); resolve both
    # sides so the comparison doesn't care which form is used.
    assert path.resolve().parent == (tmp_path / "outputs").resolve()
    assert path.suffix == ".md"
    assert "my-topic" in path.name
    assert path.read_text(encoding="utf-8") == "# Hello world\n\nbody.\n"
