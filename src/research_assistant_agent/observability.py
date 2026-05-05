"""Langfuse + CrewAI tracing bootstrap.

Single entry point for wiring up observability. Called at the top of every
runtime entry point (`main.py`, `app.py`) AFTER `load_dotenv()` so the
Langfuse credentials are visible.

WHY a dedicated module rather than inlining in main/app?

The setup must be idempotent (importing the Flow from a notebook + calling
`run()` from the CLI must not re-instrument CrewAI twice — that produces
duplicate spans for every operation). Centralizing the guard makes that
trivial. It also gives us one place to change the integration approach if
Langfuse / OpenInference ever drop the auto-instrumentation path.

WHY OpenInference instead of hand-rolled `@observe` everywhere?

Per SPEC §10, Langfuse v3+ moved from a manual-span SDK to an OpenTelemetry
exporter. The canonical CrewAI integration is the OpenInference
instrumentor (`openinference-instrumentation-crewai`), which auto-traces:
  * each Crew kickoff
  * each agent step (plan → action → observation)
  * each tool call (incl. SerperDevTool and our CleanArticleExtractor)
  * each LLM call, with tokens + cost
That gives us 90% of the observability we want from one line of init.
We add explicit `@observe` only at the Flow stage boundary so the tree
shows `plan → search → critique → [retry] → write` as parent spans.

Behavior when keys are missing (e.g., a stranger clones the repo):
  * print one stderr line
  * return False
  * never raise — observability must not be a hard dependency
"""

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator

_SETUP_DONE = False
_SETUP_RESULT = False


def setup_observability() -> bool:
    """Initialize Langfuse + CrewAI tracing if env vars are present.

    Returns True if tracing was successfully wired up, False otherwise.
    Safe to call multiple times — only the first call has effect.
    """
    global _SETUP_DONE, _SETUP_RESULT
    if _SETUP_DONE:
        return _SETUP_RESULT
    _SETUP_DONE = True

    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        print(
            "[observability] LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set — tracing disabled.",
            file=sys.stderr,
        )
        return False

    # Imports are inside the function so that:
    #  (1) pytest runs without importing langfuse OTel machinery into every
    #      test module (faster collection, fewer side effects);
    #  (2) a clone without the optional deps installed still fails loudly
    #      here rather than at module-import time elsewhere.
    from langfuse import get_client
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from openinference.instrumentation.crewai import CrewAIInstrumentor

    # Two instrumentors:
    #   1. CrewAIInstrumentor wraps Crew/Agent/Task/Tool spans — gives the
    #      structural picture of "which agent did what" plus per-tool
    #      observability (Serper / CleanArticleExtractor invocations).
    #   2. AnthropicInstrumentor wraps `anthropic.Anthropic.messages.create`,
    #      which CrewAI 1.14.4's native Anthropic provider (no LiteLLM)
    #      uses for plain completions — i.e., the planner / critic / writer.
    #
    # KNOWN GAP: the searcher's tool-use loop goes through
    # `client.beta.messages.create(...)` instead, and OpenInference 0.1.x
    # does NOT wrap the beta endpoint. So the searcher's LLM calls never
    # appear as GENERATION spans in Langfuse. Consequence: per-stage cost
    # in the trace is correct only for the 3 Sonnet stages; the searcher's
    # Haiku spend is reported in stdout's `[run summary]` line via
    # CrewOutput.token_usage, but not rolled up into the Langfuse trace
    # root. Patching this requires a beta-aware wrapper or pinning a newer
    # anthropic SDK (which conflicts with CrewAI 1.14.4's pins). Tracked as
    # v2 polish — see SPEC §10.
    CrewAIInstrumentor().instrument(skip_dep_check=True)
    AnthropicInstrumentor().instrument(skip_dep_check=True)

    client = get_client()
    if not client.auth_check():
        print(
            "[observability] Langfuse auth_check failed — verify "
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL.",
            file=sys.stderr,
        )
        return False

    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    print(f"[observability] Langfuse tracing active → {host}", file=sys.stderr)
    _SETUP_RESULT = True
    return True


def log_stage(*, input_: object | None = None, output: object | None = None) -> None:
    """Attach `input` / `output` summaries to the current Langfuse span.

    Intended to be called from inside an `@observe`-decorated Flow stage
    method. Without this, our stage spans show up in the Langfuse UI with
    `Input: empty object` and `Output: undefined` (because the decorator
    can only see the function's positional/keyword args, and our stage
    methods take only `self` and return `None`). With this, clicking
    `flow.search` in a trace surfaces a readable summary of what that
    stage saw and produced — without having to drill into the child
    generation spans.

    Pass compact summaries, not full prompts. Full prompts live in the
    child `generation`-typed spans (emitted by `AnthropicInstrumentor`);
    duplicating them here only inflates trace storage.

    No-op when observability is disabled (no Langfuse keys) — calling
    code never has to branch on "is tracing on".
    """
    if not _SETUP_RESULT:
        return
    from langfuse import get_client

    langfuse = get_client()
    payload: dict[str, object] = {}
    if input_ is not None:
        payload["input"] = input_
    if output is not None:
        payload["output"] = output
    if payload:
        langfuse.update_current_span(**payload)


@contextlib.contextmanager
def research_trace(topic: str, depth: str) -> Iterator[None]:
    """Wrap a single research request in ONE root Langfuse trace.

    Without this, every stage (`@observe(name="flow.X")`) and every CrewAI
    auto-instrumented `crew.kickoff()` becomes its OWN root trace because
    nothing in the parent context tells OpenTelemetry "you're inside a
    larger operation." The Langfuse UI then shows a flat soup of dozens of
    independent traces per request, and the per-request cost is scattered
    across all of them.

    Opening a `start_as_current_span` here pushes a span onto the OTel
    context. Every downstream OTel-aware call — `@observe`, the
    OpenInference CrewAI instrumentor, any LLM SDK auto-instrumentation —
    inherits that context and nests UNDER this root. Result: one trace per
    request, with topic/depth in its name and aggregate cost on the root.

    Behavior when observability is disabled (no keys, or auth failed):
    yields immediately as a no-op. Calling code never has to branch on
    "is tracing on" — it just wraps unconditionally.

    Generator-friendly: the `with` block stays open across `yield` calls
    in caller generators (Gradio's `run_research`), because Python keeps
    the generator's frame — and its with-stack — alive until the generator
    is exhausted or closed.
    """
    if not _SETUP_RESULT:
        yield
        return

    # Imported lazily to avoid pulling Langfuse into pytest's collect path.
    from langfuse import get_client

    langfuse = get_client()
    # Langfuse v3 SDK: `start_as_current_observation` is the OTel-aware
    # context manager. `as_type="chain"` is the right semantic label for a
    # multi-step pipeline (planner → searcher → critic → writer); it shows
    # up as a CHAIN-typed root in the Langfuse UI, distinguishing it from
    # individual SPAN/AGENT/TOOL nodes. The root observation's name +
    # input + metadata become the trace's identity in the UI.
    with langfuse.start_as_current_observation(
        name=f"research:{topic[:80]}",
        as_type="chain",
        input={"topic": topic, "depth": depth},
        metadata={"depth": depth},
    ):
        yield


def format_token_summary(token_usages: list[object]) -> str:
    """Format an aggregated token summary across multiple Crew kickoffs.

    Each item is a CrewAI `UsageMetrics` instance from `CrewOutput.token_usage`.
    Kept generic (`object`) to avoid a hard import dependency on CrewAI's
    internal usage type, which has moved between versions.

    Why print this when Langfuse already shows costs? Two reasons:
      1. Local debugging without opening a browser — the most useful number
         (total tokens) lands in the same terminal where stdout already is.
      2. CI / CLI runs without Langfuse credentials still get a cost signal.
    """
    total_prompt = 0
    total_completion = 0
    total_all = 0
    for u in token_usages:
        total_prompt += int(getattr(u, "prompt_tokens", 0) or 0)
        total_completion += int(getattr(u, "completion_tokens", 0) or 0)
        total_all += int(getattr(u, "total_tokens", 0) or 0)
    return (
        f"Tokens: {total_prompt} prompt + {total_completion} completion "
        f"= {total_all} total across {len(token_usages)} stage(s)"
    )
