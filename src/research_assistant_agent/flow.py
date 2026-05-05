"""ResearchFlow — orchestrates planner → searcher → critic → writer
with a bounded one-shot verifier retry.

WHY a Flow rather than a sequential Crew?

A `Crew` runs tasks in a fixed order, threading each output forward as
context. That's enough for a linear pipeline, but SPEC §2.2 has a real
conditional branch:

    if critic.ok:  writer
    else:          retry_searcher → writer

`Crew` can't model "run X conditionally based on the output of Y." A Flow
can: `@router` returns a string label, `@listen("label")` binds to that
branch, and `or_(method, "label")` lets the writer fire on either path.
The retry budget is enforced in plain Python state — not by an
LLM-driven loop — so it cannot run away on cost.

WIRING DIAGRAM (matches SPEC §2.2):

    @start  plan ─► @listen  search ─► @listen  critique
                                              │
                                       @router decide
                                              │
                            ┌─ "go_write" ────┘
                            │                 │
                            │       └─ "go_retry" ─► retry_search
                            │                              │
                            └─ @listen(or_("go_write", retry_search)) ─► write
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TypeVar, cast

from crewai.flow.flow import Flow, listen, or_, router, start
from langfuse import observe
from pydantic import BaseModel

from research_assistant_agent.crew import ResearchAssistantAgent
from research_assistant_agent.models import (
    CritiqueResult,
    Depth,
    Findings,
    Report,
    Subtopics,
)
from research_assistant_agent.observability import log_stage

# Depth → numeric parameters used to interpolate into task prompts.
# Source-of-truth for the SPEC §2.4 depth table.
DEPTH_PROFILE: dict[Depth, dict[str, int]] = {
    "quick": {
        "subtopic_count": 3,
        "sources_per_subtopic": 3,
        "total_min": 5,
        "total_max": 8,
    },
    "standard": {
        "subtopic_count": 5,
        "sources_per_subtopic": 6,
        "total_min": 8,
        "total_max": 15,
    },
    "deep": {
        "subtopic_count": 8,
        "sources_per_subtopic": 12,
        "total_min": 15,
        "total_max": 25,
    },
}

# Hard retry cap (SPEC §2.2). One retry, no more — bounded so cost can't
# run away. If you find yourself wanting to bump this, fix the prompts
# instead.
MAX_RETRIES = 1


class ResearchState(BaseModel):
    """All Flow-internal state.

    Pydantic so the state survives serialization (e.g., for Flow `replay`
    or `@persist`). Defaults let `flow.kickoff(inputs=…)` populate fields
    without us declaring them as required.
    """

    # Inputs (provided to flow.kickoff via inputs=).
    topic: str = ""
    depth: Depth = "standard"

    # Stage outputs — populated as the Flow advances.
    subtopics: Subtopics | None = None
    findings: Findings | None = None
    critique: CritiqueResult | None = None
    report: Report | None = None

    # Retry bookkeeping.
    retries_used: int = 0


class ResearchFlow(Flow[ResearchState]):
    """The 4-stage research pipeline with a bounded verifier retry."""

    def __init__(self) -> None:
        super().__init__()
        # The factory is constructed once per Flow instance. Each per-stage
        # `*_crew()` call still returns a fresh Crew so stages don't share
        # implicit state.
        self._factory = ResearchAssistantAgent()
        # Accumulator for per-stage token usage. Lives outside ResearchState
        # because (a) it's runtime telemetry, not domain state, and (b) the
        # CrewAI UsageMetrics type is not Pydantic-friendly.
        self._token_usages: list[object] = []

    # ---- Stages ---------------------------------------------------------

    # WHY this decorator order: CrewAI Flow decorators (`@start`, `@listen`,
    # `@router`) stamp marker attributes (e.g. `_is_start_method`) on the
    # function and the Flow engine introspects the class for those markers
    # when wiring stages. `@observe` from Langfuse wraps the function with
    # `functools.wraps`, which preserves __name__/__doc__ but NOT custom
    # attributes — so if `@observe` were outermost, the wrapped function
    # would lose the Flow markers and the stage would silently never fire.
    # Putting `@observe` ABOVE `@start()` keeps the Flow's marker on the
    # inner function (the one the engine ultimately calls).
    @start()
    @observe(name="flow.plan")
    def plan(self) -> None:
        """Stage 1 — planner produces N sub-questions."""
        profile = DEPTH_PROFILE[self.state.depth]
        stage_input = {
            "topic": self.state.topic,
            "depth": self.state.depth,
            "subtopic_count": profile["subtopic_count"],
        }
        result = self._factory.planning_crew().kickoff(inputs=stage_input)
        self._token_usages.append(result.token_usage)
        self.state.subtopics = expect_pydantic(result.pydantic, Subtopics, result.raw)
        log_stage(
            input_=stage_input,
            output={"subtopics": [s.model_dump() for s in self.state.subtopics.items]},
        )

    @listen(plan)
    @observe(name="flow.search")
    def search(self) -> None:
        """Stage 2 — searcher executes the initial brief."""
        assert self.state.subtopics is not None
        profile = DEPTH_PROFILE[self.state.depth]
        brief = format_initial_brief(self.state.subtopics)
        result = self._factory.search_crew().kickoff(
            inputs={
                "search_brief": brief,
                "sources_per_subtopic": profile["sources_per_subtopic"],
                "total_sources_min": profile["total_min"],
                "total_sources_max": profile["total_max"],
            }
        )
        self._token_usages.append(result.token_usage)
        self.state.findings = expect_pydantic(result.pydantic, Findings, result.raw)
        log_stage(
            input_={
                "brief": brief,
                "sources_per_subtopic": profile["sources_per_subtopic"],
                "total_sources_max": profile["total_max"],
            },
            output={
                "findings_count": len(self.state.findings.items),
                "urls": [f.source_url for f in self.state.findings.items],
            },
        )

    @listen(search)
    @observe(name="flow.critique")
    def critique(self) -> None:
        """Stage 3 — critic verifies grounding and coverage."""
        assert self.state.subtopics is not None
        assert self.state.findings is not None
        result = self._factory.critique_crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "subtopics_json": self.state.subtopics.model_dump_json(),
                "findings_json": self.state.findings.model_dump_json(),
            }
        )
        self._token_usages.append(result.token_usage)
        self.state.critique = expect_pydantic(result.pydantic, CritiqueResult, result.raw)
        log_stage(
            input_={
                "topic": self.state.topic,
                "subtopics_count": len(self.state.subtopics.items),
                "findings_count": len(self.state.findings.items),
            },
            output=self.state.critique.model_dump(),
        )

    @router(critique)
    def critique_decision(self) -> str:
        """Pick a branch based on the critic's verdict.

        Two outcomes — both string labels picked up by `@listen` below:
          - "go_write" : findings are good, OR the retry budget is spent
          - "go_retry" : critic flagged gaps and we still have a retry

        WHY the labels are prefixed `go_` instead of plain "write" / "retry":
        a Flow method's own name is also an implicit label (each method
        emits its name on completion). If a router label collides with a
        downstream method name, that method's own completion can re-trigger
        it through any `@listen` it has on that label — an infinite loop.
        Prefixed labels make the router's emissions unambiguous and keep
        the loop detector quiet.
        """
        assert self.state.critique is not None
        if self.state.critique.ok or self.state.retries_used >= MAX_RETRIES:
            return "go_write"
        return "go_retry"

    @listen("go_retry")
    @observe(name="flow.retry_search")
    def retry_search(self) -> None:
        """Stage 2b — scoped re-search to address the critic's gaps.

        New findings are *appended* to the existing ones rather than
        replacing them — the original findings are still grounded
        evidence, even if the critique called for more.
        """
        assert self.state.critique is not None
        assert self.state.findings is not None
        profile = DEPTH_PROFILE[self.state.depth]
        retry_brief = format_retry_brief(self.state.critique)
        result = self._factory.search_crew().kickoff(
            inputs={
                "search_brief": retry_brief,
                "sources_per_subtopic": profile["sources_per_subtopic"],
                "total_sources_min": profile["total_min"],
                "total_sources_max": profile["total_max"],
            }
        )
        self._token_usages.append(result.token_usage)
        new = expect_pydantic(result.pydantic, Findings, result.raw)
        self.state.findings = Findings(items=[*self.state.findings.items, *new.items])
        self.state.retries_used += 1
        log_stage(
            input_={
                "retry_brief": retry_brief,
                "missing_claims": self.state.critique.missing_claims,
                "weak_claims": self.state.critique.weak_claims,
            },
            output={
                "new_findings_count": len(new.items),
                "new_urls": [f.source_url for f in new.items],
                "total_findings_after_retry": len(self.state.findings.items),
            },
        )

    @listen(or_("go_write", retry_search))
    @observe(name="flow.write")
    def write(self) -> Report:
        """Stage 4 (terminal) — writer composes the final markdown report.

        Listens on BOTH branches: the router's "go_write" label (no retry
        needed) AND the completion of `retry_search` (after the one
        permitted retry). Either way, write fires exactly once because
        only one branch executes per Flow run.
        """
        assert self.state.findings is not None
        result = self._factory.writing_crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "depth": self.state.depth,
                "findings_json": self.state.findings.model_dump_json(),
                "today": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }
        )
        self._token_usages.append(result.token_usage)
        self.state.report = expect_pydantic(result.pydantic, Report, result.raw)
        log_stage(
            input_={
                "topic": self.state.topic,
                "depth": self.state.depth,
                "findings_count": len(self.state.findings.items),
            },
            output={
                "markdown_chars": len(self.state.report.markdown),
                "sources_count": len(self.state.report.sources),
                "sources": self.state.report.sources,
            },
        )
        return self.state.report


# ---- Helpers ------------------------------------------------------------


def format_initial_brief(subtopics: Subtopics) -> str:
    return "\n".join(f"- {s.question} (rationale: {s.rationale})" for s in subtopics.items)


def format_retry_brief(critique: CritiqueResult) -> str:
    lines = ["The previous research had these gaps. Address each one:"]
    for c in critique.missing_claims:
        lines.append(f"- MISSING: {c}")
    for c in critique.weak_claims:
        lines.append(f"- WEAK SUPPORT (find better evidence): {c}")
    return "\n".join(lines)


_T = TypeVar("_T", bound=BaseModel)


def expect_pydantic(value: BaseModel | None, expected: type[_T], raw: str) -> _T:
    """Assert the crew returned a parsed Pydantic instance of the expected type.

    `result.pydantic` is None when the LLM emitted output that didn't match
    `output_pydantic`'s schema. Bailing out loudly with the raw text is far
    more debuggable than letting a None propagate three stages downstream.
    """
    if value is None:
        raise RuntimeError(
            f"Expected {expected.__name__} from crew but got None. Raw output:\n{raw}"
        )
    if not isinstance(value, expected):
        raise RuntimeError(
            f"Expected {expected.__name__} from crew but got {type(value).__name__}."
        )
    return cast(_T, value)
