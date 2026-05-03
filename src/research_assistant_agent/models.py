"""Pydantic models — the structured outputs that flow between agent stages.

WHY structured outputs everywhere?

Per CLAUDE.md: "Every task returns a Pydantic model from models.py via
output_pydantic=. Don't pass freeform strings between stages." This forces the
contracts between agents to be explicit and catches malformed LLM output at
the deserialization boundary, instead of letting a typo or missing field
silently propagate three stages downstream.

NAMING CONVENTION

A *single-instance* result keeps its semantic name (`CritiqueResult`,
`Report`). A *collection* result is wrapped in a plural container
(`Subtopics`, `Findings`) because CrewAI's `output_pydantic` needs a single
root model — you can't bind a task to `list[Subtopic]` directly. The
wrapper's only job is to give that list a named root.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Single source of truth for the depth enum. Used by ResearchInput, Report,
# and (via Field aliasing) the UI dropdown.
Depth = Literal["quick", "standard", "deep"]


class ResearchInput(BaseModel):
    """Validated user input — what the UI sends in and the Flow starts from.

    The 1–500 char bound mirrors SPEC §6 ("Gradio input validation"). Putting
    it on the model gives us defense in depth — even if the UI layer is
    bypassed (e.g., direct CLI invocation), bad input fails fast here.
    """

    topic: str = Field(..., min_length=1, max_length=500)
    depth: Depth = "standard"


class Subtopic(BaseModel):
    """One angle the planner wants the searcher to investigate."""

    question: str
    rationale: str = Field(..., description="Why the planner chose this subtopic.")


class Subtopics(BaseModel):
    """Container for the planner's output (a list of subtopics)."""

    items: list[Subtopic]


class Finding(BaseModel):
    """One factual claim, traced to a source the searcher actually fetched.

    The `excerpt` cap matters: it forces the searcher to quote a *specific*
    supporting passage rather than dump the whole article into context. The
    critic agent then has a tractable amount of evidence to verify against.
    """

    claim: str
    source_url: str
    source_title: str
    excerpt: str = Field(
        ...,
        max_length=800,
        description="≤800 chars from the source that supports the claim.",
    )
    fetched_at: datetime


class Findings(BaseModel):
    """Container for the searcher's output (all findings across subtopics)."""

    items: list[Finding]


class CritiqueResult(BaseModel):
    """The critic's verdict on whether the findings are sufficient and grounded.

    `ok=True` short-circuits the verifier loop and proceeds to the writer.
    `ok=False` triggers exactly one re-search, scoped to `missing_claims`
    (SPEC §2.2). After the re-search, the writer runs unconditionally.
    """

    ok: bool
    weak_claims: list[str] = Field(
        default_factory=list,
        description="Claims with shaky evidence — should be hedged or dropped.",
    )
    missing_claims: list[str] = Field(
        default_factory=list,
        description="Gaps the report should cover; passed to the retry searcher.",
    )
    notes: str = ""


class Report(BaseModel):
    """The writer's final output — what the user sees and downloads."""

    topic: str
    depth: Depth
    markdown: str
    sources: list[str] = Field(
        default_factory=list,
        description="Deduped source URLs cited in the report's footnotes.",
    )
    generated_at: datetime
