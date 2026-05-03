"""Unit tests for the Pydantic models in models.py.

We don't test that Pydantic itself works — only the constraints we've layered
on top (length limits, defaults, depth enum, round-trip serialization).
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from research_assistant_agent.models import (
    CritiqueResult,
    Finding,
    Report,
    ResearchInput,
    Subtopic,
    Subtopics,
)


class TestResearchInput:
    def test_default_depth_is_standard(self) -> None:
        assert ResearchInput(topic="x").depth == "standard"

    def test_topic_must_be_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            ResearchInput(topic="")

    def test_topic_max_length_500(self) -> None:
        with pytest.raises(ValidationError):
            ResearchInput(topic="x" * 501)

    def test_depth_must_be_valid_enum(self) -> None:
        with pytest.raises(ValidationError):
            ResearchInput(topic="x", depth="exhaustive")  # type: ignore[arg-type]


class TestFinding:
    def _build(self, excerpt: str) -> Finding:
        return Finding(
            claim="The sky is blue on Earth.",
            source_url="https://example.com/sky",
            source_title="Why the sky is blue",
            excerpt=excerpt,
            fetched_at=datetime.now(),
        )

    def test_excerpt_at_800_chars_accepted(self) -> None:
        # Exactly at the limit — must not raise.
        self._build("x" * 800)

    def test_excerpt_over_800_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._build("x" * 801)


class TestCritiqueResult:
    def test_default_lists_and_notes_are_empty(self) -> None:
        result = CritiqueResult(ok=True)
        assert result.weak_claims == []
        assert result.missing_claims == []
        assert result.notes == ""


class TestReport:
    def test_round_trip_serialization(self) -> None:
        """A Report must JSON-serialize and round-trip cleanly.

        The writer agent emits Report-shaped JSON; CrewAI deserializes it back
        via Pydantic; the UI reads the markdown field. If round-trip ever
        breaks (e.g., a datetime serializer regression), the whole output
        channel breaks silently.
        """
        original = Report(
            topic="Rust async runtimes",
            depth="standard",
            markdown="# Hello\n\nbody",
            sources=["https://example.com/a", "https://example.com/b"],
            generated_at=datetime.now(),
        )
        revived = Report.model_validate_json(original.model_dump_json())
        assert revived == original


class TestSubtopics:
    def test_container_holds_items(self) -> None:
        # Single-field wrapper — exists so CrewAI's `output_pydantic` has
        # a single root model to bind to (it can't bind directly to list[X]).
        wrapped = Subtopics(items=[Subtopic(question="q1", rationale="r1")])
        assert len(wrapped.items) == 1
        assert wrapped.items[0].question == "q1"
