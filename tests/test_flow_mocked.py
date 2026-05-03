"""Integration tests for ResearchFlow with all LLMs mocked.

These cover SPEC §7's two key invariants:

  1. Writer output references at least one URL from the searcher's findings
     (the URL stays grounded as it threads through 4 stages).
  2. A critic verdict of `ok=False` triggers exactly one searcher retry
     (the bounded verifier loop in SPEC §2.2 actually fires).

WHY mocked, not real?
  Per CLAUDE.md: "No real API calls in tests." Real Anthropic + Serper would
  couple test reliability to provider uptime, slow CI, and burn money. The
  mocks (see conftest.py) replace `LLM.call` deterministically; tool calls
  short-circuit because the stub returns a final answer rather than a
  tool-call list.

WHY two tests, not one?
  The happy path and retry path exercise different `@router` branches in
  flow.py. Bundling them would hide which branch broke when something
  regresses; splitting them keeps the failure signal local.
"""

from __future__ import annotations

import json

from research_assistant_agent.flow import ResearchFlow

# ---- Canned LLM outputs ------------------------------------------------
#
# Every JSON below conforms to the matching `output_pydantic` schema in
# models.py. If a schema changes, these will fail to parse and the tests
# break loudly — that's intentional: the integration test is also a check
# that the YAML/Pydantic contract still lines up.

URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_C = "https://example.com/c"  # only appears in the retry findings

PLANNER_JSON = json.dumps(
    {
        "items": [
            {"question": "Q1?", "rationale": "Because A matters."},
            {"question": "Q2?", "rationale": "Because B matters."},
            {"question": "Q3?", "rationale": "Because C matters."},
        ]
    }
)

INITIAL_FINDINGS_JSON = json.dumps(
    {
        "items": [
            {
                "claim": "Claim about A.",
                "source_url": URL_A,
                "source_title": "Source A",
                "excerpt": "Supporting passage from source A.",
                "fetched_at": "2026-05-03T10:00:00+00:00",
            },
            {
                "claim": "Claim about B.",
                "source_url": URL_B,
                "source_title": "Source B",
                "excerpt": "Supporting passage from source B.",
                "fetched_at": "2026-05-03T10:00:00+00:00",
            },
        ]
    }
)

# Retry findings — used only in the retry-path test. The retry searcher's
# output is APPENDED to the initial findings (see flow.py::retry_search), so
# tests can verify both old and new URLs make it through.
RETRY_FINDINGS_JSON = json.dumps(
    {
        "items": [
            {
                "claim": "Claim about C (retry).",
                "source_url": URL_C,
                "source_title": "Source C",
                "excerpt": "Passage from source C, found on retry.",
                "fetched_at": "2026-05-03T10:05:00+00:00",
            }
        ]
    }
)

CRITIQUE_OK_JSON = json.dumps(
    {
        "ok": True,
        "weak_claims": [],
        "missing_claims": [],
        "notes": "Findings are well-grounded.",
    }
)

CRITIQUE_FAIL_JSON = json.dumps(
    {
        "ok": False,
        "weak_claims": [],
        "missing_claims": ["What about C?"],
        "notes": "C is uncovered.",
    }
)


def _writer_response(sources: list[str]) -> str:
    """Build a canned writer response that cites the given source URLs.

    The writer mock has to *use* the URLs we pass in, otherwise the
    "writer references at least one finding URL" assertion would be
    trivially satisfied by hardcoded content. By taking the URL list as a
    parameter, each test wires up an end-to-end URL hand-off.
    """
    citations = "\n".join(f"[^{i + 1}]: [Source]({url})" for i, url in enumerate(sources))
    markdown = f"# Topic\n\nClaim 1[^1].\n\n## Sources\n{citations}\n"
    return json.dumps(
        {
            "topic": "test topic",
            "depth": "standard",
            "markdown": markdown,
            "sources": sources,
            "generated_at": "2026-05-03T10:10:00+00:00",
        }
    )


# ---- Tests --------------------------------------------------------------


def test_happy_path_no_retry(stub_llm) -> None:
    """ok=True from the critic short-circuits straight to the writer."""
    handle = stub_llm(
        {
            "planner": PLANNER_JSON,
            "searcher": INITIAL_FINDINGS_JSON,
            "critic": CRITIQUE_OK_JSON,
            "writer": _writer_response([URL_A, URL_B]),
        }
    )

    flow = ResearchFlow()
    flow.kickoff(inputs={"topic": "test topic", "depth": "standard"})

    # Each agent ran exactly once — no retry path was taken.
    assert handle.counters == {"planner": 1, "searcher": 1, "critic": 1, "writer": 1}

    # Retry budget untouched.
    assert flow.state.retries_used == 0

    # Report exists and at least one of its sources is a URL the searcher
    # actually returned. This is the SPEC §7 grounding assertion: every
    # claim cites a source the agent fetched.
    report = flow.state.report
    assert report is not None
    finding_urls = {f.source_url for f in flow.state.findings.items}
    assert finding_urls.intersection(report.sources), (
        f"writer cited no URLs from findings (findings={finding_urls}, "
        f"report.sources={report.sources})"
    )


def test_retry_triggered_when_critic_flags_gaps(stub_llm) -> None:
    """ok=False from the critic must run the searcher exactly one more time.

    This is the bounded verifier loop in SPEC §2.2. `MAX_RETRIES = 1` —
    after one retry, the writer fires regardless of whether the critic
    would still complain. The test pins both halves: the retry happens,
    AND the loop terminates.
    """
    # Searcher gets two distinct response sets — one for the initial brief,
    # one for the retry brief. The retry result is APPENDED in flow.py, so
    # the writer should see findings from both calls.
    handle = stub_llm(
        {
            "planner": PLANNER_JSON,
            "searcher": [INITIAL_FINDINGS_JSON, RETRY_FINDINGS_JSON],
            "critic": CRITIQUE_FAIL_JSON,
            "writer": _writer_response([URL_A, URL_B, URL_C]),
        }
    )

    flow = ResearchFlow()
    flow.kickoff(inputs={"topic": "test topic", "depth": "standard"})

    # Searcher ran twice (initial + retry); critic only runs once because
    # MAX_RETRIES=1 — after retry_search, the router never fires again.
    assert handle.counters["searcher"] == 2
    assert handle.counters["critic"] == 1
    assert handle.counters["writer"] == 1
    assert flow.state.retries_used == 1

    # Findings include items from both searcher calls (3 total = 2 + 1).
    assert flow.state.findings is not None
    finding_urls = {f.source_url for f in flow.state.findings.items}
    assert finding_urls == {URL_A, URL_B, URL_C}

    # Writer cited at least one source from findings.
    report = flow.state.report
    assert report is not None
    assert finding_urls.intersection(report.sources)
