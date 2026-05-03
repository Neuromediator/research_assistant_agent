"""Shared test fixtures.

The big one here is `stub_llm` — a fixture that replaces CrewAI's
`get_llm_response` with a deterministic, role-routed stub. Every Flow
integration test needs this so we can run the full 4-stage pipeline offline,
in <1s, at $0.

WHY patch `get_llm_response` and not `LLM.call`?

`LLM.call` is abstract — each provider has its own concrete subclass
(`AnthropicCompletion`, `OpenAICompletion`, ...). Patching the base class
doesn't override the subclass methods. `get_llm_response` is the single
function the executor (`CrewAgentExecutor`) calls regardless of provider:

    crewai.agents.crew_agent_executor:
        from crewai.utilities.agent_utils import get_llm_response, ...
        answer = get_llm_response(llm=..., from_agent=..., ...)

So we patch the binding *inside the executor module* — that's where the call
actually originates. Patching `crewai.utilities.agent_utils.get_llm_response`
alone wouldn't help because the executor already imported the original.

WHY route by `from_agent.role` instead of inspecting the prompt?

`get_llm_response` receives `from_agent` as a kwarg, and each role we set in
`agents.yaml` is unique. A substring match on the role is stable, doesn't
depend on prompt wording, and makes the stub readable.

WHY return a JSON string and not a parsed Pydantic instance?

The non-tools executor path wraps strings into `AgentFinish.text`; the tools
path does the same when the answer is a string (i.e., not a tool-call list).
Either way, `Task._export_output` then parses the JSON against the task's
`output_pydantic` model — exactly the same contract the real LLM would have
to satisfy. Returning a string keeps us as close as possible to "what would a
real Anthropic response look like" without depending on whether structured
output is enabled.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from crewai.agents import crew_agent_executor as _executor_mod

# ---- Stub LLM ----------------------------------------------------------

# Role substrings that uniquely identify each of our 4 agents (see
# `src/research_assistant_agent/config/agents.yaml`). Substring match keeps
# the stub robust against trailing whitespace / minor wording tweaks in the
# YAML role lines.
_ROLE_KEYS: dict[str, str] = {
    "Strategist": "planner",
    "Operative": "searcher",
    "Editor": "critic",
    "Writer": "writer",
}


def _agent_kind(from_agent: Any | None) -> str:
    """Map a CrewAI agent (or None) to a short kind label.

    `from_agent` should always be set when the executor calls the LLM via
    `get_llm_response`. If we ever see a None or an unrecognized role, we
    fail loudly — that means the test or the YAML changed in a way the stub
    doesn't understand.
    """
    role = (getattr(from_agent, "role", "") or "").strip()
    for needle, kind in _ROLE_KEYS.items():
        if needle in role:
            return kind
    raise AssertionError(
        f"stub_llm: could not classify agent (role={role!r}). "
        "Add the new role substring to _ROLE_KEYS."
    )


class StubLLMHandle:
    """Test-side handle for the stubbed LLM.

    `responses` maps a kind ("planner"/"searcher"/"critic"/"writer") to either:
      - a single JSON string (returned every time that agent is called), or
      - a list of JSON strings (one per call, in order; last value is reused
        if the agent is called more times than the list provides).

    `counters` records how many times each kind was invoked, which is how the
    tests assert control-flow properties like "the searcher ran twice."
    """

    def __init__(self, responses: dict[str, str | list[str]]) -> None:
        self.responses = responses
        self.counters: dict[str, int] = {kind: 0 for kind in _ROLE_KEYS.values()}

    def reply_for(self, kind: str) -> str:
        if kind not in self.responses:
            raise AssertionError(
                f"stub_llm: no canned response configured for {kind!r}. "
                f"Configured kinds: {list(self.responses)}"
            )
        value = self.responses[kind]
        idx = self.counters[kind]
        self.counters[kind] += 1
        if isinstance(value, list):
            return value[min(idx, len(value) - 1)]
        return value


StubBuilder = Callable[[dict[str, str | list[str]]], StubLLMHandle]


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch) -> StubBuilder:
    """Return a builder that installs the stub for the test's chosen responses.

    Usage:

        handle = stub_llm({"planner": PLANNER_JSON, ...})
        # ... run the flow ...
        assert handle.counters["searcher"] == 2

    We expose a builder (rather than installing a fixed stub) because each
    test wants different canned responses — happy path vs. retry path, etc.
    """

    def _install(responses: dict[str, str | list[str]]) -> StubLLMHandle:
        handle = StubLLMHandle(responses)

        def fake_get_llm_response(
            llm: Any,
            messages: Any,
            callbacks: Any,
            printer: Any,
            tools: Any = None,
            available_functions: Any = None,
            from_task: Any = None,
            from_agent: Any = None,
            response_model: Any = None,
            executor_context: Any = None,
            verbose: bool = True,
        ) -> str:
            kind = _agent_kind(from_agent)
            return handle.reply_for(kind)

        # Patch BOTH the sync and async entry points the executor uses.
        # Path: rebind the names imported into crew_agent_executor — patching
        # `crewai.utilities.agent_utils.get_llm_response` would miss because
        # the executor already imported the original at module load.
        monkeypatch.setattr(_executor_mod, "get_llm_response", fake_get_llm_response)
        monkeypatch.setattr(_executor_mod, "aget_llm_response", fake_get_llm_response)
        return handle

    return _install


# ---- Misc fixtures -----------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip API keys from the env for every test.

    Belt-and-braces: we mock `LLM.call`, but if a regression caused a real
    network call to slip through, missing env vars would fail loud and fast
    instead of quietly burning money. SERPER_API_KEY removal also ensures
    SerperDevTool can't accidentally hit the wire.
    """
    for var in ("ANTHROPIC_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
