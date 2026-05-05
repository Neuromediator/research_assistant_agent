"""Crew factory — wires the 4 research agents and their tasks.

WHY @CrewBase?

The decorator gives us three things we'd otherwise hand-roll:

1. **YAML loading.** `agents_config = "config/agents.yaml"` is parsed once at
   instantiation. Agent prompts (role/goal/backstory) live in YAML, not in
   Python — per CLAUDE.md, "configure agents in YAML, not Python."
2. **Method-name → key binding.** `def planner(self)` matches `planner:` in
   YAML automatically; mismatches surface fast as KeyError.
3. **Auto-collected `self.agents` / `self.tasks`** — populated from the
   `@agent` / `@task` decorators. The default `crew()` method below uses
   them for the bundled "run everything sequentially" mode.

WHY a per-stage Crew rather than `Agent.kickoff()` directly?

Either approach works inside a Flow. We choose Crew-per-stage because:
  - It preserves the Task abstraction (`output_pydantic`, `expected_output`)
    that gives us the structured-output contracts in models.py.
  - Variable interpolation (`{topic}`, `{depth}`, etc.) is built into
    `Crew.kickoff(inputs=…)` — no manual `.format()` calls.
  - The pattern matches the official CrewAI docs, so a future maintainer
    reading this file recognizes the abstractions immediately.

The Flow (`flow.py`, Step 6) calls these `*_crew()` helpers one at a time,
inspects each result's `.pydantic`, and decides what to do next.
"""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

from research_assistant_agent.models import (
    CritiqueResult,
    Findings,
    Report,
    Subtopics,
)
from research_assistant_agent.tools import CleanArticleExtractor


@CrewBase
class ResearchAssistantAgent:
    """Factory: 4 agents + 4 tasks + 4 single-task Crews, ready for the Flow."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ----- Agents (config from agents.yaml) ------------------------------

    @agent
    def planner(self) -> Agent:
        return Agent(config=self.agents_config["planner"])

    @agent
    def searcher(self) -> Agent:
        # Tools are attached at the agent level so both the initial search
        # task AND a future retry task share the same tool set without
        # duplicating config. SerperDevTool reads SERPER_API_KEY from env.
        #
        # `max_iter=12` is a hard ceiling on the ReAct loop, NOT the normal
        # operating count. With ~9 sources for quick depth, the disciplined
        # path is roughly: 3 search calls + 9 extractor calls + a final
        # synthesis = 13. Capping at 12 forces the agent to consolidate
        # rather than enter a "let me re-check that source" doom loop —
        # which was a contributing factor in the v1 600k-token regression
        # (CrewAI's default `max_iter=25` allowed too many redundant
        # think-act cycles, each one re-prompting with the full conversation
        # history including all prior tool observations).
        return Agent(
            config=self.agents_config["searcher"],
            tools=[SerperDevTool(), CleanArticleExtractor()],
            max_iter=12,
        )

    @agent
    def critic(self) -> Agent:
        return Agent(config=self.agents_config["critic"])

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"])

    # ----- Tasks (config from tasks.yaml; output_pydantic from models.py) -

    @task
    def plan_subtopics(self) -> Task:
        return Task(
            config=self.tasks_config["plan_subtopics"],
            output_pydantic=Subtopics,
        )

    @task
    def search_sources(self) -> Task:
        return Task(
            config=self.tasks_config["search_sources"],
            output_pydantic=Findings,
        )

    @task
    def critique_findings(self) -> Task:
        return Task(
            config=self.tasks_config["critique_findings"],
            output_pydantic=CritiqueResult,
        )

    @task
    def write_report(self) -> Task:
        return Task(
            config=self.tasks_config["write_report"],
            output_pydantic=Report,
        )

    # ----- Per-stage Crews (consumed by flow.py) -------------------------
    #
    # Each helper returns a fresh single-agent / single-task Crew. The Flow
    # builds them just-in-time so each stage gets a clean execution context
    # and we don't carry leftover state (e.g., prior task outputs) into the
    # next stage where it isn't wanted.

    def planning_crew(self) -> Crew:
        return Crew(
            agents=[self.planner()],
            tasks=[self.plan_subtopics()],
            process=Process.sequential,
            verbose=True,
        )

    def search_crew(self) -> Crew:
        return Crew(
            agents=[self.searcher()],
            tasks=[self.search_sources()],
            process=Process.sequential,
            verbose=True,
        )

    def critique_crew(self) -> Crew:
        return Crew(
            agents=[self.critic()],
            tasks=[self.critique_findings()],
            process=Process.sequential,
            verbose=True,
        )

    def writing_crew(self) -> Crew:
        return Crew(
            agents=[self.writer()],
            tasks=[self.write_report()],
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Default Crew bundle.

        Rarely used directly — the Flow drives execution stage-by-stage via
        the `*_crew()` helpers above. Kept so `crewai run` has a default
        target and the `@crew` decorator's bookkeeping is satisfied.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
