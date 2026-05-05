"""Article extractor — the project's one custom CrewAI tool.

WHY a custom tool rather than `crewai-tools`' built-in `ScrapeWebsiteTool`?

1. We want **main-article extraction** (strip nav, ads, comments, sidebars).
   `trafilatura` is the gold-standard library for that; `ScrapeWebsiteTool`
   returns raw text dumps that include all the boilerplate.
2. We want a **per-URL on-disk cache** so verifier retries (SPEC §2.2) and dev
   iteration don't re-hit the network. SQLite gives us that with no extra
   service to run.

WHY `BaseTool` subclass instead of the `@tool` decorator? The tool carries
state (cache path, timeout, user-agent) and a non-trivial `_run` body. The
`@tool` form is best for one-shot pure functions; `BaseTool` is the
right shape for anything with configuration or lifecycle. (See
`.claude/skills/design-agent/references/custom-tools.md`.)

OUTPUT CONTRACT — CrewAI tools return strings; the string is fed back into the
LLM verbatim. We deliberately encode success/failure into the prefix:

    Success ▸ "Title: …\\nAuthor: …\\nPublished: …\\nURL: …\\n\\n---\\n\\n<markdown body>"
    Failure ▸ "ERROR: Could not fetch <url> (<short reason>)"

The searcher agent's prompt is told to drop any URL whose tool output starts
with "ERROR:" and move on. Returning a string (rather than raising) keeps the
agent's control flow simple — no try/except inside the prompt.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import requests
import trafilatura
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

DEFAULT_USER_AGENT = (
    "ResearchAssistantAgent/0.1 (+https://github.com/Neuromediator/research_assistant_agent)"
)
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_CACHE_PATH = Path(".cache/articles.db")

# Hard cap on the article body returned to the LLM (NOT cached size — see
# `_run`'s caching order). Most factual claims sit in the lede / first
# section of an article; 6000 chars (~1500 tokens) gives enough lookahead
# to find a Finding.excerpt of ≤800 chars while keeping the searcher's
# ReAct conversation history bounded. This is the SPEC §2.3 "full body
# stays out of LLM prompts" contract — a v1 600k-prompt-token regression
# was traced directly to the absence of this cap. Tune carefully: doubling
# this roughly doubles the searcher's input cost.
MAX_BODY_CHARS_FOR_LLM = 6000


class ArticleExtractorInput(BaseModel):
    """Input schema for the article extractor tool.

    The Pydantic schema is what the agent sees when deciding how to call the
    tool — field descriptions become part of the tool spec sent to the LLM.
    """

    url: str = Field(
        ...,
        description="The HTTP(S) URL of an article or web page to fetch and extract.",
    )


class CleanArticleExtractor(BaseTool):
    """Fetch a URL and return its main-article content as clean text + metadata."""

    name: str = "Clean Article Extractor"
    description: str = (
        "Fetch a web URL and return the main article text as clean markdown — "
        "stripped of navigation, ads, comments, and sidebars. Use this AFTER "
        "search results give you a URL, to read the actual page content. The "
        "output begins with metadata (Title, Author, Published, URL) followed "
        "by '---' and the article body. If the fetch fails (timeout, 404, or "
        "an unparseable page), the output begins with 'ERROR:' — when you see "
        "that, drop the URL and try the next one."
    )
    args_schema: type[BaseModel] = ArticleExtractorInput

    cache_path: Path = DEFAULT_CACHE_PATH
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    user_agent: str = DEFAULT_USER_AGENT

    def _ensure_cache(self) -> None:
        """Create the cache directory and table if missing. Idempotent."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS articles ("
                "url TEXT PRIMARY KEY, "
                "payload TEXT NOT NULL, "
                "fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _cache_get(self, url: str) -> str | None:
        with sqlite3.connect(self.cache_path) as conn:
            row = conn.execute("SELECT payload FROM articles WHERE url = ?", (url,)).fetchone()
        return row[0] if row else None

    def _cache_put(self, url: str, payload: str) -> None:
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO articles (url, payload) VALUES (?, ?)",
                (url, payload),
            )

    def _run(self, url: str) -> str:
        self._ensure_cache()

        cached = self._cache_get(url)
        if cached is not None:
            # Cache stores the FULL payload; we truncate at return-time so
            # raising MAX_BODY_CHARS_FOR_LLM later doesn't require
            # re-fetching. See module-level constant for rationale.
            return _truncate_for_llm(cached)

        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            # Failures are NOT cached — we want a retry to actually retry.
            return f"ERROR: Could not fetch {url} (request failed: {type(exc).__name__})"

        body = trafilatura.extract(
            response.text,
            output_format="markdown",
            include_comments=False,
            include_tables=False,
            url=url,
        )
        if not body or not body.strip():
            return f"ERROR: Could not fetch {url} (no main content extracted)"

        metadata = trafilatura.extract_metadata(response.text)
        title = (metadata.title if metadata else None) or "(untitled)"
        author = metadata.author if metadata else None
        date = metadata.date if metadata else None

        header_lines = [f"Title: {title}"]
        if author:
            header_lines.append(f"Author: {author}")
        if date:
            header_lines.append(f"Published: {date}")
        header_lines.append(f"URL: {url}")
        payload = "\n".join(header_lines) + "\n\n---\n\n" + body.strip()

        self._cache_put(url, payload)
        return _truncate_for_llm(payload)


def _truncate_for_llm(payload: str) -> str:
    """Cap the article-body portion of the payload at MAX_BODY_CHARS_FOR_LLM.

    Operates only on the body (the part after the `---` separator); the
    metadata header is always preserved in full so the agent still gets
    Title/URL/etc. If the body is short enough, returns unchanged.
    """
    separator = "\n\n---\n\n"
    parts = payload.split(separator, 1)
    if len(parts) != 2:
        # Defensive: every successful payload is built with the separator
        # above. If it's missing, treat the whole thing as body.
        body = payload
        header = ""
    else:
        header, body = parts
    if len(body) <= MAX_BODY_CHARS_FOR_LLM:
        return payload
    truncated = body[:MAX_BODY_CHARS_FOR_LLM].rstrip()
    note = (
        f"\n\n[...truncated for LLM context budget; original body was "
        f"{len(body)} chars. The visible portion above is the article's "
        f"opening — usually enough to extract a Finding.excerpt. "
        f"Do not re-fetch this URL hoping for more.]"
    )
    return (header + separator if header else "") + truncated + note
