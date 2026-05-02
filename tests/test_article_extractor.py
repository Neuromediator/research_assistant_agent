"""Unit tests for CleanArticleExtractor.

These tests are fully offline — `requests.get` is patched in every test that
would otherwise hit the network. Per CLAUDE.md: "No real API calls in tests."
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from research_assistant_agent.tools.article_extractor import CleanArticleExtractor

# A minimal but realistic page: <article> body wrapped in nav/aside/footer
# boilerplate that trafilatura should strip.
SAMPLE_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <title>How Trafilatura Works</title>
  <meta name="author" content="Jane Doe">
  <meta property="article:published_time" content="2024-01-15T00:00:00Z">
</head>
<body>
  <nav>HOME ABOUT CONTACT — these nav links should be stripped</nav>
  <article>
    <h1>How Trafilatura Works</h1>
    <p>Trafilatura extracts the main content from a web page, ignoring
       boilerplate like navigation, sidebars, and footer links.</p>
    <p>It uses a combination of heuristics and DOM rules to identify the body
       of an article and discard the rest of the page.</p>
  </article>
  <aside>SIDEBAR_CONTENT_MARKER — should be stripped</aside>
  <footer>FOOTER_CONTENT_MARKER — also should be stripped</footer>
</body>
</html>
"""

# Where requests.get is *imported* — patch the binding, not the source module.
# (Common gotcha: `patch("requests.get")` would miss it because we imported
# `requests` and call `requests.get`, but the binding lives in our module.)
REQUESTS_GET = "research_assistant_agent.tools.article_extractor.requests.get"


@pytest.fixture
def extractor(tmp_path: Path) -> CleanArticleExtractor:
    """Build an extractor with an isolated SQLite cache for each test.

    Without `tmp_path`, the default `./.cache/articles.db` would leak state
    across tests and pollute the project directory.
    """
    return CleanArticleExtractor(cache_path=tmp_path / "articles.db")


def _ok_response(html: str) -> MagicMock:
    response = MagicMock()
    response.text = html
    response.status_code = 200
    response.raise_for_status = MagicMock()
    return response


class TestExtraction:
    def test_extracts_main_content(self, extractor: CleanArticleExtractor) -> None:
        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response(SAMPLE_HTML)
            output = extractor._run("https://example.com/article")

        assert "Trafilatura extracts the main content" in output
        assert "How Trafilatura Works" in output

    def test_strips_boilerplate(self, extractor: CleanArticleExtractor) -> None:
        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response(SAMPLE_HTML)
            output = extractor._run("https://example.com/article")

        assert "SIDEBAR_CONTENT_MARKER" not in output
        assert "FOOTER_CONTENT_MARKER" not in output

    def test_includes_metadata_header(self, extractor: CleanArticleExtractor) -> None:
        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response(SAMPLE_HTML)
            output = extractor._run("https://example.com/article")

        assert output.startswith("Title:")
        assert "URL: https://example.com/article" in output
        # Body separator is present.
        assert "\n---\n" in output


class TestCache:
    def test_cache_hit_skips_network(self, extractor: CleanArticleExtractor) -> None:
        url = "https://example.com/article"

        # First call: populate the cache.
        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response(SAMPLE_HTML)
            first = extractor._run(url)
            assert mock_get.call_count == 1

        # Second call: must NOT hit the network. The mock raises if called.
        with patch(REQUESTS_GET) as mock_get:
            mock_get.side_effect = AssertionError("network must not be called on cache hit")
            second = extractor._run(url)
            assert mock_get.call_count == 0

        assert first == second

    def test_failed_fetch_is_not_cached(self, extractor: CleanArticleExtractor) -> None:
        """A failure on attempt N must not poison attempt N+1.

        WHY this matters: the verifier loop in SPEC §2.2 may re-search after a
        gap is found. If transient failures (e.g., a flaky DNS) were cached,
        the retry would silently re-serve the old error and never recover.
        """
        url = "https://broken.example.com/article"

        with patch(REQUESTS_GET) as mock_get:
            mock_get.side_effect = requests.ConnectionError("simulated DNS failure")
            first = extractor._run(url)
            assert first.startswith("ERROR:")

        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response(SAMPLE_HTML)
            second = extractor._run(url)
            assert not second.startswith("ERROR:")
            assert mock_get.call_count == 1


class TestErrorHandling:
    def test_request_timeout_returns_error_string(self, extractor: CleanArticleExtractor) -> None:
        with patch(REQUESTS_GET) as mock_get:
            mock_get.side_effect = requests.Timeout("simulated timeout")
            output = extractor._run("https://slow.example.com/")

        assert output.startswith("ERROR:")
        assert "https://slow.example.com/" in output
        assert "Timeout" in output

    def test_unparseable_html_returns_error_string(self, extractor: CleanArticleExtractor) -> None:
        with patch(REQUESTS_GET) as mock_get:
            mock_get.return_value = _ok_response("<html><body></body></html>")
            output = extractor._run("https://empty.example.com/")

        assert output.startswith("ERROR:")
        assert "no main content" in output

    def test_never_raises_on_failure(self, extractor: CleanArticleExtractor) -> None:
        """Contract: `_run` must always return a string, never propagate.

        The agent's prompt assumes a string return — an exception escaping
        here would crash the whole Crew, not just one URL.
        """
        with patch(REQUESTS_GET) as mock_get:
            mock_get.side_effect = requests.RequestException("anything")
            output = extractor._run("https://example.com/")

        assert isinstance(output, str)
        assert output.startswith("ERROR:")
