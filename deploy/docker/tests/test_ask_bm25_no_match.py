"""
Regression tests for the /ask endpoint BM25 no-match behavior.

Bug (introduced in commit 5297e36): when a search query shared no terms with
the Crawl4AI context corpus, the *relative* score cutoff
(``cutoff = max_score * score_ratio``) degenerated to ``0.0`` because every
BM25 score was ``0.0``. The filter ``score >= cutoff`` then became
``0.0 >= 0.0`` and admitted *every* chunk/section, so callers received up to
``max_results`` irrelevant ``score=0.0`` chunks instead of an empty result
set. ``score_ratio`` was useless in that case (even ``1.0`` admitted all
chunks). The same defect existed in the doc-section path.

Fix: an explicit zero-match check (``max_score <= 0``) short-circuits to an
empty result list for each context type, consistent with the established
``BM25ContentFilter`` pattern in ``crawl4ai/content_filter_strategy.py``.

These tests exercise the running FastAPI app via Starlette's ``TestClient``
(see ``conftest.py``). The ``/ask`` handler only reads two local markdown
files, so it returns 200 even without the FastAPI lifespan (no browser pool /
Redis is started by the test client).
"""

import pytest

pytestmark = pytest.mark.cve  # regression test for a specific reported bug

from auth import create_access_token  # noqa: E402


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _auth_headers() -> dict:
    return _bearer(create_access_token({"sub": "u@x.com"}))


# ────────────────────────────────────────────────────────────────────────
# Query fixtures anchored to the real repo context files (deploy/docker/).
# Confirmed via direct BM25Okapi scoring against c4ai-*-context.md:
#   "selenium webdriver"               -> code max=0.0, doc max=0.0  (no match)
#   "supercalifragilistic zzzzzz"      -> code max=0.0, doc max=0.0  (no match)
#   "selenium webdriver integration"   -> code max=0.0, doc max=6.54 (doc only)
#   "playwright browser"              -> code max=4.93, doc max=5.73 (both match)
# ────────────────────────────────────────────────────────────────────────

NO_MATCH_QUERIES = ["selenium webdriver", "supercalifragilistic zzzzzz"]
DOC_ONLY_MATCH_QUERY = "selenium webdriver integration"  # code=0, doc>0
BOTH_MATCH_QUERY = "playwright browser"                  # both > 0


def _get(client, params):
    return client.get("/ask", params=params, headers=_auth_headers())


class TestAskBm25NoMatch:
    """A query with zero BM25 matches must NOT return irrelevant score=0.0
    chunks; it must return an empty list for each requested corpus.
    """

    @pytest.mark.parametrize("query", NO_MATCH_QUERIES, ids=NO_MATCH_QUERIES)
    def test_code_no_match_returns_empty(self, stock_client, query):
        """context_type=code with no code matches -> code_results == []."""
        r = _get(stock_client, {"context_type": "code", "query": query})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("code_results") == [], (
            f"no-match query {query!r} returned {len(body.get('code_results', []))} "
            f"code chunks with irrelevant score=0.0 results"
        )
        assert "doc_results" not in body

    @pytest.mark.parametrize("query", NO_MATCH_QUERIES, ids=NO_MATCH_QUERIES)
    def test_doc_no_match_returns_empty(self, stock_client, query):
        """context_type=doc with no doc matches -> doc_results == []."""
        r = _get(stock_client, {"context_type": "doc", "query": query})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body.get("doc_results") == [], (
            f"no-match query {query!r} returned {len(body.get('doc_results', []))} "
            f"doc sections with irrelevant score=0.0 results"
        )
        assert "code_results" not in body

    @pytest.mark.parametrize("query", NO_MATCH_QUERIES, ids=NO_MATCH_QUERIES)
    def test_all_no_match_returns_empty_both(self, stock_client, query):
        """context_type=all with no matches anywhere -> both lists empty."""
        r = _get(stock_client, {"context_type": "all", "query": query})
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body.keys()) == {"code_results", "doc_results"}, body.keys()
        assert body["code_results"] == [], (
            f"code_results not empty for no-match query {query!r}"
        )
        assert body["doc_results"] == [], (
            f"doc_results not empty for no-match query {query!r}"
        )

    @pytest.mark.parametrize(
        "ratio", [0.0, 0.5, 1.0], ids=["r0", "r0.5", "r1.0"],
    )
    def test_no_match_empty_regardless_of_score_ratio(self, stock_client, ratio):
        """The bug made score_ratio ineffective for no-match queries: even
        score_ratio=1.0 admitted every chunk. The fix must yield empty results
        for ANY score_ratio when there are no matches.
        """
        query = "selenium webdriver"  # zero matches in both corpora
        r = _get(stock_client, {
            "context_type": "all", "query": query, "score_ratio": ratio,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code_results"] == [], (
            f"score_ratio={ratio}: code_results not empty for no-match query"
        )
        assert body["doc_results"] == [], (
            f"score_ratio={ratio}: doc_results not empty for no-match query"
        )


class TestAskBm25HappyPath:
    """Regression checks: matching queries still return scored results, so
    the fix does not over-suppress real matches.
    """

    def test_code_match_returns_positive_scored_results(self, stock_client):
        r = _get(stock_client, {"context_type": "code", "query": BOTH_MATCH_QUERY})
        assert r.status_code == 200, r.text
        results = r.json().get("code_results") or []
        assert len(results) > 0, "matching query returned no code results"
        # Every returned code chunk must have a strictly positive score:
        # code has no neighbor expansion, so 0.0-scored chunks must not appear.
        for item in results:
            assert isinstance(item, dict), item
            assert "text" in item and "score" in item, item
            assert item["score"] > 0, f"non-positive score returned: {item['score']}"
        scores = [item["score"] for item in results]
        assert scores == sorted(scores, reverse=True), scores

    def test_doc_match_returns_results(self, stock_client):
        r = _get(stock_client, {"context_type": "doc", "query": BOTH_MATCH_QUERY})
        assert r.status_code == 200, r.text
        results = r.json().get("doc_results") or []
        assert len(results) > 0, "matching query returned no doc results"
        # At least one returned section must carry a strictly positive score
        # (doc results may include 0-score neighbor sections for expansion).
        assert any(item["score"] > 0 for item in results), (
            "no positively-scored section in doc results"
        )
        for item in results:
            assert isinstance(item, dict) and "text" in item and "score" in item


class TestAskBm25MixedMatch:
    """A query that matches one corpus but not the other must empty the
    non-matching side while still returning the matching side (no
    over-suppression).
    """

    def test_code_empty_doc_nonempty(self, stock_client):
        # "selenium webdriver integration": code max=0, doc max>0
        r = _get(stock_client, {
            "context_type": "all", "query": DOC_ONLY_MATCH_QUERY,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["code_results"] == [], (
            "code corpus has no matches; code_results must be empty"
        )
        assert len(body["doc_results"]) > 0, (
            "doc corpus has matches; doc_results must be non-empty"
        )
        for item in body["doc_results"]:
            assert "text" in item and "score" in item
        assert any(item["score"] > 0 for item in body["doc_results"])
