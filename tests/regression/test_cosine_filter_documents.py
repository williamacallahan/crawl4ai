"""
Regression tests for ``CosineStrategy.filter_documents_embeddings``.

``at_least_k`` is documented as a *minimum* number of documents to return, but
the method used to end with ``return filtered_docs[:at_least_k]``, turning the
floor into a hard cap and silently dropping every above-threshold chunk past
position ``at_least_k``. These tests lock in the restored contract: all
above-threshold documents are returned (in input order), and the documented
padding floor still fills short corpora up to ``at_least_k``.

Runs fully offline. The only mocked boundaries are ``CosineStrategy.get_embeddings``
(returns crafted numpy embeddings so the real ``all-MiniLM-L6-v2`` model is not
loaded) and ``sklearn.metrics.pairwise.cosine_similarity`` (a textbook cosine
formula, identical in behaviour to the one the production method imports). The
filtering / sorting / padding logic in ``crawl4ai/extraction_strategy.py`` runs
verbatim; it is not copied or paraphrased.

Run:
    PYTHONPATH="$PWD:$PWD/deploy/docker" .venv/bin/python -m pytest -xvs \
        tests/regression/test_cosine_filter_documents.py
"""

import sys
import types

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test harness: build a CosineStrategy without loading torch / the HF model,
# and stub only the two heavy boundaries the method touches.
# ---------------------------------------------------------------------------


def _install_fake_sklearn(monkeypatch, cosine_similarity_fn):
    """Install a fake ``sklearn.metrics.pairwise`` module so the method's lazy
    ``from sklearn.metrics.pairwise import cosine_similarity`` resolves to our
    numpy implementation regardless of whether scikit-learn is installed.

    Using ``monkeypatch.setitem`` means real sklearn (if present) is restored
    after the test, and the entries are deleted if they were absent.
    """
    sk = types.ModuleType("sklearn")
    skm = types.ModuleType("sklearn.metrics")
    skp = types.ModuleType("sklearn.metrics.pairwise")
    skp.cosine_similarity = cosine_similarity_fn
    skm.pairwise = skp
    sk.metrics = skm
    monkeypatch.setitem(sys.modules, "sklearn", sk)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", skm)
    monkeypatch.setitem(sys.modules, "sklearn.metrics.pairwise", skp)


def _numpy_cosine_similarity(X, Y):
    """Textbook cosine similarity, equivalent to sklearn's implementation for
    the row-vector inputs the production method passes."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    if Y.ndim == 1:
        Y = Y[None, :]
    num = X @ Y.T
    den = np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(
        Y, axis=1, keepdims=False
    )
    return num / den


def _build_strategy(monkeypatch, sim_threshold, target_sims):
    """Create a ``CosineStrategy`` whose ``get_embeddings`` returns embeddings
    with the requested cosine similarities (``target_sims``) to a single query,
    and install a numpy ``cosine_similarity`` in place of sklearn.

    Returns ``(strategy, documents)`` where ``documents[i]`` has cosine
    similarity ``target_sims[i]`` to the semantic filter.
    """
    from crawl4ai.extraction_strategy import CosineStrategy

    _install_fake_sklearn(monkeypatch, _numpy_cosine_similarity)

    rng = np.random.default_rng(0)
    dim = max(8, len(target_sims) + 1)
    q = rng.standard_normal(dim)
    q /= np.linalg.norm(q)

    doc_rows = []
    for sim in target_sims:
        o = rng.standard_normal(dim)
        o -= o.dot(q) * q  # make orthogonal to q
        norm_o = np.linalg.norm(o)
        if norm_o < 1e-12:
            o = rng.standard_normal(dim)
            o -= o.dot(q) * q
            norm_o = np.linalg.norm(o)
        o /= norm_o
        s = float(np.clip(sim, -1.0, 1.0))
        d = s * q + np.sqrt(max(0.0, 1.0 - s * s)) * o
        d /= np.linalg.norm(d)
        doc_rows.append(d)
    doc_rows = np.vstack(doc_rows) if doc_rows else np.zeros((0, dim))
    query_row = q[None, :]

    strategy = CosineStrategy.__new__(CosineStrategy)
    strategy.sim_threshold = sim_threshold
    strategy.verbose = False

    def fake_get_embeddings(sentences, batch_size=None, bypass_buffer=False):
        # The method calls get_embeddings([filter])[0] for the query and
        # get_embeddings(documents) for the corpus. Distinguish by input length.
        if len(sentences) == 1:
            return [query_row[0].tolist()]
        return doc_rows

    strategy.get_embeddings = fake_get_embeddings

    documents = [f"chunk {i}" for i in range(len(target_sims))]
    return strategy, documents


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_semantic_filter_returns_documents_unchanged(monkeypatch):
    """With no semantic filter the method short-circuits and returns the input
    verbatim — order and length preserved."""
    strategy, documents = _build_strategy(monkeypatch, sim_threshold=0.3,
                                          target_sims=[0.9] * 10)
    result = strategy.filter_documents_embeddings(documents, None, at_least_k=20)
    assert result is documents


def test_above_threshold_not_truncated_above_at_least_k(monkeypatch):
    """Regression for the reported bug: when *more* than ``at_least_k`` chunks
    clear the threshold, *all* of them are returned — not just the first
    ``at_least_k`` by position. Previously 30 above-threshold chunks -> 20."""
    n = 30
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=[0.99] * n
    )
    result = strategy.filter_documents_embeddings(documents, "relevant",
                                                  at_least_k=20)
    assert len(result) == n, f"expected all {n} above-threshold chunks, got {len(result)}"
    assert set(result) == set(documents)


def test_short_corpus_all_above_threshold_not_halved(monkeypatch):
    """A short corpus retains every above-threshold chunk."""
    n = 12
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=[0.99] * n
    )
    result = strategy.filter_documents_embeddings(documents, "relevant",
                                                  at_least_k=20)
    assert len(result) == n, f"expected all {n}, got {len(result)}"
    assert set(result) == set(documents)


def test_short_all_below_threshold_exhausts_the_corpus(monkeypatch):
    target_sims = [0.01 * i for i in range(12)]
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=target_sims
    )

    result = strategy.filter_documents_embeddings(documents, "relevant", at_least_k=20)

    assert result == list(reversed(documents))


def test_above_threshold_docs_preserve_input_order(monkeypatch):
    """Above-threshold documents are returned in their original input order
    (the method never re-sorts them; only the below-threshold padding is
    sorted by similarity)."""
    target_sims = [0.9, 0.95, 0.85, 0.99, 0.7, 0.8, 0.92, 0.88, 0.6, 0.5]
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=target_sims
    )
    result = strategy.filter_documents_embeddings(documents, "relevant",
                                                  at_least_k=5)
    expected = [d for d, s in zip(documents, target_sims) if s >= 0.3]
    assert result[: len(expected)] == expected


def test_padding_fills_up_to_at_least_k(monkeypatch):
    """When fewer than ``at_least_k`` documents pass the threshold, the result
    is padded with the best below-threshold docs up to ``at_least_k``. No
    above-threshold doc is lost."""
    # 4 above threshold, 16 below -> pad with 16 best below to reach 20.
    target_sims = [0.9, 0.9, 0.9, 0.9] + [0.1 + 0.01 * i for i in range(16)]
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=target_sims
    )
    result = strategy.filter_documents_embeddings(documents, "relevant",
                                                  at_least_k=20)
    assert len(result) == 20
    for i in range(4):  # all 4 above-threshold docs are kept
        assert documents[i] in result


def test_padding_keeps_best_below_threshold_in_descending_similarity(monkeypatch):
    """The padding docs are the highest-similarity below-threshold documents,
    appended in descending similarity order after the above-threshold set."""
    # 2 above threshold; below-threshold sims are strictly increasing so we
    # can identify which ones the padding must pick and in what order.
    below_sims = [0.05, 0.10, 0.20, 0.28, 0.15]
    target_sims = [0.9, 0.9] + below_sims
    strategy, documents = _build_strategy(
        monkeypatch, sim_threshold=0.3, target_sims=target_sims
    )
    result = strategy.filter_documents_embeddings(documents, "relevant",
                                                  at_least_k=5)
    # 2 above + 3 best below = 5. Best 3 below: 0.28, 0.20, 0.15 (descending).
    assert len(result) == 5
    assert result[:2] == [documents[0], documents[1]]  # above-threshold, input order
    # Padding: indices of sims 0.28 (idx 5), 0.20 (idx 4), 0.15 (idx 6).
    assert result[2:] == [documents[5], documents[4], documents[6]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-xvs"]))
