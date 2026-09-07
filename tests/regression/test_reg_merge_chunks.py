"""
Regression tests for crawl4ai.utils.merge_chunks.

Guards the per-chunk token-budget invariant after the fix for the
currency-mismatch bug where the distribution loop counted `curr_size` in raw
words while `target_size` was specified in estimated tokens (scaled by
`word_token_ratio`). Both production callers (`LLMExtractionStrategy._merge`
and `LLMContentFilter._merge_chunks`) wire `chunk_token_threshold` as
`target_size` and `WORD_TOKEN_RATE=1.3` as `word_token_ratio`; the bug inflated
every filled chunk to `target_size * word_token_ratio` estimated tokens.
"""

from crawl4ai.utils import merge_chunks
from crawl4ai.config import CHUNK_TOKEN_THRESHOLD, OVERLAP_RATE, WORD_TOKEN_RATE


def _est_tokens(chunk: str, ratio: float) -> int:
    return int(len(chunk.split()) * ratio)


def _within_budget(chunks, target_size: int, ratio: float) -> bool:
    return all(_est_tokens(c, ratio) <= target_size for c in chunks)


def _overflow_count(chunks, target_size: int, ratio: float) -> int:
    return sum(1 for c in chunks if _est_tokens(c, ratio) > target_size)


# ===================================================================
# Per-chunk token-budget invariant (the bug being fixed)
# ===================================================================

class TestPerChunkTokenBudgetInvariant:
    """Every emitted chunk must satisfy int(len(chunk.split()) * word_token_ratio) <= target_size."""

    def test_production_defaults_respects_budget(self):
        """Bug-report repro at production defaults: 0 overflowing chunks (was 10)."""
        doc = " ".join(f"w{i}" for i in range(20000))
        target = CHUNK_TOKEN_THRESHOLD
        overlap = int(target * OVERLAP_RATE)
        ratio = WORD_TOKEN_RATE
        chunks = merge_chunks([doc], target_size=target, overlap=overlap, word_token_ratio=ratio)
        assert len(chunks) >= 13
        assert _overflow_count(chunks, target, ratio) == 0
        assert _within_budget(chunks, target, ratio)
        assert max(_est_tokens(c, ratio) for c in chunks) <= target

    def test_ratio_below_one_respects_budget(self):
        """word_token_ratio < 1.0 (matches chunk_documents default of 0.75) must respect the budget."""
        doc = " ".join(f"w{i}" for i in range(5000))
        for ratio in (0.5, 0.75):
            for target in (50, 100, 2048):
                chunks = merge_chunks([doc], target_size=target, overlap=int(target * 0.1), word_token_ratio=ratio)
                assert _within_budget(chunks, target, ratio), (ratio, target)

    def test_ratio_above_one_respects_budget(self):
        """word_token_ratio > 1.0 must respect the budget across realistic targets."""
        doc = " ".join(f"w{i}" for i in range(5000))
        for ratio in (1.3, 2.0, 3.0):
            for target in (20, 50, 100, 2048):
                chunks = merge_chunks([doc], target_size=target, overlap=int(target * 0.1), word_token_ratio=ratio)
                assert _within_budget(chunks, target, ratio), (ratio, target)

    def test_small_targets_strict_invariant(self):
        """Small target_size + non-1.0 ratio is the regime where the bare `>= target_size`
        boundary would overflow by 1 est token; the look-ahead check must keep the
        invariant strict here."""
        doc = " ".join(f"w{i}" for i in range(200))
        for target, ratio in [(8, 1.3), (3, 1.4), (5, 1.3), (7, 1.3), (10, 1.3), (100, 1.3)]:
            chunks = merge_chunks([doc], target_size=target, overlap=0, word_token_ratio=ratio)
            assert _within_budget(chunks, target, ratio), (target, ratio)


# ===================================================================
# Single-chunk overflow path
# ===================================================================

class TestSingleChunkOverflowPath:
    """Inputs with raw_words in (target/ratio, target] previously fit in a single
    oversized chunk; the fix must split them so each chunk satisfies the invariant."""

    def test_raw_words_in_split_range_splits(self):
        """2000 raw words at production defaults used to emit 1 chunk at 2600 est (overflow)."""
        target = CHUNK_TOKEN_THRESHOLD
        overlap = int(target * OVERLAP_RATE)
        ratio = WORD_TOKEN_RATE
        doc = " ".join(f"w{i}" for i in range(2000))
        chunks = merge_chunks([doc], target_size=target, overlap=overlap, word_token_ratio=ratio)
        assert len(chunks) >= 2
        assert _overflow_count(chunks, target, ratio) == 0
        assert _within_budget(chunks, target, ratio)


# ===================================================================
# No regression for word_token_ratio == 1.0 (function default)
# ===================================================================

class TestRatioOneNoRegression:
    """At word_token_ratio=1.0 the original loop was internally consistent; the
    fix must preserve that exact behaviour."""

    def test_ratio_one_chunks_have_exact_target_size(self):
        """ratio=1.0 with no overlap: every chunk must have exactly target_size words."""
        target = 100
        doc = " ".join(f"w{i}" for i in range(target * 4))
        chunks = merge_chunks([doc], target_size=target, overlap=0, word_token_ratio=1.0)
        assert len(chunks) == 4
        for c in chunks:
            assert len(c.split()) == target


# ===================================================================
# Multi-document input (production call shape)
# ===================================================================

class TestMultiDocumentInput:
    """LLMExtractionStrategy._merge passes a list of pre-split documents,
    not a single string. The invariant must hold across document boundaries."""

    def test_five_docs_varying_lengths_production_defaults(self):
        """5-doc input (23500 raw words total) at production defaults: every chunk
        within budget, including the tail (was: tail could overflow when the
        pre-allocated cap was reached)."""
        docs = [
            " ".join(f"w{i}" for i in range(5000)),
            " ".join(f"x{i}" for i in range(8000)),
            " ".join(f"y{i}" for i in range(3000)),
            " ".join(f"z{i}" for i in range(6000)),
            " ".join(f"a{i}" for i in range(1500)),
        ]
        target = CHUNK_TOKEN_THRESHOLD
        overlap = int(target * OVERLAP_RATE)
        ratio = WORD_TOKEN_RATE
        chunks = merge_chunks(docs, target_size=target, overlap=overlap, word_token_ratio=ratio)
        assert len(chunks) >= 15
        assert _overflow_count(chunks, target, ratio) == 0
        assert _within_budget(chunks, target, ratio)
        assert _est_tokens(chunks[-1], ratio) <= target


# ===================================================================
# Dynamic chunk count — overlap must not force an oversize tail
# ===================================================================

class TestDynamicChunkCount:
    """The pre-fix code pre-allocated num_chunks and capped with
    `curr_chunk < num_chunks - 1`, which forced surplus into the final chunk
    when overlap was enabled. The fix grows chunks dynamically."""

    def test_overlap_does_not_oversize_tail(self):
        """A doc sized so pre-allocation would undercount must not have an oversize tail."""
        target = 100
        overlap = int(target * 0.1)
        ratio = 1.3
        doc = " ".join(f"w{i}" for i in range(600))
        chunks = merge_chunks([doc], target_size=target, overlap=overlap, word_token_ratio=ratio)
        assert _within_budget(chunks, target, ratio)
        assert _est_tokens(chunks[-1], ratio) <= target


# ===================================================================
# Empty / edge cases
# ===================================================================

class TestEmptyAndEdgeCases:
    """Empty and degenerate inputs must not raise and must respect the early-return contract."""

    def test_empty_input_list(self):
        """Empty docs list returns []."""
        assert merge_chunks([], target_size=100) == []

    def test_only_empty_docs(self):
        """A list of only empty/whitespace strings returns []."""
        assert merge_chunks(["", "   ", ""], target_size=100) == []

    def test_single_small_doc_no_chunking(self):
        """A single small doc returns a single-chunk list with the doc."""
        result = merge_chunks(["hello world"], target_size=100, word_token_ratio=1.3)
        assert result == ["hello world"]


# ===================================================================
# Overlap continuity
# ===================================================================

class TestOverlapContinuity:
    """With overlap > 0, consecutive chunks must share `overlap` boundary words."""

    def test_overlap_words_preserved_across_chunks(self):
        """For every consecutive (chunks[i], chunks[i+1]) pair, the last `overlap`
        words of chunks[i] equal the first `overlap` words of chunks[i+1]."""
        target = 50
        overlap = 5
        doc = " ".join(f"w{i}" for i in range(500))
        chunks = merge_chunks([doc], target_size=target, overlap=overlap, word_token_ratio=1.0)
        assert len(chunks) > 1
        for i in range(len(chunks) - 1):
            assert chunks[i].split()[-overlap:] == chunks[i + 1].split()[:overlap]


# ===================================================================
# No token loss / order preservation
# ===================================================================

class TestNoTokenLoss:
    """Concatenating chunks (minus the overlap dups) must reconstruct the source."""

    def test_concatenation_preserves_words(self):
        """For a 5000-word doc, dropping the first `overlap` words of each chunk
        (after the first) and concatenating reconstructs the original word list."""
        doc_words = [f"w{i}" for i in range(5000)]
        doc = " ".join(doc_words)
        target = CHUNK_TOKEN_THRESHOLD
        overlap = int(target * OVERLAP_RATE)
        ratio = WORD_TOKEN_RATE
        chunks = merge_chunks([doc], target_size=target, overlap=overlap, word_token_ratio=ratio)
        reconstructed = list(chunks[0].split())
        for c in chunks[1:]:
            reconstructed.extend(c.split()[overlap:])
        assert reconstructed == doc_words


# ===================================================================
# Caller wiring invariant
# ===================================================================

class TestCallerWiring:
    """Mirror the exact wiring used by LLMExtractionStrategy._merge to assert the
    contract that caller relies on."""

    def test_extraction_strategy_wiring_respects_threshold(self):
        """target_size = chunk_token_threshold, overlap = int(threshold * overlap_rate),
        ratio = WORD_TOKEN_RATE: no chunk exceeds threshold in the function's estimator."""
        threshold = CHUNK_TOKEN_THRESHOLD
        overlap = int(threshold * OVERLAP_RATE)
        ratio = WORD_TOKEN_RATE
        doc = " ".join(f"w{i}" for i in range(20000))
        chunks = merge_chunks([doc], target_size=threshold, overlap=overlap, word_token_ratio=ratio)
        assert _within_budget(chunks, threshold, ratio)
        assert _overflow_count(chunks, threshold, ratio) == 0


# ===================================================================
# Consistency with chunk_documents (the same module's correct path)
# ===================================================================

class TestChunkDocumentsConsistency:
    """The same module's chunk_documents already implements the correct per-word
    scaling; merge_chunks must satisfy the same invariant for the same input."""

    def test_both_satisfy_invariant(self):
        """For identical inputs both merge_chunks and chunk_documents produce chunks
        that satisfy int(len(c.split()) * ratio) <= threshold."""
        from crawl4ai.utils import chunk_documents

        threshold = 100
        overlap = 10
        ratio = 1.3
        doc = " ".join(f"w{i}" for i in range(2000))
        merged = merge_chunks([doc], target_size=threshold, overlap=overlap, word_token_ratio=ratio)
        chunked = list(chunk_documents([doc], chunk_token_threshold=threshold, overlap=overlap, word_token_rate=ratio))
        for c in merged:
            assert _est_tokens(c, ratio) <= threshold
        for c in chunked:
            assert _est_tokens(c, ratio) <= threshold
