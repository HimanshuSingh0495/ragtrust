"""Tests for the similarity gate: thresholds, ordering, edge cases."""

import pytest

from ragtrust import GateResult, ScoredChunk, SimilarityGate


def _chunk(cid: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=f"text of {cid}", score=score)


def test_threshold_boundary_is_inclusive():
    gate = SimilarityGate(threshold=0.35)
    result = gate.filter([_chunk("a", 0.9), _chunk("b", 0.35), _chunk("c", 0.349)])
    assert [c.chunk_id for c in result.kept] == ["a", "b"]
    assert [c.chunk_id for c in result.dropped] == ["c"]


def test_kept_sorted_descending():
    gate = SimilarityGate(threshold=0.1)
    result = gate.filter([_chunk("a", 0.2), _chunk("b", 0.9), _chunk("c", 0.5)])
    assert [c.score for c in result.kept] == [0.9, 0.5, 0.2]


def test_empty_input():
    gate = SimilarityGate()
    result = gate.filter([])
    assert isinstance(result, GateResult)
    assert result.kept == () and result.dropped == ()
    assert result.top_score is None
    stats = gate.stats(result)
    assert stats.n_kept == 0 and stats.mean_score is None


def test_stats_over_kept():
    gate = SimilarityGate(threshold=0.5)
    result = gate.filter([_chunk("a", 0.8), _chunk("b", 0.6), _chunk("c", 0.1)])
    stats = gate.stats(result)
    assert (stats.n_kept, stats.n_dropped) == (2, 1)
    assert stats.min_score == pytest.approx(0.6)
    assert stats.max_score == pytest.approx(0.8)
    assert stats.mean_score == pytest.approx(0.7)


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        SimilarityGate(threshold=-0.1)
    with pytest.raises(ValueError):
        SimilarityGate(threshold=1.5)


def test_zero_threshold_keeps_everything():
    gate = SimilarityGate(threshold=0.0)
    result = gate.filter([_chunk("a", 0.0), _chunk("b", 0.01)])
    assert result.n_kept == 2 and result.n_dropped == 0
