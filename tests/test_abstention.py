"""Tests for the abstention policy."""

from ragtrust import (
    ABSTAIN_MESSAGE,
    AbstainedResponse,
    AbstentionPolicy,
    GateResult,
    ScoredChunk,
)


def _gated(kept_scores, dropped_scores=()) -> GateResult:
    kept = tuple(
        ScoredChunk(f"k{i}", "text", s) for i, s in enumerate(kept_scores)
    )
    dropped = tuple(
        ScoredChunk(f"d{i}", "text", s) for i, s in enumerate(dropped_scores)
    )
    return GateResult(kept=kept, dropped=dropped, threshold=0.35)


def test_abstain_message_is_plain_and_stable():
    assert ABSTAIN_MESSAGE == "I could not find a reliable answer."


def test_abstains_when_nothing_retrieved():
    policy = AbstentionPolicy()
    resp = policy.decide(_gated([]), n_retrieved_before_gate=0)
    assert isinstance(resp, AbstainedResponse)
    assert resp.reason == "no_chunks_retrieved"
    assert resp.to_text() == ABSTAIN_MESSAGE


def test_abstains_when_nothing_passes_gate():
    policy = AbstentionPolicy()
    resp = policy.decide(_gated([], dropped_scores=[0.2, 0.1]), n_retrieved_before_gate=2)
    assert resp is not None
    assert resp.reason == "no_chunks_passed_gate"
    assert resp.chunks_considered == 2


def test_abstains_when_top_score_below_minimum():
    policy = AbstentionPolicy(min_top_score=0.5)
    resp = policy.decide(_gated([0.4, 0.3]))
    assert resp is not None
    assert resp.reason == "top_score_below_minimum"
    assert resp.top_score == 0.4


def test_proceeds_when_evidence_is_good():
    policy = AbstentionPolicy(min_top_score=0.35)
    assert policy.decide(_gated([0.9, 0.5])) is None


def test_boundary_top_score_equal_to_minimum_proceeds():
    policy = AbstentionPolicy(min_top_score=0.5)
    assert policy.decide(_gated([0.5])) is None


def test_abstained_response_carries_diagnostics():
    policy = AbstentionPolicy()
    resp = policy.decide(_gated([], dropped_scores=[0.2]), n_retrieved_before_gate=1)
    assert resp is not None
    assert resp.top_score is None
    assert resp.chunks_considered == 1
