"""Similarity gate: weak matches never reach the model.

A retrieved chunk is only useful if it is actually similar to the question.
The gate drops everything below a configurable score threshold so the
generator (and the abstention policy downstream) only ever sees evidence
worth reasoning about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


@dataclass(frozen=True)
class ScoredChunk:
    """A retrieved chunk with a similarity score in [0, 1]; higher is better."""

    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True)
class GateResult:
    """Outcome of applying a :class:`SimilarityGate` to a retrieval result."""

    kept: Tuple[ScoredChunk, ...]
    dropped: Tuple[ScoredChunk, ...]
    threshold: float

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)

    @property
    def top_score(self) -> float | None:
        """Highest score among kept chunks, or None when nothing was kept."""
        return self.kept[0].score if self.kept else None


@dataclass(frozen=True)
class GateStats:
    """Descriptive stats over the kept chunks; None when nothing was kept."""

    n_kept: int
    n_dropped: int
    min_score: float | None
    max_score: float | None
    mean_score: float | None


class SimilarityGate:
    """Drop retrieved chunks scoring below ``threshold``.

    Chunks scoring exactly at the threshold are kept (>= comparison).
    Kept chunks are returned sorted by score, descending.
    """

    def __init__(self, threshold: float = 0.35) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        self.threshold = threshold

    def filter(self, chunks: Iterable[ScoredChunk]) -> GateResult:
        kept: list[ScoredChunk] = []
        dropped: list[ScoredChunk] = []
        for chunk in chunks:
            (kept if chunk.score >= self.threshold else dropped).append(chunk)
        kept.sort(key=lambda c: c.score, reverse=True)
        dropped.sort(key=lambda c: c.score, reverse=True)
        return GateResult(kept=tuple(kept), dropped=tuple(dropped), threshold=self.threshold)

    def stats(self, result: GateResult) -> GateStats:
        scores = [c.score for c in result.kept]
        if not scores:
            return GateStats(
                n_kept=0,
                n_dropped=result.n_dropped,
                min_score=None,
                max_score=None,
                mean_score=None,
            )
        return GateStats(
            n_kept=len(scores),
            n_dropped=result.n_dropped,
            min_score=min(scores),
            max_score=max(scores),
            mean_score=sum(scores) / len(scores),
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(threshold={self.threshold})"
