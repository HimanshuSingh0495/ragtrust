"""Low-confidence abstention: say plainly when there is no reliable answer.

A wrong answer delivered with confidence destroys trust; an honest
"I could not find a reliable answer" preserves it. The abstention policy
sits downstream of the similarity gate and decides, from the gated
retrieval result, whether the pipeline may answer at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .similarity_gate import GateResult

ABSTAIN_MESSAGE = "I could not find a reliable answer."

#: Machine-readable reason codes attached to abstentions.
REASON_NO_CHUNKS_RETRIEVED = "no_chunks_retrieved"
REASON_NO_CHUNKS_PASSED_GATE = "no_chunks_passed_gate"
REASON_TOP_SCORE_BELOW_MINIMUM = "top_score_below_minimum"


@dataclass(frozen=True)
class AbstainedResponse:
    """What the pipeline returns instead of an answer when it abstains."""

    message: str = ABSTAIN_MESSAGE
    reason: str = ""
    top_score: float | None = None
    chunks_considered: int = 0

    def to_text(self) -> str:
        """Render the user-facing text. The message stays plain by design."""
        return self.message


class AbstentionPolicy:
    """Decide whether a gated retrieval result justifies answering.

    Abstains when:
    - nothing was retrieved at all, or
    - nothing survived the similarity gate, or
    - the best surviving chunk scores below ``min_top_score``.

    Returns an :class:`AbstainedResponse` when the pipeline must abstain,
    or ``None`` when it may proceed to generation.
    """

    def __init__(
        self,
        min_top_score: float = 0.35,
        message: str = ABSTAIN_MESSAGE,
    ) -> None:
        if not 0.0 <= min_top_score <= 1.0:
            raise ValueError(f"min_top_score must be in [0, 1], got {min_top_score!r}")
        self.min_top_score = min_top_score
        self.message = message

    def decide(
        self,
        gate_result: GateResult,
        n_retrieved_before_gate: int | None = None,
    ) -> AbstainedResponse | None:
        total = (
            n_retrieved_before_gate
            if n_retrieved_before_gate is not None
            else gate_result.n_kept + gate_result.n_dropped
        )
        if total == 0:
            return AbstainedResponse(
                message=self.message,
                reason=REASON_NO_CHUNKS_RETRIEVED,
                top_score=None,
                chunks_considered=0,
            )
        if gate_result.n_kept == 0:
            return AbstainedResponse(
                message=self.message,
                reason=REASON_NO_CHUNKS_PASSED_GATE,
                top_score=None,
                chunks_considered=total,
            )
        top = gate_result.top_score
        assert top is not None  # guaranteed by n_kept > 0
        if top < self.min_top_score:
            return AbstainedResponse(
                message=self.message,
                reason=REASON_TOP_SCORE_BELOW_MINIMUM,
                top_score=top,
                chunks_considered=total,
            )
        return None

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(min_top_score={self.min_top_score})"
