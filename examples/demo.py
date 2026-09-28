"""Minimal end-to-end ragtrust demo. Zero API keys, stdlib only.

Wires a fake keyword retriever and an extractive generator through the
similarity gate and the abstention policy, exposing a ``pipeline``
function that ``ragtrust eval`` can grade:

    ragtrust eval --eval-set examples/data/eval_set.jsonl --pipeline examples.demo:pipeline
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from ragtrust import (
    ABSTAIN_MESSAGE,
    AbstentionPolicy,
    PipelineResult,
    ScoredChunk,
    SimilarityGate,
    check_citations,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the and or but if then else when where which who whom what this that "
    "these those is are was were be been being has have had do does did will "
    "would can could should may might must shall of in on at to for with by "
    "from as into over after before it its i you he she we they them his her "
    "our your their not no yes so such than too very just about also there here "
    "how does".split()
)


def _tokens(text: str) -> frozenset:
    return frozenset(t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS)


class FakeRetriever:
    """Keyword-overlap retriever. Deterministic; for plumbing, not quality."""

    def __init__(self, docs_path: Path = DATA_DIR / "docs.jsonl") -> None:
        self.docs: List[Tuple[str, str]] = []
        with open(docs_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    self.docs.append((doc["id"], doc["text"]))

    def retrieve(self, question: str, k: int = 4) -> List[ScoredChunk]:
        q_tokens = _tokens(question)
        scored = []
        for doc_id, text in self.docs:
            d_tokens = _tokens(text)
            score = len(q_tokens & d_tokens) / len(q_tokens) if q_tokens else 0.0
            scored.append(ScoredChunk(doc_id, text, score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]


RETRIEVER = FakeRetriever()
GATE = SimilarityGate(threshold=0.25)
POLICY = AbstentionPolicy(min_top_score=0.25)


def _generate(question: str, chunks: List[ScoredChunk]) -> Tuple[str, List[str]]:
    """Extractive generator: answer from the top chunk, cited.

    The citation goes *inside* the sentence (``... ninety minutes [d2].``)
    so the citation checker attributes it to the right sentence.
    """
    top = chunks[0]
    text = top.text.rstrip()
    if text.endswith((".", "!", "?")):
        text = text[:-1]
    return f"{text} [{top.chunk_id}].", [top.chunk_id]


def pipeline(question: str) -> PipelineResult:
    """Full trust-layered pipeline: retrieve -> gate -> abstain-or-answer."""
    retrieved = RETRIEVER.retrieve(question)
    gated = GATE.filter(retrieved)
    abstained = POLICY.decide(gated, n_retrieved_before_gate=len(retrieved))
    if abstained is not None:
        return PipelineResult(
            answer=abstained.to_text(),
            cited_ids=(),
            chunks=(),
            abstained=True,
        )
    answer, cited_ids = _generate(question, list(gated.kept))
    return PipelineResult(
        answer=answer,
        cited_ids=tuple(cited_ids),
        chunks=gated.kept,
        abstained=False,
    )


def _demo() -> None:
    questions = [
        "How do I reset the Acme Widget?",
        "What color options does the Nebula Drone come in?",
    ]
    for question in questions:
        print(f"Q: {question}")
        result = pipeline(question)
        if result.abstained:
            print(f"A: {result.answer}  (abstained)")
        else:
            report = check_citations(result.answer, result.chunks)
            print(f"A: {result.answer}")
            print(f"   citation_precision={report.citation_precision}")
        print()


if __name__ == "__main__":
    _demo()
