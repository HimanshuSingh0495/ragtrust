"""Citation grounding: every claim traceable to a retrieved chunk.

This module checks, heuristically, whether the sentences of a generated
answer are supported by the chunks they cite. Citations are expected in
bracket form, e.g. ``The battery lasts 90 minutes [d2].``

The check is deliberately heuristic (content-word overlap), not semantic:
it catches the common failure modes -- uncited claims and citations that
point at unrelated chunks -- without needing a model call. Treat its
verdict as a signal for the eval harness, not as proof of correctness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .similarity_gate import ScoredChunk

CITATION_RE = re.compile(r"\[([\w][\w\-\.]*)\]")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")

#: Small stopword list so overlap is measured on content words.
STOPWORDS = frozenset(
    """
    a an the and or but if then else when where which who whom what
    this that these those is are was were be been being has have had
    do does did will would can could should may might must shall
    of in on at to for with by from as into over after before
    it its it’s i you he she we they them his her our your their
    not no yes so such than too very just about also there here
    """.split()
)

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        tok for tok in TOKEN_RE.findall(text.lower()) if tok not in STOPWORDS
    )


def _overlap_ratio(sentence: str, chunk_text: str) -> float:
    """Fraction of the sentence's content words appearing in the chunk."""
    sent_tokens = _content_tokens(sentence)
    if not sent_tokens:
        return 1.0  # nothing checkable; don't penalize greetings/filler
    chunk_tokens = _content_tokens(chunk_text)
    return len(sent_tokens & chunk_tokens) / len(sent_tokens)


@dataclass(frozen=True)
class SentenceSupport:
    sentence: str
    cited_ids: Tuple[str, ...]
    supported: bool
    best_overlap: float
    best_chunk_id: str | None


@dataclass(frozen=True)
class CitationReport:
    sentences: Tuple[SentenceSupport, ...]
    citation_precision: float | None
    unsupported_sentences: Tuple[str, ...]

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_unsupported(self) -> int:
        return len(self.unsupported_sentences)


def check_citations(
    answer: str,
    chunks: Iterable[ScoredChunk],
    min_overlap: float = 0.5,
) -> CitationReport:
    """Check each sentence of ``answer`` against ``chunks``.

    A sentence is *supported* when its content-word overlap with the best
    matching chunk (restricted to cited chunks when the sentence carries
    citations) is at least ``min_overlap``.

    ``citation_precision`` is the fraction of cited sentences that are
    supported, or None when no sentence carries a citation.
    """
    chunk_map: Dict[str, ScoredChunk] = {c.chunk_id: c for c in chunks}
    sentence_supports: List[SentenceSupport] = []

    for match in SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        cited_ids = tuple(CITATION_RE.findall(sentence))
        if cited_ids:
            candidates = [chunk_map[cid] for cid in cited_ids if cid in chunk_map]
            if not candidates:
                # Cites chunks that were never retrieved: unsupported.
                sentence_supports.append(
                    SentenceSupport(
                        sentence=sentence,
                        cited_ids=cited_ids,
                        supported=False,
                        best_overlap=0.0,
                        best_chunk_id=None,
                    )
                )
                continue
        else:
            candidates = list(chunk_map.values())

        best_id: str | None = None
        best_overlap = 0.0
        for chunk in candidates:
            overlap = _overlap_ratio(sentence, chunk.text)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = chunk.chunk_id
        sentence_supports.append(
            SentenceSupport(
                sentence=sentence,
                cited_ids=cited_ids,
                supported=best_overlap >= min_overlap,
                best_overlap=best_overlap,
                best_chunk_id=best_id,
            )
        )

    cited = [s for s in sentence_supports if s.cited_ids]
    precision = (
        sum(1 for s in cited if s.supported) / len(cited) if cited else None
    )
    unsupported = tuple(s.sentence for s in sentence_supports if not s.supported)
    return CitationReport(
        sentences=tuple(sentence_supports),
        citation_precision=precision,
        unsupported_sentences=unsupported,
    )
