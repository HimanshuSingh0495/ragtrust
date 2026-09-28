"""ragtrust: eval-gated retrieval primitives for RAG pipelines.

Small, dependency-free building blocks that encode a simple discipline:

1. Gate retrieval on similarity: weak matches never reach the model.
2. Abstain plainly when confidence is low: "I could not find a reliable answer."
3. Ground answers in citations: every claim traceable to a retrieved chunk.
4. Prove it with a labeled eval set: pass/fail, citation precision, abstention rate.
5. Ingest incrementally: hash sources, only re-embed what changed.

Bring your own retriever, generator, and embedder. ragtrust owns the
trust layer around them, not the models themselves.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .abstention import ABSTAIN_MESSAGE, AbstainedResponse, AbstentionPolicy
from .citations import CitationReport, SentenceSupport, check_citations
from .eval_harness import (
    CaseResult,
    EvalCase,
    EvalHarness,
    EvalReport,
    PipelineResult,
    load_eval_set,
    run_eval,
)
from .ingest import IngestReport, Ingestor, Source, chunk_text, default_embed, hash_text
from .similarity_gate import GateResult, ScoredChunk, SimilarityGate

__all__ = [
    "__version__",
    "ABSTAIN_MESSAGE",
    "AbstainedResponse",
    "AbstentionPolicy",
    "CaseResult",
    "CitationReport",
    "EvalCase",
    "EvalHarness",
    "EvalReport",
    "GateResult",
    "IngestReport",
    "Ingestor",
    "PipelineResult",
    "ScoredChunk",
    "SentenceSupport",
    "SimilarityGate",
    "Source",
    "check_citations",
    "chunk_text",
    "default_embed",
    "hash_text",
    "load_eval_set",
    "run_eval",
]
