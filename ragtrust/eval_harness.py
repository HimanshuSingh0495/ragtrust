"""Labeled eval harness: grade what production grades, not what demos grade.

The harness runs a labeled JSONL eval set against any RAG pipeline
function and reports:

- pass/fail per case (expected substrings present, expected chunks cited,
  abstention behavior correct),
- citation precision (mean over answerable cases),
- abstention rate (and whether abstentions were the *correct* ones).

A pipeline is any callable ``question: str -> PipelineResult``. The
harness never sees your retriever or generator; it only grades outcomes,
which keeps it honest about what production actually experiences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from .similarity_gate import ScoredChunk

#: A RAG pipeline under test: question in, structured result out.
PipelineFn = Callable[[str], "PipelineResult"]


@dataclass(frozen=True)
class EvalCase:
    """One labeled eval case loaded from JSONL."""

    id: str
    question: str
    expected_contains: Tuple[str, ...] = ()
    expected_cited_ids: Tuple[str, ...] = ()
    expect_abstain: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "EvalCase":
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            expected_contains=tuple(data.get("expected_contains", []) or []),
            expected_cited_ids=tuple(data.get("expected_cited_ids", []) or []),
            expect_abstain=bool(data.get("expect_abstain", False)),
        )


@dataclass(frozen=True)
class PipelineResult:
    """Structured outcome of one pipeline run, as returned by the pipeline."""

    answer: str
    cited_ids: Tuple[str, ...]
    chunks: Tuple[ScoredChunk, ...]
    abstained: bool
    latency_s: float | None = None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    reasons: Tuple[str, ...]
    abstained: bool
    latency_s: float | None
    citation_precision: float | None


@dataclass(frozen=True)
class EvalReport:
    results: Tuple[CaseResult, ...]
    accuracy: float
    abstention_rate: float
    abstention_correctness: float | None
    citation_precision: float | None
    mean_latency_s: float | None

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def summary_table(self) -> str:
        lines = [
            f"{'case':<28}{'pass':<8}{'abstained':<11}{'latency_s':<11}reasons",
            "-" * 90,
        ]
        for r in self.results:
            lat = f"{r.latency_s:.2f}" if r.latency_s is not None else "-"
            reasons = "; ".join(r.reasons) if r.reasons else ""
            lines.append(
                f"{r.case_id:<28}{'PASS' if r.passed else 'FAIL':<8}"
                f"{str(r.abstained):<11}{lat:<11}{reasons}"
            )
        lines.append("-" * 90)
        lines.append(f"accuracy:             {self.accuracy:.3f} ({self.n_passed}/{self.n})")
        lines.append(f"abstention_rate:      {self.abstention_rate:.3f}")
        if self.abstention_correctness is not None:
            lines.append(
                f"abstention_correct:   {self.abstention_correctness:.3f}"
            )
        if self.citation_precision is not None:
            lines.append(f"citation_precision:   {self.citation_precision:.3f}")
        if self.mean_latency_s is not None:
            lines.append(f"mean_latency_s:       {self.mean_latency_s:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "abstention_rate": self.abstention_rate,
            "abstention_correctness": self.abstention_correctness,
            "citation_precision": self.citation_precision,
            "mean_latency_s": self.mean_latency_s,
            "n": self.n,
            "n_passed": self.n_passed,
            "results": [
                {
                    "case_id": r.case_id,
                    "passed": r.passed,
                    "reasons": list(r.reasons),
                    "abstained": r.abstained,
                    "latency_s": r.latency_s,
                    "citation_precision": r.citation_precision,
                }
                for r in self.results
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def load_eval_set(path: str | Path) -> List[EvalCase]:
    """Load labeled eval cases from a JSONL file (one case per line)."""
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid eval case: {exc}") from exc
    if not cases:
        raise ValueError(f"{path}: eval set is empty")
    return cases


def _grade_case(case: EvalCase, result: PipelineResult) -> CaseResult:
    reasons: List[str] = []

    if case.expect_abstain:
        passed = result.abstained
        if not passed:
            reasons.append("expected abstention but pipeline answered")
        return CaseResult(
            case_id=case.id,
            passed=passed,
            reasons=tuple(reasons),
            abstained=result.abstained,
            latency_s=result.latency_s,
            citation_precision=None,
        )

    if result.abstained:
        reasons.append("pipeline abstained on an answerable case")
        return CaseResult(
            case_id=case.id,
            passed=False,
            reasons=tuple(reasons),
            abstained=True,
            latency_s=result.latency_s,
            citation_precision=None,
        )

    answer_lower = result.answer.lower()
    for expected in case.expected_contains:
        if expected.lower() not in answer_lower:
            reasons.append(f"answer missing expected text: {expected!r}")

    cited = set(result.cited_ids)
    per_case_precision: float | None = None
    if case.expected_cited_ids:
        expected_set = set(case.expected_cited_ids)
        missing = expected_set - cited
        if missing:
            reasons.append(f"expected citations missing: {sorted(missing)}")
        if cited:
            per_case_precision = len(expected_set & cited) / len(cited)

    return CaseResult(
        case_id=case.id,
        passed=not reasons,
        reasons=tuple(reasons),
        abstained=False,
        latency_s=result.latency_s,
        citation_precision=per_case_precision,
    )


def _mean(values: List[float]) -> float | None:
    return sum(values) / len(values) if values else None


class EvalHarness:
    """Run an eval set against a pipeline and aggregate the metrics."""

    def __init__(self, cases: Iterable[EvalCase]) -> None:
        self.cases = list(cases)
        if not self.cases:
            raise ValueError("eval set is empty")

    def run(self, pipeline: PipelineFn) -> EvalReport:
        results = tuple(
            _grade_case(case, pipeline(case.question)) for case in self.cases
        )
        n = len(results)
        accuracy = sum(1 for r in results if r.passed) / n
        abstention_rate = sum(1 for r in results if r.abstained) / n

        abstain_cases = [c for c in self.cases if c.expect_abstain]
        abstention_correctness: float | None = None
        if abstain_cases:
            correct = sum(
                1
                for c in abstain_cases
                for r in results
                if r.case_id == c.id and r.abstained
            )
            abstention_correctness = correct / len(abstain_cases)

        precisions = [r.citation_precision for r in results if r.citation_precision is not None]
        latencies = [r.latency_s for r in results if r.latency_s is not None]

        return EvalReport(
            results=results,
            accuracy=accuracy,
            abstention_rate=abstention_rate,
            abstention_correctness=abstention_correctness,
            citation_precision=_mean(precisions),
            mean_latency_s=_mean(latencies),
        )


def run_eval(cases: Iterable[EvalCase], pipeline: PipelineFn) -> EvalReport:
    """Convenience wrapper: build a harness and run it."""
    return EvalHarness(cases).run(pipeline)
