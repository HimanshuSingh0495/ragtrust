"""Tests for the eval harness: loading, grading, and metrics."""

from pathlib import Path

import pytest

import examples.demo as demo
from ragtrust import (
    EvalCase,
    EvalHarness,
    PipelineResult,
    load_eval_set,
    run_eval,
)

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"


def test_load_eval_set():
    cases = load_eval_set(DATA / "eval_set.jsonl")
    assert len(cases) == 4
    assert cases[0].id == "reset-how"
    assert cases[2].expect_abstain is True
    assert cases[0].expected_cited_ids == ("d1",)


def test_load_eval_set_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(bad)


def test_load_eval_set_rejects_empty(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_eval_set(empty)


def test_demo_pipeline_scores_perfectly_on_fixture():
    cases = load_eval_set(DATA / "eval_set.jsonl")
    report = run_eval(cases, demo.pipeline)
    assert report.accuracy == pytest.approx(1.0)
    assert report.abstention_rate == pytest.approx(0.25)
    assert report.abstention_correctness == pytest.approx(1.0)
    assert report.citation_precision == pytest.approx(1.0)
    assert report.n_passed == 4


def test_always_abstain_pipeline_fails_answerable_cases():
    def always_abstain(question: str) -> PipelineResult:
        from ragtrust import ABSTAIN_MESSAGE

        return PipelineResult(
            answer=ABSTAIN_MESSAGE, cited_ids=(), chunks=(), abstained=True
        )

    cases = load_eval_set(DATA / "eval_set.jsonl")
    report = run_eval(cases, always_abstain)
    # Only the expect_abstain case passes.
    assert report.accuracy == pytest.approx(0.25)
    assert report.abstention_rate == pytest.approx(1.0)
    assert report.n_passed == 1


def test_wrong_answer_fails_with_reasons():
    def wrong(question: str) -> PipelineResult:
        return PipelineResult(
            answer="something unrelated entirely",
            cited_ids=(),
            chunks=(),
            abstained=False,
        )

    cases = [EvalCase(id="c1", question="q?", expected_contains=["ten seconds"])]
    report = run_eval(cases, wrong)
    assert report.accuracy == pytest.approx(0.0)
    assert "ten seconds" in report.results[0].reasons[0]


def test_missing_expected_citation_fails():
    def no_cite(question: str) -> PipelineResult:
        return PipelineResult(
            answer="ten seconds",
            cited_ids=(),
            chunks=(),
            abstained=False,
        )

    cases = [
        EvalCase(
            id="c1",
            question="q?",
            expected_contains=["ten seconds"],
            expected_cited_ids=["d1"],
        )
    ]
    report = run_eval(cases, no_cite)
    assert report.results[0].passed is False
    assert any("citation" in r for r in report.results[0].reasons)


def test_report_serializes_to_json():
    cases = load_eval_set(DATA / "eval_set.jsonl")
    report = run_eval(cases, demo.pipeline)
    import json

    data = json.loads(report.to_json())
    assert data["accuracy"] == pytest.approx(1.0)
    assert len(data["results"]) == 4
    assert "case" in report.summary_table() or "reset-how" in report.summary_table()
