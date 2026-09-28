"""Command-line interface for ragtrust."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .eval_harness import EvalHarness, load_eval_set
from .ingest import Ingestor
from .similarity_gate import ScoredChunk, SimilarityGate


def _import_callable(dotted: str):
    """Import ``pkg.mod:attr`` or ``pkg.mod.attr`` and return the callable."""
    if ":" in dotted:
        mod_name, _, attr_path = dotted.partition(":")
    else:
        mod_name, _, attr_path = dotted.rpartition(".")
    if not mod_name or not attr_path:
        raise ValueError(
            f"could not parse callable reference {dotted!r}; "
            "expected 'package.module:callable' or 'package.module.callable'"
        )
    module = importlib.import_module(mod_name)
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise ValueError(f"{dotted!r} is not callable")
    return obj


def cmd_eval(args: argparse.Namespace) -> int:
    cases = load_eval_set(args.eval_set)
    # Make the pipeline importable from wherever the user runs the CLI.
    for extra in (args.import_path or []):
        sys.path.insert(0, extra)
    sys.path.insert(0, str(Path.cwd()))
    pipeline = _import_callable(args.pipeline)
    report = EvalHarness(cases).run(pipeline)
    print(report.summary_table())
    if args.json_out:
        Path(args.json_out).write_text(report.to_json(), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    if args.fail_under is not None and report.accuracy < args.fail_under:
        print(
            f"\naccuracy {report.accuracy:.3f} is below --fail-under {args.fail_under}",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_gate_demo(args: argparse.Namespace) -> int:
    gate = SimilarityGate(threshold=args.threshold)
    chunks = [
        ScoredChunk("c1", "the widget reset procedure", 0.91),
        ScoredChunk("c2", "battery charging time", 0.62),
        ScoredChunk("c3", "unrelated marketing copy", 0.21),
        ScoredChunk("c4", "shipping information", 0.08),
    ]
    result = gate.filter(chunks)
    stats = gate.stats(result)
    print(f"threshold: {gate.threshold}\n")
    print(f"{'chunk':<8}{'score':<8}verdict")
    print("-" * 28)
    for chunk in chunks:
        verdict = "KEPT" if chunk in result.kept else "dropped"
        print(f"{chunk.chunk_id:<8}{chunk.score:<8.2f}{verdict}")
    print("-" * 28)
    mean = f"{stats.mean_score:.2f}" if stats.mean_score is not None else "-"
    print(f"kept={stats.n_kept} dropped={stats.n_dropped} mean_kept_score={mean}")
    print("\nWeak matches never reach the model: only KEPT chunks are passed on.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    sources_dir = Path(args.sources)
    if not sources_dir.is_dir():
        print(f"error: --sources is not a directory: {sources_dir}", file=sys.stderr)
        return 2
    paths = [
        p
        for p in sources_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    ]
    if not paths:
        print(f"error: no .txt/.md files under {sources_dir}", file=sys.stderr)
        return 2
    ingestor = Ingestor(
        args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    report = ingestor.ingest_paths(paths, base_dir=sources_dir)
    print(f"index: {args.index}")
    print(report.summary())
    for sid in report.added:
        print(f"  + added     {sid}")
    for sid in report.updated:
        print(f"  ~ updated   {sid}  (re-embedded)")
    for sid in report.removed:
        print(f"  - removed   {sid}  (pruned)")
    if report.unchanged:
        print(f"  = unchanged {len(report.unchanged)} source(s) skipped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragtrust",
        description="Eval-gated retrieval primitives for RAG pipelines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="Run a labeled JSONL eval set against a pipeline.")
    p_eval.add_argument("--eval-set", required=True, help="Path to eval set JSONL.")
    p_eval.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline callable as 'package.module:callable'. "
        "Takes a question string, returns a ragtrust.eval_harness.PipelineResult.",
    )
    p_eval.add_argument(
        "--import-path",
        action="append",
        default=[],
        help="Extra sys.path entry so --pipeline is importable (repeatable).",
    )
    p_eval.add_argument("--json-out", default=None, help="Write the JSON report here.")
    p_eval.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit 1 when accuracy is below this value (for CI gates).",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_gate = sub.add_parser("gate-demo", help="Show the similarity gate on toy chunks.")
    p_gate.add_argument("--threshold", type=float, default=0.35)
    p_gate.set_defaults(func=cmd_gate_demo)

    p_ing = sub.add_parser("ingest", help="Incrementally ingest .txt/.md sources.")
    p_ing.add_argument("--sources", required=True, help="Directory of source files.")
    p_ing.add_argument("--index", required=True, help="Index directory (created if missing).")
    p_ing.add_argument("--chunk-size", type=int, default=500)
    p_ing.add_argument("--chunk-overlap", type=int, default=50)
    p_ing.set_defaults(func=cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
