# ragtrust

[![PyPI version](https://img.shields.io/pypi/v/ragtrust.svg)](https://pypi.org/project/ragtrust/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen.svg)](tests/)

Eval-gated retrieval primitives for RAG pipelines. Small, dependency-free
building blocks that encode one discipline: **grade what production grades,
not what demos grade.**

Inspired by the production notes in [*Trust Is the Only Metric: Building a
RAG Assistant 50+ Engineers Actually Use*](https://muse.ai/s/technical-blog-post-xkv663xbixaifxo).

## The idea

Most RAG evals grade the words. Production grades the actions: did the
pipeline retrieve the right evidence, abstain when it had none, and cite
what it claimed? `ragtrust` owns that trust layer around your retriever
and generator. It does not embed, retrieve, or generate for you.

```
                        +-------------------+
                        |   Eval Harness    |  labeled JSONL set:
                        |  `ragtrust eval`  |  pass/fail, citation
                        +---------+---------+  precision, abstention rate
                                  |  gates every change
                                  v
+-----------+   +----------------+--+----------------+------------+
| Ingest    |   | Similarity Gate   | Abstention       | Citations  |
|           |   |                   |                  |            |
| hash each |   | weak matches      | low confidence   | answers    |
| source;   |   | never reach the   | -> "I could not  | grounded   |
| re-embed  |   | model             | find a reliable  | to chunk   |
| only what |   |                   | answer"          | IDs        |
| changed   |   |                   |                  |            |
+-----------+   +-------------------+------------------+------------+
      ^                      |                 |               |
      |                      v                 v               v
 your embedder        your retriever    your generator     your UI
 (bring your own)     (bring your own)  (bring your own)
```

## Install

```bash
pip install ragtrust
# or from source
pip install -e .
```

Core has **zero required dependencies** (stdlib only). Optional:

```bash
pip install "ragtrust[embed]"  # sentence-transformers, for real embeddings
pip install "ragtrust[dev]"    # pytest, for running the test suite
```

## Quickstart

### 1. Gate retrieval on similarity

```python
from ragtrust import SimilarityGate, ScoredChunk

gate = SimilarityGate(threshold=0.35)
result = gate.filter([
    ScoredChunk("d1", "reset: hold power ten seconds", 0.91),
    ScoredChunk("d2", "unrelated marketing copy", 0.12),
])
# result.kept -> (d1,), result.dropped -> (d2,)
```

```bash
ragtrust gate-demo --threshold 0.35
```

### 2. Abstain instead of hallucinating

```python
from ragtrust import AbstentionPolicy

policy = AbstentionPolicy(min_top_score=0.35)
abstained = policy.decide(result, n_retrieved_before_gate=2)
if abstained is not None:
    print(abstained.to_text())  # "I could not find a reliable answer."
```

### 3. Check citation grounding

```python
from ragtrust import check_citations

report = check_citations(
    "Hold power for ten seconds [d1].",
    result.kept,
)
print(report.citation_precision)  # 1.0
```

### 4. Run a labeled eval set

Write `eval_set.jsonl` (one case per line):

```jsonl
{"id": "reset-how", "question": "How do I reset the widget?", "expected_contains": ["ten seconds"], "expected_cited_ids": ["d1"]}
{"id": "refund-policy", "question": "What is the refund policy?", "expect_abstain": true}
```

Point the harness at any pipeline function `(question: str) -> PipelineResult`:

```bash
ragtrust eval --eval-set eval_set.jsonl --pipeline mypackage.pipeline:answer
ragtrust eval --eval-set eval_set.jsonl --pipeline mypackage.pipeline:answer \
    --json-out report.json --fail-under 0.8   # CI gate
```

Output is a per-case table plus `accuracy`, `abstention_rate`,
`abstention_correctness`, `citation_precision`, and `mean_latency_s`.

### 5. Ingest incrementally

```bash
ragtrust ingest --sources ./docs --index ./index
ragtrust ingest --sources ./docs --index ./index   # second run: only changed files re-embed
```

A `manifest.json` tracks per-source content hashes; unchanged sources are
skipped and deleted sources are pruned from the index.

## End-to-end demo (no API keys)

```bash
python examples/demo.py
ragtrust eval --eval-set examples/data/eval_set.jsonl --pipeline examples.demo:pipeline
```

The demo wires a fake keyword retriever and an extractive generator through
the gate and the abstention policy, so the whole loop runs on stdlib alone.

## API overview

| Module | What it does |
|---|---|
| `ragtrust.similarity_gate` | `SimilarityGate(threshold)` drops weak chunks before generation |
| `ragtrust.abstention` | `AbstentionPolicy(min_top_score)` returns a plain-language abstention instead of a guess |
| `ragtrust.citations` | `check_citations(answer, chunks)` heuristically verifies claims against cited chunks |
| `ragtrust.eval_harness` | `EvalHarness(cases).run(pipeline)` grades pass/fail, citation precision, abstention rate |
| `ragtrust.ingest` | `Ingestor(index_dir, embed_fn)` re-embeds only new/changed sources via content hashing |
| `ragtrust.cli` | `ragtrust eval | gate-demo | ingest` |

## What this is not

- Not a vector database. Bring your own index; `Ingestor` stores a simple JSON index for small corpora.
- Not an LLM client. Bring your own retriever, generator, and embedder.
- Not a semantic judge. Citation checks are token-overlap heuristics; treat them as eval signals, not proof.
- No benchmark claims. The bundled embedder is a deterministic hashed
  bag-of-words for plumbing and tests, not for retrieval quality.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
