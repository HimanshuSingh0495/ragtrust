"""Incremental ingestion: hash sources, only re-embed what changed.

Re-embedding an entire corpus on every refresh is the expensive default.
The ingestor keeps a JSON manifest of per-source content hashes; on each
run it embeds only new or changed sources, leaves unchanged ones alone,
and prunes sources that disappeared.

Embeddings are pluggable: pass any ``embed(texts) -> vectors`` callable.
The bundled :func:`default_embed` is a deterministic hashed bag-of-words
(stdlib only) so demos and tests run with zero API keys. For real use,
bring a real embedder (e.g. sentence-transformers) and inject it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

EmbedFn = Callable[[List[str]], List[List[float]]]

MANIFEST_NAME = "manifest.json"
VECTORS_NAME = "vectors.json"
CHUNKS_NAME = "chunks.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def hash_text(text: str) -> str:
    """Stable sha256 over normalized text (whitespace collapsed)."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def default_embed(texts: List[str], dim: int = 128) -> List[List[float]]:
    """Deterministic hashed bag-of-words embedding (stdlib only).

    Not a semantic embedding: use it for demos, tests, and plumbing, not
    for retrieval quality. Vectors are L2-normalized.
    """
    vectors: List[List[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    return vectors


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """Split text into word-boundary chunks of ~``chunk_size`` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    words = text.split()
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > chunk_size:
            chunks.append(" ".join(current))
            # carry overlap words into the next chunk
            overlap: List[str] = []
            overlap_len = 0
            for w in reversed(current):
                if overlap_len + len(w) + 1 > chunk_overlap:
                    break
                overlap.append(w)
                overlap_len += len(w) + 1
            current = list(reversed(overlap))
            current_len = sum(len(w) for w in current) + max(len(current) - 1, 0)
        current.append(word)
        current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


@dataclass(frozen=True)
class Source:
    """One ingestible source document."""

    source_id: str
    text: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    content_hash: str
    chunk_ids: Tuple[str, ...]
    embedded_at: str


@dataclass(frozen=True)
class IngestReport:
    added: Tuple[str, ...]
    updated: Tuple[str, ...]
    unchanged: Tuple[str, ...]
    removed: Tuple[str, ...]

    @property
    def n_embedded(self) -> int:
        return len(self.added) + len(self.updated)

    def summary(self) -> str:
        return (
            f"added={len(self.added)} updated={len(self.updated)} "
            f"unchanged={len(self.unchanged)} removed={len(self.removed)}"
        )


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ingestor:
    """Incrementally embed sources into a JSON-backed index directory."""

    def __init__(
        self,
        index_dir: str | Path,
        embed_fn: EmbedFn | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embed_dim: int = 128,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embed_fn: EmbedFn = embed_fn or (lambda texts: default_embed(texts, dim=embed_dim))
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_dim = embed_dim

    # -- manifest / store I/O -------------------------------------------------
    def _load_json(self, name: str) -> dict:
        path = self.index_dir / name
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_json(self, name: str, data: dict) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_dir / name, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_manifest(self) -> Dict[str, SourceRecord]:
        raw = self._load_json(MANIFEST_NAME).get("sources", {})
        return {
            sid: SourceRecord(
                source_id=sid,
                content_hash=rec["content_hash"],
                chunk_ids=tuple(rec["chunk_ids"]),
                embedded_at=rec["embedded_at"],
            )
            for sid, rec in raw.items()
        }

    def _save_manifest(self, records: Dict[str, SourceRecord]) -> None:
        self._save_json(
            MANIFEST_NAME,
            {
                "version": 1,
                "sources": {
                    sid: {
                        "content_hash": rec.content_hash,
                        "chunk_ids": list(rec.chunk_ids),
                        "embedded_at": rec.embedded_at,
                    }
                    for sid, rec in records.items()
                },
            },
        )

    # -- ingestion ------------------------------------------------------------
    def _embed_source(self, source: Source) -> Tuple[Tuple[str, ...], List[List[float]], List[str]]:
        chunks = chunk_text(source.text, self.chunk_size, self.chunk_overlap)
        chunk_ids = tuple(f"{source.source_id}#chunk-{i}" for i in range(len(chunks)))
        vectors = self.embed_fn(list(chunks)) if chunks else []
        return chunk_ids, vectors, chunks

    def ingest(self, sources: Iterable[Source]) -> IngestReport:
        """Embed new/changed sources; skip unchanged; prune removed."""
        sources = list(sources)
        manifest = self._load_manifest()
        vectors_store: dict = self._load_json(VECTORS_NAME).get("vectors", {})
        chunks_store: dict = self._load_json(CHUNKS_NAME).get("chunks", {})

        seen: set[str] = set()
        added: List[str] = []
        updated: List[str] = []
        unchanged: List[str] = []

        for source in sources:
            if source.source_id in seen:
                raise ValueError(f"duplicate source_id: {source.source_id!r}")
            seen.add(source.source_id)
            content_hash = hash_text(source.text)
            record = manifest.get(source.source_id)

            if record is not None and record.content_hash == content_hash:
                unchanged.append(source.source_id)
                continue

            # New or changed: drop stale chunks/vectors first.
            if record is not None:
                for cid in record.chunk_ids:
                    vectors_store.pop(cid, None)
                    chunks_store.pop(cid, None)
                updated.append(source.source_id)
            else:
                added.append(source.source_id)

            chunk_ids, vectors, chunk_texts = self._embed_source(source)
            for cid, text, vec in zip(chunk_ids, chunk_texts, vectors):
                chunks_store[cid] = text
                vectors_store[cid] = vec
            manifest[source.source_id] = SourceRecord(
                source_id=source.source_id,
                content_hash=content_hash,
                chunk_ids=chunk_ids,
                embedded_at=_utcnow(),
            )

        removed = sorted(set(manifest) - seen)
        for sid in removed:
            record = manifest.pop(sid)
            for cid in record.chunk_ids:
                vectors_store.pop(cid, None)
                chunks_store.pop(cid, None)

        self._save_manifest(manifest)
        self._save_json(VECTORS_NAME, {"version": 1, "vectors": vectors_store})
        self._save_json(CHUNKS_NAME, {"version": 1, "chunks": chunks_store})

        return IngestReport(
            added=tuple(added),
            updated=tuple(updated),
            unchanged=tuple(unchanged),
            removed=tuple(removed),
        )

    def ingest_paths(
        self, paths: Iterable[str | Path], base_dir: str | Path | None = None
    ) -> IngestReport:
        """Ingest text files; source_id is the path relative to ``base_dir``."""
        base = Path(base_dir) if base_dir else None
        file_paths = [Path(p) for p in paths]
        # Deterministic order so reports are stable.
        file_paths.sort(key=lambda p: str(p))
        sources = []
        for path in file_paths:
            text = path.read_text(encoding="utf-8")
            if base is not None:
                try:
                    sid = str(path.relative_to(base))
                except ValueError:
                    sid = str(path)
            else:
                sid = path.name
            sources.append(Source(source_id=sid, text=text))
        return self.ingest(sources)
