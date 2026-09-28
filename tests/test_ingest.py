"""Tests for incremental ingestion: hashing, change detection, pruning."""

from ragtrust import Ingestor, Source, hash_text


def _sources(texts: dict) -> list:
    return [Source(source_id=sid, text=text) for sid, text in texts.items()]


def test_hash_text_is_stable_and_sensitive():
    assert hash_text("hello world") == hash_text("hello  world\n")
    assert hash_text("hello world") != hash_text("hello there")


def test_first_ingest_embeds_everything(tmp_path):
    ing = Ingestor(tmp_path / "index")
    report = ing.ingest(_sources({"a": "alpha beta gamma", "b": "delta epsilon"}))
    assert set(report.added) == {"a", "b"}
    assert report.updated == () and report.unchanged == () and report.removed == ()
    assert report.n_embedded == 2
    assert (tmp_path / "index" / "manifest.json").exists()


def test_second_ingest_skips_unchanged(tmp_path):
    ing = Ingestor(tmp_path / "index")
    texts = {"a": "alpha beta gamma", "b": "delta epsilon"}
    ing.ingest(_sources(texts))
    report = ing.ingest(_sources(texts))
    assert set(report.unchanged) == {"a", "b"}
    assert report.added == () and report.updated == ()
    assert report.n_embedded == 0


def test_changed_source_is_reembedded(tmp_path):
    ing = Ingestor(tmp_path / "index")
    ing.ingest(_sources({"a": "alpha beta gamma", "b": "delta epsilon"}))
    report = ing.ingest(_sources({"a": "alpha beta gamma CHANGED", "b": "delta epsilon"}))
    assert report.updated == ("a",)
    assert report.unchanged == ("b",)
    assert report.n_embedded == 1


def test_removed_source_is_pruned(tmp_path):
    ing = Ingestor(tmp_path / "index")
    ing.ingest(_sources({"a": "alpha beta gamma", "b": "delta epsilon"}))
    report = ing.ingest(_sources({"a": "alpha beta gamma"}))
    assert report.removed == ("b",)
    manifest_sources = ing._load_manifest()
    assert "b" not in manifest_sources


def test_duplicate_source_ids_rejected(tmp_path):
    ing = Ingestor(tmp_path / "index")
    try:
        ing.ingest([Source("a", "x"), Source("a", "y")])
    except ValueError as exc:
        assert "duplicate source_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ingest_paths_from_directory(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    (src / "one.md").write_text("first document text", encoding="utf-8")
    (src / "two.txt").write_text("second document text", encoding="utf-8")
    ing = Ingestor(tmp_path / "index")
    report = ing.ingest_paths([src / "one.md", src / "two.txt"], base_dir=src)
    assert set(report.added) == {"one.md", "two.txt"}
