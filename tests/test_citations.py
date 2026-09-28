"""Tests for citation grounding checks (heuristic)."""

from ragtrust import ScoredChunk, check_citations


def _chunk(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=text, score=0.9)


def test_supported_citation():
    chunks = [_chunk("d1", "hold the power button for ten seconds until it blinks")]
    report = check_citations("Hold the power button for ten seconds [d1].", chunks)
    assert report.citation_precision == 1.0
    assert report.n_unsupported == 0


def test_citation_to_unrelated_chunk_is_unsupported():
    chunks = [_chunk("d9", "the weather in Lisbon is mild in spring")]
    report = check_citations("Hold the power button for ten seconds [d9].", chunks)
    assert report.citation_precision == 0.0
    assert report.n_unsupported == 1


def test_citation_to_unknown_chunk_id_is_unsupported():
    chunks = [_chunk("d1", "hold the power button for ten seconds")]
    report = check_citations("Hold the power button [nope].", chunks)
    assert report.n_unsupported == 1


def test_no_citations_gives_none_precision():
    chunks = [_chunk("d1", "hold the power button for ten seconds")]
    report = check_citations("Hold the power button for ten seconds.", chunks)
    assert report.citation_precision is None


def test_mixed_sentences():
    chunks = [
        _chunk("d1", "hold the power button for ten seconds until it blinks"),
        _chunk("d2", "the battery charges over USB-C in ninety minutes"),
    ]
    answer = "Hold the power button for ten seconds [d1]. The moon is cheese [d2]."
    report = check_citations(answer, chunks)
    assert report.citation_precision == 0.5
    assert report.n_unsupported == 1
