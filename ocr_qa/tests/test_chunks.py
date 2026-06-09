"""Tests for the output_chunks.json integration: parsing, page attachment
(0-based↔1-based offset, multi-page chunks), and its effect on XSA / TSI / BCC.
Run: pytest ocr_qa/tests/test_chunks.py -q
"""

from __future__ import annotations

import os

import pytest

from ocr_qa.config import Config
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.ocr.chandra_parser import attach_chunks, parse_chunks
from ocr_qa.ocr.client import MockChandraClient
from ocr_qa.scoring.aggregate import build_document_report

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))

# A trimmed copy of the REAL output_chunks.json the user provided.
REAL_CHUNKS = [
    {
        "page_start": 0, "page_end": 1, "page_ids": [0, 1], "header_path": [],
        "chunk_title": "Nemluvio Wave 7 (Q1 '26) ATU Survey", "token_count": 71,
        "content": "GALDERMA\n\n# Nemluvio Wave 7 (Q1 '26) ATU Survey\n\nGlobal Report",
        "table_metrics": {"has_table": False, "table_count": 0, "table_density": 0.0},
    },
    {
        "page_start": 2, "page_end": 3, "page_ids": [2, 3], "header_path": [],
        "chunk_title": "ATU Overview and Context", "token_count": 590,
        "content": "## ATU Overview and Context\n\n| Milestone | Date |\n| --- | --- |",
        "table_metrics": {"has_table": True, "table_count": 1, "table_density": 0.52},
    },
]


def _load(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    return c.parser.parse(b.output_json, b.output_md, b.metadata, b.output_chunks)


# --------------------------------------------------------------------------- #
def test_parse_real_chunk_shape():
    chunks = parse_chunks(REAL_CHUNKS)
    assert len(chunks) == 2
    assert chunks[0].page_ids == [0, 1]
    assert chunks[1].table_metrics["has_table"] is True
    assert chunks[1].table_metrics["table_count"] == 1
    assert "ATU Overview" in chunks[1].content


def test_parse_chunks_defensive():
    assert parse_chunks(None) == []
    assert parse_chunks("garbage") == []
    assert parse_chunks({"chunks": REAL_CHUNKS})  # wrapper form
    # missing keys tolerated
    c = parse_chunks([{"content": "x"}])
    assert len(c) == 1 and c[0].page_ids == []


def test_attach_offset_and_multipage():
    """1-based pages (1,2) must map to 0-based chunk ids (0,1)."""

    class P:  # minimal stand-in with the attributes attach_chunks touches
        def __init__(self, n):
            self.page_no = n
            self.chunk_text = None
            self.chunk_meta = {}

    pages = [P(1), P(2)]
    chunks = parse_chunks(
        [{"page_ids": [0, 1], "content": "shared chunk content",
          "table_metrics": {"has_table": True, "table_count": 1}}]
    )
    attach_chunks(pages, chunks)
    assert pages[0].chunk_text == "shared chunk content"
    assert pages[1].chunk_text == "shared chunk content"
    assert pages[0].chunk_meta["covered"] is True
    assert pages[0].chunk_meta["single_page"] is False  # spans 2 pages


def test_uncovered_page_flagged():
    class P:
        def __init__(self, n):
            self.page_no = n
            self.chunk_text = None
            self.chunk_meta = {}

    pages = [P(1), P(2)]
    chunks = parse_chunks([{"page_ids": [0], "content": "only covers page 1"}])
    attach_chunks(pages, chunks)
    assert pages[0].chunk_meta["covered"] is True
    assert pages[1].chunk_meta == {"chunks_present": True, "covered": False}


# --------------------------------------------------------------------------- #
def test_good_fixture_has_chunks():
    pages = _load("good_page")
    assert all(p.chunk_text for p in pages)
    assert pages[0].chunk_meta["covered"] is True


def test_xsa_uses_chunk_subpart_clean():
    pages = _load("good_page")
    ctx = build_context(pages, Config())
    m = {r.key: r for r in run_page(pages[0], None, ctx)}
    # chunk subpart present and high recall keeps XSA healthy
    assert "JSON↔chunk" in m["XSA"].how_computed
    assert m["XSA"].score >= 60
    assert any("chunk coverage" in e for e in m["XSA"].evidence)


def test_xsa_chunk_divergence_bad():
    pages = _load("bad_page")
    ctx = build_context(pages, Config())
    m = {r.key: r for r in run_page(pages[0], None, ctx)}
    # bad chunk content diverges -> low coverage noted
    assert any("disagrees with output_chunks" in e or "chunk coverage" in e
               for e in m["XSA"].evidence)


def test_tsi_detects_dropped_table_via_chunk():
    pages = _load("bad_page")
    ctx = build_context(pages, Config())
    # page 2 has no parsed table but its single-page chunk reports has_table
    m2 = {r.key: r for r in run_page(pages[1], None, ctx)}
    assert m2["TSI"].applicable
    assert m2["TSI"].score <= 30
    assert any("dropped table" in e or "table_count" in e for e in m2["TSI"].evidence)


# --------------------------------------------------------------------------- #
def test_verdicts_unchanged_with_chunks():
    good = build_document_report(_load("good_page"), None, Config())
    bad = build_document_report(_load("bad_page"), None, Config())
    assert good.document_verdict == "PASSED"
    assert bad.document_verdict == "FAILED"
