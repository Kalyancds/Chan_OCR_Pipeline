"""Tests for the output.html integration: per-page HTML split/attachment and
its JSON↔HTML agreement sub-part in XSA.
Run: pytest ocr_qa/tests/test_html.py -q
"""

from __future__ import annotations

import os

from ocr_qa.config import Config
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.ocr.chandra_parser import split_html
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))

REAL_HTML = (
    '<!DOCTYPE html><html><body>'
    '<div class="page" data-page-id="0"><p data-block-id="/page/0/Text/0">'
    'GALDERMA</p><h1>Nemluvio Wave 7</h1></div>'
    '<div class="page" data-page-id="1"><h2>Table of Contents</h2>'
    '<p><a href="#block-2-0">ATU Overview</a></p></div>'
    '</body></html>'
)


def _load(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    return c.parser.parse(b.output_json, b.output_md, b.metadata,
                          b.output_chunks, b.output_html)


def test_split_html_real_shape():
    seg = split_html(REAL_HTML)
    assert set(seg.keys()) == {0, 1}
    assert "GALDERMA" in seg[0]
    assert "Table of Contents" in seg[1]


def test_split_html_defensive():
    assert split_html(None) == {}
    assert split_html("") == {}
    assert split_html("<html><body>no page divs</body></html>") == {}


def test_good_fixture_has_html():
    pages = _load("good_page")
    assert all(p.html_text for p in pages)
    assert "Fachinformation" in pages[0].html_text
    assert "<table" in pages[0].html_text


def test_html_fallback_without_file():
    """Without output.html, html_text falls back to block HTML."""
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, "good_page"))
    pages = c.parser.parse(b.output_json, b.output_md, b.metadata, b.output_chunks)  # no html
    assert pages[0].html_text  # built from block htmls
    assert "<" in pages[0].html_text


def test_xsa_includes_json_html_subpart():
    pages = _load("good_page")
    ctx = build_context(pages, Config())
    m = {r.key: r for r in run_page(pages[0], None, ctx)}
    assert "JSON↔HTML" in m["XSA"].how_computed
    assert any("JSON↔HTML" in e for e in m["XSA"].evidence)
    # html derives from same blocks => high similarity => XSA still healthy on clean page
    assert m["XSA"].score >= 60


def test_verdicts_unchanged_with_html():
    from ocr_qa.scoring.aggregate import build_document_report

    assert build_document_report(_load("good_page"), None, Config()).document_verdict == "PASSED"
    assert build_document_report(_load("bad_page"), None, Config()).document_verdict == "FAILED"
