"""Tests for the enhancement metrics: EMP (empty blocks), FIG (figure caption),
DUP (duplicate page). Run: pytest ocr_qa/tests/test_enhancements.py -q
"""

from __future__ import annotations

import os

from ocr_qa.config import Config
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.ocr.chandra_parser import ChandraParser
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _page(children, page_no=1):
    obj = {"id": f"/page/{page_no}/Page/{page_no}", "block_type": "Page",
           "bbox": [0, 0, 1000, 1000], "children": children}
    return obj


def _run(pages):
    ctx = build_context(pages, Config())
    return [{r.key: r for r in run_page(p, None, ctx)} for p in pages]


# --------------------------------------------------------------------------- #
def test_emp_flags_empty_content_block():
    p = ChandraParser().parse([_page([
        {"id": "/page/1/Text/0", "block_type": "Text", "html": "<p>Real content here with words.</p>"},
        {"id": "/page/1/Text/1", "block_type": "Text", "html": "<p></p>"},  # empty
    ])])
    m = _run(p)[0]["EMP"]
    assert m.raw_value == 0.5
    assert m.score < 100
    assert any(L["kind"] == "empty_block" and L["confidence"] == "definite"
               for L in m.locations)


def test_emp_clean_when_all_filled():
    p = ChandraParser().parse([_page([
        {"id": "/page/1/Text/0", "block_type": "Text", "html": "<p>One paragraph.</p>"},
        {"id": "/page/1/Text/1", "block_type": "Text", "html": "<p>Another paragraph.</p>"},
    ])])
    assert _run(p)[0]["EMP"].score == 100.0


def test_fig_flags_missing_description():
    p = ChandraParser().parse([_page([
        {"id": "/page/1/Figure/0", "block_type": "Figure", "html": "<img src='a.png'/>"},  # no desc
        {"id": "/page/1/Figure/1", "block_type": "Figure",
         "html": "<img src='b.png'/><div class=\"img-description\">A clear caption.</div>"},
    ])])
    m = _run(p)[0]["FIG"]
    assert m.applicable
    assert 0.0 < m.raw_value < 1.0  # one of two missing
    assert any(L["kind"] == "figure_no_caption" for L in m.locations)


def test_fig_not_applicable_without_figures():
    p = ChandraParser().parse([_page([
        {"id": "/page/1/Text/0", "block_type": "Text", "html": "<p>No figures here at all.</p>"},
    ])])
    assert _run(p)[0]["FIG"].applicable is False


def test_dup_detects_duplicate_pages():
    txt = ("<p>The committee reviewed the submitted evidence and concluded that the "
           "treatment provides a measurable benefit for the target population across "
           "all assessed clinical endpoints in the study.</p>")
    pages = ChandraParser().parse([
        _page([{"id": "/page/1/Text/0", "block_type": "Text", "html": txt}], 1),
        _page([{"id": "/page/2/Text/0", "block_type": "Text", "html": txt}], 2),
    ])
    res = _run(pages)
    m1 = res[0]["DUP"]
    assert m1.applicable
    assert m1.raw_value >= 0.98  # identical text
    assert m1.score < 50
    assert any(L["kind"] == "duplicate_page" and L["confidence"] == "definite"
               for L in m1.locations)


def test_dup_unique_pages_pass():
    pages = ChandraParser().parse([
        _page([{"id": "/page/1/Text/0", "block_type": "Text",
                "html": "<p>Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu.</p>"}], 1),
        _page([{"id": "/page/2/Text/0", "block_type": "Text",
                "html": "<p>Completely different words here: pharma dosage subkutan injection efficacy safety.</p>"}], 2),
    ])
    assert _run(pages)[0]["DUP"].score >= 70


def test_enhancements_keep_fixture_verdicts():
    from ocr_qa.scoring.aggregate import build_document_report

    def load(folder):
        c = MockChandraClient(); b = c.from_dir(os.path.join(SAMPLE, folder))
        return c.parser.parse(b.output_json, b.output_md, b.metadata,
                              b.output_chunks, b.output_html)

    assert build_document_report(load("good_page"), None, Config()).document_verdict == "PASSED"
    assert build_document_report(load("bad_page"), None, Config()).document_verdict == "FAILED"
