"""Tests for the PDF report builder (review/pdf_report.py).
Run: pytest ocr_qa/tests/test_pdf_report.py -q
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("reportlab")

from ocr_qa.config import Config
from ocr_qa.ocr.chandra_parser import audit_page_counts
from ocr_qa.ocr.client import MockChandraClient
from ocr_qa.review.pdf_report import LATEX, build_pdf_report
from ocr_qa.scoring.aggregate import build_document_report

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _doc(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    pages = c.parser.parse(b.output_json, b.output_md, b.metadata,
                           b.output_chunks, b.output_html)
    audit = audit_page_counts(b.output_json, b.output_md, b.metadata,
                              b.output_chunks, b.output_html, source_pages=2)
    return pages, build_document_report(pages, None, Config()), audit


def _assert_pdf(path):
    assert os.path.exists(path)
    with open(path, "rb") as fh:
        head = fh.read(5)
    assert head == b"%PDF-"
    assert os.path.getsize(path) > 2000  # non-trivial


def test_report_for_failed_doc(tmp_path):
    pages, doc, audit = _doc("bad_page")
    out = str(tmp_path / "bad_report.pdf")
    build_pdf_report(doc, None, Config(), out, page_audit=audit, source_name="bad.pdf")
    assert doc.document_verdict == "FAILED"
    _assert_pdf(out)


def test_report_for_passed_doc(tmp_path):
    pages, doc, audit = _doc("good_page")
    out = str(tmp_path / "good_report.pdf")
    build_pdf_report(doc, None, Config(), out, page_audit=audit, source_name="good.pdf")
    assert doc.document_verdict == "PASSED"
    _assert_pdf(out)


def test_every_metric_has_a_latex_formula():
    from ocr_qa.metrics.registry import METRIC_KEYS
    for k in METRIC_KEYS:
        assert k in LATEX and LATEX[k].strip(), f"missing LaTeX for {k}"
