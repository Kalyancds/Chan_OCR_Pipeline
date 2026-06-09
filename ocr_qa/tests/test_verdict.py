"""M2 scoring + verdict tests: clean doc PASSES, garbled doc FAILS with a
numeric reason and named HARD rules; problems[] are numeric; determinism.
Run: pytest ocr_qa/tests/test_verdict.py -q
"""

from __future__ import annotations

import os

import pytest

from ocr_qa.config import Config
from ocr_qa.models import PageOCR
from ocr_qa.ocr.client import MockChandraClient
from ocr_qa.scoring.aggregate import build_document_report

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _load(folder: str) -> list[PageOCR]:
    client = MockChandraClient()
    b = client.from_dir(os.path.join(SAMPLE, folder))
    return client.parser.parse(b.output_json, b.output_md, b.metadata)


def test_clean_document_passes():
    doc = build_document_report(_load("good_page"), None, Config())
    assert doc.document_verdict == "PASSED"
    assert doc.document_score >= 75
    assert doc.faulty_pages == []
    assert "PASSED" in doc.verdict_reason
    assert doc.summary["hard_pages"] == []


def test_garbled_document_fails():
    doc = build_document_report(_load("bad_page"), None, Config())
    assert doc.document_verdict == "FAILED"
    assert len(doc.faulty_pages) >= 1
    assert "FAILED" in doc.verdict_reason
    # numeric reason present
    assert "/100" in doc.verdict_reason
    # named HARD rules surfaced
    assert doc.summary["hard_rules"], "expected at least one hard rule"
    assert "inference_failed content" in doc.verdict_reason


def test_inference_failed_forces_faulty():
    doc = build_document_report(_load("bad_page"), None, Config())
    # page 1 has an inference_failed content block -> must be faulty
    assert 1 in doc.faulty_pages
    assert 1 in doc.summary["inference_failed_pages"]


def test_problems_are_numeric():
    doc = build_document_report(_load("bad_page"), None, Config())
    p1 = next(r for r in doc.page_reports if r.page_no == 1)
    assert p1.problems, "faulty page must list problems"
    joined = " ".join(p1.problems)
    # at least one digit-bearing, metric-tagged problem
    assert any(ch.isdigit() for ch in joined)
    assert any(p.startswith("[") for p in p1.problems)


def test_determinism():
    cfg = Config()
    pages = _load("bad_page")
    d1 = build_document_report(pages, None, cfg)
    d2 = build_document_report(_load("bad_page"), None, cfg)
    assert d1.document_verdict == d2.document_verdict
    assert d1.verdict_reason == d2.verdict_reason
    assert d1.document_score == d2.document_score
    assert [r.page_score for r in d1.page_reports] == [
        r.page_score for r in d2.page_reports
    ]


def test_pqs_within_bounds():
    for folder in ("good_page", "bad_page"):
        doc = build_document_report(_load(folder), None, Config())
        for r in doc.page_reports:
            assert 0.0 <= r.page_score <= 100.0
        assert 0.0 <= doc.document_score <= 100.0
