"""Tests for the preprocessing page-count reconciliation (audit_page_counts).
Run: pytest ocr_qa/tests/test_audit.py -q
"""

from __future__ import annotations

import os

from ocr_qa.ocr.chandra_parser import audit_page_counts
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _bundle(folder):
    return MockChandraClient().from_dir(os.path.join(SAMPLE, folder))


def test_audit_all_artefacts_agree():
    b = _bundle("good_page")
    a = audit_page_counts(b.output_json, b.output_md, b.metadata, b.output_chunks, b.output_html)
    assert a["ocr_consistent"] is True
    assert a["ocr_pages"] == 2
    assert a["status"] == "ok"
    # each artefact counted at 2
    assert a["counts"]["output.json"] == 2
    assert a["counts"]["output.md"] == 2
    assert a["counts"]["output.html"] == 2
    assert a["counts"]["metadata"] == 2


def test_audit_source_mismatch():
    b = _bundle("good_page")
    a = audit_page_counts(b.output_json, b.output_md, b.metadata, b.output_chunks,
                          b.output_html, source_pages=5)
    assert a["status"] == "source_mismatch"
    assert a["delta_ocr_minus_source"] == 2 - 5
    assert any("drift" in m for m in a["messages"])


def test_audit_source_matches():
    b = _bundle("good_page")
    a = audit_page_counts(b.output_json, b.output_md, b.metadata, b.output_chunks,
                          b.output_html, source_pages=2)
    assert a["status"] == "ok"
    assert a["delta_ocr_minus_source"] == 0


def test_audit_ocr_inconsistent():
    # JSON has 2 pages but metadata claims 3 -> inconsistent
    b = _bundle("good_page")
    bad_meta = {"page_stats": [{"page_id": i, "num_blocks": 1} for i in range(3)]}
    a = audit_page_counts(b.output_json, b.output_md, bad_meta, b.output_chunks, b.output_html)
    assert a["ocr_consistent"] is False
    assert a["status"] == "ocr_inconsistent"


def test_audit_defensive_minimal():
    a = audit_page_counts([{"id": "/page/0/Page/0", "children": []}])
    assert a["ocr_pages"] == 1
    assert a["status"] == "ok"
