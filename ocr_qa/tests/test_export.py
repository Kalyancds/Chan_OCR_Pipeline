"""M5 export tests: faulty-only review package contains review.json/csv/md with
numeric problems and full metric evidence; send_for_review is a safe stub.
Run: pytest ocr_qa/tests/test_export.py -q
"""

from __future__ import annotations

import json
import os
import zipfile

from ocr_qa.config import Config
from ocr_qa.ocr.client import MockChandraClient
from ocr_qa.review.export import build_review_package, send_for_review
from ocr_qa.scoring.aggregate import build_document_report

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _doc(folder: str):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    pages = c.parser.parse(b.output_json, b.output_md, b.metadata)
    return build_document_report(pages, None, Config())


def test_package_contents(tmp_path):
    doc = _doc("bad_page")
    out = str(tmp_path / "review.zip")
    build_review_package(doc, None, out)
    assert os.path.exists(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "review.json" in names
        assert "review.csv" in names
        assert "review.md" in names
        data = json.loads(zf.read("review.json"))
        assert data["document_verdict"] == "FAILED"
        assert data["faulty_pages"], "expected faulty pages in package"
        page = data["faulty_pages"][0]
        assert "metrics" in page and len(page["metrics"]) == 14
        # every metric carries evidence + justification
        assert all("evidence" in m for m in page["metrics"])
        assert page["problems"], "faulty page must carry numeric problems"
        csv_text = zf.read("review.csv").decode()
        assert "page_no" in csv_text
        md_text = zf.read("review.md").decode()
        assert "Faults on this page" in md_text


def test_send_for_review_is_stub(tmp_path):
    out = str(tmp_path / "r.zip")
    doc = _doc("bad_page")
    build_review_package(doc, None, out)
    res = send_for_review(out)
    assert res["status"] == "not_sent"
    assert res["package"] == out
