"""UI smoke tests via streamlit.testing.AppTest: the app script runs without
raising on initial load AND after a real analysis is injected — exercising every
stage renderer (gauge, faulty side-by-side + md panel, verdict, export) through
the persisted-stage navigation. Run: pytest ocr_qa/tests/test_app_smoke.py -q
"""

from __future__ import annotations

import os

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ocr_qa.config import Config  # noqa: E402
from ocr_qa.ocr.client import MockChandraClient  # noqa: E402
from ocr_qa.scoring.aggregate import build_document_report  # noqa: E402

HERE = os.path.dirname(__file__)
APP = os.path.normpath(os.path.join(HERE, "..", "app.py"))
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))
STAGES = ["Ingest", "Processing", "Faulty Pages", "Passed Pages", "Verdict", "Export"]


def _doc(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    pages = c.parser.parse(b.output_json, b.output_md, b.metadata, b.output_chunks)
    return pages, build_document_report(pages, None, Config())


def _seed(at, pages, doc, ingested=None):
    at.session_state["doc"] = doc
    at.session_state["pages"] = pages
    at.session_state["ingested"] = ingested
    at.session_state["parse_log"] = [{"page_no": 1, "blocks_found": 5}]
    at.session_state["faulty_pick"] = doc.faulty_pages[0] if doc.faulty_pages else None


def test_app_initial_load():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception


def test_every_stage_renders_after_analysis():
    # bad doc exercises Faulty stage; good doc exercises Passed stage.
    for folder in ("bad_page", "good_page"):
        pages, doc = _doc(folder)
        for stage in STAGES:
            at = AppTest.from_file(APP, default_timeout=60)
            at.run()
            _seed(at, pages, doc)
            at.session_state["stage"] = stage
            at.run()
            assert not at.exception, f"{folder} stage {stage!r} raised: {at.exception}"


def test_faulty_page_shows_image_and_md(tmp_path):
    """Side-by-side image + markdown extract path with a real ingested image."""
    from PIL import Image

    from ocr_qa.models import IngestedDocument, PageImage

    pages, doc = _doc("bad_page")
    faulty = doc.faulty_pages[0]
    img_path = str(tmp_path / "page.png")
    Image.new("RGB", (120, 160), (240, 240, 240)).save(img_path)
    ingested = IngestedDocument(
        doc_id="bad", source_type="pdf",
        pages=[PageImage(page_no=faulty, image_path=img_path, width=120, height=160)],
    )
    # re-score so report.image_path is populated with offset-aware lookup
    doc = build_document_report(pages, ingested, Config())

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _seed(at, pages, doc, ingested)
    at.session_state["stage"] = "Faulty Pages"
    at.run()
    assert not at.exception, f"faulty view raised: {at.exception}"


def test_clean_doc_passes_verdict():
    pages, doc = _doc("good_page")
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _seed(at, pages, doc)
    at.session_state["stage"] = "Verdict"
    at.run()
    assert not at.exception
    assert doc.document_verdict == "PASSED"
