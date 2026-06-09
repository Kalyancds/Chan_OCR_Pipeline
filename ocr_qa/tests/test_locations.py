"""Tests for error localization (locations) + definite/statistical confidence.
Run: pytest ocr_qa/tests/test_locations.py -q
"""

from __future__ import annotations

import os

import pytest

from ocr_qa.config import Config
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _load(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    return c.parser.parse(b.output_json, b.output_md, b.metadata,
                          b.output_chunks, b.output_html)


@pytest.fixture(scope="module")
def bad():
    pages = _load("bad_page")
    return pages, build_context(pages, Config())


def _by_key(pages, ctx, pno):
    page = next(p for p in pages if p.page_no == pno)
    return {r.key: r for r in run_page(page, None, ctx)}


def test_ifr_locates_failed_block_definite(bad):
    pages, ctx = bad
    m = _by_key(pages, ctx, 1)["IFR"]
    assert m.locations, "IFR should localize the failed block"
    L = m.locations[0]
    assert L["confidence"] == "definite"
    assert L["block_id"]  # has a concrete block id
    assert L["kind"] == "inference_failed"


def test_ced_locates_mojibake_definite(bad):
    pages, ctx = bad
    m = _by_key(pages, ctx, 1)["CED"]
    kinds = {L["kind"] for L in m.locations}
    assert "mojibake" in kinds
    assert all(L["confidence"] == "definite" for L in m.locations)


def test_xsa_locates_repeated_span_definite(bad):
    pages, ctx = bad
    m = _by_key(pages, ctx, 1)["XSA"]
    reps = [L for L in m.locations if L["kind"] == "repetition"]
    assert reps and reps[0]["confidence"] == "definite"
    assert "patient" in reps[0]["snippet"]  # the repeated phrase


def test_lvr_splits_definite_vs_statistical(bad):
    pages, ctx = bad
    m = _by_key(pages, ctx, 2)["LVR"]  # page 2 is gibberish-heavy
    confs = {L["confidence"] for L in m.locations}
    # unpronounceable gibberish -> definite present
    assert "definite" in confs
    assert any(L["kind"] == "gibberish_word" for L in m.locations)


def test_geometry_locates_bad_box_with_bbox(bad):
    pages, ctx = bad
    m = _by_key(pages, ctx, 2)["BGC"]  # page 2 has a degenerate box
    assert m.locations
    assert any(L["bbox"] for L in m.locations)
    assert all(L["confidence"] == "definite" for L in m.locations)


def test_clean_page_has_no_definite_errors():
    pages = _load("good_page")
    ctx = build_context(pages, Config())
    page = pages[0]
    locs = [L for r in run_page(page, None, ctx) for L in r.locations]
    assert not any(L["confidence"] == "definite" for L in locs)


def test_highlight_snippets_marks_text():
    from ocr_qa.models import PageReport
    from ocr_qa.ui.components import highlight_snippets

    pages = _load("bad_page")
    ctx = build_context(pages, Config())
    page = next(p for p in pages if p.page_no == 1)
    report = PageReport(page_no=1, metrics=run_page(page, None, ctx))
    out = highlight_snippets(page.text, report)
    assert "<mark" in out  # something was highlighted
