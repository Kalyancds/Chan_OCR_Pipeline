"""M0 parser tests: nested JSON-tree flatten, HTML strip, MD split, metadata,
defensive behaviour. Run with: pytest ocr_qa/tests/test_parser.py -q
"""

from __future__ import annotations

import os

import pytest

from ocr_qa.models import PageOCR
from ocr_qa.ocr.chandra_parser import (
    ChandraParser,
    parse_metadata,
    split_markdown,
    strip_html,
)
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))
GOOD = os.path.join(SAMPLE, "good_page")
BAD = os.path.join(SAMPLE, "bad_page")


# --------------------------------------------------------------------------- #
# unit helpers
# --------------------------------------------------------------------------- #
def test_strip_html_basic():
    assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert strip_html("") == ""
    assert "footnote" in strip_html("text<sup>1</sup> footnote").lower()


def test_strip_html_handles_br():
    out = strip_html("line one<br/>line two")
    assert "line one" in out and "line two" in out


def test_split_markdown_pages():
    md = "1------------------------------------------------\nalpha\n2----------------\nbeta"
    seg = split_markdown(md)
    assert seg[1].strip() == "alpha"
    assert seg[2].strip() == "beta"


def test_parse_metadata_page_stats():
    meta = {"page_stats": [{"page_id": 1, "num_blocks": 5}, {"page_id": 2, "num_blocks": 3}]}
    counts = parse_metadata(meta)
    assert counts == {1: 5, 2: 3}


def test_parse_metadata_defensive():
    assert parse_metadata(None) == {}
    assert parse_metadata({"page_stats": "nope"}) == {}
    assert parse_metadata({"page_stats": [{"bad": 1}]}) == {}


# --------------------------------------------------------------------------- #
# end-to-end on fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def good_pages() -> list[PageOCR]:
    client = MockChandraClient()
    b = client.from_dir(GOOD)
    return client.parser.parse(b.output_json, b.output_md, b.metadata)


@pytest.fixture
def bad_pages() -> list[PageOCR]:
    client = MockChandraClient()
    b = client.from_dir(BAD)
    return client.parser.parse(b.output_json, b.output_md, b.metadata)


def test_good_fixture_structure(good_pages):
    assert len(good_pages) == 2
    p1 = good_pages[0]
    # footer with html="" is skipped -> 5 emitted blocks
    assert len(p1.blocks) == 5
    types = [b.block_type for b in p1.blocks]
    assert "SectionHeader" in types and "Table" in types and "Figure" in types
    assert "PageFooter" not in types  # empty footer skipped
    # table / figure flagged
    assert any(b.is_table for b in p1.blocks)
    assert any(b.is_figure for b in p1.blocks)
    # no inference failures in clean doc
    assert all(not b.inference_failed for b in p1.blocks)
    # metadata count read (direct-children convention => 6 incl. empty footer)
    assert p1.num_blocks_meta == 6
    assert p1.num_blocks_json == 6  # direct children of the Page block
    # md cross-source text present
    assert p1.md_text and "Nemluvio" in p1.md_text


def test_good_fixture_reading_order_and_text(good_pages):
    p1 = good_pages[0]
    orders = [b.reading_order for b in p1.blocks]
    assert orders == sorted(orders)  # preserved, monotonic
    assert "Nemolizumab" in p1.text
    # accents preserved (multilingual)
    assert "gemäß" in p1.text


def test_bad_fixture_inference_failed(bad_pages):
    p1 = bad_pages[0]
    failed = [b for b in p1.blocks if b.inference_failed]
    assert len(failed) == 1
    assert failed[0].is_content  # Text block => forces faulty later
    # metadata mismatch present (json 5 vs meta 13)
    assert len(p1.content_blocks()) >= 1
    assert p1.num_blocks_meta == 13


def test_bad_fixture_table_and_md_divergence(bad_pages):
    p1 = bad_pages[0]
    assert any(b.is_table for b in p1.blocks)
    # md exists but diverges from json content
    assert p1.md_text is not None
    assert "different paragraph" in p1.md_text


def test_parse_log_records_findings(good_pages):
    # parse_log populated via the same parser instance used in fixture
    client = MockChandraClient()
    b = client.from_dir(GOOD)
    client.parser.parse(b.output_json, b.output_md, b.metadata)
    log = client.parser.parse_log
    assert len(log) == 2
    assert log[0]["blocks_found"] == 5
    assert log[0]["skipped_empty_headfoot"] == 1


def test_defensive_empty_input():
    p = ChandraParser()
    assert p.parse([]) == []
    assert p.parse(None) == []
    # missing children key tolerated
    pages = p.parse([{"id": "/page/1/Page/1", "html": "<p>hi</p>"}])
    assert len(pages) == 1
