"""M1 metric tests: clean fixture scores high, garbled fixture scores low and
trips the HARD conditions; multilingual / pharma proper nouns not falsely
flagged; tables not scored as prose. Run: pytest ocr_qa/tests/test_metrics.py -q
"""

from __future__ import annotations

import os

import pytest

from ocr_qa.config import Config
from ocr_qa.metrics.base import is_pronounceable
from ocr_qa.metrics.registry import METRIC_KEYS, build_context, run_page
from ocr_qa.models import PageOCR
from ocr_qa.ocr.client import MockChandraClient

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))


def _load(folder: str) -> list[PageOCR]:
    client = MockChandraClient()
    b = client.from_dir(os.path.join(SAMPLE, folder))
    return client.parser.parse(b.output_json, b.output_md, b.metadata)


@pytest.fixture(scope="module")
def good():
    pages = _load("good_page")
    ctx = build_context(pages, Config())
    return pages, ctx


@pytest.fixture(scope="module")
def bad():
    pages = _load("bad_page")
    ctx = build_context(pages, Config())
    return pages, ctx


def _by_key(results):
    return {r.key: r for r in results}


# --------------------------------------------------------------------------- #
def test_all_14_metrics_run(good):
    pages, ctx = good
    res = run_page(pages[0], None, ctx)
    assert len(res) == 14
    assert sorted(r.key for r in res) == sorted(METRIC_KEYS)
    # every metric carries at least one evidence example
    for r in res:
        assert r.evidence, f"{r.key} has no evidence"


def test_applicable_metrics_have_worked_examples(good, bad):
    # Every applicable metric should expose a plain-English worked example.
    for pages, ctx in (good, bad):
        for p in pages:
            for r in run_page(p, None, ctx):
                if r.applicable:
                    assert r.example.strip(), f"{r.key} (page {p.page_no}) has no example"


def test_examples_cite_real_values(bad):
    pages, ctx = bad
    m = {r.key: r for r in run_page(pages[1], None, ctx)}
    # LVR example should cite the real validity percentage / counts
    assert "%" in m["LVR"].example
    assert any(ch.isdigit() for ch in m["LVR"].example)
    # XSA example should reference repetition or chunk/MD divergence concretely
    assert any(w in m["XSA"].example.lower()
               for w in ("repeat", "chunk", "markdown", "consistent", "error"))


def test_clean_page_scores_high(good):
    pages, ctx = good
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["IFR"].score == 100.0 and not m["IFR"].hard_fail
    assert m["LVR"].score >= 80 and not m["LVR"].hard_fail
    assert m["CED"].score >= 70 and not m["CED"].hard_fail
    assert m["XSA"].score >= 60 and not m["XSA"].hard_fail
    assert m["HMW"].score >= 80
    assert m["BCC"].score >= 80 and not m["BCC"].hard_fail


def test_clean_pharma_nouns_not_flagged(good):
    pages, ctx = good
    m = _by_key(run_page(pages[0], None, ctx))
    # LVR should consider the page largely valid despite brand/German words
    assert m["LVR"].raw_value >= 0.90
    # pronounceability gate keeps brand names plausible
    assert is_pronounceable("Nemluvio")
    assert is_pronounceable("Galderma")
    assert not is_pronounceable("rnixqk")
    assert not is_pronounceable("qwzxfp")


def test_tables_not_scored_as_prose(good):
    pages, ctx = good
    p = pages[0]
    # table block exists and is excluded from prose
    assert any(b.is_table for b in p.blocks)
    assert "30 mg" not in p.prose_text()  # table content not in prose
    m = _by_key(run_page(p, None, ctx))
    assert m["TSI"].applicable  # table metric ran
    assert m["TSI"].score >= 70


# --------------------------------------------------------------------------- #
def test_garbled_page_inference_hard_fail(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["IFR"].hard_fail is True
    assert m["IFR"].score == 0.0


def test_garbled_page_lexical_low(bad):
    pages, ctx = bad
    # page 1 is dominated by a repeated valid phrase (caught by XSA/TTR), so the
    # genuine low-LVR / gibberish page is page 2.
    m2 = _by_key(run_page(pages[1], None, ctx))
    assert m2["LVR"].raw_value < 0.80
    assert m2["LVR"].hard_fail is True
    # OOV separates gibberish from plausible (on page 1 too)
    m1 = _by_key(run_page(pages[0], None, ctx))
    assert m1["OOV"].raw_value > 0.0


def test_garbled_page_mojibake_hard(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["CED"].hard_fail is True
    assert m["CED"].score <= 30


def test_garbled_page_repetition_hard(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["XSA"].hard_fail is True
    assert any("VLM loop" in e or "×" in e for e in m["XSA"].evidence)


def test_garbled_block_count_mismatch(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["BCC"].hard_fail is True
    assert "13" in " ".join(m["BCC"].evidence)


def test_garbled_markup_and_table(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["HMW"].score < 100  # malformed block present
    assert m["TSI"].score < 80   # ragged table


def test_garbled_geometry(bad):
    pages, ctx = bad
    # page 2 has a degenerate / out-of-bounds box
    m = _by_key(run_page(pages[1], None, ctx))
    assert m["BGC"].score < 100


def test_garbled_wordshape(bad):
    pages, ctx = bad
    m = _by_key(run_page(pages[0], None, ctx))
    assert m["WSA"].raw_value > 0.0
    assert m["WSA"].score < 100
