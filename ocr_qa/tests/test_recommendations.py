"""Tests for the metric-review recommendations: hard/soft tier framework, the
XSA redesign (split sub-metrics, no worst-of hard gating, native gate), the
per-metric DEFINITE triggers, the decision-support context fields, and the
"1 hard OR >=3 soft" review rule. Run: pytest ocr_qa/tests/test_recommendations.py -q
"""

from __future__ import annotations

import os

from ocr_qa.config import Config, HARD_METRICS
from ocr_qa.metrics.agreement import XSAMetric
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.ocr.chandra_parser import ChandraParser
from ocr_qa.ocr.client import MockChandraClient
from ocr_qa.scoring.aggregate import build_document_report

HERE = os.path.dirname(__file__)
SAMPLE = os.path.normpath(os.path.join(HERE, "..", "sample"))
WORDS = " ".join("alpha beta gamma delta epsilon zeta eta theta iota kappa "
                 "lambda mu nu xi omicron pi rho sigma tau".split())
OTHER = " ".join("one two three four five six seven eight nine ten eleven "
                 "twelve thirteen fourteen fifteen sixteen".split())


def _page(children, page_no=1):
    return {"id": f"/page/{page_no}/Page/{page_no}", "block_type": "Page",
            "bbox": [0, 0, 1000, 1000], "children": children}


def _text_block(txt, pno=1, i=0, bt="Text"):
    return {"id": f"/page/{pno}/{bt}/{i}", "block_type": bt, "html": f"<p>{txt}</p>"}


def _load(folder):
    c = MockChandraClient()
    b = c.from_dir(os.path.join(SAMPLE, folder))
    return c.parser.parse(b.output_json, b.output_md, b.metadata,
                          b.output_chunks, b.output_html)


def _ctx_and_metrics(pages):
    ctx = build_context(pages, Config())
    return ctx, [{r.key: r for r in run_page(p, None, ctx)} for p in pages]


# --------------------------------------------------------------------------- #
# tier framework
# --------------------------------------------------------------------------- #
def test_soft_metrics_never_hard_fail():
    for folder in ("good_page", "bad_page"):
        _, mlist = _ctx_and_metrics(_load(folder))
        for m in mlist:
            for k, r in m.items():
                if r.tier == "soft":
                    assert not r.hard_fail, f"soft metric {k} hard-failed"


def test_hard_metric_set_matches_tiers():
    _, mlist = _ctx_and_metrics(_load("bad_page"))
    for r in mlist[0].values():
        expected = "hard" if r.key in HARD_METRICS else "soft"
        assert r.tier == expected


def test_lvr_is_soft_not_hard():
    _, mlist = _ctx_and_metrics(_load("bad_page"))
    # page 2 is gibberish-heavy; LVR must NOT be a hard trigger anymore
    assert mlist[1]["LVR"].tier == "soft"
    assert mlist[1]["LVR"].hard_fail is False


# --------------------------------------------------------------------------- #
# XSA redesign
# --------------------------------------------------------------------------- #
def _xsa_for(json_pages, md=None, chunks=None, html=None, native=None,
             native_reliable=False):
    pages = ChandraParser().parse(json_pages, md, None, chunks, html)
    ctx = build_context(pages, Config())
    ctx.native_reliable[pages[0].page_no] = native_reliable
    return XSAMetric().compute(pages[0], native, ctx), pages[0]


def test_xsa_single_artefact_disagreement_not_hard():
    md = "1------------------------------------------------\n" + OTHER  # only MD diverges
    res, _ = _xsa_for([_page([_text_block(WORDS)])], md=md)
    assert res.submetrics.get("json_md", 100) < 60   # MD disagrees
    assert res.hard_fail is False                     # but ONE artefact != hard


def test_xsa_two_artefact_disagreements_hard():
    md = "1------------------------------------------------\n" + OTHER
    chunks = [{"page_ids": [1], "content": OTHER}]    # MD and chunk both diverge
    res, _ = _xsa_for([_page([_text_block(WORDS)])], md=md, chunks=chunks)
    assert res.submetrics.get("json_md", 100) < 60
    assert res.submetrics.get("json_chunk", 100) < 60
    assert res.hard_fail is True                       # >=2 independent failures


def test_xsa_repetition_loop_is_hard():
    loop = ("the patient must take " * 8).strip()
    res, _ = _xsa_for([_page([_text_block(loop)])])
    assert res.submetrics["repetition"] == 0.0
    assert res.hard_fail is True


def test_xsa_native_cer_gated_by_reliability():
    garbled = "xqz wkpr mnbv lkjh teh teh teh of teh data was corrupted here now"
    native = WORDS  # very different => high CER
    # unreliable native -> CER skipped, not hard
    res_unrel, _ = _xsa_for([_page([_text_block(garbled)])], native=native,
                            native_reliable=False)
    assert "native_cer" not in res_unrel.submetrics
    # reliable native -> CER applies
    res_rel, _ = _xsa_for([_page([_text_block(garbled)])], native=native,
                          native_reliable=True)
    assert "native_cer" in res_rel.submetrics


def test_native_reliability_requires_alignment():
    """A native layer that shares words but is RE-ORDERED (book reading-order /
    headers-footers) must be judged unreliable, so CER cannot falsely hard-fail."""
    from ocr_qa.metrics.registry import native_reliability
    from ocr_qa.ocr.chandra_parser import ChandraParser

    body = ("the committee reviewed the submitted clinical evidence and concluded "
            "that the treatment provides a measurable and durable benefit across all "
            "assessed primary and secondary endpoints in adult study patients here")
    pages = ChandraParser().parse([_page([_text_block(body)])])
    p = pages[0]
    # aligned native (same order) -> reliable
    assert native_reliability(p, body) is True
    # same words, shuffled order (misaligned) -> NOT reliable
    shuffled = " ".join(reversed(body.split()))
    assert native_reliability(p, shuffled) is False


def test_xsa_native_cer_not_hard_when_misaligned():
    """The exact book false-positive: words present but reordered => native CER
    must be skipped, so XSA does not hard-fail a clean page."""
    body = ("annual insights over twenty years of market research across many "
            "global regions covering adoption attitudes and behaviours of clinicians "
            "and patients in detail throughout the full reporting period here now")
    misaligned = " ".join(sorted(body.split()))  # same words, wrong order
    res, _ = _xsa_for([_page([_text_block(body)])], native=misaligned,
                      native_reliable=False)  # gate would mark it unreliable
    assert "native_cer" not in res.submetrics
    assert res.hard_fail is False


def test_xsa_lone_native_cer_not_hard():
    """A reliable+aligned-prose native CER mismatch, with everything else
    agreeing, must NOT hard-fail on its own — native CER now corroborates (needs
    a 2nd independent disagreement). This is the financial/clinical-slide
    false-positive fix."""
    res, _ = _xsa_for([_page([_text_block(WORDS)])], native=OTHER,
                      native_reliable=True)
    assert "native_cer" in res.submetrics       # native applied (prose + reliable)
    assert res.submetrics["native_cer"] < 50    # high CER
    assert res.hard_fail is False               # but alone => NOT hard


def test_xsa_native_plus_one_artefact_is_hard():
    """Native CER + one artefact disagreement = 2 independent sources => hard."""
    md = "1------------------------------------------------\n" + OTHER  # MD diverges
    res, _ = _xsa_for([_page([_text_block(WORDS)])], md=md, native=OTHER,
                      native_reliable=True)
    assert res.hard_fail is True


def test_xsa_not_worst_of():
    # one zero sub-part should not collapse the whole XSA score to 0
    md = "1------------------------------------------------\n" + OTHER
    res, _ = _xsa_for([_page([_text_block(WORDS)])], md=md)
    assert res.score > 20  # mean, not worst-of


# --------------------------------------------------------------------------- #
# per-metric DEFINITE triggers
# --------------------------------------------------------------------------- #
def test_emp_empty_content_block_is_hard():
    pages = ChandraParser().parse([_page([
        _text_block("Real sentence with plenty of words here for the page body.", i=0),
        {"id": "/page/1/Text/1", "block_type": "Text", "html": "<p></p>"},
    ])])
    _, m = _ctx_and_metrics(pages)
    assert m[0]["EMP"].hard_fail is True


def test_hmw_hard_only_on_content_block():
    pages = ChandraParser().parse([_page([
        {"id": "/page/1/Text/0", "block_type": "Text",
         "html": "<p>unterminated <b>bold tag with no closing element at all here"},
    ])])
    _, m = _ctx_and_metrics(pages)
    assert m[0]["HMW"].hard_fail is True


def test_bgc_hard_only_for_severe_content_geometry():
    bad = {"id": "/page/1/Text/0", "block_type": "Text", "html": "<p>content text here</p>",
           "bbox": [-50, 800, 30, 760]}  # degenerate + out of bounds, content block
    pages = ChandraParser().parse([_page([bad])])
    _, m = _ctx_and_metrics(pages)
    assert m[0]["BGC"].hard_fail is True


def test_dup_hard_only_for_near_exact():
    txt = ("The committee reviewed the submitted evidence and concluded that the "
           "treatment provides a measurable and durable clinical benefit across all "
           "of the assessed primary and secondary study endpoints in adult patients.")
    pages = ChandraParser().parse([_page([_text_block(txt, pno=1)], 1),
                                   _page([_text_block(txt, pno=2)], 2)])
    _, m = _ctx_and_metrics(pages)
    assert m[0]["DUP"].hard_fail is True


# --------------------------------------------------------------------------- #
# review rule & context
# --------------------------------------------------------------------------- #
def test_context_fields_populated():
    doc = build_document_report(_load("good_page"), None, Config())
    r = doc.page_reports[0]
    assert r.page_type in {"prose-heavy", "mixed", "title-cover", "table-heavy",
                           "image-heavy", "list-caption"}
    assert r.text_density >= 0
    assert 0.0 <= r.reading_order_conf <= 1.0


def test_soft_metrics_gated_on_non_prose_page():
    # a title/cover-style page: header + tiny text => soft text metrics low-reliability
    pages = ChandraParser().parse([_page([
        {"id": "/page/1/SectionHeader/0", "block_type": "SectionHeader",
         "html": "<h2>Annual Report 2026</h2>"},
        _text_block("Global summary", i=1),
    ])])
    ctx, m = _ctx_and_metrics(pages)
    assert ctx.page_types[pages[0].page_no] in {"title-cover", "list-caption",
                                                "image-heavy", "mixed"}
    # SFC must not count as a soft failure on a non-prose page
    assert m[0]["SFC"].reliability == "low"


def test_clean_doc_passes_and_garbled_fails():
    assert build_document_report(_load("good_page"), None, Config()).document_verdict == "PASSED"
    assert build_document_report(_load("bad_page"), None, Config()).document_verdict == "FAILED"
