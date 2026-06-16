"""Metric registry: builds the document context (language, median blocks, PPL
calibration) and runs all 14 metrics on a page (Sec 4/5).

Metric order matches the spec groups so the UI lists them coherently.
"""

from __future__ import annotations

import statistics
from typing import Optional

from ocr_qa.config import Config
from ocr_qa.metrics.agreement import XSAMetric
from ocr_qa.metrics.base import BaseMetric, DocContext, detect_language
from ocr_qa.metrics.block_count import BCCMetric
from ocr_qa.metrics.diversity import TTRMetric
from ocr_qa.metrics.duplicate import DUPMetric, build_duplicate_map
from ocr_qa.metrics.emptiness import EMPMetric
from ocr_qa.metrics.entropy import CEDMetric
from ocr_qa.metrics.figures import FIGMetric
from ocr_qa.metrics.geometry import BGCMetric
from ocr_qa.metrics.inference_failed import IFRMetric
from ocr_qa.metrics.lexical import LVRMetric
from ocr_qa.metrics.markup import HMWMetric
from ocr_qa.metrics.perplexity import PPLMetric, build_ppl_context
from ocr_qa.metrics.pronounceability import OOVMetric
from ocr_qa.metrics.punctuation import PSWMetric
from ocr_qa.metrics.stopwords import SFCMetric
from ocr_qa.metrics.tables import TSIMetric
from ocr_qa.metrics.wordshape import WSAMetric
from ocr_qa.models import MetricResult, PageOCR

# Ordered list of all metrics (14 core + 3 enhancement).
METRICS: list[BaseMetric] = [
    IFRMetric(),   # A — engine & structure
    BCCMetric(),
    BGCMetric(),
    EMPMetric(),   # A+ empty content-block rate (enhancement)
    HMWMetric(),   # B — markup
    TSIMetric(),
    FIGMetric(),   # B+ figure & caption integrity (enhancement)
    LVRMetric(),   # C — lexical
    OOVMetric(),
    SFCMetric(),
    CEDMetric(),   # D — statistical text
    PPLMetric(),
    TTRMetric(),
    WSAMetric(),   # E — surface
    PSWMetric(),
    XSAMetric(),   # F — agreement
    DUPMetric(),   # F+ duplicate-page detection (enhancement)
]

METRIC_KEYS = [m.key for m in METRICS]
CORE_METRIC_KEYS = ["IFR", "BCC", "BGC", "HMW", "TSI", "LVR", "OOV", "SFC",
                    "CED", "PPL", "TTR", "WSA", "PSW", "XSA"]
assert len(METRIC_KEYS) == 17, "expected 14 core + 3 enhancement metrics"


def build_context(pages: list[PageOCR], config: Config) -> DocContext:
    """One pre-pass over the document to set up shared context."""
    ctx = DocContext(config=config)

    # languages per page (fallback to doc-dominant if a page is short)
    page_prose: dict[int, str] = {}
    for p in pages:
        prose = p.prose_text()
        page_prose[p.page_no] = prose
        lang = detect_language(prose) or detect_language(p.text)
        if lang:
            ctx.languages[p.page_no] = lang
            p.language = lang

    # doc-dominant language fills the gaps
    if ctx.languages:
        dominant = statistics.mode(list(ctx.languages.values()))
        for p in pages:
            if p.page_no not in ctx.languages:
                ctx.languages[p.page_no] = dominant
                p.language = dominant

    # median block count (use the direct-child count to match metadata/BCC)
    counts = [
        (p.num_blocks_json if p.num_blocks_json is not None else len(p.blocks))
        for p in pages
        if (p.num_blocks_json or p.blocks)
    ]
    ctx.doc_median_blocks = statistics.median(counts) if counts else None

    # PPL calibration on the doc's own prose
    build_ppl_context(ctx, page_prose)

    # near-duplicate page map (document-level, for DUP)
    ctx.duplicate_of = build_duplicate_map(pages)

    # decision-support context per page (page type, density, reading order)
    for p in pages:
        ctx.page_types[p.page_no] = classify_page_type(p)
        ctx.text_density[p.page_no] = text_density(p)
        ctx.reading_order_conf[p.page_no] = reading_order_confidence(p)
    return ctx


# --------------------------------------------------------------------------- #
# decision-support context helpers (metric-review recommendations)
# --------------------------------------------------------------------------- #
def classify_page_type(page: PageOCR) -> str:
    """prose-heavy / table-heavy / image-heavy / title-cover / list-caption /
    mixed / empty — used to decide which soft metrics actually apply."""
    blocks = page.blocks
    if not blocks:
        return "empty"
    from ocr_qa.metrics.base import word_tokens

    prose_toks = len(word_tokens(page.prose_text()))
    n_tables = len(page.table_blocks())
    n_figs = len(page.figure_blocks())
    n_list = sum(1 for b in blocks if b.block_type in ("ListItem", "Caption"))
    n_content = max(1, len([b for b in blocks if b.is_content]))
    has_header = any(b.block_type == "SectionHeader" for b in blocks)

    if prose_toks < 30 and has_header and len(blocks) <= 4:
        return "title-cover"
    if n_tables and n_tables / n_content >= 0.4:
        return "table-heavy"
    if n_figs and prose_toks < 40:
        return "image-heavy"
    if n_list >= max(2, 0.5 * len(blocks)) and prose_toks < 60:
        return "list-caption"
    if prose_toks >= 60:
        return "prose-heavy"
    return "mixed"


def text_density(page: PageOCR) -> float:
    """Extracted characters per 1e6 px^2 of page area (OCR coverage proxy)."""
    pb = page.raw.get("bbox") if isinstance(page.raw, dict) else None
    area = 0.0
    if isinstance(pb, list) and len(pb) == 4:
        area = max(1.0, (pb[2] - pb[0]) * (pb[3] - pb[1]))
    return round(len(page.text) / (area / 1e6), 2) if area else 0.0


def reading_order_confidence(page: PageOCR) -> float:
    """1.0 = blocks read top-to-bottom coherently; lower => order inversions."""
    blocks = [b for b in page.blocks if b.bbox and len(b.bbox) == 4]
    ordered = sorted(blocks, key=lambda b: b.reading_order)
    if len(ordered) < 2:
        return 1.0
    pb = page.raw.get("bbox") if isinstance(page.raw, dict) else None
    ph = pb[3] if (isinstance(pb, list) and len(pb) == 4 and pb[3]) else \
        max((b.bbox[3] for b in ordered), default=1000.0)
    inv = sum(1 for prev, cur in zip(ordered, ordered[1:])
              if cur.bbox[1] < prev.bbox[1] - 0.25 * ph)
    return round(1.0 - inv / (len(ordered) - 1), 3)


def native_reliability(page: PageOCR, native_text) -> bool:
    """Is the PDF's native text layer clean AND ALIGNED enough to trust CER/WER?

    The metric-review doc warns that CER/WER is misleading when the native layer
    is misaligned (different reading order, headers/footers, hidden text), even
    when the words match. So we require THREE things, not just vocabulary
    overlap: (1) a substantial native layer, (2) similar length, and crucially
    (3) sequence ALIGNMENT — the OCR and native token streams must line up, not
    merely share a bag of words. Re-ordered book pages fail (3) and are excluded,
    which prevents the native-CER false positives.
    """
    import difflib

    from ocr_qa.metrics.base import word_tokens

    if not native_text or not native_text.strip():
        return False
    nat = [t.lower() for t in word_tokens(native_text)]
    ocr = [t.lower() for t in word_tokens(page.text)]
    if len(nat) < 20 or len(ocr) < 20:
        return False
    # (1) vocabulary overlap — the content is actually present
    if len(set(nat) & set(ocr)) / max(1, len(set(ocr))) < 0.6:
        return False
    # (2) comparable length — no large hidden/extra native text
    if min(len(nat), len(ocr)) / max(len(nat), len(ocr)) < 0.6:
        return False
    # (3) sequence alignment — same reading order, not just same words
    ratio = difflib.SequenceMatcher(
        None, " ".join(ocr[:400]), " ".join(nat[:400])
    ).ratio()
    return ratio >= 0.6


def run_page(
    page: PageOCR,
    native_text: Optional[str],
    ctx: DocContext,
) -> list[MetricResult]:
    """Run all metrics on a page, never raising (defensive). Each result is
    tagged with its tier ('hard' = definite-eligible, 'soft' = supporting)."""
    from ocr_qa.config import HARD_METRICS

    results: list[MetricResult] = []
    for metric in METRICS:
        try:
            r = metric.compute(page, native_text, ctx)
            r.tier = "hard" if r.key in HARD_METRICS else "soft"
            # SOFT metrics never carry a hard trigger (safety net).
            if r.tier == "soft":
                r.hard_fail = False
            results.append(r)
            continue
        except Exception as exc:  # pragma: no cover - safety net
            results.append(
                MetricResult(
                    key=metric.key,
                    name=metric.name,
                    raw_value=0.0,
                    score=100.0,
                    status="na",
                    threshold={},
                    what_it_measures=metric.what_it_measures,
                    how_computed="metric errored",
                    evidence=[f"error: {type(exc).__name__}: {exc}"],
                    justification="Metric could not be computed; excluded from "
                    "scoring.",
                    reliability="low",
                    applicable=False,
                    tier="hard" if metric.key in HARD_METRICS else "soft",
                )
            )
    return results
