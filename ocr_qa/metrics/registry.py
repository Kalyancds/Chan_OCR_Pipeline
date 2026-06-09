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
    return ctx


def run_page(
    page: PageOCR,
    native_text: Optional[str],
    ctx: DocContext,
) -> list[MetricResult]:
    """Run all 14 metrics on a page, never raising (defensive)."""
    results: list[MetricResult] = []
    for metric in METRICS:
        try:
            results.append(metric.compute(page, native_text, ctx))
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
                )
            )
    return results
