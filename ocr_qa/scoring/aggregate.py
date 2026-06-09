"""Scoring aggregation (Sec 5) + document-report orchestration.

PQS = weighted sum of AVAILABLE metric scores (weights renormalised over the
metrics that actually applied).
DQS = token-count-weighted mean of PQS, with each page's weight floored so a
2-line cover page can't dominate (and a content-empty page isn't ignored).

``build_document_report`` ties parsing -> metrics -> scoring -> verdict ->
problems into one deterministic call used by both the UI and the tests.
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.config import Config
from ocr_qa.metrics.base import word_tokens
from ocr_qa.metrics.registry import build_context, run_page
from ocr_qa.models import (
    DocumentReport,
    IngestedDocument,
    MetricResult,
    PageOCR,
    PageReport,
)
from ocr_qa.scoring.explain import build_problems
from ocr_qa.scoring.verdict import decide_document, decide_page


def page_quality_score(metrics: list[MetricResult], config: Config) -> float:
    """PQS over available metrics with renormalised weights."""
    available = [m for m in metrics if m.applicable]
    if not available:
        return 100.0
    weights = config.normalized_weights([m.key for m in available])
    return round(sum(weights[m.key] * m.score for m in available), 1)


def document_quality_score(reports: list[PageReport], config: Config) -> float:
    """DQS = token-weighted mean of PQS with a per-page weight floor."""
    if not reports:
        return 100.0
    floor = float(config.short_page_token_floor)
    num = 0.0
    den = 0.0
    for r in reports:
        w = max(float(r.token_count), floor)
        num += w * r.page_score
        den += w
    return round(num / den, 1) if den else 100.0


def build_page_report(
    page: PageOCR,
    native_text: Optional[str],
    ctx,
    config: Config,
    image_path: Optional[str] = None,
) -> PageReport:
    metrics = run_page(page, native_text, ctx)
    pqs = page_quality_score(metrics, config)
    token_count = len(word_tokens(page.prose_text()))
    report = PageReport(
        page_no=page.page_no,
        metrics=metrics,
        page_score=pqs,
        token_count=token_count,
        image_path=image_path,
    )
    decide_page(report, config)  # sets verdict
    report.problems = build_problems(metrics, config.t_page) if report.verdict == "review" else []
    return report


def build_document_report(
    pages: list[PageOCR],
    ingested: Optional[IngestedDocument],
    config: Config,
    progress_cb=None,
) -> DocumentReport:
    """Full deterministic pipeline: context -> per-page metrics/scoring/verdict
    -> document verdict. ``progress_cb(i, total, page_no)`` is optional."""
    ctx = build_context(pages, config)

    # Align parsed pages (1-based after renumbering) with rendered images, which
    # are also 1-based but may differ in count. An offset is detected just in
    # case either side uses a different base.
    raw_images: dict[int, str] = {}
    offset = 0
    if ingested and ingested.pages:
        raw_images = {pi.page_no: pi.image_path for pi in ingested.pages}
        offset = min(raw_images) - min(p.page_no for p in pages)

    page_reports: list[PageReport] = []
    total = len(pages)
    for i, page in enumerate(pages, start=1):
        native = ingested.native_for(page.page_no + offset) if ingested else None
        report = build_page_report(
            page, native, ctx, config, raw_images.get(page.page_no + offset)
        )
        page_reports.append(report)
        if progress_cb:
            progress_cb(i, total, page.page_no)

    dqs = document_quality_score(page_reports, config)
    faulty = sorted(r.page_no for r in page_reports if r.verdict == "review")

    doc = DocumentReport(
        doc_id=ingested.doc_id if ingested else "document",
        document_score=dqs,
        page_reports=page_reports,
        faulty_pages=faulty,
    )
    decide_document(doc, config)  # sets verdict + reason + summary
    return doc
