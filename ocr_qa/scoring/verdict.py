"""Page and document verdicts (Sec 5 & 6).

HARD conditions are carried on the individual ``MetricResult.hard_fail`` flags
(set by IFR, CED, XSA, LVR, BCC), so the verdict logic stays in one place and
the named rules are derived directly from which metrics tripped.

Everything here is deterministic: identical inputs => identical verdict AND
identical wording.
"""

from __future__ import annotations

from ocr_qa.config import Config
from ocr_qa.models import DocumentReport, PageReport

# Human-readable names for the HARD rules, keyed by metric.
_HARD_RULE_NAMES = {
    "IFR": "inference_failed on a content block",
    "CED": "mojibake/encoding corruption above bound",
    "XSA": "repetition loop / native-text CER over bound",
    "LVR": "lexical validity below 80%",
    "BCC": "block-count mismatch beyond tolerance",
}


def page_hard_fails(report: PageReport) -> list[str]:
    """Return the metric keys that tripped a HARD condition on this page."""
    return [m.key for m in report.metrics if m.hard_fail and m.applicable]


def decide_page(report: PageReport, config: Config) -> PageReport:
    """Set report.verdict to 'review' if PQS < T_page OR any HARD condition."""
    hard = page_hard_fails(report)
    if report.page_score < config.t_page or hard:
        report.verdict = "review"
    else:
        report.verdict = "ok"
    return report


def decide_document(doc: DocumentReport, config: Config) -> DocumentReport:
    """Set document verdict PASSED/FAILED with a numeric reason (Sec 6)."""
    total = len(doc.page_reports)
    faulty = len(doc.faulty_pages)
    ratio = (faulty / total) if total else 0.0

    # gather all hard-failing pages and the rules they hit
    hard_pages: list[int] = []
    hard_rules: set[str] = set()
    inference_pages: list[int] = []
    for r in doc.page_reports:
        hf = page_hard_fails(r)
        if hf:
            hard_pages.append(r.page_no)
            for k in hf:
                hard_rules.add(k)
            if "IFR" in hf:
                inference_pages.append(r.page_no)

    cond_dqs = doc.document_score >= config.t_doc
    cond_ratio = ratio <= config.doc_flag_ratio
    cond_hard = len(hard_pages) == 0

    passed = cond_dqs and cond_ratio and cond_hard
    doc.document_verdict = "PASSED" if passed else "FAILED"

    # ---- numeric one-line reason + named failing rules -------------------
    failed_rules: list[str] = []
    if not cond_dqs:
        failed_rules.append(
            f"document score {doc.document_score:.0f} < {config.t_doc:.0f}"
        )
    if not cond_ratio:
        failed_rules.append(
            f"{faulty}/{total} pages ({ratio * 100:.0f}%) need review "
            f"> {config.doc_flag_ratio * 100:.0f}% allowed"
        )
    if not cond_hard:
        named = ", ".join(
            _HARD_RULE_NAMES.get(k, k) for k in sorted(hard_rules)
        )
        failed_rules.append(
            f"{len(hard_pages)} page(s) hit a HARD condition [{named}]"
        )

    if passed:
        reason = (
            f"PASSED: document score {doc.document_score:.0f}/100; "
            f"{faulty}/{total} pages ({ratio * 100:.0f}%) flagged; "
            f"no HARD conditions."
        )
    else:
        head = (
            f"FAILED: {faulty}/{total} pages ({ratio * 100:.0f}%) need review; "
            f"doc score {doc.document_score:.0f}/100"
        )
        if inference_pages:
            head += f"; {len(inference_pages)} page(s) had inference_failed content"
        reason = head + ". Rule(s): " + "; ".join(failed_rules) + "."

    doc.verdict_reason = reason
    doc.summary = {
        "total_pages": total,
        "passed_pages": total - faulty,
        "faulty_pages": faulty,
        "faulty_ratio": round(ratio, 3),
        "dqs": doc.document_score,
        "hard_pages": sorted(hard_pages),
        "hard_rules": sorted(hard_rules),
        "inference_failed_pages": sorted(inference_pages),
        "thresholds": {
            "t_doc": config.t_doc,
            "t_page": config.t_page,
            "doc_flag_ratio": config.doc_flag_ratio,
        },
        "failed_rules": failed_rules,
    }
    return doc
