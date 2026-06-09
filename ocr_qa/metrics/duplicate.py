"""DUP — Duplicate / Near-Duplicate Page (enhancement metric, Group F).

Source: output.json (page text across the whole document).

Flags a page whose text is a near-duplicate of ANOTHER page. Genuine OCR
duplication (a page emitted twice) inflates the page count and is a real defect;
it is also a useful diagnostic when the OCR page count exceeds the source PDF
(e.g. 191 vs 178). Repeated section dividers / TOC pages on slide decks are
common, so the bar is set high and only EXACT duplicates are 'definite'.

Cross-page comparison is done once in the registry pre-pass (``build_duplicate_map``)
and stored on the DocContext; this metric just reads the result for its page.
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, loc, word_tokens
from ocr_qa.models import PageOCR

# Tunables (kept here; high bar to avoid flagging legitimate repeated layouts).
DUP_WARN = 0.85   # similarity at/above this starts lowering the score
DUP_EXACT = 0.98  # at/above this => treated as a true duplicate (definite)
MIN_TOKENS = 20   # ignore very short pages (trivially similar)


def _shingles(text: str) -> set:
    toks = [t.lower() for t in word_tokens(text)]
    return set(toks)


def build_duplicate_map(pages: list[PageOCR]) -> dict[int, tuple]:
    """Return {page_no: (other_page_no, jaccard)} for each page's most similar
    OTHER page (Jaccard of token sets). O(n^2) but cheap for typical docs."""
    sigs = {}
    for p in pages:
        toks = _shingles(p.text or p.prose_text())
        sigs[p.page_no] = toks if len(toks) >= MIN_TOKENS else None

    out: dict[int, tuple] = {}
    items = [(pn, s) for pn, s in sigs.items() if s]
    for i, (pn_a, sa) in enumerate(items):
        best_pn, best_sim = None, 0.0
        for pn_b, sb in items:
            if pn_b == pn_a:
                continue
            inter = len(sa & sb)
            if inter == 0:
                continue
            union = len(sa | sb)
            sim = inter / union if union else 0.0
            if sim > best_sim:
                best_sim, best_pn = sim, pn_b
        if best_pn is not None:
            out[pn_a] = (best_pn, round(best_sim, 3))
    return out


class DUPMetric(BaseMetric):
    key = "DUP"
    name = "Duplicate / Near-Duplicate Page"
    what_it_measures = (
        "Whether this page's text is a near-duplicate of another page in the "
        "document (token-set Jaccard similarity). True duplicates inflate the "
        "page count and indicate the OCR emitted a page twice."
    )

    def compute(self, page: PageOCR, native_text: Optional[str], ctx: DocContext):
        match = ctx.duplicate_of.get(page.page_no)
        if not match:
            return self.not_applicable("no comparable page text (too short / unique)")
        other, sim = match

        score = lerp_score(sim, DUP_WARN, 1.0)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        is_dup = sim >= DUP_WARN
        confidence = "definite" if sim >= DUP_EXACT else "statistical"

        locations = []
        if is_dup:
            locations.append(loc("duplicate_page", confidence, block=None,
                                 snippet=f"~{sim*100:.0f}% identical to page {other}"))

        if sim >= DUP_EXACT:
            example = (
                f"This page is {sim*100:.0f}% identical (token overlap) to page "
                f"{other} — effectively a duplicate, which inflates the page count."
            )
        elif is_dup:
            example = (
                f"This page shares {sim*100:.0f}% of its words with page {other} "
                f"(threshold {DUP_WARN*100:.0f}%). Could be a repeated layout "
                f"(divider/TOC) or a real duplicate — worth a glance."
            )
        else:
            example = (
                f"Most similar other page is page {other} at only {sim*100:.0f}% "
                f"overlap (well below {DUP_WARN*100:.0f}%) → this page is unique."
            )

        return self.result(
            raw_value=sim,
            score=score,
            status=status,
            threshold={"warn": DUP_WARN, "exact_dup": DUP_EXACT, "min_tokens": MIN_TOKENS},
            how_computed="max token-set Jaccard similarity vs every other page; "
            "score falls as similarity approaches 100%.",
            example=example,
            locations=locations,
            evidence=[f"most similar page: {other} (Jaccard {sim*100:.0f}%)"],
            justification=(
                f"{sim*100:.0f}% token overlap with page {other}"
                + (" — near-duplicate." if is_dup else " — distinct page.")
            ),
        )
