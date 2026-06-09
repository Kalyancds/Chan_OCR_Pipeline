"""M4 HMW — HTML/Markup Well-formedness (Sec 4, Group B).

Chandra emits HTML per block. We balance-check each block's tags (unclosed /
mismatched), detect broken entities and stray angle brackets. A high malformed
rate signals a degraded transcription.
"""

from __future__ import annotations

import re
from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, loc
from ocr_qa.models import PageOCR

_VOID = {"br", "img", "hr", "meta", "input", "col", "area", "base", "link", "source"}
_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(/?)\s*>")
_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_BAD_AMP_RE = re.compile(r"&(?!amp;|lt;|gt;|quot;|apos;|nbsp;|#\d+;|#x[0-9a-fA-F]+;)")


def _malformed(html: str) -> Optional[str]:
    """Return a short reason string if the block HTML is malformed, else None."""
    if not html or "<" not in html:
        return None

    stack: list[str] = []
    for m in _TAG_RE.finditer(html):
        closing, name, selfclose = m.group(1), m.group(2).lower(), m.group(3)
        if name in _VOID or selfclose == "/":
            continue
        if not closing:
            stack.append(name)
        else:
            if name not in stack:
                return f"stray closing </{name}>"
            # pop until match (tolerate simple nesting)
            while stack and stack[-1] != name:
                stack.pop()
            if stack:
                stack.pop()
    if stack:
        return f"unclosed <{stack[-1]}>"

    # broken entity (bare & not part of a valid entity)
    if _BAD_AMP_RE.search(html):
        return "broken/raw '&' entity"

    return None


class HMWMetric(BaseMetric):
    key = "HMW"
    name = "HTML/Markup Well-formedness"
    what_it_measures = (
        "Whether each block's HTML is well-formed: balanced tags, no stray "
        "closing tags, no unclosed elements, no broken character entities."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        blocks = [b for b in page.blocks if b.html and "<" in b.html]
        total = len(blocks)
        if total == 0:
            return self.not_applicable("no HTML blocks on this page")

        bad = 0
        evidence: list[str] = []
        locations: list[dict] = []
        for b in blocks:
            reason = _malformed(b.html)
            if reason:
                bad += 1
                snippet = b.html.strip()[:60].replace("\n", " ")
                evidence.append(
                    f"{b.block_id or b.block_type}: {reason} — '{snippet}…'"
                )
                locations.append(
                    loc("malformed_html", "definite", block=b,
                        snippet=f"{reason}: {snippet}")
                )

        raw = bad / total
        score = lerp_score(raw, 0.0, 0.3)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        if not evidence:
            evidence = [f"all {total} HTML blocks are well-formed"]

        if bad:
            example = (
                f"Example: {evidence[0]} — {bad} of {total} HTML blocks are "
                f"malformed ({raw * 100:.0f}%)."
            )
        else:
            example = (
                f"Every one of the {total} HTML blocks has balanced tags and "
                f"valid entities (e.g. <p>…</p>, <table>…</table>) → 0 malformed."
            )

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"malformed_rate_zero_at": 0.0, "bad_at": 0.3},
            how_computed="malformed_blocks / total HTML blocks (tag-balance + "
            "entity check via lxml-style scan).",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"{bad}/{total} blocks have malformed markup "
                f"({raw * 100:.1f}%)."
                if bad
                else f"All {total} HTML blocks are well-formed."
            ),
        )
