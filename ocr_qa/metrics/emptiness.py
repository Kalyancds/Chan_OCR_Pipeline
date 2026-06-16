"""EMP — Empty Content-Block Rate (enhancement metric, Group A).

Source: output.json (block tree).

BCC checks the NUMBER of blocks matches metadata; EMP checks those blocks
actually carry content. A content block (Text / SectionHeader / ListItem /
Caption) that strips to empty text means the OCR emitted a placeholder where
content should be — a concrete, reproducible content-loss signal (DEFINITE).
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, loc
from ocr_qa.models import PageOCR

_EMP_TYPES = {"Text", "SectionHeader", "ListItem", "Caption", "Footnote"}


class EMPMetric(BaseMetric):
    key = "EMP"
    name = "Empty Content-Block Rate"
    what_it_measures = (
        "Share of textual content blocks (Text/SectionHeader/ListItem/Caption) "
        "that are empty after stripping HTML — i.e. the OCR produced a block but "
        "captured no text in it (dropped content)."
    )

    def compute(self, page: PageOCR, native_text: Optional[str], ctx: DocContext):
        blocks = [b for b in page.blocks if b.block_type in _EMP_TYPES]
        total = len(blocks)
        if total == 0:
            return self.not_applicable("no textual content blocks on this page")

        empty = [b for b in blocks if not (b.text or "").strip()]
        raw = len(empty) / total
        score = lerp_score(raw, 0.0, 0.3)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        locations = [
            loc("empty_block", "definite", block=b, snippet=f"empty {b.block_type}")
            for b in empty
        ]
        evidence = [
            f"{b.block_id or b.block_type}: empty after HTML strip" for b in empty
        ] or [f"all {total} content blocks carry text"]

        if empty:
            example = (
                f"{len(empty)} of {total} content blocks are empty — e.g. "
                f"{empty[0].block_id or empty[0].block_type} has a box but no text. "
                f"That is {raw * 100:.0f}% empty (content likely dropped)."
            )
        else:
            example = (
                f"All {total} content blocks contain text after HTML stripping → "
                f"0% empty."
            )

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"empty_rate_good_at": 0.0, "bad_at": 0.3},
            how_computed="empty_content_blocks / total content blocks "
            "(text-bearing block types).",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"{len(empty)}/{total} content blocks are empty "
                f"({raw * 100:.0f}%) — meaningful content appears dropped."
                if empty
                else f"None of the {total} content blocks are empty."
            ),
            # DEFINITE: an empty text/content block means content was dropped.
            hard_fail=bool(empty),
        )
