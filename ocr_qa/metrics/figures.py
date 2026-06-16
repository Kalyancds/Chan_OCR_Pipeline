"""FIG — Figure & Caption Integrity (enhancement metric, Group B).

Source: output.json (Figure / Picture blocks; their <img> + img-description).

Chandra represents figures as an <img> with a sibling
``<div class="img-description">…</div>`` (and/or a Caption block). A figure with
NO usable description means the visual content was not described/captured — a
gap a human should check. If the page has no figures the metric is N/A.
"""

from __future__ import annotations

import re
from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, loc
from ocr_qa.models import PageOCR

_IMGDESC_RE = re.compile(r'img-description"?\s*>(.*?)</', re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


def _description_text(block) -> str:
    """Best-effort description for a figure block: img-description div, else the
    stripped block text (excluding the <img> tag itself)."""
    html = block.html or ""
    m = _IMGDESC_RE.search(html)
    if m:
        return _TAG_RE.sub(" ", m.group(1)).strip()
    return (block.text or "").strip()


class FIGMetric(BaseMetric):
    key = "FIG"
    name = "Figure & Caption Integrity"
    what_it_measures = (
        "Whether each figure/image block has a usable description or caption "
        "(Chandra's <img> + img-description). Figures with no description mean "
        "visual content was left undescribed."
    )

    def compute(self, page: PageOCR, native_text: Optional[str], ctx: DocContext):
        figs = page.figure_blocks()
        if not figs:
            return self.not_applicable("page has no figures/images")

        missing = []
        for b in figs:
            desc = _description_text(b)
            if len(desc) < 3:
                missing.append(b)

        total = len(figs)
        raw = len(missing) / total
        score = round((1.0 - raw) * 100.0, 1)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        locations = [
            loc("figure_no_caption", "definite", block=b, snippet="figure without description")
            for b in missing
        ]
        evidence = [
            f"{b.block_id or 'figure'}: no img-description/caption" for b in missing
        ] or [f"all {total} figure(s) have a description"]

        if missing:
            example = (
                f"{len(missing)} of {total} figure(s) have no description — e.g. "
                f"{missing[0].block_id or 'a figure'} has an <img> but no "
                f"img-description, so its content is uncaptured ({raw * 100:.0f}%)."
            )
        else:
            example = (
                f"All {total} figure(s) carry an img-description/caption → fully "
                f"described."
            )

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"undescribed_rate_good_at": 0.0, "bad_at": 1.0},
            how_computed="figures_without_description / total figures.",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"{len(missing)}/{total} figure(s) lack a description "
                f"({raw * 100:.0f}%)."
                if missing
                else f"All {total} figure(s) are described."
            ),
        )
