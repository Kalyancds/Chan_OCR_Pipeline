"""M3 BGC — Block-Geometry Coherence (Sec 4, Group A).

On block bboxes/polygons: degenerate boxes (zero/negative area), boxes outside
the page bounds, heavy overlap between content blocks, and reading-order
inversions (y-order incoherent). No token geometry exists — only block boxes.
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, loc
from ocr_qa.models import Block, PageOCR


def _area(b: list[float]) -> float:
    if len(b) != 4:
        return 0.0
    return (b[2] - b[0]) * (b[3] - b[1])


def _degenerate(b: list[float]) -> bool:
    if len(b) != 4:
        return True
    return (b[2] - b[0]) <= 0 or (b[3] - b[1]) <= 0


def _outside(b: list[float], pw: float, ph: float, tol: float = 2.0) -> bool:
    if len(b) != 4:
        return False
    return b[0] < -tol or b[1] < -tol or b[2] > pw + tol or b[3] > ph + tol


def _overlap_area(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


class BGCMetric(BaseMetric):
    key = "BGC"
    name = "Block-Geometry Coherence"
    what_it_measures = (
        "Geometric sanity of block boxes: degenerate (zero/negative-area) "
        "boxes, boxes outside the page, heavily overlapping content blocks, and "
        "reading-order vs vertical-position inversions."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        blocks = [b for b in page.blocks if b.bbox]
        total = len(blocks)
        if total == 0:
            return self.not_applicable("no block bboxes on this page")

        pbbox = page.raw.get("bbox") if isinstance(page.raw, dict) else None
        if isinstance(pbbox, list) and len(pbbox) == 4:
            pw, ph = float(pbbox[2]), float(pbbox[3])
        else:
            pw = max((b.bbox[2] for b in blocks), default=2548.0)
            ph = max((b.bbox[3] for b in blocks), default=3508.0)

        bad_ids: set[str] = set()
        evidence: list[str] = []

        for b in blocks:
            if _degenerate(b.bbox):
                bad_ids.add(b.block_id)
                evidence.append(
                    f"{b.block_id or b.block_type}: degenerate box {b.bbox}"
                )
            elif _outside(b.bbox, pw, ph):
                bad_ids.add(b.block_id)
                evidence.append(
                    f"{b.block_id or b.block_type}: box {b.bbox} outside page "
                    f"(0..{pw:.0f} x 0..{ph:.0f})"
                )

        # heavy overlap between content blocks
        content = [b for b in blocks if b.is_content and not _degenerate(b.bbox)]
        for i in range(len(content)):
            for j in range(i + 1, len(content)):
                ov = _overlap_area(content[i].bbox, content[j].bbox)
                amin = min(_area(content[i].bbox), _area(content[j].bbox))
                if amin > 0 and ov / amin > 0.5:
                    bad_ids.add(content[i].block_id)
                    bad_ids.add(content[j].block_id)
                    evidence.append(
                        f"{content[i].block_id or content[i].block_type} & "
                        f"{content[j].block_id or content[j].block_type} overlap "
                        f"{ov / amin * 100:.0f}%"
                    )

        # reading-order inversions: blocks in reading order should not jump
        # sharply UP the page repeatedly.
        ordered = sorted(
            [b for b in blocks if not _degenerate(b.bbox)],
            key=lambda b: b.reading_order,
        )
        inversions = 0
        for prev, cur in zip(ordered, ordered[1:]):
            if cur.bbox[1] < prev.bbox[1] - 0.25 * ph:
                inversions += 1
        if inversions:
            evidence.append(
                f"{inversions} reading-order inversion(s) vs vertical position"
            )

        bad = len(bad_ids) + inversions
        raw = min(1.0, bad / total)
        score = lerp_score(raw, 0.0, 0.4)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        if not evidence:
            evidence = [f"all {total} boxes have valid geometry within the page"]

        if bad and evidence:
            example = (
                f"Example: {evidence[0]}. That makes {bad} of {total} boxes "
                f"geometrically broken ({raw * 100:.0f}%)."
            )
        else:
            example = (
                f"All {total} block boxes have positive area and sit inside the "
                f"{pw:.0f}×{ph:.0f} page, in top-to-bottom reading order → 0% bad."
            )

        locations = [
            loc("bad_geometry", "definite", block=b,
                snippet=(b.text or b.block_type)[:60])
            for b in blocks
            if b.block_id in bad_ids
        ]

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"bad_rate_zero_at": 0.0, "bad_rate_bad_at": 0.4},
            how_computed="(degenerate + out-of-bounds + heavily-overlapping + "
            "order-inverted blocks) / total blocks.",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"{bad}/{total} blocks have geometry problems "
                f"({raw * 100:.1f}%)."
                if bad
                else f"All {total} blocks have coherent geometry."
            ),
        )
