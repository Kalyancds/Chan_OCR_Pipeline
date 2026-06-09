"""M2 BCC — Block-Count & Source Consistency (Sec 4, Group A).

Compares blocks parsed from output.json against metadata.page_stats.num_blocks
for this page (dropped/extra blocks), and flags pages whose block count is
abnormally low versus the document's own median (possible content drop / blank
page). A mismatch beyond tolerance is a HARD condition (Sec 5).
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score
from ocr_qa.models import PageOCR


class BCCMetric(BaseMetric):
    key = "BCC"
    name = "Block-Count & Source Consistency"
    what_it_measures = (
        "Whether the number of blocks parsed from output.json agrees with the "
        "count Chandra recorded in metadata, and whether the page has an "
        "abnormally low block count versus the rest of the document."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        # metadata.num_blocks counts the Page's DIRECT children, so compare
        # against that count when available (falls back to flattened count).
        json_blocks = (
            page.num_blocks_json
            if page.num_blocks_json is not None
            else len(page.blocks)
        )
        meta = page.num_blocks_meta
        tol = ctx.config.bcc_tolerance
        evidence: list[str] = []
        threshold = {"tolerance": tol, "low_vs_median_factor": 0.5}

        if meta is None:
            # No metadata to compare; fall back to median check only.
            return self._apply_chunk(
                self._median_only(page, ctx, json_blocks, threshold), page
            )

        denom = max(meta, 1)
        raw = abs(json_blocks - meta) / denom
        evidence.append(f"json={json_blocks} blocks vs metadata={meta} blocks")

        # median context
        if ctx.doc_median_blocks:
            evidence.append(f"document median = {ctx.doc_median_blocks:.0f} blocks")

        score = lerp_score(raw, 0.0, max(tol * 2.5, 0.5))
        hard = raw > tol
        status = "bad" if hard else ("warn" if score < 70 else "good")

        if hard:
            direction = "fewer" if json_blocks < meta else "more"
            just = (
                f"output.json has {json_blocks} blocks but metadata expects "
                f"{meta} ({raw * 100:.1f}% mismatch, {direction} than recorded) "
                f"— exceeds the ±{tol * 100:.0f}% tolerance; possible "
                f"dropped/extra content."
            )
        else:
            just = (
                f"json={json_blocks} vs metadata={meta} "
                f"({raw * 100:.1f}% within ±{tol * 100:.0f}% tolerance)."
            )

        if hard:
            example = (
                f"output.json gave {json_blocks} blocks for this page but the "
                f"metadata says it should have {meta}. |{json_blocks}−{meta}|/"
                f"{meta} = {raw * 100:.0f}%, past the ±{tol * 100:.0f}% limit — "
                f"so blocks were likely dropped or duplicated."
            )
        else:
            example = (
                f"output.json gave {json_blocks} blocks and metadata expected "
                f"{meta}: |{json_blocks}−{meta}|/{meta} = {raw * 100:.0f}%, "
                f"comfortably within the ±{tol * 100:.0f}% tolerance."
            )
        return self._apply_chunk(
            self.result(
                raw_value=raw,
                score=score,
                status=status,
                threshold=threshold,
                how_computed="|json_blocks - metadata_blocks| / metadata_blocks; "
                "HARD when above tolerance.",
                example=example,
                evidence=evidence,
                justification=just,
                hard_fail=hard,
            ),
            page,
        )

    def _apply_chunk(self, res, page):
        """Lower the score and add evidence if the page is absent from
        output_chunks (a possible chunk/content drop). Never raises the score."""
        cm = page.chunk_meta or {}
        if cm.get("chunks_present") and not cm.get("covered"):
            res.applicable = True
            res.evidence.append(
                "page not covered by any output_chunks entry — possible "
                "chunk/content drop"
            )
            res.score = min(res.score, 55.0)
            if res.status in ("good", "na"):
                res.status = "warn"
            res.justification += (
                " Also: page is absent from output_chunks (possible content drop)."
            )
        return res

    def _median_only(self, page, ctx, json_blocks, threshold):
        median = ctx.doc_median_blocks
        if not median:
            return self.not_applicable(
                "no metadata num_blocks and no document median available"
            )
        ratio = json_blocks / median if median else 1.0
        low = ratio < 0.5
        raw = max(0.0, 1.0 - ratio)
        score = 40.0 if low else 100.0
        return self.result(
            raw_value=raw,
            score=score,
            status="warn" if low else "good",
            threshold=threshold,
            how_computed="metadata absent; compared block count to document "
            "median (flag if < 50% of median).",
            evidence=[
                f"json={json_blocks} blocks vs document median "
                f"{median:.0f} (ratio {ratio:.2f})"
            ],
            justification=(
                f"No metadata count; page has {json_blocks} blocks vs median "
                f"{median:.0f}"
                + (" — abnormally low (possible content drop)." if low else ".")
            ),
        )
