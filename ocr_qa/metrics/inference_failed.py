"""M1 IFR — Inference-Failure Rate (Sec 4, Group A).

PRIMARY engine signal. Chandra sets ``inference_failed=true`` on blocks its own
VLM could not transcribe. Any failure on a CONTENT block (Text/SectionHeader/
Table/ListItem) is a HARD FAIL that forces the page into the review queue.
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, loc
from ocr_qa.models import PageOCR


class IFRMetric(BaseMetric):
    key = "IFR"
    name = "Inference-Failure Rate"
    what_it_measures = (
        "Fraction of blocks Chandra's own engine flagged as inference_failed "
        "(it could not reliably transcribe them). This is the OCR engine's "
        "direct admission of failure."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        total = len(page.blocks)
        if total == 0:
            return self.not_applicable("page has no blocks")

        failed = [b for b in page.blocks if b.inference_failed]
        content_failed = [b for b in failed if b.is_content]
        raw = len(failed) / total

        evidence = [
            f"{b.block_id or '(no id)'} [{b.block_type}] inference_failed=true"
            for b in failed
        ] or ["no blocks reported inference_failed"]

        threshold = {"any_content_block_failed": "HARD FAIL", "rate_bad_at": 0.20}

        if content_failed:
            ids = ", ".join(b.block_id or b.block_type for b in content_failed)
            first = content_failed[0]
            example = (
                f"Block '{first.block_id or first.block_type}' "
                f"({first.block_type}) came back with inference_failed=true — "
                f"Chandra itself could not read it. That is {len(failed)} of "
                f"{total} blocks; because it is a content block, the page is "
                f"sent to review regardless of other scores."
            )
            return self.result(
                raw_value=raw,
                score=0.0,
                status="bad",
                threshold=threshold,
                how_computed="failed_blocks / total_blocks; HARD FAIL when any "
                "Text/SectionHeader/Table/ListItem block failed.",
                example=example,
                locations=[
                    loc("inference_failed", "definite", block=b,
                        snippet=(b.text or b.block_type)[:80])
                    for b in failed
                ],
                evidence=evidence,
                justification=(
                    f"{len(content_failed)} content block(s) returned "
                    f"inference_failed=true ({ids}) — the engine could not read "
                    f"page content, so the page MUST be reviewed."
                ),
                hard_fail=True,
            )

        if failed:
            from ocr_qa.metrics.base import lerp_score

            score = lerp_score(raw, 0.0, 0.20)
            return self.result(
                raw_value=raw,
                score=score,
                status="warn" if score >= 50 else "bad",
                threshold=threshold,
                how_computed="failed_blocks / total_blocks (non-content failures "
                "penalised but not a hard fail).",
                example=(
                    f"{len(failed)} of {total} non-content blocks (e.g. a header/"
                    f"footer) failed = {raw * 100:.0f}%. No Text/Table block "
                    f"failed, so it is penalised but not a hard fail."
                ),
                evidence=evidence,
                justification=(
                    f"{len(failed)}/{total} non-content blocks "
                    f"({raw * 100:.1f}%) had inference_failed=true."
                ),
            )

        return self.result(
            raw_value=0.0,
            score=100.0,
            status="good",
            threshold=threshold,
            how_computed="failed_blocks / total_blocks.",
            example=(
                f"All {total} blocks on this page were transcribed "
                f"(0 inference_failed) → 0/{total} = 0%, full marks."
            ),
            evidence=evidence,
            justification="No block reported inference_failed; engine read the "
            "whole page.",
        )
