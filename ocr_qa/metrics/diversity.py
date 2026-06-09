"""M11 TTR — Lexical Diversity / Type-Token Anomaly (Sec 4, Group D).

Type-Token Ratio compared to an expected band for the text length. A LOW tail
means looping/repetition (the model repeats a few words); a HIGH tail with junk
types means fragmentation/garble (almost every token is unique nonsense).
"""

from __future__ import annotations

import math
from typing import Optional

from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    is_pronounceable,
    word_tokens,
)
from ocr_qa.models import PageOCR


def _expected_band(n: int) -> tuple[float, float]:
    """Rough expected TTR band as a function of token count (natural prose).
    Longer texts naturally repeat function words, lowering TTR."""
    if n <= 0:
        return 0.0, 1.0
    # expected center decays ~ 1/sqrt-ish
    center = max(0.35, min(0.95, 1.6 / math.sqrt(n) + 0.30))
    low = max(0.20, center - 0.22)
    high = min(0.99, center + 0.20)
    return low, high


class TTRMetric(BaseMetric):
    key = "TTR"
    name = "Lexical Diversity / Type-Token Anomaly"
    what_it_measures = (
        "Whether vocabulary variety (unique words / total words) sits in the "
        "expected band for the text length. Too low = looping/repetition; too "
        "high with nonsense types = fragmentation."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        toks = [t.lower() for t in word_tokens(page.prose_text())]
        n = len(toks)
        if n < 10:
            return self.not_applicable("too few prose tokens for diversity (<10)")

        types = set(toks)
        ttr = len(types) / n
        low, high = _expected_band(n)

        tail = "in-band"
        if ttr < low:
            tail = "low (looping/repetition)"
            dist = low - ttr
        elif ttr > high:
            tail = "high (possible fragmentation)"
            dist = ttr - high
        else:
            dist = 0.0

        # For a high-tail page, check whether the unique types are mostly junk.
        junk_note = ""
        if ttr > high:
            junk = sum(1 for t in types if not is_pronounceable(t))
            jr = junk / len(types)
            junk_note = f"; {jr * 100:.0f}% of unique types are unpronounceable"
            if jr > 0.3:
                dist += 0.1

        score = max(0.0, 100.0 - (dist / 0.30) * 100.0)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        n_types = len(types)
        if score >= 70:
            example = (
                f"{n_types} distinct words across {n} tokens (TTR {ttr:.2f}) sits "
                f"within the healthy {low:.2f}–{high:.2f} band for this length."
            )
        elif tail.startswith("low"):
            example = (
                f"Only {n_types} distinct words across {n} tokens (TTR {ttr:.2f}), "
                f"below the {low:.2f}–{high:.2f} band expected here — a sign the "
                f"same words repeat (looping)."
            )
        else:
            example = (
                f"{n_types} distinct words across {n} tokens (TTR {ttr:.2f}), above "
                f"the {low:.2f}–{high:.2f} band{junk_note} — near-unique nonsense "
                f"suggests fragmentation."
            )

        return self.result(
            raw_value=ttr,
            score=score,
            status=status,
            threshold={"expected_low": round(low, 2), "expected_high": round(high, 2)},
            how_computed="unique_words / total_words vs length-adjusted expected "
            "band; distance outside band drives the score.",
            example=example,
            evidence=[
                f"TTR={ttr:.2f} over {n} tokens; expected {low:.2f}–{high:.2f}; "
                f"tail={tail}{junk_note}"
            ],
            justification=(
                f"Type-token ratio {ttr:.2f} ({tail}); expected band "
                f"{low:.2f}–{high:.2f} for {n} tokens."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
        )
