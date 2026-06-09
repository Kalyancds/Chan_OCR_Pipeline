"""M7 OOV — OOV Pronounceability (Sec 4, Group C).

Of the tokens LVR marked invalid, what share are genuine GIBBERISH (illegal
char structure, e.g. 'rnixqk') versus PLAUSIBLE real-but-rare / brand words
(e.g. 'Nemluvio')? This separates OCR garble from legitimate domain vocabulary.

raw = gibberish_oov / total_oov. Reuses ``invalid_tokens`` from the lexical
module so the two metrics agree on the OOV set.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, is_pronounceable, lerp_score
from ocr_qa.metrics.lexical import invalid_tokens
from ocr_qa.models import PageOCR


class OOVMetric(BaseMetric):
    key = "OOV"
    name = "OOV Pronounceability"
    what_it_measures = (
        "Among out-of-vocabulary words, the fraction that are unpronounceable "
        "gibberish (OCR garble) rather than plausible rare/brand words. Keeps "
        "legitimate proper nouns from being treated as errors."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        lang = ctx.languages.get(page.page_no) or "en"
        oov = invalid_tokens(page.prose_text(), lang)
        total_oov = len(oov)
        if total_oov == 0:
            return self.result(
                raw_value=0.0,
                score=100.0,
                status="good",
                threshold={"gibberish_rate_good_at": 0.0, "bad_at": 0.5},
                how_computed="gibberish_oov / total_oov (no OOV words here).",
                example="Every word on this page is in-vocabulary, so there is "
                "nothing to classify as gibberish → full score.",
                evidence=["no out-of-vocabulary words"],
                justification="No out-of-vocabulary words to classify.",
                reliability="low" if self.low_reliability(page, ctx) else "high",
            )

        gibberish = [t for t in oov if not is_pronounceable(t)]
        plausible = [t for t in oov if is_pronounceable(t)]
        raw = len(gibberish) / total_oov
        score = lerp_score(raw, 0.0, 0.5)

        gc = Counter(gibberish)
        pc = Counter(plausible)
        evidence = []
        if gc:
            evidence.append(
                "gibberish: " + ", ".join(f"'{t}'×{c}" for t, c in gc.most_common(5))
            )
        if pc:
            evidence.append(
                "plausible: " + ", ".join(f"'{t}'×{c}" for t, c in pc.most_common(5))
            )

        g_ex = ", ".join(f"'{t}'" for t, _ in gc.most_common(2))
        p_ex = ", ".join(f"'{t}'" for t, _ in pc.most_common(2))
        example = (
            f"{total_oov} words are out-of-vocabulary. {len(gibberish)} are "
            f"unpronounceable gibberish"
            + (f" (e.g. {g_ex})" if g_ex else "")
            + f" and {len(plausible)} look like plausible names"
            + (f" (e.g. {p_ex})" if p_ex else "")
            + f" → {raw * 100:.0f}% gibberish."
        )

        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"gibberish_rate_good_at": 0.0, "bad_at": 0.5},
            how_computed="unpronounceable_oov / total_oov (vowel ratio, "
            "consonant-run, illegal-bigram gate).",
            example=example,
            evidence=evidence,
            justification=(
                f"{len(gibberish)}/{total_oov} OOV words are unpronounceable "
                f"gibberish ({raw * 100:.0f}%); {len(plausible)} look like "
                f"plausible rare/brand words."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
        )
