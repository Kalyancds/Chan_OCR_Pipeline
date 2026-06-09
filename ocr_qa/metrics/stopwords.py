"""M8 SFC — Stopword Frequency Conformance (Sec 4, Group C).

Natural prose has a characteristic share of function words (the, of, and / der,
und, die …). OCR garble and looping/fragmented text deviate from this band. We
compare the observed stopword ratio to a per-language expected band.
"""

from __future__ import annotations

from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, word_tokens
from ocr_qa.models import PageOCR

# Small, deterministic stopword sets + expected ratio per language.
_STOP = {
    "en": {
        "the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "as",
        "was", "with", "be", "by", "on", "not", "this", "are", "or", "an",
        "at", "from", "but", "had", "have", "has", "were", "their", "which",
        "been", "its", "they", "we", "he", "she", "all", "can", "will",
    },
    "de": {
        "der", "die", "das", "und", "in", "den", "von", "zu", "mit", "des",
        "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine", "als", "auch",
        "es", "an", "werden", "aus", "er", "hat", "dass", "sie", "nach", "wird",
        "bei", "einer", "um", "am", "sind", "noch", "wie", "einem", "über",
    },
}
_EXPECTED = {"en": 0.45, "de": 0.42}
_BAND = 0.18  # acceptable absolute deviation before score drops to 0


class SFCMetric(BaseMetric):
    key = "SFC"
    name = "Stopword Frequency Conformance"
    what_it_measures = (
        "How closely the share of function/stopwords matches what natural prose "
        "in the detected language exhibits. Garbled or looping text deviates."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        lang = (ctx.languages.get(page.page_no) or "en")[:2]
        if lang not in _STOP:
            lang = "en"
        toks = [t.lower() for t in word_tokens(page.prose_text())]
        total = len(toks)
        if total == 0:
            return self.not_applicable("no prose tokens on this page")

        stop = _STOP[lang]
        n_stop = sum(1 for t in toks if t in stop)
        observed = n_stop / total
        expected = _EXPECTED[lang]
        raw = abs(observed - expected)
        score = lerp_score(raw, 0.0, _BAND)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        base = (
            f"{n_stop} of {total} words here are function words "
            f"('the/of/and'-type) = {observed * 100:.0f}%; natural {lang} prose "
            f"is ~{expected * 100:.0f}% (a {raw * 100:.0f}% gap)"
        )
        if score >= 70:
            example = base + " — within the normal band."
        elif self.low_reliability(page, ctx):
            example = (
                base + f". But this is a very short/heading-style page "
                f"({total} prose words), where few function words is expected — "
                f"treat as low-confidence."
            )
        else:
            example = base + " — unusually low for real prose, hinting at garbled or looping text."

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"expected": expected, "band": _BAND, "lang": lang},
            how_computed="|observed_stopword_ratio - expected_ratio| for the "
            "detected language.",
            example=example,
            evidence=[
                f"observed {observed * 100:.1f}% vs expected {expected * 100:.0f}% "
                f"({lang}); deviation {raw * 100:.1f}%"
            ],
            justification=(
                f"Stopword share {observed * 100:.1f}% deviates {raw * 100:.1f}% "
                f"from the expected {expected * 100:.0f}% for {lang} "
                f"(band ±{_BAND * 100:.0f}%)."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
        )
