"""M13 PSW — Punctuation & Sentence Well-formedness (Sec 4, Group E).

Unmatched brackets/quotes/parens, sentence-ending-punctuation density, abnormal
spacing, and sentence-length plausibility — on PROSE only (table/caption blocks
are skipped).
"""

from __future__ import annotations

import re
from typing import Optional

from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    lerp_score,
    sentences,
    word_tokens,
)
from ocr_qa.models import PageOCR

_PAIRS = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = {v: k for k, v in _PAIRS.items()}
_MULTISPACE_RE = re.compile(r" {3,}")
_SPACE_BEFORE_PUNC_RE = re.compile(r"\s+[,.;:!?]")


def _bracket_defects(text: str) -> tuple[int, list[str]]:
    stack: list[str] = []
    unmatched = 0
    ev: list[str] = []
    for ch in text:
        if ch in _PAIRS:
            stack.append(ch)
        elif ch in _CLOSERS:
            if stack and stack[-1] == _CLOSERS[ch]:
                stack.pop()
            else:
                unmatched += 1
    unmatched += len(stack)
    if stack:
        ev.append(f"{len(stack)} unclosed {' '.join(stack[:5])}")
    # straight double quotes parity
    dq = text.count('"')
    if dq % 2 == 1:
        unmatched += 1
        ev.append('odd number of " quotes')
    return unmatched, ev


class PSWMetric(BaseMetric):
    key = "PSW"
    name = "Punctuation & Sentence Well-formedness"
    what_it_measures = (
        "Punctuation sanity of prose: matched brackets/quotes, reasonable "
        "sentence-ending punctuation, normal spacing, and plausible sentence "
        "lengths."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        text = page.prose_text()
        if not text.strip():
            return self.not_applicable("no prose text on this page")

        n_words = max(1, len(word_tokens(text)))
        unmatched, ev = _bracket_defects(text)

        sents = sentences(text)
        # sentence-ending punctuation density: sentences ending with . ! ?
        ended = sum(1 for s in sents if s and s[-1] in ".!?")
        end_ratio = ended / len(sents) if sents else 0.0

        # implausible sentence lengths (very long run with no terminator)
        long_sents = [s for s in sents if len(word_tokens(s)) > 60]

        multispace = len(_MULTISPACE_RE.findall(text))
        space_punc = len(_SPACE_BEFORE_PUNC_RE.findall(text))

        # normalise: defects per 100 words
        defects = unmatched + len(long_sents) + 0.5 * multispace + 0.25 * space_punc
        raw = defects / (n_words / 100.0)
        # also fold in low end-punctuation density as a penalty
        if end_ratio < 0.5 and len(sents) >= 2:
            raw += 1.0

        score = lerp_score(raw, 0.5, 6.0)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        if unmatched:
            ev.append(f"{unmatched} unmatched bracket/quote(s)")
        if long_sents:
            ev.append(f"{len(long_sents)} implausibly long sentence(s)")
        if multispace:
            ev.append(f"{multispace} abnormal multi-space gap(s)")
        if end_ratio < 0.5 and len(sents) >= 2:
            ev.append(f"only {end_ratio * 100:.0f}% of sentences end with . ! ?")
        if not ev:
            ev = ["punctuation well-formed"]

        if unmatched or long_sents or multispace:
            example = (
                f"Example: {ev[0]}. Overall ~{raw:.1f} punctuation defects per "
                f"100 words (good ≤0.5)."
            )
        else:
            example = (
                f"Brackets and quotes are balanced, {end_ratio * 100:.0f}% of "
                f"sentences end in . ! ?, and spacing is normal → ~{raw:.1f} "
                f"defects/100 words."
            )

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"defects_per_100w_good": 0.5, "bad": 6.0},
            how_computed="(unmatched brackets/quotes + over-long sentences + "
            "spacing anomalies) per 100 words, plus end-punctuation density.",
            example=example,
            evidence=ev,
            justification=(
                f"{unmatched} unmatched bracket/quote(s) and "
                f"{raw:.1f} punctuation defects per 100 words "
                f"(good ≤0.5, bad ≥6)."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
        )
