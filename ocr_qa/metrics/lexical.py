"""M6 LVR — Lexical Validity Ratio (Sec 4, Group C).

valid_alpha_tokens / total_alpha_tokens on PROSE text only. A token is valid if
it is a real word (wordfreq Zipf > 0 in the page language or English), known to
the spellchecker, on the pharma/brand whitelist, or a pronounceable proper noun.
Numbers / URLs / emails are excluded from the denominator.

Normalisation (Sec 4): >=0.95 -> 100, 0.80 -> 60, <=0.60 -> 0. LVR < 0.80 is a
HARD condition (Sec 5).

This module also exposes ``invalid_tokens`` which the OOV metric reuses, so the
two metrics agree on what counts as garbled.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ocr_qa.config import DOMAIN_WHITELIST
from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    find_block_for,
    is_pronounceable,
    is_numbery,
    is_url_or_email,
    loc,
    word_tokens,
)
from ocr_qa.models import PageOCR

# spellchecker is lazy + cached per language
_SPELL: dict[str, object] = {}


def _spell(lang: str):
    lang = (lang or "en")[:2]
    if lang in _SPELL:
        return _SPELL[lang]
    try:
        from spellchecker import SpellChecker

        sc = SpellChecker(language=lang if lang in {"en", "es", "fr", "de", "pt", "it"} else "en")
    except Exception:
        sc = None
    _SPELL[lang] = sc
    return sc


def _zipf(token: str, lang: str) -> float:
    try:
        from wordfreq import zipf_frequency

        z = zipf_frequency(token, lang)
        if z > 0:
            return z
        if lang != "en":
            return zipf_frequency(token, "en")
        return z
    except Exception:
        return 0.0


def token_is_valid(token: str, lang: str) -> bool:
    low = token.lower().strip("'’-")
    if not low:
        return True
    if low in DOMAIN_WHITELIST:
        return True
    if _zipf(low, lang) > 0:
        return True
    sc = _spell(lang)
    if sc is not None:
        try:
            if low in sc:  # known word
                return True
        except Exception:
            pass
    # proper-noun-like: capitalised AND pronounceable => accept (brand names)
    if token[:1].isupper() and is_pronounceable(token):
        return True
    # all-caps short acronym
    if token.isupper() and len(token) <= 5:
        return True
    return False


def countable_tokens(text: str) -> list[str]:
    """Alpha tokens that count toward LVR (numbers/urls/emails excluded)."""
    out = []
    for t in word_tokens(text):
        if is_numbery(t) or is_url_or_email(t):
            continue
        out.append(t)
    return out


def invalid_tokens(text: str, lang: str) -> list[str]:
    return [t for t in countable_tokens(text) if not token_is_valid(t, lang)]


def _norm_lvr(ratio: float) -> float:
    # piecewise: 0.60->0, 0.80->60, 0.95->100
    if ratio >= 0.95:
        return 100.0
    if ratio <= 0.60:
        return 0.0
    if ratio >= 0.80:
        return 60.0 + (ratio - 0.80) / (0.95 - 0.80) * 40.0
    return (ratio - 0.60) / (0.80 - 0.60) * 60.0


class LVRMetric(BaseMetric):
    key = "LVR"
    name = "Lexical Validity Ratio"
    what_it_measures = (
        "Share of alphabetic prose words that are real, known words (dictionary "
        "/ word-frequency / brand whitelist). Numbers, URLs and emails are "
        "excluded; pharma proper nouns are not penalised."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        lang = ctx.languages.get(page.page_no) or "en"
        prose = page.prose_text()
        toks = countable_tokens(prose)
        total = len(toks)
        if total == 0:
            return self.not_applicable("no prose tokens on this page")

        bad = [t for t in toks if not token_is_valid(t, lang)]
        ratio = (total - len(bad)) / total
        score = _norm_lvr(ratio)
        hard = ratio < ctx.config.lexical_warn  # < 0.80

        counts = Counter(bad)
        evidence = [f"'{tok}' ×{c}" for tok, c in counts.most_common(8)] or [
            "no invalid tokens detected"
        ]

        sample_bad = ", ".join(f"'{t}'" for t, _ in counts.most_common(3))
        if bad:
            example = (
                f"This page has {total} alphabetic words; {len(bad)} are not "
                f"real/known words — e.g. {sample_bad}. So valid = "
                f"{total - len(bad)}/{total} = {ratio * 100:.0f}% "
                f"(target {ctx.config.lexical_good * 100:.0f}%)."
            )
        else:
            example = (
                f"All {total} words are recognised dictionary, multilingual or "
                f"whitelisted brand words (e.g. 'Nemluvio', 'Galderma') → "
                f"{total}/{total} = 100% valid."
            )

        locations = []
        for tok, _c in counts.most_common(8):
            gib = not is_pronounceable(tok)
            b = find_block_for(page, tok)
            locations.append(loc(
                "gibberish_word" if gib else "rare_word",
                "definite" if gib else "statistical",
                block=b, snippet=tok,
            ))

        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        return self.result(
            raw_value=ratio,
            score=score,
            status=status,
            threshold={
                "good": ctx.config.lexical_good,
                "warn_hard_below": ctx.config.lexical_warn,
                "zero": ctx.config.lexical_zero,
            },
            locations=locations,
            how_computed="valid_words / countable_words (Zipf>0 OR known OR "
            "whitelisted OR pronounceable proper noun).",
            example=example,
            evidence=evidence,
            justification=(
                f"Lexical validity {ratio * 100:.0f}% "
                f"(threshold {ctx.config.lexical_good * 100:.0f}%; "
                f"hard-fail below {ctx.config.lexical_warn * 100:.0f}%); "
                f"{len(bad)}/{total} prose words are not recognised."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
            hard_fail=hard,
        )
