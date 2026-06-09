"""M12 WSA — Word-Shape / Tokenization Anomaly (Sec 4, Group E).

Surface-form anomalies that betray bad OCR segmentation:
  * merged words      : tokens longer than L_max (default 25 chars)
  * over-segmentation : lone single-char alpha tokens (excl. a, I, à, …)
  * mid-word breaks   : tokens ending in a hyphen (line-break hyphenation)
  * alnum confusion   : letter/digit mixes (l<->1, O<->0) within a word
"""

from __future__ import annotations

import re
from typing import Optional

from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    find_block_for,
    is_numbery,
    is_url_or_email,
    lerp_score,
    loc,
    raw_tokens,
)
from ocr_qa.models import PageOCR

_ALLOWED_SINGLE = set("aIiAàáâäéèêëÀ")  # legitimate one-letter tokens
_ALNUM_CONF_RE = re.compile(r"^(?=.*[A-Za-zÀ-ÿ])(?=.*[0-9lO])[A-Za-z0-9lO]*[0-9][A-Za-z]")
_DIGIT_IN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][0-9]|[0-9][A-Za-zÀ-ÿ]")


def _is_alnum_confusion(tok: str) -> bool:
    if is_numbery(tok) or is_url_or_email(tok):
        return False
    has_alpha = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if not (has_alpha and has_digit):
        return False
    # genuine alnum codes (e.g. "B12", "COVID19") are usually short/uppercase;
    # confusion = lower-case letters interleaved with 0/1 repeatedly.
    return bool(_DIGIT_IN_WORD_RE.search(tok)) and len(tok) >= 4


class WSAMetric(BaseMetric):
    key = "WSA"
    name = "Word-Shape / Tokenization Anomaly"
    what_it_measures = (
        "Surface tokenization defects: merged words (too long), over-segmented "
        "single characters, mid-word hyphen breaks, and letter/digit confusion "
        "(l↔1, O↔0)."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        toks = raw_tokens(page.prose_text())
        total = len(toks)
        if total == 0:
            return self.not_applicable("no prose tokens on this page")

        lmax = ctx.config.l_max_token
        merged, overseg, hyphen, alnum = [], [], [], []

        for t in toks:
            core = t.strip(".,;:!?()[]\"'“”·")
            if not core:
                continue
            if len(core) > lmax and any(c.isalpha() for c in core):
                merged.append(core)
            if len(core) == 1 and core.isalpha() and core not in _ALLOWED_SINGLE:
                overseg.append(core)
            if core.endswith("-") and len(core) > 1:
                hyphen.append(core)
            if _is_alnum_confusion(core):
                alnum.append(core)

        bad = len(merged) + len(overseg) + len(hyphen) + len(alnum)
        raw = bad / total
        score = lerp_score(raw, ctx.config.wsa_good, ctx.config.wsa_bad)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        evidence: list[str] = []
        if merged:
            evidence.append(f"merged (>{lmax}c): '{merged[0][:40]}…'")
        if overseg:
            evidence.append(f"over-segmented singles: {', '.join(overseg[:6])}")
        if hyphen:
            evidence.append(f"mid-word hyphen break: '{hyphen[0]}'")
        if alnum:
            evidence.append(f"alnum confusion (l↔1,O↔0): {', '.join(alnum[:5])}")
        if not evidence:
            evidence = [f"no word-shape anomalies in {total} tokens"]

        if bad:
            parts = []
            if merged:
                parts.append(f"a {len(merged[0])}-char merged word '{merged[0][:24]}…'")
            if alnum:
                parts.append(f"letter/digit confusion '{alnum[0]}' (l↔1, O↔0)")
            if overseg:
                parts.append(f"stray single letters ({', '.join(overseg[:3])})")
            if hyphen:
                parts.append(f"a line-break hyphen '{hyphen[0]}'")
            example = (
                "Found " + "; ".join(parts) + f". That is {bad}/{total} tokens = "
                f"{raw * 100:.1f}% (good ≤{ctx.config.wsa_good * 100:.0f}%)."
            )
        else:
            example = (
                f"All {total} tokens are normally shaped — none over {lmax} chars, "
                f"no l↔1/O↔0 confusion, no broken hyphenation → {raw * 100:.1f}%."
            )

        locations = []
        for t in merged[:3]:
            locations.append(loc("merged_word", "definite",
                                 block=find_block_for(page, t), snippet=t))
        for t in alnum[:3]:
            locations.append(loc("alnum_confusion", "definite",
                                 block=find_block_for(page, t), snippet=t))
        for t in hyphen[:2]:
            locations.append(loc("hyphen_break", "statistical",
                                 block=find_block_for(page, t), snippet=t))

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"good": ctx.config.wsa_good, "bad": ctx.config.wsa_bad,
                       "l_max": lmax},
            locations=locations,
            how_computed="(merged + over-segmented + hyphen-break + alnum-confusion "
            "tokens) / total tokens.",
            example=example,
            evidence=evidence,
            justification=(
                f"Word-shape anomaly rate {raw * 100:.1f}% "
                f"(good ≤{ctx.config.wsa_good * 100:.0f}%, "
                f"bad ≥{ctx.config.wsa_bad * 100:.0f}%); "
                f"{bad}/{total} tokens affected."
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
        )
