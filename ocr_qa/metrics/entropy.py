"""M9 CED — Character Entropy & Mojibake Divergence (Sec 4, Group D).

Shannon entropy of the prose plus KL divergence of its character distribution
against a language baseline, with EXPLICIT mojibake detection (U+FFFD, Ã/Â/â€
sequences, control chars). Accented Latin letters (à é ü ç) are VALID and are
NOT mojibake. A mojibake count at/above the configured bound is a HARD
condition (Sec 5).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Optional

from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    find_block_for,
    lang_baseline,
    lerp_score,
    loc,
    shannon_entropy,
)
from ocr_qa.models import PageOCR

# Classic mojibake signatures (UTF-8 mis-decoded as Latin-1/cp1252).
_MOJIBAKE_RE = re.compile(
    r"�"  # replacement char
    r"|Ã[\x80-\xbf€‚ƒ„…†‡ˆ‰Š‹ŒŽ]"  # Ã + high byte
    r"|Â[\xa0-\xbf]"  # Â + nbsp-range
    r"|â€[\x9c\x9d\x99\x9c\x9d™œ“”˜]"  # smart-quote mojibake
    r"|â€"  # generic â€ sequence
)


def _mojibake_hits(text: str) -> list[str]:
    hits = [m.group(0) for m in _MOJIBAKE_RE.finditer(text)]
    # control chars (excluding tab/newline)
    for ch in text:
        if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\t\n\r":
            hits.append(f"U+{ord(ch):04X}")
    return hits


def _kl_divergence(text: str, baseline: dict[str, float]) -> tuple[float, list[str]]:
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return 0.0, []
    n = len(letters)
    counts = Counter(letters)
    kl = 0.0
    over = []
    eps = 1e-6
    for ch, p_count in counts.items():
        p = p_count / n
        q = baseline.get(ch, eps)
        if q <= 0:
            q = eps
        contrib = p * math.log2(p / q)
        kl += contrib
        if contrib > 0.15 and ch not in baseline:
            over.append(ch)
    return max(0.0, kl), over


class CEDMetric(BaseMetric):
    key = "CED"
    name = "Character Entropy & Mojibake Divergence"
    what_it_measures = (
        "Whether the page's character distribution matches the language baseline "
        "(KL divergence) and whether mojibake/encoding corruption is present. "
        "Accented Latin letters are treated as valid."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        lang = ctx.languages.get(page.page_no) or "en"
        text = page.prose_text()
        if not text.strip():
            return self.not_applicable("no prose text on this page")

        baseline = lang_baseline(lang)
        kl, over = _kl_divergence(text, baseline)
        entropy = shannon_entropy(text)
        mojibake = _mojibake_hits(text)
        n_moji = len(mojibake)

        good, bad = ctx.config.ced_kl_good, ctx.config.ced_kl_bad
        score = lerp_score(kl, good, bad)

        hard = n_moji >= ctx.config.ced_mojibake_bad
        if n_moji > 0:
            score = min(score, max(0.0, 60.0 - 15.0 * n_moji))
        if hard:
            score = min(score, 15.0)

        evidence: list[str] = []
        if mojibake:
            uniq = list(dict.fromkeys(mojibake))[:6]
            evidence.append("mojibake: " + " ".join(repr(x) for x in uniq))
        if over:
            evidence.append("over-represented odd chars: " + " ".join(over[:8]))
        evidence.append(f"entropy={entropy:.2f} bits/char, KL={kl:.3f}")

        if n_moji:
            moji_ex = " ".join(repr(x) for x in list(dict.fromkeys(mojibake))[:3])
            example = (
                f"Found {n_moji} mojibake/encoding artefact(s) — e.g. {moji_ex} "
                f"(garbled UTF-8). "
                + (
                    f"{n_moji} ≥ the {ctx.config.ced_mojibake_bad}-hit limit, so "
                    f"this is flagged as encoding corruption."
                    if hard
                    else "Accented letters like é/ü would NOT count — these are real garble."
                )
            )
        else:
            example = (
                f"Character mix matches {lang} (KL {kl:.2f} ≤ {good}); no '�' or "
                f"'Ã©'-style mojibake. Accented letters are treated as valid."
            )

        locations = []
        for hit in list(dict.fromkeys(mojibake))[:8]:
            b = find_block_for(page, hit)
            locations.append(loc("mojibake", "definite", block=b, snippet=hit))

        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        return self.result(
            raw_value=kl,
            score=score,
            status=status,
            threshold={
                "kl_good": good,
                "kl_bad": bad,
                "mojibake_hard_at": ctx.config.ced_mojibake_bad,
            },
            locations=locations,
            how_computed="KL divergence of char frequencies vs language baseline "
            "+ explicit mojibake count.",
            example=example,
            evidence=evidence,
            justification=(
                f"KL divergence {kl:.3f} (good ≤{good}, bad ≥{bad}); "
                f"{n_moji} mojibake/encoding hit(s)"
                + (
                    f" ≥ hard threshold {ctx.config.ced_mojibake_bad} — encoding "
                    "corruption."
                    if hard
                    else "."
                )
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
            hard_fail=hard,
        )
