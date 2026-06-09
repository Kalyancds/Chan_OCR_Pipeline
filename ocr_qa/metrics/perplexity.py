"""M10 PPL — N-gram Perplexity / Fluency (Sec 4, Group D).

Pluggable. The DEFAULT 'ngram' backend trains a character-trigram model on the
document's OWN prose (so multilingual / domain text is the baseline, not penalised)
and scores each page by mean surprisal. Calibration bounds are derived from the
distribution of the document's per-page surprisals (good_bound -> 100,
bad_bound -> 0). Garbled pages contain many trigrams that are rare relative to
the rest of the document and therefore score high surprisal / low fluency.

Optional 'kenlm' / 'gpt2' backends are stubs that fall back to 'ngram' with a
clear note (no model download in the default path).

The registry builds the model + per-page raw surprisal in a pre-pass and stores
them on DocContext; this metric maps the page's raw value to a score and lists
the highest-perplexity sentences as evidence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, lerp_score, sentences
from ocr_qa.models import PageOCR


class NgramModel:
    """Tiny char-trigram language model with add-k smoothing. Deterministic."""

    def __init__(self, k: float = 0.1, n: int = 3):
        self.k = k
        self.n = n
        self.context_counts: dict[str, int] = defaultdict(int)
        self.gram_counts: dict[str, int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.trained = False

    @staticmethod
    def _prep(text: str) -> str:
        return " " + " ".join(text.lower().split()) + " "

    def train(self, texts: list[str]) -> None:
        for text in texts:
            s = self._prep(text)
            for ch in s:
                self.vocab.add(ch)
            for i in range(len(s) - self.n + 1):
                gram = s[i : i + self.n]
                ctx = gram[:-1]
                self.gram_counts[gram] += 1
                self.context_counts[ctx] += 1
        self.trained = bool(self.gram_counts)

    def _logprob_char(self, gram: str) -> float:
        ctx = gram[:-1]
        v = max(len(self.vocab), 1)
        num = self.gram_counts.get(gram, 0) + self.k
        den = self.context_counts.get(ctx, 0) + self.k * v
        return math.log2(num / den)

    def surprisal(self, text: str) -> float:
        """Mean negative log2 prob per char (bits). Higher = less fluent."""
        s = self._prep(text)
        if len(s) < self.n:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(s) - self.n + 1):
            total += -self._logprob_char(s[i : i + self.n])
            count += 1
        return total / count if count else 0.0


def build_ppl_context(ctx: DocContext, page_prose: dict[int, str]) -> None:
    """Pre-pass: train the model on all prose and compute per-page surprisal +
    calibration bounds. Stores everything on ``ctx``."""
    backend = ctx.config.perplexity_backend
    texts = [t for t in page_prose.values() if t and t.strip()]
    model = NgramModel()
    if texts:
        model.train(texts)
    ctx.ppl_model = model

    raws: dict[int, float] = {}
    for pno, prose in page_prose.items():
        raws[pno] = model.surprisal(prose) if (prose and prose.strip()) else 0.0
    ctx.ppl_page_raw = raws

    vals = sorted(v for v in raws.values() if v > 0)
    if len(vals) >= 4:
        # robust bounds from the doc's own distribution
        lo = vals[max(0, int(0.25 * len(vals)) - 1)]
        hi = vals[min(len(vals) - 1, int(0.90 * len(vals)))]
        ctx.ppl_good_bound = lo
        ctx.ppl_bad_bound = max(hi, lo * 1.5 + 0.5)
    elif vals:
        med = vals[len(vals) // 2]
        ctx.ppl_good_bound = med
        ctx.ppl_bad_bound = med * 1.8 + 0.8
    else:
        ctx.ppl_good_bound = 0.0
        ctx.ppl_bad_bound = 1.0


class PPLMetric(BaseMetric):
    key = "PPL"
    name = "N-gram Perplexity / Fluency"
    what_it_measures = (
        "How fluent the prose is relative to the rest of THIS document, via a "
        "character n-gram model trained on the document itself (so domain / "
        "multilingual text is the baseline, not penalised)."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        prose = page.prose_text()
        if not prose.strip():
            return self.not_applicable("no prose text on this page")

        model: Optional[NgramModel] = ctx.ppl_model  # type: ignore[assignment]
        raw = ctx.ppl_page_raw.get(page.page_no)
        if raw is None and model is not None:
            raw = model.surprisal(prose)
        raw = raw or 0.0

        good = ctx.ppl_good_bound if ctx.ppl_good_bound is not None else 0.0
        bad = ctx.ppl_bad_bound if ctx.ppl_bad_bound is not None else max(good + 1.0, 1.0)
        score = lerp_score(raw, good, bad)

        # evidence: 3 highest-surprisal sentences
        evidence: list[str] = []
        if model is not None:
            scored = sorted(
                ((model.surprisal(s), s) for s in sentences(prose)),
                key=lambda x: -x[0],
            )
            for sv, s in scored[:3]:
                evidence.append(f"[{sv:.2f} bits] {s[:80]}")
        if not evidence:
            evidence = [f"page surprisal {raw:.2f} bits/char"]

        backend = ctx.config.perplexity_backend
        note = ""
        if backend in {"kenlm", "gpt2"}:
            note = f" (requested '{backend}' backend unavailable; used ngram fallback)"

        worst_sentence = ""
        if model is not None:
            ss = sorted(sentences(prose), key=lambda s: -model.surprisal(s))
            if ss:
                worst_sentence = ss[0][:70]
        example = (
            f"This page reads at {raw:.2f} bits/char of surprise; the document's "
            f"clean pages sit around {good:.2f}. "
            + (
                f"Its least-fluent line — '{worst_sentence}…' — drives the score "
                f"down."
                if (worst_sentence and score < 70)
                else "That is well within the fluent range for this document."
            )
        )

        low_rel = self.low_reliability(page, ctx) or backend == "ngram"
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"good_bound": round(good, 3), "bad_bound": round(bad, 3),
                       "backend": backend},
            how_computed="mean char-trigram surprisal (bits/char) calibrated on "
            "the document's own per-page distribution." + note,
            example=example,
            evidence=evidence,
            justification=(
                f"Page surprisal {raw:.2f} bits/char vs document calibration "
                f"(fluent ≤{good:.2f}, degraded ≥{bad:.2f})."
            ),
            reliability="low" if low_rel else "high",
        )
