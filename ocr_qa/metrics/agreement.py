"""M14 XSA — Repetition & Cross-Source/Native Agreement (Sec 4, Group F).

The strongest available integrity check given there is NO confidence signal.
Composite of up to three sub-parts; the score is the worst of the AVAILABLE
sub-parts (unavailable ones are dropped):

  (a) Repetition / hallucination: trigram repetition rate, longest repeated
      span + count, gzip compressibility. HARD FLAG if any 5-gram repeats >5x.
  (b) JSON <-> MD agreement: similarity between the page text from output.json
      and the page's segment in output.md (pipeline-consistency check).
  (c) Native-text agreement (only when the source PDF is digital-born): CER+WER
      (jiwer) of OCR prose vs embedded native text as pseudo-ground-truth.
      CER > 0.25 is a HARD condition.
"""

from __future__ import annotations

import difflib
import gzip
import re
from collections import Counter
from typing import Optional

from ocr_qa.metrics.base import (
    BaseMetric,
    DocContext,
    find_block_for,
    lerp_score,
    loc,
    word_tokens,
)
from ocr_qa.models import PageOCR

_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip().lower())


# --------------------------------------------------------------------------- #
# (a) repetition
# --------------------------------------------------------------------------- #
def _repetition(text: str, ngram: int, hard_max: int) -> dict:
    toks = word_tokens(text.lower())
    out = {
        "trigram_rep": 0.0,
        "max_ngram_count": 0,
        "longest_span": "",
        "gzip_ratio": 1.0,
        "hard": False,
    }
    if len(toks) >= 3:
        tris = [" ".join(toks[i : i + 3]) for i in range(len(toks) - 2)]
        if tris:
            uniq = len(set(tris))
            out["trigram_rep"] = 1.0 - uniq / len(tris)
    if len(toks) >= ngram:
        grams = [" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]
        c = Counter(grams)
        gram, cnt = c.most_common(1)[0]
        out["max_ngram_count"] = cnt
        out["longest_span"] = gram
        out["hard"] = cnt > hard_max
    if text.strip():
        raw = text.encode("utf-8")
        comp = gzip.compress(raw)
        out["gzip_ratio"] = len(raw) / max(1, len(comp))
    return out


# --------------------------------------------------------------------------- #
# (b) json <-> md
# --------------------------------------------------------------------------- #
def _json_html_similarity(json_text: str, html_text: Optional[str]) -> Optional[float]:
    """Similarity between the JSON page text and the output.html page text
    (both derive from the same blocks, so divergence flags a pipeline bug)."""
    if not html_text or not html_text.strip():
        return None
    from ocr_qa.ocr.chandra_parser import strip_html

    a, b = _norm(json_text), _norm(strip_html(html_text))
    if not a or not b:
        return None
    return difflib.SequenceMatcher(None, a, b).ratio()


def _text_recall(page_text: str, other_text: Optional[str]) -> Optional[float]:
    """Fraction of the page's distinct words that also appear in ``other_text``.

    Used for both JSON↔MD and JSON↔chunk: it measures CONTENT overlap, so it is
    robust to markdown/formatting differences (links, '##', '|' tables) and only
    drops when content is genuinely missing. Returns None when there is too
    little prose to judge reliably.
    """
    if not other_text or not other_text.strip():
        return None
    page_toks = {t.lower() for t in word_tokens(page_text)}
    if len(page_toks) < 10:
        return None
    other_toks = {t.lower() for t in word_tokens(other_text)}
    if not other_toks:
        return None
    return len(page_toks & other_toks) / len(page_toks)


# --------------------------------------------------------------------------- #
# (c) native CER/WER
# --------------------------------------------------------------------------- #
def _native_agreement(ocr_text: str, native: Optional[str]) -> Optional[dict]:
    if not native or not native.strip() or not ocr_text.strip():
        return None
    try:
        import jiwer

        ref = _norm(native)
        hyp = _norm(ocr_text)
        cer = jiwer.cer(ref, hyp)
        wer = jiwer.wer(ref, hyp)
        return {"cer": float(cer), "wer": float(wer)}
    except Exception:
        return None


class XSAMetric(BaseMetric):
    key = "XSA"
    name = "Repetition & Cross-Source Agreement"
    what_it_measures = (
        "Integrity via three lenses: (a) hallucination/looping repetition, "
        "(b) agreement between the JSON and Markdown outputs, and (c) when a "
        "digital-born PDF is available, character/word error vs the embedded "
        "native text."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        cfg = ctx.config
        text = page.prose_text() or page.text
        evidence: list[str] = []
        sub: dict[str, float] = {}          # submetric breakdown (0-100)
        warn = cfg.xsa_md_sim_warn
        native_ok = ctx.native_reliable.get(page.page_no, False)

        # (a) repetition / hallucination loop -----------------------------
        rep = _repetition(text, cfg.xsa_repeat_ngram, cfg.xsa_repeat_max)
        rep_score = 100.0
        rep_score = min(rep_score, lerp_score(rep["trigram_rep"], 0.1, 0.6))
        rep_score = min(rep_score, lerp_score(rep["gzip_ratio"], 3.0, 8.0))
        if rep["max_ngram_count"] > 1:
            rep_score = min(
                rep_score, lerp_score(rep["max_ngram_count"], 1, cfg.xsa_repeat_max + 3)
            )
        if rep["hard"]:
            rep_score = 0.0
            evidence.append(
                f"repeated span '{rep['longest_span']}' ×{rep['max_ngram_count']} "
                f"(>{cfg.xsa_repeat_max}) — definite VLM loop"
            )
        elif rep["max_ngram_count"] > 1:
            evidence.append(f"top {cfg.xsa_repeat_ngram}-gram "
                            f"'{rep['longest_span']}' ×{rep['max_ngram_count']}")
        sub["repetition"] = round(rep_score, 1)

        # (b/d/e) artefact-agreement sub-parts (JSON vs MD / chunk / HTML) -
        artefact_fail = 0           # independent artefact disagreements
        sim = _text_recall(page.text, page.md_text)
        if sim is not None:
            sub["json_md"] = lerp_score(1.0 - sim, 1.0 - 0.85, 1.0 - 0.40)
            if sim < warn:
                artefact_fail += 1
                evidence.append(f"JSON↔MD word coverage {sim * 100:.0f}% (< "
                                f"{warn * 100:.0f}%) — content missing from MD")
            else:
                evidence.append(f"JSON↔MD word coverage {sim * 100:.0f}%")
        recall = _text_recall(text, page.chunk_text)
        if recall is not None:
            sub["json_chunk"] = lerp_score(1.0 - recall, 1.0 - 0.85, 1.0 - 0.40)
            if recall < warn:
                artefact_fail += 1
                evidence.append(f"JSON↔chunk coverage {recall * 100:.0f}% (< "
                                f"{warn * 100:.0f}%) — disagrees with output_chunks")
            else:
                evidence.append(f"JSON↔chunk coverage {recall * 100:.0f}%")
        html_sim = _json_html_similarity(page.text, page.html_text)
        if html_sim is not None:
            sub["json_html"] = lerp_score(1.0 - html_sim, 1.0 - 0.9, 1.0 - 0.3)
            if html_sim < warn:
                artefact_fail += 1
                evidence.append(f"JSON↔HTML similarity {html_sim * 100:.0f}% (< "
                                f"{warn * 100:.0f}%) — JSON and HTML disagree")
            else:
                evidence.append(f"JSON↔HTML similarity {html_sim * 100:.0f}%")

        # (c) native CER/WER — ONLY when the native text layer is reliable --
        nat = None
        cer_hard = False
        if native_text and native_text.strip():
            if native_ok:
                nat = _native_agreement(text, native_text)
                if nat is not None:
                    sub["native_cer"] = lerp_score(nat["cer"], 0.02, cfg.xsa_cer_bad * 2)
                    cer_hard = nat["cer"] > cfg.xsa_cer_bad
                    evidence.append(
                        f"CER {nat['cer'] * 100:.1f}%, WER {nat['wer'] * 100:.1f}% "
                        f"vs native text" + (" (> bound)" if cer_hard else ""))
            else:
                evidence.append("native text layer unreliable/unaligned — "
                                "CER/WER skipped (gated)")

        # ---- new HARD rule: definite repetition OR reliable native CER OR
        #      >=2 independent artefact-agreement failures (no worst-of gating) -
        hard = bool(rep["hard"]) or (cer_hard and native_ok) or (artefact_fail >= 2)

        # score: MEAN of available sub-parts (not worst-of, so one weak check
        # no longer collapses XSA); clamped low when a hard condition fires.
        scores = list(sub.values())
        score = round(sum(scores) / len(scores), 1) if scores else 100.0
        if hard:
            score = min(score, 20.0)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        locations = []
        if rep["max_ngram_count"] > 1 and rep["longest_span"]:
            locations.append(loc(
                "repetition", "definite" if rep["hard"] else "statistical",
                block=find_block_for(page, rep["longest_span"]),
                snippet=rep["longest_span"]))

        if rep["hard"]:
            example = (f"The phrase '{rep['longest_span']}' repeats "
                       f"{rep['max_ngram_count']}× (> {cfg.xsa_repeat_max}) — a definite "
                       f"hallucination loop, so the page is flagged.")
        elif cer_hard:
            example = (f"Against the (reliable) native PDF text the OCR has "
                       f"{nat['cer'] * 100:.0f}% character error (> "
                       f"{cfg.xsa_cer_bad * 100:.0f}%).")
        elif artefact_fail >= 2:
            example = (f"{artefact_fail} independent artefact checks disagree "
                       f"(JSON vs MD/chunk/HTML) — multiple sources lost content, so "
                       f"this is treated as a definite cross-source failure.")
        elif artefact_fail == 1:
            example = ("Only one artefact disagrees — recorded as a soft signal, NOT "
                       "a hard failure (a single weak sub-check no longer forces review).")
        else:
            example = ("No repetition loop and the JSON/MD/chunk/HTML texts agree → "
                       "the page is internally consistent.")

        return self.result(
            raw_value=rep["trigram_rep"],
            score=score,
            status=status,
            threshold={
                "repeat_hard_ngram": cfg.xsa_repeat_ngram,
                "repeat_hard_max": cfg.xsa_repeat_max,
                "artefact_fail_for_hard": 2,
                "md_sim_warn": warn,
                "cer_bad": cfg.xsa_cer_bad,
                "native_reliable": native_ok,
            },
            how_computed="split sub-metrics (repetition, JSON↔MD, JSON↔chunk, "
            "JSON↔HTML, native CER). HARD only on a definite repetition loop, a "
            "reliable native-CER mismatch, or ≥2 independent artefact "
            "disagreements; score = mean of sub-parts.",
            example=example,
            locations=locations,
            evidence=evidence,
            submetrics=sub,
            justification=(
                f"Agreement {score:.0f}/100 across {len(sub)} sub-metric(s); "
                f"{artefact_fail} artefact disagreement(s)"
                + (" — DEFINITE." if hard else " — soft/consistent.")
            ),
            reliability="high",  # cross-source agreement applies to any page type
            hard_fail=hard,
        )
