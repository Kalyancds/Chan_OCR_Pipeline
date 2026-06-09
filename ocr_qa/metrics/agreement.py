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
        subscores: list[float] = []
        hard = False

        # (a) repetition --------------------------------------------------
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
            hard = True
            evidence.append(
                f"repeated span '{rep['longest_span']}' ×{rep['max_ngram_count']} "
                f"(>{cfg.xsa_repeat_max}) — likely VLM loop"
            )
        elif rep["max_ngram_count"] > 1:
            evidence.append(
                f"top {cfg.xsa_repeat_ngram}-gram '{rep['longest_span']}' "
                f"×{rep['max_ngram_count']}"
            )
        evidence.append(
            f"trigram repetition {rep['trigram_rep'] * 100:.0f}%, "
            f"gzip ratio {rep['gzip_ratio']:.1f}"
        )
        subscores.append(rep_score)

        # (b) json <-> md  (word coverage: robust to markdown formatting; only
        #     genuine content loss lowers it, not '##'/'|'/link syntax) --------
        sim = _text_recall(page.text, page.md_text)
        if sim is not None:
            md_score = lerp_score(1.0 - sim, 1.0 - 0.85, 1.0 - 0.40)
            subscores.append(md_score)
            if sim < cfg.xsa_md_sim_warn:
                evidence.append(
                    f"JSON↔MD word coverage {sim * 100:.0f}% (< "
                    f"{cfg.xsa_md_sim_warn * 100:.0f}%) — content missing from MD"
                )
            else:
                evidence.append(f"JSON↔MD word coverage {sim * 100:.0f}%")

        # (c) native ------------------------------------------------------
        nat = _native_agreement(text, native_text)
        if nat is not None:
            cer_score = lerp_score(nat["cer"], 0.02, cfg.xsa_cer_bad * 2)
            subscores.append(cer_score)
            if nat["cer"] > cfg.xsa_cer_bad:
                hard = True
                evidence.append(
                    f"CER {nat['cer'] * 100:.1f}% (> {cfg.xsa_cer_bad * 100:.0f}%) "
                    f"vs native text, WER {nat['wer'] * 100:.1f}%"
                )
            else:
                evidence.append(
                    f"CER {nat['cer'] * 100:.1f}%, WER {nat['wer'] * 100:.1f}% "
                    f"vs native text"
                )

        # (d) json <-> chunk (output_chunks.json) -------------------------
        recall = _text_recall(text, page.chunk_text)
        if recall is not None:
            chunk_score = lerp_score(1.0 - recall, 1.0 - 0.85, 1.0 - 0.40)
            subscores.append(chunk_score)
            if recall < cfg.xsa_md_sim_warn:
                evidence.append(
                    f"JSON↔chunk coverage {recall * 100:.0f}% (< "
                    f"{cfg.xsa_md_sim_warn * 100:.0f}%) — page content disagrees "
                    f"with output_chunks"
                )
            else:
                evidence.append(f"JSON↔chunk coverage {recall * 100:.0f}%")

        # (e) json <-> html (output.html) --------------------------------
        html_sim = _json_html_similarity(page.text, page.html_text)
        if html_sim is not None:
            html_score = lerp_score(1.0 - html_sim, 1.0 - 0.9, 1.0 - 0.3)
            subscores.append(html_score)
            if html_sim < cfg.xsa_md_sim_warn:
                evidence.append(
                    f"JSON↔HTML similarity {html_sim * 100:.0f}% (< "
                    f"{cfg.xsa_md_sim_warn * 100:.0f}%) — JSON and HTML disagree"
                )
            else:
                evidence.append(f"JSON↔HTML similarity {html_sim * 100:.0f}%")

        score = min(subscores) if subscores else 100.0
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")

        locations = []
        if rep["max_ngram_count"] > 1 and rep["longest_span"]:
            b = find_block_for(page, rep["longest_span"])
            locations.append(loc(
                "repetition",
                "definite" if rep["hard"] else "statistical",
                block=b,
                snippet=rep["longest_span"],
            ))

        parts = ["repetition"]
        if sim is not None:
            parts.append("JSON↔MD")
        if nat is not None:
            parts.append("native-CER")
        if recall is not None:
            parts.append("JSON↔chunk")
        if html_sim is not None:
            parts.append("JSON↔HTML")

        # worked example from the worst available sub-part
        if rep["hard"]:
            example = (
                f"The phrase '{rep['longest_span']}' repeats {rep['max_ngram_count']}× "
                f"on this page — more than the {cfg.xsa_repeat_max}× limit. That is a "
                f"classic VLM hallucination loop, so the page is flagged."
            )
        elif nat is not None and nat["cer"] > cfg.xsa_cer_bad:
            example = (
                f"Against the PDF's own text layer the OCR has {nat['cer'] * 100:.0f}% "
                f"character error (limit {cfg.xsa_cer_bad * 100:.0f}%) — i.e. about "
                f"1 in {max(1, round(1 / max(nat['cer'], 1e-6)))} characters is wrong."
            )
        elif recall is not None and recall < cfg.xsa_md_sim_warn:
            example = (
                f"Only {recall * 100:.0f}% of this page's words also appear in the "
                f"output_chunks text for it — the two Chandra artefacts disagree, "
                f"suggesting content was lost or shuffled."
            )
        elif sim is not None and sim < cfg.xsa_md_sim_warn:
            example = (
                f"Only {sim * 100:.0f}% of this page's words appear in the Markdown "
                f"output — content seems to be missing from output.md (formatting "
                f"differences alone would not lower this)."
            )
        else:
            example = (
                f"No repeated phrases (top {cfg.xsa_repeat_ngram}-gram seen "
                f"{rep['max_ngram_count']}×) and the JSON/MD/chunk texts agree → the "
                f"page is internally consistent."
            )

        return self.result(
            raw_value=rep["trigram_rep"],
            score=score,
            status=status,
            threshold={
                "repeat_hard_ngram": cfg.xsa_repeat_ngram,
                "repeat_hard_max": cfg.xsa_repeat_max,
                "md_sim_warn": cfg.xsa_md_sim_warn,
                "cer_bad": cfg.xsa_cer_bad,
            },
            how_computed="worst of available sub-parts: "
            + " + ".join(parts)
            + ".",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"Worst-of agreement score {score:.0f}/100 across {len(subscores)} "
                f"sub-part(s) ({', '.join(parts)})"
                + (" — HARD repetition/native flag." if hard else ".")
            ),
            reliability="low" if self.low_reliability(page, ctx) else "high",
            hard_fail=hard,
        )
