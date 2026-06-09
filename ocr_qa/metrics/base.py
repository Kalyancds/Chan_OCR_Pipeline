"""Metric interface, document context, and shared text utilities (Sec 4).

Every metric implements ``compute(page, native_text, ctx) -> MetricResult`` and
returns a fully self-describing card: raw_value, score(0-100), status, the
threshold it used, a static ``what_it_measures``, a plain-English
``how_computed``, concrete ``evidence`` from THIS page, a templated
``justification`` citing the number AND threshold, and a ``reliability`` flag.

Justifications are DETERMINISTIC and templated — never an LLM (operating
agreement). Heavy deps (wordfreq, spellchecker, langdetect, jiwer) are imported
lazily so the module loads even before they are installed.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from ocr_qa.config import Config
from ocr_qa.models import MetricResult, MetricStatus, PageOCR


# --------------------------------------------------------------------------- #
# Document-level context (built once per document, shared by all page metrics)
# --------------------------------------------------------------------------- #
@dataclass
class DocContext:
    config: Config
    doc_median_blocks: Optional[float] = None
    # PPL calibration: raw surprisal bounds derived from the doc's own pages.
    ppl_good_bound: Optional[float] = None
    ppl_bad_bound: Optional[float] = None
    ppl_model: object = None  # trained NgramModel (set by registry pre-pass)
    ppl_page_raw: dict[int, float] = field(default_factory=dict)
    # per-page detected language {page_no: lang}
    languages: dict[int, str] = field(default_factory=dict)
    # near-duplicate pages: {page_no: (other_page_no, jaccard_similarity)}
    duplicate_of: dict[int, tuple] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Tokenisation helpers (unicode-aware: accented Latin letters are VALID)
# --------------------------------------------------------------------------- #
# A "word" = run of letters (any script, incl. àéüç…) with internal '-' or '’'.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_RAW_TOKEN_RE = re.compile(r"\S+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_NUMERIC_RE = re.compile(r"^[\d.,:/%\-+°×x ]+$")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def word_tokens(text: str) -> list[str]:
    """Letter-run word tokens (accents preserved)."""
    if not text:
        return []
    return _WORD_RE.findall(text)


def raw_tokens(text: str) -> list[str]:
    """Whitespace-delimited tokens (for length / shape analysis)."""
    if not text:
        return []
    return _RAW_TOKEN_RE.findall(text)


def is_numbery(tok: str) -> bool:
    return bool(_NUMERIC_RE.match(tok))


def is_url_or_email(tok: str) -> bool:
    return bool(_URL_RE.search(tok) or _EMAIL_RE.search(tok))


def sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def lerp_score(value: float, good: float, bad: float) -> float:
    """Linear map value->[0,100] where value<=good => 100 and value>=bad => 0.

    Works whether good<bad (lower is better) or good>bad (higher is better).
    """
    if good == bad:
        return 100.0 if value == good else 0.0
    # normalise to fraction along good->bad
    frac = (value - good) / (bad - good)
    frac = max(0.0, min(1.0, frac))
    return round((1.0 - frac) * 100.0, 1)


def status_from_score(score: float, warn: float = 70.0, bad: float = 50.0) -> MetricStatus:
    if score >= warn:
        return "good"
    if score >= bad:
        return "warn"
    return "bad"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Error-localization helpers (build the `locations` list for highlighting)
# --------------------------------------------------------------------------- #
def loc(
    kind: str,
    confidence: str,
    *,
    block: "object" = None,
    block_id: str = "",
    bbox: Optional[list] = None,
    snippet: str = "",
) -> dict:
    """Build a location record. ``confidence`` is 'definite' (a reproducible
    structural fact) or 'statistical' (a threshold-based flag)."""
    if block is not None:
        block_id = block_id or getattr(block, "block_id", "")
        bbox = bbox if bbox is not None else getattr(block, "bbox", [])
    return {
        "kind": kind,
        "confidence": "definite" if confidence == "definite" else "statistical",
        "block_id": block_id or "",
        "bbox": list(bbox or []),
        "snippet": (snippet or "")[:120],
    }


def find_block_for(page, needle: str):
    """Return the first block whose text/html contains ``needle`` (to attribute a
    text snippet to a concrete block + bbox). None if not found."""
    if not needle:
        return None
    low = needle.lower()
    for b in page.blocks:
        if low in (b.text or "").lower() or low in (b.html or "").lower():
            return b
    return None


# --------------------------------------------------------------------------- #
# Language detection (lazy, cached)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def detect_language(text: str) -> Optional[str]:
    sample = (text or "").strip()
    if len(sample) < 20:
        return None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic
        return detect(sample[:2000])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Char-frequency baselines for entropy / KL (Sec 4 M9). Coarse but deterministic.
# Values are approximate relative frequencies of letters in large corpora.
# --------------------------------------------------------------------------- #
_EN_FREQ = {
    "e": 0.127, "t": 0.091, "a": 0.082, "o": 0.075, "i": 0.070, "n": 0.067,
    "s": 0.063, "h": 0.061, "r": 0.060, "d": 0.043, "l": 0.040, "c": 0.028,
    "u": 0.028, "m": 0.024, "w": 0.024, "f": 0.022, "g": 0.020, "y": 0.020,
    "p": 0.019, "b": 0.015, "v": 0.010, "k": 0.008, "j": 0.0015, "x": 0.0015,
    "q": 0.001, "z": 0.0007,
}
_DE_FREQ = {
    "e": 0.165, "n": 0.098, "i": 0.076, "s": 0.073, "r": 0.070, "a": 0.065,
    "t": 0.061, "d": 0.051, "h": 0.048, "u": 0.044, "l": 0.034, "c": 0.027,
    "g": 0.030, "m": 0.025, "o": 0.025, "b": 0.019, "w": 0.019, "f": 0.017,
    "k": 0.012, "z": 0.011, "p": 0.0067, "v": 0.0067, "ü": 0.0065, "ä": 0.0054,
    "ö": 0.0030, "j": 0.0027, "y": 0.0004, "x": 0.0003, "q": 0.0002,
}

LANG_FREQ = {"en": _EN_FREQ, "de": _DE_FREQ}


def lang_baseline(lang: Optional[str]) -> dict[str, float]:
    return LANG_FREQ.get((lang or "en")[:2], _EN_FREQ)


_VOWELS = set("aeiouyàáâäãåéèêëíìîïóòôöõúùûüøœæ")
_ILLEGAL_BIGRAMS = {
    "qx", "xq", "qz", "zq", "qj", "jq", "jx", "xj", "vq", "qv", "fq",
    "kq", "qk", "vx", "xv", "wx", "kx", "zx", "xz", "bx", "fx", "gx", "cx",
}


def is_pronounceable(token: str) -> bool:
    """Heuristic gate: does a token look like a pronounceable word (any Latin
    language) rather than OCR gibberish? Brand/proper nouns (Nemluvio,
    Galderma) pass; gibberish (rnixqk, qwzxfp) fails. Deterministic."""
    t = "".join(c for c in token.lower() if c.isalpha())
    n = len(t)
    if n <= 2:
        return True
    vowels = sum(1 for c in t if c in _VOWELS)
    vratio = vowels / n
    if vowels == 0:
        return False
    if vratio < 0.18 or vratio > 0.90:
        return False
    # longest consonant run
    run = best = 0
    for c in t:
        if c in _VOWELS:
            run = 0
        else:
            run += 1
            best = max(best, run)
    if best > 4:
        return False
    # illegal bigrams
    for i in range(n - 1):
        if t[i : i + 2] in _ILLEGAL_BIGRAMS:
            return False
    return True


def shannon_entropy(text: str) -> float:
    letters = [c.lower() for c in text if c.isalpha()]
    if not letters:
        return 0.0
    from collections import Counter

    n = len(letters)
    counts = Counter(letters)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# --------------------------------------------------------------------------- #
# Metric base class
# --------------------------------------------------------------------------- #
class BaseMetric(ABC):
    key: str = "XXX"
    name: str = "Unnamed metric"
    what_it_measures: str = ""
    # default group weight is read from Config.weights[key]; this is just docs.

    @abstractmethod
    def compute(
        self,
        page: PageOCR,
        native_text: Optional[str],
        ctx: DocContext,
    ) -> MetricResult:
        ...

    # ---- shared result builder -------------------------------------------
    def result(
        self,
        *,
        raw_value: float,
        score: float,
        status: MetricStatus,
        threshold: dict,
        how_computed: str,
        evidence: list[str],
        justification: str,
        example: str = "",
        locations: Optional[list[dict]] = None,
        reliability: str = "high",
        applicable: bool = True,
        hard_fail: bool = False,
    ) -> MetricResult:
        return MetricResult(
            key=self.key,
            name=self.name,
            raw_value=round(float(raw_value), 4),
            score=round(float(score), 1),
            status=status,
            threshold=threshold,
            what_it_measures=self.what_it_measures,
            how_computed=how_computed,
            example=example,
            evidence=evidence[:8],
            locations=(locations or [])[:12],
            justification=justification,
            reliability="low" if reliability == "low" else "high",
            applicable=applicable,
            hard_fail=hard_fail,
        )

    def not_applicable(self, reason: str) -> MetricResult:
        return self.result(
            raw_value=0.0,
            score=100.0,
            status="na",
            threshold={},
            how_computed=reason,
            evidence=[],
            justification=f"Not applicable: {reason}",
            applicable=False,
        )

    # ---- shared reliability gate -----------------------------------------
    @staticmethod
    def low_reliability(page: PageOCR, ctx: DocContext) -> bool:
        toks = word_tokens(page.prose_text())
        if len(toks) < ctx.config.short_page_token_floor:
            return True
        if not ctx.languages.get(page.page_no):
            return True
        return False
