"""Central, sidebar-editable configuration for the OCR QA app.

Every threshold, bound and metric weight lives here so the pipeline is
config-driven (operating agreement) and the Streamlit sidebar can override
defaults at runtime without touching compute code.

Nothing in this module imports heavy deps, so it is safe to load anywhere.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Metric weights (Sec 5). Keys are metric `key`s used by the registry.
# Must conceptually sum to 1.0; the aggregator renormalises over AVAILABLE
# metrics anyway, so editing one weight never breaks scoring.
# --------------------------------------------------------------------------- #
DEFAULT_WEIGHTS: dict[str, float] = {
    "IFR": 0.16,  # Inference-Failure Rate  (primary engine signal)
    "BCC": 0.06,  # Block-Count & Source Consistency
    "BGC": 0.06,  # Block-Geometry Coherence
    "HMW": 0.07,  # HTML/Markup Well-formedness
    "TSI": 0.05,  # Table Structure Integrity
    "LVR": 0.14,  # Lexical Validity Ratio
    "OOV": 0.07,  # OOV Pronounceability
    "SFC": 0.04,  # Stopword Frequency Conformance
    "CED": 0.07,  # Character Entropy & Mojibake Divergence
    "PPL": 0.08,  # N-gram Perplexity / Fluency
    "TTR": 0.04,  # Lexical Diversity / Type-Token Anomaly
    "WSA": 0.04,  # Word-Shape / Tokenization Anomaly
    "PSW": 0.04,  # Punctuation & Sentence Well-formedness
    "XSA": 0.08,  # Repetition & Cross-Source/Native Agreement
    # ---- enhancement metrics (added after the original 14) ----------------
    "EMP": 0.04,  # Empty Content-Block Rate (content-loss)
    "FIG": 0.03,  # Figure & Caption Integrity
    "DUP": 0.04,  # Duplicate / Near-Duplicate Page
    # weights need not sum to 1.0: the aggregator renormalises over the
    # metrics that actually applied to each page.
}


class Config(BaseModel):
    """Runtime knobs. All fields are sidebar-editable (Sec 9)."""

    # ---- page / document verdict thresholds -------------------------------
    # Decisioning (per the metric-review recommendations): a page is flagged for
    # review when EITHER one DEFINITE (hard) failure fires, OR at least
    # `soft_fail_threshold` SOFT signals fail on a page with enough text, OR the
    # PQS falls below `pqs_backstop` (a low safety net). `t_page` is retained for
    # display/back-compat but no longer the primary trigger.
    t_page: float = Field(70.0, description="PQS reference line (display)")
    pqs_backstop: float = Field(
        50.0, description="PQS below this => review (low safety net)"
    )
    soft_fail_threshold: int = Field(
        3, description="this many soft-signal failures (on a text-rich page) => review"
    )
    t_doc: float = Field(75.0, description="DQS below this => document FAILED")
    doc_flag_ratio: float = Field(
        0.20, description="faulty/total above this => document FAILED"
    )

    # ---- token / page sizing ---------------------------------------------
    l_max_token: int = Field(25, description="token length above this = merged-word")
    short_page_token_floor: int = Field(
        40, description="prose tokens below this => low-reliability metrics"
    )

    # ---- lexical (LVR / OOV) ---------------------------------------------
    lexical_good: float = Field(0.92, description="LVR at/above => score 100")
    lexical_warn: float = Field(0.80, description="LVR below => HARD condition")
    lexical_zero: float = Field(0.60, description="LVR at/below => score 0")

    # ---- entropy / mojibake (CED) ---------------------------------------
    ced_kl_good: float = Field(0.15, description="KL divergence at/below => 100")
    ced_kl_bad: float = Field(1.50, description="KL divergence at/above => 0")
    ced_mojibake_bad: int = Field(
        3, description="mojibake hits at/above => HARD condition"
    )

    # ---- word-shape (WSA) ------------------------------------------------
    wsa_good: float = Field(0.02, description="anomaly rate at/below => 100")
    wsa_bad: float = Field(0.15, description="anomaly rate at/above => 0")

    # ---- cross-source / repetition (XSA) --------------------------------
    xsa_repeat_ngram: int = Field(5, description="n-gram size for repetition flag")
    xsa_repeat_max: int = Field(
        5, description="same n-gram repeating MORE than this => HARD flag"
    )
    xsa_cer_bad: float = Field(0.25, description="CER above (native text) => HARD")
    xsa_md_sim_warn: float = Field(
        0.60, description="json<->md similarity below => warn"
    )

    # ---- block-count consistency (BCC) ----------------------------------
    bcc_tolerance: float = Field(
        0.20, description="relative json-vs-metadata mismatch tolerated (+/-)"
    )

    # ---- perplexity backend ---------------------------------------------
    perplexity_backend: str = Field(
        "ngram", description="one of: ngram | kenlm | gpt2"
    )

    # ---- weights ---------------------------------------------------------
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    # ---- rendering -------------------------------------------------------
    render_dpi: int = Field(200, description="PyMuPDF render DPI for page images")

    def normalized_weights(self, available_keys: list[str]) -> dict[str, float]:
        """Renormalise weights over the metrics actually present (Sec 5)."""
        sub = {k: self.weights.get(k, 0.0) for k in available_keys}
        total = sum(sub.values())
        if total <= 0:
            # Degenerate: equal weight.
            n = len(available_keys) or 1
            return {k: 1.0 / n for k in available_keys}
        return {k: v / total for k, v in sub.items()}


# A module-level default others can import directly.
DEFAULT_CONFIG = Config()


# --------------------------------------------------------------------------- #
# Metric tiers (metric-review recommendations).
#  * HARD metrics may, when a DEFINITE condition fires, send a page straight to
#    review (carried on MetricResult.hard_fail).
#  * SOFT metrics never trigger review alone; they only count toward the
#    ">= soft_fail_threshold soft failures" rule and act as supporting evidence.
# --------------------------------------------------------------------------- #
HARD_METRICS: set[str] = {"IFR", "BCC", "BGC", "EMP", "HMW", "TSI", "DUP", "CED", "XSA"}
SOFT_METRICS: set[str] = {"SFC", "TTR", "PPL", "PSW", "LVR", "OOV", "WSA"}


# Pharma / brand proper nouns that must never be flagged as garbled (Sec 4).
# Lower-cased for matching. Extendable; the pronounceability gate (M7) is the
# general fallback, this is the precise allowlist.
DOMAIN_WHITELIST: set[str] = {
    "nemluvio",
    "nemolizumab",
    "galderma",
    "gemeinsamer",
    "bundesausschuss",
    "biologika",
    "atopische",
    "dermatitis",
    "prurigo",
    "nodularis",
    "subkutan",
    "fertigpen",
    "fertigspritze",
    "anwendungsgebiet",
    "wirkstoff",
    "arzneimittel",
    "pharmakovigilanz",
    "ema",
    "fda",
    "ich",
    "gcp",
}
