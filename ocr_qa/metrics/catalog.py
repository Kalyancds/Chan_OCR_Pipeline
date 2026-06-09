"""Static reference catalog for the Metrics Guide tab.

Names and `what_it_measures` come from the metric classes (single source of
truth); this module adds the human-facing reference fields: group, which files
each metric reads, a compact formula, the confidence type, and a concrete
illustrative example. Keep entries in the same order as the registry.
"""

from __future__ import annotations

from ocr_qa.config import DEFAULT_WEIGHTS

# Group label + accent colour (for the guide's section styling).
GROUPS = {
    "A": ("Engine & Structure", "#2f6fed"),
    "B": ("Markup Integrity", "#7b4fc0"),
    "C": ("Lexical / Dictionary", "#1a9850"),
    "D": ("Statistical Text", "#e8870c"),
    "E": ("Surface / Format", "#d6663c"),
    "F": ("Repetition & Agreement", "#c0397b"),
}

# confidence buckets
DEF = "Definite"          # reproducible structural fact
STAT = "Statistical"      # threshold-based flag
MIX = "Mixed"             # definite for hard cases, statistical otherwise

# key -> reference fields
CATALOG: list[dict] = [
    {
        "key": "IFR", "group": "A",
        "reads": ["output.json"],
        "formula": "failed_blocks / total_blocks  ·  HARD if any content block failed",
        "confidence": DEF,
        "example": "Block /page/7/Text/3 has inference_failed=true → 1 of 8 blocks "
                   "failed; because it is content, the page is forced to review.",
    },
    {
        "key": "BCC", "group": "A",
        "reads": ["output.json", "output.metadata.json", "output_chunks.json"],
        "formula": "|json_blocks − metadata.num_blocks| / metadata.num_blocks  ·  "
                   "HARD beyond ±tolerance",
        "confidence": DEF,
        "example": "output.json parses 7 blocks but metadata records 13 → 46% "
                   "mismatch → blocks were dropped/duplicated.",
    },
    {
        "key": "BGC", "group": "A",
        "reads": ["output.json"],
        "formula": "(degenerate + out-of-bounds + overlapping + order-inverted "
                   "blocks) / total blocks",
        "confidence": DEF,
        "example": "A block bbox [-50, 800, 30, 760] has negative height (degenerate) "
                   "→ 1 of 5 boxes invalid = 20%.",
    },
    {
        "key": "EMP", "group": "A",
        "reads": ["output.json"],
        "formula": "empty_content_blocks / content_blocks",
        "confidence": DEF,
        "example": "A <Text> block whose HTML is '<p></p>' strips to empty — a box "
                   "with no text means content was dropped.",
    },
    {
        "key": "HMW", "group": "B",
        "reads": ["output.json"],
        "formula": "malformed_blocks / total HTML blocks  (tag-balance + entities)",
        "confidence": DEF,
        "example": "'<p>… <b>bold' leaves <b> (and <p>) unclosed → that block's "
                   "markup is malformed.",
    },
    {
        "key": "TSI", "group": "B",
        "reads": ["output.json", "output_chunks.json"],
        "formula": "ragged-row + 0.5·empty-cell + no-header + stray-colspan, per "
                   "<table>; cross-checked vs chunk table_metrics",
        "confidence": DEF,
        "example": "Header has 4 columns but row 3 has 3 → ragged table; or chunks "
                   "report a table but none was parsed → dropped table.",
    },
    {
        "key": "FIG", "group": "B",
        "reads": ["output.json"],
        "formula": "figures_without_description / total figures",
        "confidence": DEF,
        "example": "An <img> with no sibling <div class=\"img-description\"> → the "
                   "figure's visual content was left undescribed.",
    },
    {
        "key": "LVR", "group": "C",
        "reads": ["output.json", "wordfreq", "pyspellchecker", "domain whitelist"],
        "formula": "valid_words / countable_words  (Zipf>0 OR known OR whitelisted "
                   "OR pronounceable proper noun)",
        "confidence": MIX,
        "example": "'rnixqk' is no real word → invalid; 'Nemluvio' is on the pharma "
                   "whitelist → valid. 27/63 invalid = 57% (HARD below 80%).",
    },
    {
        "key": "OOV", "group": "C",
        "reads": ["output.json", "pronounceability gate"],
        "formula": "unpronounceable_oov / total_oov",
        "confidence": MIX,
        "example": "'rnixqk' (illegal vowel/consonant pattern) = gibberish; "
                   "'Galderma' = plausible brand → keeps real names from being errors.",
    },
    {
        "key": "SFC", "group": "C",
        "reads": ["output.json", "langdetect", "stopword lists"],
        "formula": "|observed_stopword_ratio − expected_ratio(language)|",
        "confidence": STAT,
        "example": "Prose with 12% function words ('the/of/and') vs ~45% expected for "
                   "English → 33% deviation (low on garbled/looping text).",
    },
    {
        "key": "CED", "group": "D",
        "reads": ["output.json", "language baselines"],
        "formula": "KL(char_freq ‖ language_baseline) + explicit mojibake count",
        "confidence": DEF,
        "example": "'Ã©' or '�' are mojibake; ≥3 hits → HARD. An accented 'é' is "
                   "valid and is NOT counted as mojibake.",
    },
    {
        "key": "PPL", "group": "D",
        "reads": ["output.json (all pages)"],
        "formula": "mean char-trigram surprisal (bits/char), calibrated on the "
                   "document's own cleaner pages",
        "confidence": STAT,
        "example": "A garbled line reads 3.2 bits/char vs the document's clean ~1.8 "
                   "→ low fluency for THIS document.",
    },
    {
        "key": "TTR", "group": "D",
        "reads": ["output.json"],
        "formula": "unique_words / total_words vs a length-adjusted expected band",
        "confidence": STAT,
        "example": "12 unique of 40 tokens (TTR 0.30) is below the band → the same "
                   "words repeat (looping).",
    },
    {
        "key": "WSA", "group": "E",
        "reads": ["output.json"],
        "formula": "(merged + over-segmented + hyphen-break + alnum-confusion "
                   "tokens) / total tokens",
        "confidence": MIX,
        "example": "'l1l1O0' mixes letters and 1/0 (l↔1, O↔0); a 40-char run with no "
                   "spaces = merged words.",
    },
    {
        "key": "PSW", "group": "E",
        "reads": ["output.json"],
        "formula": "(unmatched brackets/quotes + over-long sentences + spacing "
                   "anomalies) per 100 words",
        "confidence": STAT,
        "example": "'Mismatched (brackets [here' leaves '(' and '[' unclosed and an "
                   "odd quote count.",
    },
    {
        "key": "XSA", "group": "F",
        "reads": ["output.json", "output.md", "output.html", "output_chunks.json",
                  "native PDF text"],
        "formula": "worst of available: repetition · JSON↔MD · JSON↔chunk · "
                   "JSON↔HTML · native CER/WER",
        "confidence": MIX,
        "example": "The 5-gram 'the patient must take the' repeats 8× → VLM "
                   "hallucination loop (HARD); or CER 30% vs the PDF text.",
    },
    {
        "key": "DUP", "group": "F",
        "reads": ["output.json (all pages)"],
        "formula": "max token-set Jaccard similarity vs every other page",
        "confidence": MIX,
        "example": "Page 50 shares 99% of its words with page 37 → a duplicate page "
                   "(inflates the page count); ≥85% = near-duplicate.",
    },
]


def weight_for(key: str) -> float:
    return DEFAULT_WEIGHTS.get(key, 0.0)


# File → which metrics consume it (for the guide's source map).
FILE_MAP: list[tuple[str, str, str]] = [
    ("output.json", "Block tree: text, HTML, bbox, inference_failed",
     "ALL metrics (backbone)"),
    ("output.metadata.json", "page_stats.num_blocks (also embedded in output.json)",
     "BCC"),
    ("output.md", "Page-delimited markdown ({N}----)", "XSA (JSON↔MD)"),
    ("output.html", "Per-page rendered HTML (<div class='page'>)",
     "XSA (JSON↔HTML) + the HTML view"),
    ("output_chunks.json", "Chunk content + table_metrics (page_ids)",
     "BCC, TSI, XSA (JSON↔chunk)"),
    ("source PDF/PPTX (native text)", "Embedded text layer when digital-born",
     "XSA (CER/WER)"),
    ("wordfreq / pyspellchecker / whitelist", "Dictionaries & brand list",
     "LVR, OOV"),
    ("langdetect", "Per-page/block language", "SFC, CED, LVR"),
]
