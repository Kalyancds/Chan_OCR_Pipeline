"""Reusable, deterministic Streamlit UI components (Sec 7).

Metric cards are TEMPLATED — every field comes straight from the MetricResult,
never an LLM. Colours/badges are derived from status. Importing this module does
not require Streamlit until a function is actually called, so the pipeline/tests
stay dependency-light.
"""

from __future__ import annotations

import html
import re
from typing import Optional

from ocr_qa.models import DocumentReport, MetricResult, PageReport

# Location kinds whose snippet is a literal text span (can be <mark>-highlighted).
TEXT_HIGHLIGHT_KINDS = {
    "gibberish_word", "rare_word", "mojibake", "repetition",
    "merged_word", "alnum_confusion", "hyphen_break",
}
_KIND_LABEL = {
    "inference_failed": "Engine could not read block",
    "bad_geometry": "Bad block geometry",
    "malformed_html": "Malformed HTML",
    "table_defect": "Table structure defect",
    "table_unparseable": "Unparseable table",
    "dropped_table": "Dropped table",
    "mojibake": "Mojibake / encoding",
    "repetition": "Repeated span (loop)",
    "gibberish_word": "Gibberish word",
    "rare_word": "Rare / unknown word",
    "merged_word": "Merged word",
    "alnum_confusion": "Letter/digit confusion",
    "hyphen_break": "Hyphen line-break",
}

_STATUS_COLOR = {"good": "#1a9850", "warn": "#e8870c", "bad": "#d73027", "na": "#9aa0a6"}
_STATUS_LABEL = {"good": "GOOD", "warn": "WARN", "bad": "BAD", "na": "N/A"}


def status_color(status: str) -> str:
    return _STATUS_COLOR.get(status, "#9aa0a6")


# --------------------------------------------------------------------------- #
# global styling
# --------------------------------------------------------------------------- #
APP_CSS = """
<style>
:root { --good:#1a9850; --warn:#e8870c; --bad:#d73027; --na:#9aa0a6; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }
h1, h2, h3 { letter-spacing:-.01em; }

/* hero header */
.hero { background:linear-gradient(100deg,#1f2a44 0%,#2f6fed 100%); color:#fff;
  border-radius:14px; padding:1.0rem 1.3rem; margin:.1rem 0 .5rem;
  box-shadow:0 6px 18px rgba(47,111,237,.18); }
.hero h1 { color:#fff; font-size:1.5rem; margin:0; }
.hero p { color:#dbe6ff; margin:.2rem 0 0; font-size:.9rem; }

/* status legend */
.legend { display:flex; gap:1rem; flex-wrap:wrap; font-size:.8rem; color:#555;
  align-items:center; margin:.1rem 0 .2rem; }
.legend .sw { display:inline-block; width:11px; height:11px; border-radius:50%;
  margin-right:.3rem; vertical-align:middle; }

/* guide */
.guide-group { font-weight:800; font-size:1.05rem; margin:.7rem 0 .35rem;
  padding-left:.5rem; border-left:5px solid #ccc; }
.src-chip { display:inline-block; background:#eef2f7; border:1px solid #dde3ea;
  color:#33415c; padding:.06rem .45rem; border-radius:6px; margin:.1rem .2rem 0 0;
  font-size:.72rem; font-family:ui-monospace,Consolas,monospace; }
.conf { display:inline-block; padding:.05rem .45rem; border-radius:999px;
  font-size:.66rem; font-weight:700; color:#fff; }
.conf.Definite { background:#b00020; }
.conf.Statistical { background:#e8870c; }
.conf.Mixed { background:#6a51c9; }

/* nav pills */
div[data-testid="stHorizontalBlock"] .ocrqa-nav { }

/* metric card header */
.mhead { display:flex; align-items:center; gap:.55rem; margin:.1rem 0 .15rem; }
.mhead .dot { width:11px; height:11px; border-radius:50%; flex:0 0 auto;
  box-shadow:0 0 0 3px rgba(0,0,0,.04); }
.mhead .mname { font-weight:650; font-size:1rem; }
.mhead .mkey { font-size:.66rem; color:#6b7280; border:1px solid #e2e5ea;
  padding:.02rem .4rem; border-radius:6px; letter-spacing:.04em; }
.mhead .mscore { margin-left:auto; font-weight:800; font-size:1.18rem; line-height:1; }
.mhead .mscore small { font-weight:500; color:#9aa0a6; font-size:.62rem; }

/* badges */
.pill { display:inline-block; padding:.06rem .5rem; border-radius:999px;
  font-size:.66rem; font-weight:700; color:#fff; letter-spacing:.03em; vertical-align:middle;}
.pill.ghost { background:transparent; border:1px solid #d7000022; }
.tag-hard { background:#fbe9e7; color:#c62828; border:1px solid #f1b0a8;
  font-size:.64rem; font-weight:700; padding:.04rem .45rem; border-radius:6px; }
.tag-rel { background:#fff8e1; color:#a6791f; border:1px solid #f0dca0;
  font-size:.62rem; padding:.04rem .4rem; border-radius:6px; }
.tier-hard { background:#2a2f45; color:#fff; font-size:.6rem; font-weight:700;
  padding:.04rem .45rem; border-radius:6px; letter-spacing:.03em; }
.tier-soft { background:#eef2f7; color:#5b6472; border:1px solid #dde3ea;
  font-size:.6rem; font-weight:700; padding:.04rem .45rem; border-radius:6px; }
.ctxstrip { display:flex; gap:.4rem; flex-wrap:wrap; margin:.2rem 0 .4rem; }
.ctx { background:#f3f6fb; border:1px solid #e2e8f1; border-radius:7px;
  padding:.18rem .5rem; font-size:.74rem; color:#33415c; }
.ctx b { color:#1f2a44; }
.submx { display:flex; gap:.35rem; flex-wrap:wrap; margin:.2rem 0; }
.submx .s { font-size:.72rem; border:1px solid #e2e8f1; border-radius:6px;
  padding:.1rem .4rem; background:#fafbfd; }

/* score bar */
.bar { height:7px; background:#eef0f3; border-radius:7px; overflow:hidden; margin:.15rem 0 .35rem;}
.bar .fill { height:100%; border-radius:7px; transition:width .3s; }

/* fault chips */
.faultwrap { margin:.1rem 0 .2rem; }
.fault-chip { display:inline-block; background:#fff5f5; border:1px solid #f3c2c2;
  color:#9b1c1c; padding:.22rem .55rem; border-radius:9px; margin:.18rem .25rem 0 0;
  font-size:.82rem; }

/* formula box */
.formula { font-family: ui-monospace,SFMono-Regular,Consolas,monospace;
  font-size:.8rem; background:#f6f8fa; border:1px solid #eaecef; border-radius:6px;
  padding:.35rem .55rem; margin:.1rem 0 .35rem; color:#24292f; }

/* score bullets */
.scorelist { margin:.1rem 0 .4rem 1.05rem; padding:0; }
.scorelist li { margin:.12rem 0; font-size:.88rem; line-height:1.4; }

/* worked example callout */
.example { background:#eef6ff; border:1px solid #cfe3fb; border-left:4px solid #2f6fed;
  border-radius:8px; padding:.55rem .75rem; margin:.45rem 0 .15rem; font-size:.86rem;
  color:#1f2a44; line-height:1.45; }
.example b { color:#2f6fed; }

/* evidence */
.ev { font-family: ui-monospace,SFMono-Regular,Consolas,monospace; font-size:.78rem;
  background:#f6f8fa; border:1px solid #e9ecef; border-radius:6px; padding:.22rem .5rem;
  margin:.14rem 0; white-space:pre-wrap; word-break:break-word; }

/* section labels inside a card */
.lbl { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
  color:#8a909a; margin:.5rem 0 .12rem; }
.val { font-size:.9rem; color:#222; }

/* page detail header */
.pagehdr { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap; margin:.2rem 0 .5rem;}
.pagehdr .pq { font-weight:800; font-size:1.05rem; }
.colcap { font-weight:700; color:#444; margin:0 0 .35rem; font-size:.95rem; }
</style>
"""


def inject_css() -> None:
    import streamlit as st

    st.markdown(APP_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    import streamlit as st

    st.markdown(
        "<div class='hero'><h1>🔎 OCR Quality Validation — Chandra</h1>"
        "<p>Statistically validates page-wise Chandra OCR output, pinpoints where "
        "errors are, and tells you which pages a human must check.</p></div>",
        unsafe_allow_html=True,
    )


def render_status_legend() -> None:
    import streamlit as st

    st.markdown(
        "<div class='legend'>"
        f"<span><span class='sw' style='background:{_STATUS_COLOR['good']}'></span>Good</span>"
        f"<span><span class='sw' style='background:{_STATUS_COLOR['warn']}'></span>Warn</span>"
        f"<span><span class='sw' style='background:{_STATUS_COLOR['bad']}'></span>Bad</span>"
        "<span><span class='conf Definite'>DEFINITE</span> structural fact</span>"
        "<span><span class='conf Statistical'>STATISTICAL</span> threshold flag</span>"
        "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Metrics Guide
# --------------------------------------------------------------------------- #
def render_metrics_guide() -> None:
    """Self-documenting reference: how scoring works, which file feeds each
    metric, and a per-metric table with formulas + examples."""
    import streamlit as st

    from ocr_qa.config import DEFAULT_CONFIG
    from ocr_qa.metrics.catalog import CATALOG, FILE_MAP, GROUPS, weight_for
    from ocr_qa.metrics.registry import METRICS

    by_key = {m.key: m for m in METRICS}
    cfg = DEFAULT_CONFIG

    st.subheader("📖 Metrics Guide")
    st.caption("What each statistical metric measures, the file it reads, its "
               "formula, and a worked example.")

    # ---- how scoring works (tier-based decisioning) ----
    with st.container(border=True):
        st.markdown("#### How a page is sent to review")
        st.markdown(
            f"Metrics are split into two tiers (to avoid false positives):\n"
            f"- **DEFINITE (hard) triggers** — one is enough to flag a page: "
            f"inference-failed content (IFR) · major block/page-count mismatch (BCC) · "
            f"empty content blocks (EMP) · markup that breaks extraction (HMW) · "
            f"unexpected duplicate page (DUP) · mojibake/encoding corruption (CED) · "
            f"definite repetition loop / reliable native-CER / ≥2 artefact "
            f"disagreements (XSA) · severe geometry or table content loss (BGC/TSI).\n"
            f"- **SOFT signals** (LVR, OOV, WSA, PPL, TTR, PSW, SFC, KL-divergence) "
            f"are supporting evidence only — they flag a page **only when ≥ "
            f"{cfg.soft_fail_threshold} of them fail together on a text-rich page**.\n\n"
            f"**Page → review** if (1 DEFINITE failure) **OR** (≥ "
            f"{cfg.soft_fail_threshold} soft failures) **OR** PQS < "
            f"{cfg.pqs_backstop:.0f} (safety net). "
            f"**Document = PASSED** if DQS ≥ {cfg.t_doc:.0f}, faulty ratio ≤ "
            f"{cfg.doc_flag_ratio*100:.0f}%, and no page hit a DEFINITE condition.\n\n"
            f"Soft text-statistics metrics apply **only to prose-bearing pages with "
            f"enough text** (page-type / token gating), and native CER/WER is used "
            f"**only when the native PDF text layer is reliable**."
        )
        render_status_legend()

    # ---- file -> metric map ----
    with st.expander("Which file feeds which metric", expanded=False):
        st.dataframe(
            [{"File / resource": f, "Contains": c, "Used by": u}
             for f, c, u in FILE_MAP],
            use_container_width=True, hide_index=True,
        )

    # ---- summary table ----
    from ocr_qa.config import HARD_METRICS

    st.markdown("#### All 17 metrics at a glance")
    rows = []
    for i, e in enumerate(CATALOG, start=1):
        m = by_key.get(e["key"])
        rows.append({
            "#": i,
            "Key": e["key"],
            "Metric": m.name if m else e["key"],
            "Group": GROUPS[e["group"]][0],
            "Trigger": "DEFINITE (hard)" if e["key"] in HARD_METRICS else "soft signal",
            "Reads": ", ".join(e["reads"][:3]) + ("…" if len(e["reads"]) > 3 else ""),
            "Weight": f"{weight_for(e['key']):.2f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True, height=430)

    # ---- per-group detail cards ----
    st.markdown("#### Detail & examples")
    last_group = None
    for e in CATALOG:
        if e["group"] != last_group:
            last_group = e["group"]
            label, color = GROUPS[e["group"]]
            st.markdown(
                f"<div class='guide-group' style='border-left-color:{color}'>"
                f"{label}</div>", unsafe_allow_html=True,
            )
        m = by_key.get(e["key"])
        with st.container(border=True):
            st.markdown(
                f"<div class='mhead'>"
                f"<span class='mname'>{html.escape(m.name if m else e['key'])}</span>"
                f"<span class='mkey'>{e['key']}</span>"
                f"<span class='conf {e['confidence']}'>{e['confidence'].upper()}</span>"
                f"<span class='mscore' style='font-size:.8rem;color:#888'>"
                f"weight {weight_for(e['key']):.2f}</span></div>",
                unsafe_allow_html=True,
            )
            if m and m.what_it_measures:
                st.markdown(f"<span class='val'>{html.escape(m.what_it_measures)}</span>",
                            unsafe_allow_html=True)
            st.markdown("<div class='lbl'>Formula</div>"
                        f"<div class='formula'>{html.escape(e['formula'])}</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='lbl'>Reads</div>"
                        + "".join(f"<span class='src-chip'>{html.escape(s)}</span>"
                                  for s in e["reads"]),
                        unsafe_allow_html=True)
            st.markdown(f"<div class='example'>💡 <b>Example:</b> "
                        f"{html.escape(e['example'])}</div>", unsafe_allow_html=True)


def _badge(status: str) -> str:
    c = status_color(status)
    return f"<span class='pill' style='background:{c}'>{_STATUS_LABEL.get(status, status.upper())}</span>"


# --------------------------------------------------------------------------- #
# gauge
# --------------------------------------------------------------------------- #
def score_gauge(score: float, title: str, threshold: Optional[float] = None):
    import plotly.graph_objects as go

    color = "#1a9850" if score >= 75 else ("#e8870c" if score >= 50 else "#d73027")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": title, "font": {"size": 15}},
            number={"suffix": "/100", "font": {"size": 30}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#fde0dd"},
                    {"range": [50, 75], "color": "#fff7bc"},
                    {"range": [75, 100], "color": "#e5f5e0"},
                ],
                "threshold": (
                    {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold,
                    }
                    if threshold is not None
                    else None
                ),
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=46, b=8))
    return fig


# --------------------------------------------------------------------------- #
# metric card
# --------------------------------------------------------------------------- #
def _fmt_thresholds(threshold: dict) -> str:
    if not threshold:
        return "—"
    parts = []
    for k, v in threshold.items():
        if isinstance(v, float):
            v = f"{v:g}"
        parts.append(f"{k} = {v}")
    return " · ".join(parts)


def _render_card_body(m: MetricResult, page_no=None) -> None:
    """Minimalistic body: Formula → What it means → bulleted score → page example."""
    import streamlit as st

    # 1) formula
    st.markdown(
        f"<div class='lbl'>Formula</div>"
        f"<div class='formula'>{html.escape(m.how_computed)}</div>",
        unsafe_allow_html=True,
    )
    # 2) what it means
    st.markdown(
        f"<div class='lbl'>What it means</div>"
        f"<span class='val'>{html.escape(m.what_it_measures)}</span>",
        unsafe_allow_html=True,
    )
    # 3) bulleted explanation of the score
    bullets = [f"<li><b>Value</b>: {m.raw_value:g}</li>"]
    if m.threshold:
        bullets.append(f"<li><b>Thresholds</b>: {html.escape(_fmt_thresholds(m.threshold))}</li>")
    bullets.append(f"<li>{html.escape(m.justification)}</li>")
    if m.reliability == "low":
        bullets.append("<li><i>Low reliability</i> — short page / language undetermined.</li>")
    st.markdown(
        f"<div class='lbl'>What the score means</div>"
        f"<ul class='scorelist'>{''.join(bullets)}</ul>",
        unsafe_allow_html=True,
    )
    # 3b) sub-metric breakdown (XSA) — explainable split, not one worst-of score
    if m.submetrics:
        _LBL = {"repetition": "Repetition", "json_md": "JSON↔MD",
                "json_chunk": "JSON↔chunk", "json_html": "JSON↔HTML",
                "native_cer": "Native CER"}
        chips = "".join(
            f"<span class='s'>{_LBL.get(k, k)}: <b>{v:.0f}</b></span>"
            for k, v in m.submetrics.items()
        )
        st.markdown(f"<div class='lbl'>Sub-metric breakdown</div>"
                    f"<div class='submx'>{chips}</div>", unsafe_allow_html=True)
    # 4) this-page worked example
    if m.example:
        where = f"page&nbsp;{page_no}" if page_no is not None else "this page"
        st.markdown(
            f"<div class='example'>💡 <b>How {where} got {m.score:.0f}/100:</b> "
            f"{html.escape(m.example)}</div>",
            unsafe_allow_html=True,
        )
    # compact evidence (up to 3) — no nested expander
    if m.evidence:
        chips = "".join(f"<div class='ev'>{html.escape(e)}</div>" for e in m.evidence[:3])
        st.markdown(f"<div class='lbl'>Evidence</div>{chips}", unsafe_allow_html=True)


def render_metric_card(m: MetricResult, expanded: bool, page_no=None) -> None:
    import streamlit as st

    color = status_color(m.status)
    with st.container(border=True):
        extra = ""
        # tier chip: DEFINITE-eligible (hard) vs supporting-only (soft)
        if m.tier == "hard":
            extra += " <span class='tier-hard'>DEFINITE-eligible</span>"
        else:
            extra += " <span class='tier-soft'>soft signal</span>"
        if m.hard_fail:
            extra += " <span class='tag-hard'>HARD FAIL</span>"
        if m.reliability == "low":
            extra += " <span class='tag-rel'>low reliability</span>"
        if not m.applicable:
            extra += " <span class='tag-rel'>excluded from score</span>"
        st.markdown(
            f"<div class='mhead'>"
            f"<span class='dot' style='background:{color}'></span>"
            f"<span class='mname'>{html.escape(m.name)}</span>"
            f"<span class='mkey'>{m.key}</span>{_badge(m.status)}{extra}"
            f"<span class='mscore' style='color:{color}'>{m.score:.0f}"
            f"<small>/100</small></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        width = max(0, min(100, m.score))
        st.markdown(
            f"<div class='bar'><div class='fill' style='width:{width}%;"
            f"background:{color}'></div></div>",
            unsafe_allow_html=True,
        )
        if expanded:
            _render_card_body(m, page_no)
        else:
            with st.expander("Show formula, meaning & example", expanded=False):
                _render_card_body(m, page_no)


def render_metric_list(metrics: list[MetricResult], page_no=None) -> None:
    """Failing metrics expanded first, then passing collapsed (Sec 7)."""
    import streamlit as st

    failing = [m for m in metrics if m.status in ("bad", "warn") or m.hard_fail]
    passing = [m for m in metrics if m not in failing]
    failing.sort(key=lambda m: (not m.hard_fail, m.score))

    if failing:
        st.markdown(f"##### ⛔ Failing / borderline ({len(failing)})")
        for m in failing:
            render_metric_card(m, expanded=True, page_no=page_no)
    if passing:
        st.markdown(f"##### ✅ Passing ({len(passing)})")
        for m in sorted(passing, key=lambda m: -m.score):
            render_metric_card(m, expanded=False, page_no=page_no)


# --------------------------------------------------------------------------- #
# fault chips
# --------------------------------------------------------------------------- #
def render_page_context(report: PageReport) -> None:
    """Decision-support context strip (page type, density, native reliability,
    reading-order confidence, soft-signal count)."""
    import streamlit as st

    nat = "reliable" if report.native_reliable else "n/a / unreliable"
    chips = [
        f"<span class='ctx'>Page type: <b>{html.escape(report.page_type)}</b></span>",
        f"<span class='ctx'>Tokens: <b>{report.token_count}</b></span>",
        f"<span class='ctx'>Text density: <b>{report.text_density:.0f}</b>/Mpx²</span>",
        f"<span class='ctx'>Native text: <b>{nat}</b></span>",
        f"<span class='ctx'>Reading-order conf: <b>{report.reading_order_conf:.2f}</b></span>",
        f"<span class='ctx'>Soft signals: <b>{report.soft_fail_count}</b></span>",
    ]
    st.markdown(f"<div class='ctxstrip'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_fault_chips(problems: list[str]) -> None:
    import streamlit as st

    if not problems:
        st.markdown("<span class='val'>No specific faults recorded.</span>",
                    unsafe_allow_html=True)
        return
    chips = "".join(f"<span class='fault-chip'>{html.escape(p)}</span>" for p in problems)
    st.markdown(f"<div class='faultwrap'>{chips}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# verdict badge
# --------------------------------------------------------------------------- #
def render_verdict_badge(doc: DocumentReport) -> None:
    import streamlit as st

    if doc.document_verdict == "PASSED":
        st.success(f"### ✅ DOCUMENT VERDICT: PASSED\n\n{doc.verdict_reason}")
    else:
        st.error(
            f"### ❌ DOCUMENT VERDICT: FAILED — NEEDS HUMAN REVIEW\n\n{doc.verdict_reason}"
        )


# --------------------------------------------------------------------------- #
# faulty table data
# --------------------------------------------------------------------------- #
def worst_metrics(report: PageReport, n: int = 3) -> str:
    applicable = [m for m in report.metrics if m.applicable]
    worst = sorted(applicable, key=lambda m: (not m.hard_fail, m.score))[:n]
    return ", ".join(f"{m.key} {m.score:.0f}" for m in worst)


# --------------------------------------------------------------------------- #
# Error localization: collect, highlight, overlay
# --------------------------------------------------------------------------- #
def collect_locations(report: PageReport) -> list[dict]:
    """All error locations on a page, each tagged with its source metric."""
    out = []
    for m in report.metrics:
        if not m.applicable:
            continue
        for L in m.locations:
            out.append({**L, "metric": m.key, "metric_name": m.name})
    return out


def render_error_locations(report: PageReport) -> None:
    """'Where the errors are' panel, split DEFINITE vs STATISTICAL."""
    import streamlit as st

    locs = collect_locations(report)
    if not locs:
        st.caption("No localizable errors on this page.")
        return
    definite = [L for L in locs if L["confidence"] == "definite"]
    statistical = [L for L in locs if L["confidence"] != "definite"]

    def _row(L):
        label = _KIND_LABEL.get(L["kind"], L["kind"])
        where = f" · block `{L['block_id']}`" if L.get("block_id") else ""
        bbox = f" · bbox {[int(x) for x in L['bbox']]}" if L.get("bbox") else ""
        snip = f" — “{html.escape(L['snippet'])}”" if L.get("snippet") else ""
        st.markdown(
            f"<div class='ev'>[{L['metric']}] <b>{html.escape(label)}</b>{where}"
            f"{bbox}{snip}</div>",
            unsafe_allow_html=True,
        )

    if definite:
        st.markdown(
            f"<span class='pill' style='background:#b00020'>DEFINITE ×{len(definite)}</span> "
            "<span class='val'>reproducible structural facts from the files</span>",
            unsafe_allow_html=True,
        )
        for L in definite:
            _row(L)
    if statistical:
        st.markdown(
            f"<span class='pill' style='background:#e8870c'>STATISTICAL ×{len(statistical)}"
            "</span> <span class='val'>threshold-based flags (see numbers on each card)</span>",
            unsafe_allow_html=True,
        )
        for L in statistical:
            _row(L)


def highlight_snippets(text: str, report: PageReport) -> str:
    """Return HTML with definite=red / statistical=amber <mark>s around the
    offending text spans found on the page."""
    if not text:
        return ""
    spans: list[tuple[str, str]] = []  # (snippet, confidence)
    seen = set()
    for L in collect_locations(report):
        if L["kind"] in TEXT_HIGHLIGHT_KINDS and L.get("snippet"):
            s = L["snippet"]
            key = s.lower()
            if key not in seen and len(s) >= 2:
                seen.add(key)
                spans.append((s, L["confidence"]))
    # longest first so substrings don't pre-empt larger matches
    spans.sort(key=lambda x: -len(x[0]))
    out = html.escape(text)
    for snippet, conf in spans:
        color = "#ffd6d6" if conf == "definite" else "#ffeccc"
        bd = "#b00020" if conf == "definite" else "#e8870c"
        esc = re.escape(html.escape(snippet))
        mark = (
            f"<mark style='background:{color};border-bottom:2px solid {bd};"
            f"padding:0 1px'>\\g<0></mark>"
        )
        try:
            out = re.sub(esc, mark, out, flags=re.IGNORECASE)
        except re.error:
            pass
    return f"<div style='white-space:pre-wrap;line-height:1.6'>{out}</div>"


def overlay_error_boxes(image_path: str, page, report: PageReport) -> Optional[str]:
    """Draw rectangles for located block bboxes on a copy of the page image.
    Returns the new image path, or None on failure. Coordinates are scaled from
    the OCR Page bbox to the image's pixel size."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    locs = [L for L in collect_locations(report) if L.get("bbox") and len(L["bbox"]) == 4]
    if not locs:
        return None
    pbbox = page.raw.get("bbox") if (page and isinstance(page.raw, dict)) else None
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return None
    iw, ih = im.size
    if isinstance(pbbox, list) and len(pbbox) == 4 and pbbox[2] and pbbox[3]:
        pw, ph = float(pbbox[2]), float(pbbox[3])
    else:
        pw, ph = iw, ih  # assume already in pixel space
    sx, sy = iw / pw, ih / ph
    draw = ImageDraw.Draw(im, "RGBA")
    for L in locs:
        x0, y0, x1, y1 = L["bbox"]
        col = (176, 0, 32, 255) if L["confidence"] == "definite" else (232, 135, 12, 255)
        draw.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=col, width=4)
    out = image_path.rsplit(".", 1)[0] + f"_hl_{report.page_no}.png"
    try:
        im.save(out)
        return out
    except Exception:
        return None


def faulty_table_rows(doc: DocumentReport) -> list[dict]:
    rows = []
    faulty = [r for r in doc.page_reports if r.verdict == "review"]
    faulty.sort(key=lambda r: r.page_score)  # worst first
    for r in faulty:
        rows.append(
            {
                "Page": r.page_no,
                "PQS": round(r.page_score, 1),
                "Worst metrics": worst_metrics(r),
                "Top problem": r.problems[0] if r.problems else "—",
            }
        )
    return rows
