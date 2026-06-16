"""OCR Quality Validation — Streamlit app (Sec 0, 7).

Ordered stages:  Ingest → Processing → Faulty Pages → Verdict → Export.
Both OCR paths (Run Chandra / OCR-already-done) converge on the identical
compute + report path.

Navigation uses a persisted stage selector (not st.tabs) so Prev/Next inside the
Faulty Pages view never bounces the user back to Ingest on rerun.

Run:  streamlit run ocr_qa/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# Make `import ocr_qa` work whether launched as a script or a module.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

from ocr_qa.config import DEFAULT_WEIGHTS, Config
from ocr_qa.ingestion.normalizer import IngestionError, Normalizer
from ocr_qa.ocr.chandra_parser import audit_page_counts
from ocr_qa.ocr.client import MockChandraClient, RealChandraClient
from ocr_qa.review.export import build_review_package, send_for_review
from ocr_qa.review.pdf_report import build_pdf_report
from ocr_qa.scoring.aggregate import build_document_report
from ocr_qa.ui.components import (
    faulty_table_rows,
    highlight_snippets,
    inject_css,
    overlay_error_boxes,
    render_error_locations,
    render_fault_chips,
    render_hero,
    render_metric_list,
    render_metrics_guide,
    render_page_context,
    render_verdict_badge,
    score_gauge,
    status_color,
)

st.set_page_config(page_title="OCR Quality Validation", layout="wide", page_icon="🔎")

STAGES = ["Ingest", "Processing", "Faulty Pages", "Passed Pages", "Verdict",
          "Export", "Guide"]
STAGE_ICONS = ["📥", "⚙️", "🚩", "✅", "⚖️", "📦", "📖"]
# stages reachable before any analysis has run
ALWAYS_ON = {"Ingest", "Guide"}


# --------------------------------------------------------------------------- #
# session state
# --------------------------------------------------------------------------- #
def _init_state():
    ss = st.session_state
    ss.setdefault("normalizer", Normalizer(work_dir=tempfile.mkdtemp(prefix="ocrqa_ui_")))
    ss.setdefault("doc", None)
    ss.setdefault("ingested", None)
    ss.setdefault("pages", None)
    ss.setdefault("parse_log", None)
    ss.setdefault("faulty_pick", None)
    ss.setdefault("stage", "Ingest")
    ss.setdefault("human_calls", {})
    ss.setdefault("page_audit", None)
    ss.setdefault("passed_pick", None)
    ss.setdefault("config", None)


_init_state()


# --------------------------------------------------------------------------- #
# sidebar — config (Sec 9, all editable)
# --------------------------------------------------------------------------- #
def sidebar_config() -> Config:
    st.sidebar.header("⚙️ Thresholds & weights")
    cfg = Config()

    with st.sidebar.expander("Verdict thresholds", expanded=True):
        cfg.t_page = st.slider("Page review threshold (T_page)", 0, 100, int(cfg.t_page))
        cfg.t_doc = st.slider("Document pass threshold (T_doc)", 0, 100, int(cfg.t_doc))
        cfg.doc_flag_ratio = st.slider(
            "Max faulty ratio", 0.0, 1.0, float(cfg.doc_flag_ratio), 0.01
        )

    with st.sidebar.expander("Lexical / entropy / shape"):
        cfg.lexical_good = st.slider("LVR good", 0.5, 1.0, float(cfg.lexical_good), 0.01)
        cfg.lexical_warn = st.slider("LVR hard-fail below", 0.5, 1.0, float(cfg.lexical_warn), 0.01)
        cfg.ced_mojibake_bad = st.number_input("Mojibake hard-fail count", 1, 50, int(cfg.ced_mojibake_bad))
        cfg.wsa_good = st.slider("WSA good", 0.0, 0.2, float(cfg.wsa_good), 0.01)
        cfg.wsa_bad = st.slider("WSA bad", 0.0, 0.5, float(cfg.wsa_bad), 0.01)

    with st.sidebar.expander("Cross-source / repetition / block-count"):
        cfg.xsa_repeat_max = st.number_input("Repeat n-gram hard-flag (>x)", 1, 50, int(cfg.xsa_repeat_max))
        cfg.xsa_cer_bad = st.slider("CER hard-fail", 0.0, 1.0, float(cfg.xsa_cer_bad), 0.01)
        cfg.bcc_tolerance = st.slider("Block-count tolerance ±", 0.0, 1.0, float(cfg.bcc_tolerance), 0.01)
        cfg.short_page_token_floor = st.number_input(
            "Short-page token floor", 1, 500, int(cfg.short_page_token_floor)
        )

    with st.sidebar.expander("Perplexity backend"):
        cfg.perplexity_backend = st.selectbox(
            "Backend", ["ngram", "kenlm", "gpt2"],
            index=["ngram", "kenlm", "gpt2"].index(cfg.perplexity_backend),
            help="Default 'ngram' needs no model download.",
        )

    with st.sidebar.expander("Metric weights"):
        weights = {}
        for k, v in DEFAULT_WEIGHTS.items():
            weights[k] = st.number_input(f"{k}", 0.0, 1.0, float(v), 0.01, key=f"w_{k}")
        cfg.weights = weights
        st.caption(f"Σ weights = {sum(weights.values()):.2f} (renormalised at runtime)")

    # ---- re-analyze with current settings (no re-upload / re-parse needed) ----
    st.sidebar.divider()
    if st.session_state.get("pages"):
        st.sidebar.markdown("**🔄 Re-run with current settings**")
        st.sidebar.caption(
            "Edit the weights/thresholds above, then click. Your edits only take "
            "effect when you click — the analysis never restarts mid-edit."
        )
        if st.sidebar.button("🔄 Re-analyze", type="primary", use_container_width=True):
            _rerun_analysis(cfg)
    else:
        st.sidebar.caption("Run an analysis (Ingest tab) to enable re-analyze with "
                           "adjusted weights.")

    return cfg


def _rerun_analysis(cfg: Config) -> None:
    """Recompute the report from the already-parsed pages using the current
    sidebar weights/thresholds — no re-upload, no re-parse, no re-render."""
    ss = st.session_state
    if not ss.get("pages"):
        return
    bar = st.sidebar.progress(0.0, text="Re-analyzing…")
    total = len(ss.pages)

    def cb(i, tot, page_no):
        bar.progress(i / tot, text=f"Page {i}/{tot} — re-scoring (17 metrics)")

    doc = build_document_report(ss.pages, ss.ingested, cfg, progress_cb=cb)
    bar.empty()
    ss.doc = doc
    ss.config = cfg
    ss.human_calls = {}
    ss.faulty_pick = doc.faulty_pages[0] if doc.faulty_pages else None
    ss.passed_pick = None
    ss.stage = "Faulty Pages" if doc.faulty_pages else "Verdict"
    st.rerun()


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def run_analysis(cfg, source_file, ocr_status, oj, omd, omm, omc=None, omh=None):
    ss = st.session_state
    norm: Normalizer = ss.normalizer

    # 1) ingest source -> page images + native text (optional but recommended)
    ingested = None
    if source_file is not None:
        try:
            src_path = norm.save_upload(source_file)
            ingested = norm.ingest(src_path)
        except IngestionError as exc:
            st.warning(f"Ingestion degraded: {exc}")
        except Exception as exc:  # never a stack trace to the user
            st.warning(f"Could not render source images: {exc}")

    # 2) OCR -> list[PageOCR]  (+ preprocessing page-count audit)
    ss.page_audit = None
    try:
        if ocr_status == "Run Chandra OCR":
            client = RealChandraClient()
            try:
                pages = client.run(norm.save_upload(source_file) if source_file else "")
            except NotImplementedError as exc:
                st.error(str(exc))
                return
        else:  # OCR already done
            if not oj:
                st.error("Please upload at least output.json.")
                return
            client = MockChandraClient()
            bundle = client.get_bundle(oj, omd, omm, omc, omh)
            pages = client.parser.parse(
                bundle.output_json, bundle.output_md, bundle.metadata,
                bundle.output_chunks, bundle.output_html,
            )
            src_pages = len(ingested.pages) if (ingested and ingested.pages) else None
            ss.page_audit = audit_page_counts(
                bundle.output_json, bundle.output_md, bundle.metadata,
                bundle.output_chunks, bundle.output_html, src_pages,
            )
        ss.parse_log = client.parse_log
    except Exception as exc:
        st.error(f"Could not parse Chandra output: {exc}")
        return

    if not pages:
        st.error("No pages were parsed from the OCR output.")
        return

    # 3) compute (with per-page progress)
    bar = st.progress(0.0, text="Starting…")
    total = len(pages)

    def cb(i, tot, page_no):
        bar.progress(i / tot, text=f"Page {i}/{tot} — 17 metrics")

    doc = build_document_report(pages, ingested, cfg, progress_cb=cb)
    bar.empty()

    ss.doc, ss.ingested, ss.pages = doc, ingested, pages
    ss.config = cfg
    ss.human_calls = {}
    ss.faulty_pick = doc.faulty_pages[0] if doc.faulty_pages else None
    ss.stage = "Faulty Pages" if doc.faulty_pages else "Verdict"
    st.rerun()


# --------------------------------------------------------------------------- #
# stage: Ingest
# --------------------------------------------------------------------------- #
def stage_ingest(cfg):
    st.subheader("📥 Ingest")
    st.caption("Upload the source document and the Chandra OCR output.")
    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            source_file = st.file_uploader(
                "Source document (PDF / DOCX / PPTX / ZIP-of-images) — for page images",
                type=["pdf", "docx", "pptx", "zip"],
            )
        with c2:
            ocr_status = st.selectbox("OCR status", ["OCR already done", "Run Chandra OCR"])

        oj = omd = omm = omc = omh = None
        if ocr_status == "OCR already done":
            st.markdown("**Chandra output** — `output.json` required; the rest optional:")
            d1, d2 = st.columns(2)
            oj = d1.file_uploader("output.json", type=["json"], key="oj")
            omd = d2.file_uploader("output.md", type=["md", "txt"], key="omd")
            d3, d4 = st.columns(2)
            omm = d3.file_uploader("output.metadata.json (optional)", type=["json"], key="omm")
            omc = d4.file_uploader("output_chunks.json (optional)", type=["json"], key="omc")
            omh = st.file_uploader(
                "output.html (optional — richer OCR view + JSON↔HTML check)",
                type=["html", "htm"], key="omh",
            )
            st.caption("metadata is also read from inside output.json if not supplied.")
        else:
            st.info(
                "Run-Chandra-OCR is a stub in this build (no live endpoint). The "
                "parse path is real — use 'OCR already done' to validate output now."
            )

    if st.button("🔬 Analyze", type="primary", use_container_width=True):
        run_analysis(cfg, source_file, ocr_status, oj, omd, omm, omc, omh)


# --------------------------------------------------------------------------- #
# stage: Processing
# --------------------------------------------------------------------------- #
def render_page_audit():
    """Show the preprocessing page-count reconciliation across all artefacts."""
    ss = st.session_state
    audit = ss.page_audit
    if not audit:
        return
    status = audit["status"]
    head = " ".join(audit["messages"])
    if status == "ok":
        st.success(f"📐 **Page reconciliation:** {head}")
        expanded = False
    elif status == "source_mismatch":
        st.warning(f"⚠️ **Page-count mismatch (source vs OCR):** {head}")
        expanded = True
    else:  # ocr_inconsistent
        st.error(f"❌ **OCR artefacts disagree on page count:** {head}")
        expanded = True

    with st.expander("Per-artefact page counts", expanded=expanded):
        rows = [
            {"Artefact": k, "Pages": (str(v) if v is not None else "— not provided")}
            for k, v in audit["counts"].items()
        ]
        rows.append({
            "Artefact": "SOURCE (rendered images)",
            "Pages": str(audit["source_pages"]) if audit["source_pages"] is not None
            else "— source not uploaded",
        })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        for m in audit["messages"]:
            st.markdown(f"- {m}")


def stage_processing():
    st.subheader("⚙️ Processing")
    doc = st.session_state.doc
    if not doc:
        st.info("Run **Analyze** on the Ingest tab first.")
        return
    render_page_audit()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(
            score_gauge(doc.document_score, "Document Quality Score (DQS)",
                        threshold=doc.summary["thresholds"]["t_doc"]),
            use_container_width=True,
        )
    with c2:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total pages", doc.summary["total_pages"])
        m2.metric("Passed", doc.summary["passed_pages"])
        m3.metric("Need review", doc.summary["faulty_pages"])
        if doc.faulty_pages:
            if st.button("🚩 Go to faulty pages", type="primary"):
                st.session_state.stage = "Faulty Pages"
                st.rerun()
        else:
            st.success("No pages need review.")
    if st.session_state.parse_log:
        with st.expander("Parser log (found vs missing per page)"):
            st.dataframe(st.session_state.parse_log, use_container_width=True, height=300)


# --------------------------------------------------------------------------- #
# stage: Faulty Pages (centerpiece)
# --------------------------------------------------------------------------- #
def _pqs_pill(score: float) -> str:
    c = status_color("good" if score >= 75 else ("warn" if score >= 50 else "bad"))
    return (
        f"<span class='pill' style='background:{c}'>PQS {score:.0f}/100</span>"
    )


def _page_walker(state_key: str, page_nos: list[int], noun: str) -> int:
    """Prev | select | Next walker over ``page_nos`` (already in display order).
    Persists the pick in session_state[state_key]; never resets the stage."""
    ss = st.session_state
    if ss.get(state_key) not in page_nos:
        ss[state_key] = page_nos[0]
    cur = page_nos.index(ss[state_key])
    c_prev, c_sel, c_next = st.columns([1, 4, 1])
    if c_prev.button("◀ Prev", disabled=cur == 0, use_container_width=True,
                     key=f"{state_key}_prev"):
        ss[state_key] = page_nos[cur - 1]
        st.rerun()
    sel = c_sel.selectbox(
        f"{noun} ({cur + 1} of {len(page_nos)})", page_nos, index=cur,
        label_visibility="collapsed", key=f"{state_key}_sel",
    )
    if sel != ss[state_key]:
        ss[state_key] = sel
        st.rerun()
    if c_next.button("Next ▶", disabled=cur == len(page_nos) - 1,
                     use_container_width=True, key=f"{state_key}_next"):
        ss[state_key] = page_nos[cur + 1]
        st.rerun()
    return ss[state_key]


def render_page_detail(report, page, is_faulty: bool) -> None:
    """Shared side-by-side detail used by both Faulty and Passed stages."""
    pick = report.page_no
    badge = ("<span class='pill' style='background:#d73027'>NEEDS HUMAN REVIEW</span>"
             if is_faulty else
             "<span class='pill' style='background:#1a9850'>PASSED</span>")
    st.markdown(
        f"<div class='pagehdr'><span class='pq'>Page {pick}</span>"
        f"{_pqs_pill(report.page_score)}{badge}</div>",
        unsafe_allow_html=True,
    )
    render_page_context(report)

    if is_faulty:
        st.markdown("**Faults on this page**")
        render_fault_chips(report.problems)
        with st.container(border=True):
            st.markdown("**📍 Where the errors are** "
                        "(DEFINITE = structural fact · STATISTICAL = threshold flag)")
            render_error_locations(report)
    else:
        st.success("All applicable metrics passed for this page — details below verify why.")

    st.divider()

    # ---- side-by-side: original image | OCR extract ----
    left, right = st.columns(2)
    with left:
        cap = "📄 Original page"
        img_path = (report.image_path
                    if report.image_path and os.path.exists(report.image_path) else None)
        overlay = False
        if img_path and is_faulty:
            overlay = st.checkbox("🔲 Overlay error boxes on image", value=False,
                                  key=f"ov_{pick}",
                                  help="Draws located block bboxes. Positions are in "
                                       "OCR page coordinates; if the image is misaligned "
                                       "(page-count mismatch) boxes may be off.")
        st.markdown(f"<div class='colcap'>{cap}</div>", unsafe_allow_html=True)
        if img_path:
            shown = img_path
            if overlay:
                hl = overlay_error_boxes(img_path, page, report)
                shown = hl or img_path
            with st.container(border=True, height=600):
                st.image(shown, use_column_width=True)
        else:
            st.info("No rendered image for this page (source not uploaded, not "
                    "renderable, or OCR page count exceeds the source).")
    with right:
        st.markdown("<div class='colcap'>📝 OCR extract</div>", unsafe_allow_html=True)
        has_html = bool(page and page.html_text)
        opts = ((["🔦 Errors highlighted"] if is_faulty else [])
                + (["🌐 HTML (rendered)"] if has_html else [])
                + ["📝 Markdown", "</> Raw"])
        view = st.radio("view", opts, horizontal=True, label_visibility="collapsed",
                        key=f"view_{pick}")
        with st.container(border=True, height=600):
            if page is None:
                st.caption("No extract available for this page.")
            elif view.startswith("🔦"):
                # locations are derived from the JSON blocks, so highlight the
                # JSON page text (md may diverge and miss the spans).
                base = page.text or page.md_text or ""
                st.markdown(highlight_snippets(base, report), unsafe_allow_html=True)
            elif view.startswith("🌐"):
                st.html(page.html_text)
            elif view.startswith("📝"):
                md = page.md_text or page.text
                st.markdown(md if md else "_(no markdown/text)_")
            else:
                st.code(page.html_text or page.md_text or page.text or "(empty)",
                        language="html" if has_html else "markdown")
        if is_faulty:
            st.caption("🔦 = offending spans marked (red = definite, amber = statistical). "
                       "HTML view always matches the OCR.")

    st.divider()
    st.markdown("#### 🔬 Statistical metrics (17)")
    render_metric_list(report.metrics, page_no=pick)


def stage_faulty():
    st.subheader("🚩 Faulty Pages")
    ss = st.session_state
    doc = ss.doc
    if not doc:
        st.info("Run **Analyze** first.")
        return

    render_page_audit()
    page_by_no = {p.page_no: p for p in (ss.pages or [])}
    faulty = [r for r in doc.page_reports if r.verdict == "review"]
    passed = doc.summary["passed_pages"]
    if not faulty:
        st.success(f"🎉 No faulty pages — all {passed} page(s) passed. "
                   "See the **Passed Pages** tab to verify them.")
        return

    order = st.radio("Order", ["Sequential (page order)", "Worst PQS first"],
                     horizontal=True, key="faulty_order")
    if order.startswith("Sequential"):
        faulty.sort(key=lambda r: r.page_no)
    else:
        faulty.sort(key=lambda r: r.page_score)

    st.markdown(f"**{len(faulty)} page(s) need review** · {passed} passed "
                "(see Passed Pages tab).")
    with st.expander("Faulty-page table (worst first)", expanded=False):
        st.dataframe(faulty_table_rows(doc), use_container_width=True,
                     hide_index=True, height=240)

    faulty_nos = [r.page_no for r in faulty]
    pick = _page_walker("faulty_pick", faulty_nos, "Faulty page")
    report = next(r for r in doc.page_reports if r.page_no == pick)

    render_page_detail(report, page_by_no.get(pick), is_faulty=True)

    # human call
    with st.container(border=True):
        st.markdown("**🧑‍⚖️ Your call after comparing original vs OCR extract**")
        options = ["— not reviewed —", "✅ OCR looks OK (false alarm)", "❌ Confirm OCR failed"]
        prev = ss.human_calls.get(pick, options[0])
        choice = st.radio("human_call", options,
                          index=options.index(prev) if prev in options else 0,
                          horizontal=True, label_visibility="collapsed",
                          key=f"human_{pick}")
        ss.human_calls[pick] = choice
        decided = sum(1 for v in ss.human_calls.values() if v != options[0])
        st.caption(f"You have reviewed {decided}/{len(faulty_nos)} faulty pages.")


def stage_passed():
    st.subheader("✅ Passed Pages")
    ss = st.session_state
    doc = ss.doc
    if not doc:
        st.info("Run **Analyze** first.")
        return
    page_by_no = {p.page_no: p for p in (ss.pages or [])}
    passed = sorted((r for r in doc.page_reports if r.verdict == "ok"),
                    key=lambda r: r.page_no)
    if not passed:
        st.warning("No pages passed — every page is in the Faulty Pages tab.")
        return
    st.markdown(f"**{len(passed)} page(s) passed.** Inspect any to verify all 14 "
                "metrics support the pass (highest-PQS first available in the table).")
    with st.expander("Passed-page table (best first)", expanded=False):
        rows = [{"Page": r.page_no, "PQS": round(r.page_score, 1),
                 "Lowest metric": (lambda ms: f"{ms.key} {ms.score:.0f}")(
                     min((m for m in r.metrics if m.applicable),
                         key=lambda m: m.score))}
                for r in sorted(passed, key=lambda r: -r.page_score)]
        st.dataframe(rows, use_container_width=True, hide_index=True, height=240)

    passed_nos = [r.page_no for r in passed]
    pick = _page_walker("passed_pick", passed_nos, "Passed page")
    report = next(r for r in doc.page_reports if r.page_no == pick)
    render_page_detail(report, page_by_no.get(pick), is_faulty=False)


# --------------------------------------------------------------------------- #
# stage: Verdict
# --------------------------------------------------------------------------- #
def stage_verdict():
    st.subheader("⚖️ Verdict")
    doc = st.session_state.doc
    if not doc:
        st.info("Run **Analyze** first.")
        return
    render_verdict_badge(doc)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(
            score_gauge(doc.document_score, "DQS",
                        threshold=doc.summary["thresholds"]["t_doc"]),
            use_container_width=True,
        )
    with c2:
        s = doc.summary
        m1, m2, m3 = st.columns(3)
        m1.metric("Pages", s["total_pages"])
        m2.metric("Passed", s["passed_pages"])
        m3.metric("Faulty", s["faulty_pages"])
        with st.container(border=True):
            if s["failed_rules"]:
                st.markdown("**Rule(s) that caused FAIL:**")
                for fr in s["failed_rules"]:
                    st.markdown(f"- :red[{fr}]")
            else:
                st.markdown("**No failing rules.** ✅")

    if doc.faulty_pages:
        with st.expander(f"Faulty pages & top problem ({len(doc.faulty_pages)})",
                         expanded=True):
            for r in sorted((r for r in doc.page_reports if r.verdict == "review"),
                            key=lambda r: r.page_score):
                top = r.problems[0] if r.problems else "—"
                st.markdown(f"- **Page {r.page_no}** (PQS {r.page_score:.0f}): {top}")


# --------------------------------------------------------------------------- #
# stage: Export
# --------------------------------------------------------------------------- #
def stage_export():
    st.subheader("📦 Export")
    ss = st.session_state
    doc = ss.doc
    if not doc:
        st.info("Run **Analyze** first.")
        return
    cfg = ss.get("config") or Config()

    # ---- full PDF report (available for PASS or FAIL) ----
    with st.container(border=True):
        st.markdown("**📄 Full PDF report**")
        st.caption("Title + document name + PASS/FAIL + a findings table (why each "
                   "page failed, with justification) + per-page summary + a metrics "
                   "reference with LaTeX formulas and the exact file used per metric.")
        if st.button("📄 Generate PDF report", type="primary"):
            out = os.path.join(ss.normalizer.work_dir, f"{doc.doc_id}_report.pdf")
            try:
                src_name = ss.ingested.doc_id if ss.ingested else doc.doc_id
                with st.spinner("Rendering report (formulas, tables)…"):
                    path = build_pdf_report(doc, ss.ingested, cfg, out,
                                            page_audit=ss.page_audit, source_name=src_name)
                with open(path, "rb") as fh:
                    st.download_button(
                        "⬇️ Download PDF report", fh.read(),
                        file_name=os.path.basename(path), mime="application/pdf",
                        type="primary",
                    )
                st.success(f"Built {os.path.basename(path)}.")
            except Exception as exc:
                st.error(f"Could not build PDF report: {exc}")

    # ---- faulty-pages review package (only when there are faulty pages) ----
    if not doc.faulty_pages:
        st.success("No faulty pages — the document passed. (PDF report still "
                   "available above.) ✅")
        return
    with st.container(border=True):
        st.markdown("**📦 Human-review package (faulty pages only)**")
        st.caption("review.json / review.csv / review.md / page images, zipped.")
        if st.button("📦 Build review package"):
            out = os.path.join(ss.normalizer.work_dir, f"{doc.doc_id}_review.zip")
            try:
                path = build_review_package(doc, ss.ingested, out)
                with open(path, "rb") as fh:
                    st.download_button(
                        "⬇️ Download review package (zip)", fh.read(),
                        file_name=os.path.basename(path), mime="application/zip",
                    )
                st.success(f"Built {os.path.basename(path)} with "
                           f"{len(doc.faulty_pages)} faulty page(s).")
                res = send_for_review(path)
                st.caption(f"send_for_review hook: {res['status']} — {res['reason']}")
            except Exception as exc:
                st.error(f"Could not build package: {exc}")


# --------------------------------------------------------------------------- #
# navigation + main
# --------------------------------------------------------------------------- #
def render_nav():
    ss = st.session_state
    cols = st.columns(len(STAGES))
    for i, name in enumerate(STAGES):
        disabled = name not in ALWAYS_ON and ss.doc is None
        active = ss.stage == name
        if cols[i].button(
            f"{STAGE_ICONS[i]} {i + 1}·{name}",
            key=f"nav_{name}",
            use_container_width=True,
            type="primary" if active else "secondary",
            disabled=disabled and not active,
        ):
            ss.stage = name
            st.rerun()


def main():
    inject_css()
    render_hero()
    cfg = sidebar_config()
    render_nav()
    st.divider()

    stage = st.session_state.stage
    if stage == "Ingest":
        stage_ingest(cfg)
    elif stage == "Processing":
        stage_processing()
    elif stage == "Faulty Pages":
        stage_faulty()
    elif stage == "Passed Pages":
        stage_passed()
    elif stage == "Verdict":
        stage_verdict()
    elif stage == "Export":
        stage_export()
    elif stage == "Guide":
        render_metrics_guide()


if __name__ == "__main__":
    main()
