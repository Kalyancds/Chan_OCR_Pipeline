"""Full PDF report (Sec 8 extension).

Layout (ReportLab Platypus):
  1. Main title + report metadata (generated time, tool, doc id).
  2. Original source document name.
  3. PASSED / FAILED verdict badge + numeric reason + totals.
  4. Page-count reconciliation (if available).
  5. "Findings" — why it failed: per faulty page, the failing metrics WITH
     justification and the located errors.
  6. Full per-page summary table + mean-metric rollup.
  7. Metrics reference — every metric with a LaTeX-rendered formula
     (Matplotlib mathtext, no TeX install needed), the EXACT file(s) used to
     compute it, how it is computed, confidence and an example.

Formulas are rendered as crisp images via Matplotlib's mathtext so no external
LaTeX toolchain is required.
"""

from __future__ import annotations

import datetime as _dt
from io import BytesIO
from typing import Optional

from ocr_qa.config import Config
from ocr_qa.metrics.catalog import CATALOG, FILE_MAP, GROUPS, weight_for
from ocr_qa.metrics.registry import METRICS
from ocr_qa.models import DocumentReport, IngestedDocument

# --------------------------------------------------------------------------- #
# LaTeX (mathtext) formula per metric key
# --------------------------------------------------------------------------- #
LATEX: dict[str, str] = {
    "IFR": r"\mathrm{IFR}=\frac{n_{fail}}{n_{blocks}}",
    "BCC": r"\frac{|\,N_{json}-N_{meta}\,|}{N_{meta}}",
    "BGC": r"\frac{n_{bad}}{n_{blocks}}",
    "EMP": r"\frac{n_{empty}}{n_{content}}",
    "HMW": r"\frac{n_{malformed}}{n_{html}}",
    "TSI": r"\bar d=\rho_{rag}+\tfrac{1}{2}\rho_{empty}+p_{hdr}+p_{span}",
    "FIG": r"\frac{n_{nocap}}{n_{fig}}",
    "LVR": r"\mathrm{LVR}=\frac{n_{valid}}{n_{tokens}}",
    "OOV": r"\frac{n_{gibberish}}{n_{oov}}",
    "SFC": r"|\,r_{obs}-r_{exp}\,|",
    "CED": r"D_{KL}(P\|Q)=\sum_i P_i\,\log_2\frac{P_i}{Q_i}",
    "PPL": r"\frac{1}{N}\sum_i-\log_2 P(c_i\mid c_{i-2}c_{i-1})",
    "TTR": r"\mathrm{TTR}=\frac{n_{types}}{n_{tokens}}",
    "WSA": r"\frac{n_{merge}+n_{seg}+n_{hyph}+n_{alnum}}{n_{tokens}}",
    "PSW": r"\frac{n_{defects}}{n_{words}/100}",
    "XSA": r"\min(s_{rep},s_{md},s_{chunk},s_{html},s_{cer}),\;\mathrm{CER}=\frac{S+D+I}{N}",
    "DUP": r"\max_{j\neq i} J(p_i,p_j),\;J=\frac{|A\cap B|}{|A\cup B|}",
}


# --------------------------------------------------------------------------- #
# Matplotlib mathtext -> PNG flowable
# --------------------------------------------------------------------------- #
_FORMULA_CACHE: dict[str, tuple[bytes, float, float]] = {}


def _formula_png(expr: str, fontsize: int = 15) -> Optional[tuple[bytes, int, int]]:
    if expr in _FORMULA_CACHE:
        data, w, h = _FORMULA_CACHE[expr]
        return data, w, h
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${expr}$", fontsize=fontsize)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                    pad_inches=0.05, transparent=True)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image as PILImage

        im = PILImage.open(buf)
        w, h = im.size
        data = buf.getvalue()
        _FORMULA_CACHE[expr] = (data, w, h)
        return data, w, h
    except Exception:
        return None


def _formula_flowable(key: str, target_h_pt: float = 20.0):
    from reportlab.platypus import Image as RLImage, Paragraph

    expr = LATEX.get(key)
    png = _formula_png(expr) if expr else None
    if not png:
        from reportlab.lib.styles import getSampleStyleSheet
        return Paragraph(expr or "&mdash;", getSampleStyleSheet()["BodyText"])
    data, w, h = png
    scale = target_h_pt / h
    return RLImage(BytesIO(data), width=w * scale, height=h * scale)


# --------------------------------------------------------------------------- #
# Report builder
# --------------------------------------------------------------------------- #
def build_pdf_report(
    doc: DocumentReport,
    ingested: Optional[IngestedDocument],
    config: Config,
    out_path: str,
    page_audit: Optional[dict] = None,
    source_name: Optional[str] = None,
) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
        TableStyle,
    )

    by_key = {m.key: m for m in METRICS}
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=22, spaceAfter=4)
    SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=10,
                         textColor=colors.HexColor("#555555"), alignment=TA_CENTER)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=14,
                        textColor=colors.HexColor("#1f2a44"), spaceBefore=10, spaceAfter=4)
    BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9, leading=12)
    SMALL = ParagraphStyle("SMALL", parent=ss["BodyText"], fontSize=7.5, leading=9.5)
    CELL = ParagraphStyle("CELL", parent=ss["BodyText"], fontSize=8, leading=10)
    CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")

    story: list = []
    doc_name = source_name or (ingested.doc_id if ingested else None) or doc.doc_id
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- 1. title + metadata ----
    story.append(Paragraph("OCR Quality Validation Report", H1))
    story.append(Paragraph("Statistical validation of page-wise Chandra OCR output",
                           SUB))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#2f6fed"), thickness=2))
    story.append(Spacer(1, 8))
    meta = [
        ["Generated", now],
        ["Document ID", doc.doc_id],
        ["Tool", "OCR QA (Chandra) — 17 statistical metrics"],
    ]
    mt = Table(meta, colWidths=[3.2 * cm, 13 * cm])
    mt.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(mt)
    story.append(Spacer(1, 10))

    # ---- 2. original document name ----
    story.append(Paragraph("Original document", H2))
    story.append(Paragraph(f"<b>{_esc(doc_name)}</b>", BODY))
    story.append(Spacer(1, 8))

    # ---- 3. verdict badge ----
    passed = doc.document_verdict == "PASSED"
    badge_color = colors.HexColor("#1a9850") if passed else colors.HexColor("#d73027")
    verdict_text = "PASSED" if passed else "FAILED — NEEDS HUMAN REVIEW"
    vt = Table([[Paragraph(
        f"<font color='white' size=16><b>DOCUMENT VERDICT: {verdict_text}</b></font>",
        ParagraphStyle("V", parent=BODY, alignment=TA_CENTER))]],
        colWidths=[16.2 * cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    story.append(vt)
    story.append(Spacer(1, 4))
    story.append(Paragraph(_esc(doc.verdict_reason), BODY))
    story.append(Spacer(1, 6))

    s = doc.summary
    totals = [["Document score (DQS)", f"{doc.document_score:.0f} / 100"],
              ["Total pages", str(s.get("total_pages", 0))],
              ["Passed", str(s.get("passed_pages", 0))],
              ["Need review (faulty)", str(s.get("faulty_pages", 0))],
              ["Faulty ratio", f"{s.get('faulty_ratio', 0) * 100:.0f}%"],
              ["Pages with HARD conditions", str(len(s.get("hard_pages", [])))]]
    tt = Table(totals, colWidths=[6 * cm, 4 * cm])
    tt.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tt)

    if s.get("failed_rules"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Rule(s) that caused FAIL:</b>", BODY))
        for fr in s["failed_rules"]:
            story.append(Paragraph(f"• {_esc(fr)}", BODY))

    # ---- 4. reconciliation ----
    if page_audit:
        story.append(Paragraph("Page-count reconciliation", H2))
        rows = [[Paragraph("<b>Artefact</b>", CELL), Paragraph("<b>Pages</b>", CELL)]]
        for k, v in page_audit["counts"].items():
            rows.append([Paragraph(_esc(k), CELL),
                         Paragraph(str(v) if v is not None else "—", CELL)])
        sp = page_audit.get("source_pages")
        rows.append([Paragraph("SOURCE (rendered)", CELL),
                     Paragraph(str(sp) if sp is not None else "—", CELL)])
        rt = Table(rows, colWidths=[7 * cm, 3 * cm])
        rt.setStyle(_grid_style(colors))
        story.append(rt)
        for m in page_audit.get("messages", []):
            story.append(Paragraph(f"• {_esc(m)}", SMALL))

    # ---- 5. findings: why it failed ----
    story.append(Paragraph("Findings — parameters that failed & why", H2))
    faulty = [r for r in doc.page_reports if r.verdict == "review"]
    faulty.sort(key=lambda r: r.page_no)
    if not faulty:
        story.append(Paragraph("No page was flagged for review — every page met "
                               "all thresholds and tripped no HARD condition.", BODY))
    else:
        story.append(Paragraph(
            f"{len(faulty)} page(s) need human review. For each, the failing "
            f"metrics are listed with the numeric justification and the located "
            f"error (block id / snippet, tagged DEFINITE or STATISTICAL).", BODY))
        for r in faulty:
            story.append(Spacer(1, 5))
            story.append(Paragraph(
                f"<b>Page {r.page_no}</b> — PQS {r.page_score:.0f}/100", BODY))
            head = [Paragraph("<b>Metric</b>", SMALL),
                    Paragraph("<b>Score</b>", SMALL),
                    Paragraph("<b>Why (justification)</b>", SMALL),
                    Paragraph("<b>Where / confidence</b>", SMALL)]
            data = [head]
            failing = [m for m in r.metrics
                       if m.applicable and (m.status in ("bad", "warn") or m.hard_fail)]
            failing.sort(key=lambda m: (not m.hard_fail, m.score))
            for m in failing:
                where = "; ".join(
                    f"{L.get('block_id') or L['kind']} ({L['confidence'][:3]})"
                    for L in m.locations[:3]) or "—"
                tag = " [HARD]" if m.hard_fail else ""
                data.append([
                    Paragraph(f"{m.key}{tag}", SMALL),
                    Paragraph(f"{m.score:.0f}", SMALL),
                    Paragraph(_esc(m.justification), SMALL),
                    Paragraph(_esc(where), SMALL),
                ])
            ft = Table(data, colWidths=[2 * cm, 1.3 * cm, 8.4 * cm, 4.5 * cm], repeatRows=1)
            ft.setStyle(_grid_style(colors, header=True))
            story.append(ft)

    # ---- 6. full per-page summary + metric rollup ----
    story.append(PageBreak())
    story.append(Paragraph("Per-page summary (all pages)", H2))
    head = [Paragraph("<b>Page</b>", SMALL), Paragraph("<b>PQS</b>", SMALL),
            Paragraph("<b>Verdict</b>", SMALL), Paragraph("<b>Top problem / lowest metric</b>", SMALL)]
    data = [head]
    for r in sorted(doc.page_reports, key=lambda r: r.page_no):
        if r.problems:
            note = r.problems[0]
        else:
            app = [m for m in r.metrics if m.applicable]
            lo = min(app, key=lambda m: m.score) if app else None
            note = f"lowest: {lo.key} {lo.score:.0f}" if lo else "—"
        data.append([Paragraph(str(r.page_no), SMALL),
                     Paragraph(f"{r.page_score:.0f}", SMALL),
                     Paragraph(r.verdict.upper(), SMALL),
                     Paragraph(_esc(note), SMALL)])
    pt = Table(data, colWidths=[1.4 * cm, 1.4 * cm, 2.2 * cm, 11.2 * cm], repeatRows=1)
    pt.setStyle(_grid_style(colors, header=True))
    story.append(pt)

    # mean metric rollup
    story.append(Paragraph("Mean metric score across the document", H2))
    from collections import defaultdict
    agg = defaultdict(list)
    for r in doc.page_reports:
        for m in r.metrics:
            if m.applicable:
                agg[m.key].append(m.score)
    head = [Paragraph("<b>Metric</b>", SMALL), Paragraph("<b>Mean</b>", SMALL),
            Paragraph("<b>Min</b>", SMALL), Paragraph("<b>Applied on</b>", SMALL)]
    data = [head]
    for e in CATALOG:
        v = agg.get(e["key"], [])
        if v:
            data.append([Paragraph(e["key"], SMALL),
                         Paragraph(f"{sum(v) / len(v):.1f}", SMALL),
                         Paragraph(f"{min(v):.1f}", SMALL),
                         Paragraph(f"{len(v)} pages", SMALL)])
    rt = Table(data, colWidths=[3 * cm, 3 * cm, 3 * cm, 4 * cm], repeatRows=1)
    rt.setStyle(_grid_style(colors, header=True))
    story.append(rt)

    # ---- 7. metrics reference ----
    story.append(PageBreak())
    story.append(Paragraph("Metrics reference — what each parameter is, the file "
                           "used, and how it is computed", H2))
    story.append(Paragraph(
        "Every metric below states the exact file(s) it reads. <b>output.json</b> "
        "is the backbone (block tree: text, HTML, bbox, inference_failed); the "
        "other artefacts are used only by the metrics that cross-check or count.",
        BODY))
    story.append(Spacer(1, 4))

    # file -> metric source map
    head = [Paragraph("<b>File / resource</b>", SMALL),
            Paragraph("<b>Contains</b>", SMALL),
            Paragraph("<b>Used by</b>", SMALL)]
    data = [head] + [[Paragraph(_esc(f), SMALL), Paragraph(_esc(c), SMALL),
                      Paragraph(_esc(u), SMALL)] for f, c, u in FILE_MAP]
    fmt = Table(data, colWidths=[4.6 * cm, 7.4 * cm, 4.2 * cm], repeatRows=1)
    fmt.setStyle(_grid_style(colors, header=True))
    story.append(fmt)
    story.append(Spacer(1, 8))

    # per-metric reference table (grouped)
    head = [Paragraph("<b>Metric</b>", SMALL), Paragraph("<b>Formula</b>", SMALL),
            Paragraph("<b>File(s) used</b>", SMALL),
            Paragraph("<b>What it measures / how computed</b>", SMALL),
            Paragraph("<b>Conf.</b>", SMALL)]
    data = [head]
    for e in CATALOG:
        m = by_key.get(e["key"])
        name = _esc(m.name if m else e["key"])
        meas = _esc(m.what_it_measures if m else "") + f"<br/><i>{_esc(e['formula'])}</i>"
        files = "<br/>".join(f"• {_esc(x)}" for x in e["reads"])
        data.append([
            Paragraph(f"<b>{e['key']}</b><br/>{name}", SMALL),
            _formula_flowable(e["key"]),
            Paragraph(files, SMALL),
            Paragraph(meas, SMALL),
            Paragraph(e["confidence"], SMALL),
        ])
    mref = Table(data, colWidths=[3.0 * cm, 4.0 * cm, 3.4 * cm, 5.0 * cm, 1.4 * cm],
                 repeatRows=1)
    mref.setStyle(_grid_style(colors, header=True))
    story.append(mref)

    # examples
    story.append(Paragraph("Worked examples", H2))
    for e in CATALOG:
        story.append(Paragraph(f"<b>{e['key']}</b> — {_esc(e['example'])}", SMALL))

    SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.6 * cm, rightMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title=f"OCR QA Report — {doc.doc_id}",
    ).build(story, onLaterPages=_footer, onFirstPage=_footer)
    return out_path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _esc(text) -> str:
    import html as _h
    return _h.escape(str(text if text is not None else ""))


def _grid_style(colors, header: bool = False):
    from reportlab.platypus import TableStyle
    cmds = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
         [colors.white, colors.HexColor("#f6f8fa")]),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a44")),
                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
    return TableStyle(cmds)


def _footer(canvas, doc_tmpl):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColorRGB(0.5, 0.5, 0.5)
    canvas.drawString(1.6 * 28.35, 1.0 * 28.35,
                      "OCR Quality Validation — Chandra · deterministic, templated report")
    canvas.drawRightString(195 * 2.835, 1.0 * 28.35, f"Page {doc_tmpl.page}")
    canvas.restoreState()
