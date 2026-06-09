"""Full PDF report (Sec 8 extension).

Layout (ReportLab Platypus), typeset in **Computer Modern (LaTeX CMR)**:
  1. Main title + report metadata (generated time, tool, doc id).
  2. Original source document name.
  3. PASSED / FAILED verdict badge + numeric reason + totals.
  4. Page-count reconciliation (per-artefact vs source).
  5. Findings — why it failed: per faulty page, the failing metrics WITH
     justification and the located error (block id / snippet, DEFINITE/STAT).
  6. Full per-page summary table + mean-metric rollup.
  7. Metrics reference — every metric with a LaTeX-rendered formula
     (Matplotlib mathtext, ``cm`` fontset), the EXACT file(s) used, how it is
     computed, confidence and an example.

Rendering: the PRIMARY path generates a LaTeX document and compiles it with
``pdflatex`` (genuine Computer Modern / Latin Modern typesetting via lmodern,
native math formulas, full \\textwidth tables). If pdflatex is unavailable or
fails, it falls back to a ReportLab build (clean serif + Matplotlib mathtext
formulas) so a report is always produced.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import os
import shutil
import subprocess
import tempfile
import unicodedata
from io import BytesIO
from typing import Optional

from ocr_qa.config import Config
from ocr_qa.metrics.catalog import CATALOG, FILE_MAP, weight_for
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
    "XSA": r"\min(s_{rep},s_{md},s_{chunk},s_{html},s_{cer});\;\mathrm{CER}=\frac{S+D+I}{N}",
    "DUP": r"\max_{j\neq i} J(p_i,p_j);\;J=\frac{|A\cap B|}{|A\cup B|}",
}

# --------------------------------------------------------------------------- #
# text cleaning so Computer Modern (limited glyph set) renders cleanly
# --------------------------------------------------------------------------- #
_SYM = {
    "≤": "<=", "≥": ">=", "×": "x", "÷": "/", "↔": "<->", "→": "->", "←": "<-",
    "—": "-", "–": "-", "•": "*", "…": "...", "‖": "||", "≠": "!=", "·": ".",
    "“": '"', "”": '"', "‘": "'", "’": "'", "≈": "~", "√": "sqrt", "∩": " and ",
    "∪": " or ", "²": "^2", "±": "+/-", "→": "->",
}


def _clean(text) -> str:
    """ASCII-fold for the CMR body font: map symbols, transliterate accents."""
    s = str(text if text is not None else "")
    for k, v in _SYM.items():
        s = s.replace(k, v)
    # transliterate any remaining non-ASCII (accents etc.) to closest ASCII
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s


def _esc(text) -> str:
    return _html.escape(_clean(text))


# --------------------------------------------------------------------------- #
# fonts: register Computer Modern for the whole report
# --------------------------------------------------------------------------- #
# ReportLab is the FALLBACK renderer (used only if pdflatex is unavailable).
# It uses a clean serif (Times) — NOT cmr10, whose TeX OT1 encoding mangles
# ASCII '<', '>', '_'. The primary pdflatex path gives true Computer Modern.
_FONTS = {"regular": "Times-Roman", "bold": "Times-Bold", "mono": "Courier"}


def _register_fonts() -> dict:
    return _FONTS


# --------------------------------------------------------------------------- #
# Matplotlib mathtext -> PNG flowable (Computer Modern math fontset)
# --------------------------------------------------------------------------- #
_FORMULA_CACHE: dict[str, tuple[bytes, int, int]] = {}


def _formula_png(expr: str, fontsize: int = 16, dpi: int = 220):
    if expr in _FORMULA_CACHE:
        return _FORMULA_CACHE[expr]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["mathtext.fontset"] = "cm"  # Computer Modern math
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${expr}$", fontsize=fontsize, color="black")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    pad_inches=0.06, transparent=True)
        plt.close(fig)
        from PIL import Image as PILImage

        buf.seek(0)
        w, h = PILImage.open(buf).size
        out = (buf.getvalue(), w, h)
        _FORMULA_CACHE[expr] = out
        return out
    except Exception:
        return None


def _formula_flowable(key: str, max_w_pt: float, max_h_pt: float):
    """Render the formula and fit it WITHIN the column box (prevents the
    overflow-clipping that hid earlier formulas)."""
    from reportlab.platypus import Image as RLImage, Paragraph
    from reportlab.lib.styles import ParagraphStyle

    expr = LATEX.get(key)
    png = _formula_png(expr) if expr else None
    if not png:
        return Paragraph(_esc(expr or "—"),
                         ParagraphStyle("f", fontName=_FONTS["regular"], fontSize=8))
    data, w, h = png
    dpi = 220.0
    w_pt, h_pt = w * 72.0 / dpi, h * 72.0 / dpi
    scale = min(max_h_pt / h_pt, max_w_pt / w_pt)  # fit both dimensions
    return RLImage(BytesIO(data), width=w_pt * scale, height=h_pt * scale)


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
    """Build the PDF. Prefer a real LaTeX (pdflatex) build for genuine Computer
    Modern typesetting and native math; fall back to ReportLab if pdflatex is
    not available or fails."""
    src = source_name or (ingested.doc_id if ingested else None) or doc.doc_id
    try:
        if _have_pdflatex():
            return _build_latex(doc, config, out_path, page_audit, src)
    except Exception:
        pass  # fall back to ReportLab
    return _build_reportlab(doc, ingested, config, out_path, page_audit, src)


def _build_reportlab(
    doc: DocumentReport,
    ingested: Optional[IngestedDocument],
    config: Config,
    out_path: str,
    page_audit: Optional[dict] = None,
    source_name: Optional[str] = None,
) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
        TableStyle,
    )

    f = _register_fonts()
    REG, BOLD = f["regular"], f["bold"]
    by_key = {m.key: m for m in METRICS}

    LEFT = RIGHT = 1.5 * cm
    AVAIL = A4[0] - LEFT - RIGHT  # full text width

    def P(text, size=10, bold=False, color=None, align=0, leading=None):
        st = ParagraphStyle(
            "p", fontName=BOLD if bold else REG, fontSize=size,
            leading=leading or size * 1.25, alignment=align,
            textColor=color or colors.black,
        )
        return Paragraph(text, st)

    H1 = ParagraphStyle("H1", fontName=BOLD, fontSize=22, leading=26,
                        alignment=TA_CENTER, textColor=colors.HexColor("#16203a"))
    SUBS = ParagraphStyle("SUB", fontName=REG, fontSize=10.5, alignment=TA_CENTER,
                          textColor=colors.HexColor("#555555"))
    H2 = ParagraphStyle("H2", fontName=BOLD, fontSize=14, leading=17,
                        spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1f2a44"))
    BODY = ParagraphStyle("BODY", fontName=REG, fontSize=10.5, leading=13.5)
    WHITE = colors.white

    story: list = []
    doc_name = source_name or (ingested.doc_id if ingested else None) or doc.doc_id
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def grid(header=True):
        cmds = [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8ced6")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1),
             [colors.white, colors.HexColor("#eef2f7")]),
        ]
        if header:
            cmds += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2a44"))]
        return TableStyle(cmds)

    def hcell(txt):  # white header cell (Paragraph carries the colour)
        return P(f"<b>{_esc(txt)}</b>", 9.5, bold=True, color=WHITE)

    # ---- 1. title + metadata ----
    story.append(P("OCR Quality Validation Report", 22, bold=True, align=TA_CENTER,
                   color=colors.HexColor("#16203a")))
    story.append(Paragraph("Statistical validation of page-wise Chandra OCR output", SUBS))
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#2f6fed"), thickness=2))
    story.append(Spacer(1, 8))
    meta = [[P("<b>Generated</b>", 10, bold=True), P(_esc(now))],
            [P("<b>Document ID</b>", 10, bold=True), P(_esc(doc.doc_id))],
            [P("<b>Tool</b>", 10, bold=True),
             P("OCR QA (Chandra) - 17 statistical metrics")]]
    mt = Table(meta, colWidths=[AVAIL * 0.25, AVAIL * 0.75])
    mt.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story.append(mt)
    story.append(Spacer(1, 8))

    # ---- 2. original document ----
    story.append(Paragraph("Original document", H2))
    story.append(P(f"<b>{_esc(doc_name)}</b>", 12, bold=True))
    story.append(Spacer(1, 8))

    # ---- 3. verdict ----
    passed = doc.document_verdict == "PASSED"
    badge_color = colors.HexColor("#1a9850") if passed else colors.HexColor("#c0241c")
    verdict_text = "PASSED" if passed else "FAILED - NEEDS HUMAN REVIEW"
    vt = Table([[P(f"<b>DOCUMENT VERDICT: {verdict_text}</b>", 16, bold=True,
                   color=WHITE, align=TA_CENTER)]], colWidths=[AVAIL])
    vt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), badge_color),
                            ("TOPPADDING", (0, 0), (-1, -1), 9),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    story.append(vt)
    story.append(Spacer(1, 5))
    story.append(P(_esc(doc.verdict_reason)))
    story.append(Spacer(1, 6))

    s = doc.summary
    totals = [["Document score (DQS)", f"{doc.document_score:.0f} / 100"],
              ["Total pages", str(s.get("total_pages", 0))],
              ["Passed", str(s.get("passed_pages", 0))],
              ["Need review (faulty)", str(s.get("faulty_pages", 0))],
              ["Faulty ratio", f"{s.get('faulty_ratio', 0) * 100:.0f}%"],
              ["Pages with HARD conditions", str(len(s.get("hard_pages", [])))]]
    tt = Table([[P(f"<b>{a}</b>", 10, bold=True), P(b, 10)] for a, b in totals],
               colWidths=[AVAIL * 0.5, AVAIL * 0.5])
    tt.setStyle(grid(header=False))
    story.append(tt)

    if s.get("failed_rules"):
        story.append(Spacer(1, 6))
        story.append(P("<b>Rule(s) that caused FAIL:</b>", 10, bold=True))
        for fr in s["failed_rules"]:
            story.append(P(f"- {_esc(fr)}"))

    # ---- 4. reconciliation ----
    if page_audit:
        story.append(Paragraph("Page-count reconciliation", H2))
        rows = [[hcell("Artefact"), hcell("Pages")]]
        for k, v in page_audit["counts"].items():
            rows.append([P(_esc(k)), P(str(v) if v is not None else "-")])
        sp = page_audit.get("source_pages")
        rows.append([P("SOURCE (rendered)"), P(str(sp) if sp is not None else "-")])
        rt = Table(rows, colWidths=[AVAIL * 0.7, AVAIL * 0.3])
        rt.setStyle(grid())
        story.append(rt)
        for m in page_audit.get("messages", []):
            story.append(P(f"- {_esc(m)}", 9))

    # ---- 5. findings ----
    story.append(Paragraph("Findings - parameters that failed and why", H2))
    faulty = sorted((r for r in doc.page_reports if r.verdict == "review"),
                    key=lambda r: r.page_no)
    if not faulty:
        story.append(P("No page was flagged for review - every page met all "
                       "thresholds and tripped no HARD condition."))
    else:
        story.append(P(f"{len(faulty)} page(s) need human review. For each, the "
                       "failing metrics are listed with the numeric justification and "
                       "the located error (block id / snippet, tagged DEFINITE or "
                       "STATISTICAL)."))
        cw = [AVAIL * 0.13, AVAIL * 0.08, AVAIL * 0.52, AVAIL * 0.27]
        for r in faulty:
            story.append(Spacer(1, 5))
            story.append(P(f"<b>Page {r.page_no}</b> - PQS {r.page_score:.0f}/100",
                           10.5, bold=True))
            data = [[hcell("Metric"), hcell("Score"), hcell("Why (justification)"),
                     hcell("Where / confidence")]]
            failing = [m for m in r.metrics
                       if m.applicable and (m.status in ("bad", "warn") or m.hard_fail)]
            failing.sort(key=lambda m: (not m.hard_fail, m.score))
            for m in failing:
                where = "; ".join(
                    f"{L.get('block_id') or L['kind']} ({L['confidence'][:3]})"
                    for L in m.locations[:3]) or "-"
                tag = " [HARD]" if m.hard_fail else ""
                data.append([P(f"{m.key}{tag}", 9), P(f"{m.score:.0f}", 9),
                             P(_esc(m.justification), 9), P(_esc(where), 9)])
            ft = Table(data, colWidths=cw, repeatRows=1)
            ft.setStyle(grid())
            story.append(ft)

    # ---- 6. per-page summary + rollup ----
    story.append(PageBreak())
    story.append(Paragraph("Per-page summary (all pages)", H2))
    data = [[hcell("Page"), hcell("PQS"), hcell("Verdict"),
             hcell("Top problem / lowest metric")]]
    for r in sorted(doc.page_reports, key=lambda r: r.page_no):
        if r.problems:
            note = r.problems[0]
        else:
            app = [m for m in r.metrics if m.applicable]
            lo = min(app, key=lambda m: m.score) if app else None
            note = f"lowest: {lo.key} {lo.score:.0f}" if lo else "-"
        data.append([P(str(r.page_no), 9), P(f"{r.page_score:.0f}", 9),
                     P(r.verdict.upper(), 9), P(_esc(note), 9)])
    pt = Table(data, colWidths=[AVAIL * 0.1, AVAIL * 0.1, AVAIL * 0.15, AVAIL * 0.65],
               repeatRows=1)
    pt.setStyle(grid())
    story.append(pt)

    story.append(Paragraph("Mean metric score across the document", H2))
    from collections import defaultdict
    agg = defaultdict(list)
    for r in doc.page_reports:
        for m in r.metrics:
            if m.applicable:
                agg[m.key].append(m.score)
    data = [[hcell("Metric"), hcell("Mean"), hcell("Min"), hcell("Applied on")]]
    for e in CATALOG:
        v = agg.get(e["key"], [])
        if v:
            data.append([P(e["key"], 9), P(f"{sum(v) / len(v):.1f}", 9),
                         P(f"{min(v):.1f}", 9), P(f"{len(v)} pages", 9)])
    rt = Table(data, colWidths=[AVAIL * 0.25] * 4, repeatRows=1)
    rt.setStyle(grid())
    story.append(rt)

    # ---- 7. metrics reference ----
    story.append(PageBreak())
    story.append(Paragraph("Metrics reference - what each parameter is, the file "
                           "used, and how it is computed", H2))
    story.append(P("Every metric below states the exact file(s) it reads. "
                   "<b>output.json</b> is the backbone (block tree: text, HTML, bbox, "
                   "inference_failed); the other artefacts are used only by the "
                   "metrics that cross-check or count."))
    story.append(Spacer(1, 4))

    data = [[hcell("File / resource"), hcell("Contains"), hcell("Used by")]]
    for fpath, c, u in FILE_MAP:
        data.append([P(_esc(fpath), 9), P(_esc(c), 9), P(_esc(u), 9)])
    fmt = Table(data, colWidths=[AVAIL * 0.27, AVAIL * 0.45, AVAIL * 0.28], repeatRows=1)
    fmt.setStyle(grid())
    story.append(fmt)
    story.append(Spacer(1, 8))

    fcol = AVAIL * 0.24
    data = [[hcell("Metric"), hcell("Formula"), hcell("File(s) used"),
             hcell("What it measures / how computed"), hcell("Conf.")]]
    for e in CATALOG:
        m = by_key.get(e["key"])
        name = _esc(m.name if m else e["key"])
        meas = _esc(m.what_it_measures if m else "") + f"<br/><i>{_esc(e['formula'])}</i>"
        files = "<br/>".join(f"- {_esc(x)}" for x in e["reads"])
        data.append([
            P(f"<b>{e['key']}</b><br/>{name}", 9),
            _formula_flowable(e["key"], max_w_pt=fcol - 8, max_h_pt=34),
            P(files, 8.5),
            P(meas, 8.5),
            P(_esc(e["confidence"]), 8.5),
        ])
    mref = Table(data, colWidths=[AVAIL * 0.17, fcol, AVAIL * 0.20, AVAIL * 0.31,
                                  AVAIL * 0.08], repeatRows=1)
    mref.setStyle(grid())
    story.append(mref)

    story.append(Paragraph("Worked examples", H2))
    for e in CATALOG:
        story.append(P(f"<b>{e['key']}</b> - {_esc(e['example'])}", 9))

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont(REG, 7.5)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        canvas.drawString(LEFT, 1.0 * cm,
                          "OCR Quality Validation - Chandra - deterministic report")
        canvas.drawRightString(A4[0] - RIGHT, 1.0 * cm, f"Page {_doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        out_path, pagesize=A4, leftMargin=LEFT, rightMargin=RIGHT,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title=f"OCR QA Report - {doc.doc_id}",
    ).build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


# =========================================================================== #
# LaTeX (pdflatex) builder — PRIMARY path: genuine Computer Modern + native math
# =========================================================================== #
def _have_pdflatex() -> bool:
    return shutil.which("pdflatex") is not None


def _tex(text) -> str:
    """Escape text for LaTeX. ASCII-folds first (so exotic unicode never trips
    inputenc), then escapes LaTeX specials. Renders correctly under T1+lmodern."""
    s = _clean(text)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("{", r"\{"), ("}", r"\}"), ("$", r"\$"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("<", r"\textless{}"), (">", r"\textgreater{}"),
                 ("|", r"\textbar{}")):
        s = s.replace(a, b)
    return s


_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[table]{xcolor}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{lastpage}
\usepackage{fancyhdr}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\renewcommand{\arraystretch}{1.25}
\definecolor{navy}{HTML}{1F2A44}
\definecolor{rowg}{HTML}{EEF2F7}
\definecolor{passg}{HTML}{1A9850}
\definecolor{failr}{HTML}{C0241C}
\definecolor{accent}{HTML}{2F6FED}
\arrayrulecolor{gray}
\newcolumntype{L}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\newcommand{\hd}[1]{\textcolor{white}{\textbf{#1}}}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[L]{\scriptsize\textcolor{gray}{OCR Quality Validation --- Chandra --- deterministic report}}
\fancyfoot[R]{\scriptsize\textcolor{gray}{Page \thepage\ of \pageref{LastPage}}}
"""


def _ltable(header: list[str], rows: list[list[str]], colspec: str) -> str:
    """A full-width longtable with a navy header row that repeats per page."""
    out = [r"{\arrayrulecolor{gray}\begin{longtable}{" + colspec + "}", r"\hline",
           r"\rowcolor{navy} " + " & ".join(r"\hd{%s}" % h for h in header) + r" \\ \hline",
           r"\endfirsthead",
           r"\rowcolor{navy} " + " & ".join(r"\hd{%s}" % h for h in header) + r" \\ \hline",
           r"\endhead"]
    for r in rows:
        out.append(" & ".join(r) + r" \\ \hline")
    out.append(r"\end{longtable}}")
    return "\n".join(out)


def _build_latex(
    doc: DocumentReport,
    config: Config,
    out_path: str,
    page_audit: Optional[dict],
    source_name: str,
) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    by_key = {m.key: m for m in METRICS}
    s = doc.summary
    passed = doc.document_verdict == "PASSED"
    L: list[str] = [_PREAMBLE, r"\begin{document}"]

    # 1. title + metadata
    L.append(r"\begin{center}{\LARGE\bfseries OCR Quality Validation Report}\\[2pt]"
             r"{\itshape\color{gray}Statistical validation of page-wise Chandra OCR "
             r"output}\end{center}")
    L.append(r"\textcolor{accent}{\rule{\textwidth}{1.2pt}}\\[4pt]")
    L.append(_ltable(["Field", "Value"],
                     [[r"\textbf{Generated}", _tex(now)],
                      [r"\textbf{Document ID}", _tex(doc.doc_id)],
                      [r"\textbf{Tool}", _tex("OCR QA (Chandra) - 17 statistical metrics")]],
                     "L{0.25\\textwidth} L{0.71\\textwidth}"))

    # 2. original document
    L.append(r"\section*{Original document}")
    L.append(r"{\large\bfseries " + _tex(source_name) + r"}\par")

    # 3. verdict
    box = "passg" if passed else "failr"
    vtext = "PASSED" if passed else "FAILED --- NEEDS HUMAN REVIEW"
    L.append(r"\par\vspace{10pt}\noindent\colorbox{" + box +
             r"}{\parbox{\dimexpr\textwidth-2\fboxsep\relax}{\centering\color{white}"
             r"\bfseries\large\rule{0pt}{16pt} DOCUMENT VERDICT: " + vtext +
             r"\rule[-8pt]{0pt}{8pt}}}\par\vspace{8pt}")
    L.append(_tex(doc.verdict_reason))
    L.append(_ltable(["Metric", "Value"], [
        [r"\textbf{Document score (DQS)}", f"{doc.document_score:.0f} / 100"],
        [r"\textbf{Total pages}", str(s.get("total_pages", 0))],
        [r"\textbf{Passed}", str(s.get("passed_pages", 0))],
        [r"\textbf{Need review (faulty)}", str(s.get("faulty_pages", 0))],
        [r"\textbf{Faulty ratio}", f"{s.get('faulty_ratio', 0) * 100:.0f}\\%"],
        [r"\textbf{Pages with HARD conditions}", str(len(s.get("hard_pages", [])))],
    ], "L{0.5\\textwidth} L{0.46\\textwidth}"))
    if s.get("failed_rules"):
        L.append(r"\textbf{Rule(s) that caused FAIL:}\begin{itemize}\setlength\itemsep{0pt}")
        for fr in s["failed_rules"]:
            L.append(r"\item " + _tex(fr))
        L.append(r"\end{itemize}")

    # 4. reconciliation
    if page_audit:
        L.append(r"\section*{Page-count reconciliation}")
        rows = [[_tex(k), str(v) if v is not None else "-"]
                for k, v in page_audit["counts"].items()]
        sp = page_audit.get("source_pages")
        rows.append([_tex("SOURCE (rendered)"), str(sp) if sp is not None else "-"])
        L.append(_ltable(["Artefact", "Pages"], rows,
                         "L{0.7\\textwidth} L{0.26\\textwidth}"))
        for m in page_audit.get("messages", []):
            L.append(r"{\small " + _tex("- " + m) + r"}\\")

    # 5. findings
    L.append(r"\section*{Findings --- parameters that failed and why}")
    faulty = sorted((r for r in doc.page_reports if r.verdict == "review"),
                    key=lambda r: r.page_no)
    if not faulty:
        L.append(_tex("No page was flagged for review - every page met all "
                      "thresholds and tripped no HARD condition."))
    else:
        L.append(_tex(f"{len(faulty)} page(s) need human review. For each, the "
                      "failing metrics are listed with the numeric justification and "
                      "the located error (block id / snippet, tagged DEFINITE or "
                      "STATISTICAL)."))
        for r in faulty:
            L.append(r"\vspace{4pt}\textbf{Page %d --- PQS %.0f/100}\\[-2pt]" %
                     (r.page_no, r.page_score))
            rows = []
            failing = [m for m in r.metrics
                       if m.applicable and (m.status in ("bad", "warn") or m.hard_fail)]
            failing.sort(key=lambda m: (not m.hard_fail, m.score))
            for m in failing:
                where = "; ".join(
                    f"{Lc.get('block_id') or Lc['kind']} ({Lc['confidence'][:3]})"
                    for Lc in m.locations[:3]) or "-"
                tag = " [HARD]" if m.hard_fail else ""
                rows.append([_tex(m.key + tag), f"{m.score:.0f}",
                             _tex(m.justification), _tex(where)])
            L.append(_ltable(["Metric", "Score", "Why (justification)", "Where / conf."],
                             rows,
                             "L{0.13\\textwidth} L{0.06\\textwidth} L{0.50\\textwidth} L{0.21\\textwidth}"))

    # 6. per-page summary + rollup
    L.append(r"\newpage\section*{Per-page summary (all pages)}")
    rows = []
    for r in sorted(doc.page_reports, key=lambda r: r.page_no):
        if r.problems:
            note = r.problems[0]
        else:
            app = [m for m in r.metrics if m.applicable]
            lo = min(app, key=lambda m: m.score) if app else None
            note = f"lowest: {lo.key} {lo.score:.0f}" if lo else "-"
        rows.append([str(r.page_no), f"{r.page_score:.0f}", r.verdict.upper(), _tex(note)])
    L.append(_ltable(["Page", "PQS", "Verdict", "Top problem / lowest metric"], rows,
                     "L{0.08\\textwidth} L{0.08\\textwidth} L{0.13\\textwidth} L{0.61\\textwidth}"))

    from collections import defaultdict
    agg = defaultdict(list)
    for r in doc.page_reports:
        for m in r.metrics:
            if m.applicable:
                agg[m.key].append(m.score)
    L.append(r"\section*{Mean metric score across the document}")
    rows = [[e["key"], f"{sum(agg[e['key']]) / len(agg[e['key']]):.1f}",
             f"{min(agg[e['key']]):.1f}", f"{len(agg[e['key']])} pages"]
            for e in CATALOG if agg.get(e["key"])]
    L.append(_ltable(["Metric", "Mean", "Min", "Applied on"], rows,
                     "L{0.22\\textwidth} L{0.22\\textwidth} L{0.22\\textwidth} L{0.24\\textwidth}"))

    # 7. metrics reference
    L.append(r"\newpage\section*{Metrics reference --- what each parameter is, the "
             r"file used, and how it is computed}")
    L.append(_tex("Every metric below states the exact file(s) it reads. output.json "
                  "is the backbone (block tree: text, HTML, bbox, inference_failed); the "
                  "other artefacts are used only by the metrics that cross-check or count."))
    L.append(_ltable(["File / resource", "Contains", "Used by"],
                     [[_tex(f), _tex(c), _tex(u)] for f, c, u in FILE_MAP],
                     "L{0.27\\textwidth} L{0.42\\textwidth} L{0.27\\textwidth}"))

    rows = []
    for e in CATALOG:
        m = by_key.get(e["key"])
        name = _tex(m.name if m else e["key"])
        meas = _tex(m.what_it_measures if m else "")
        files = r"\newline ".join(_tex("- " + x) for x in e["reads"])
        formula = "$" + LATEX.get(e["key"], "") + "$"
        rows.append([r"\textbf{%s}\newline %s" % (_tex(e["key"]), name),
                     formula, files, meas, _tex(e["confidence"])])
    L.append(_ltable(["Metric", "Formula", "File(s) used", "What it measures",
                      "Conf."], rows,
                     "L{0.17\\textwidth} L{0.20\\textwidth} L{0.19\\textwidth} "
                     "L{0.26\\textwidth} L{0.08\\textwidth}"))

    L.append(r"\section*{Worked examples}\begin{itemize}\setlength\itemsep{1pt}")
    for e in CATALOG:
        L.append(r"\item \textbf{%s} --- %s" % (_tex(e["key"]), _tex(e["example"])))
    L.append(r"\end{itemize}")

    L.append(r"\end{document}")
    tex = "\n".join(L)

    tmp = tempfile.mkdtemp(prefix="ocrqa_tex_")
    tex_path = os.path.join(tmp, "report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    exe = shutil.which("pdflatex")
    env = dict(os.environ)
    for _ in range(2):  # twice for longtable column widths + lastpage ref
        subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=tmp, capture_output=True, timeout=180, env=env,
        )
    pdf_path = os.path.join(tmp, "report.pdf")
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) < 1000:
        raise RuntimeError("pdflatex did not produce a PDF")
    shutil.copyfile(pdf_path, out_path)
    return out_path
