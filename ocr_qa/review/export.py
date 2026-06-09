"""Human-review export package (Sec 8) — FAULTY PAGES ONLY.

Produces a single zip containing:
  * review.json  — per faulty page: page_no, PQS, every MetricResult (incl.
                   evidence), problems[], image filename
  * review.csv   — page_no, PQS, top-3 problems
  * /pages/      — the faulty page images
  * review.md    — one section per faulty page (image ref + faults + metric
                   justifications, mirroring the UI)

``send_for_review`` is a marked hook for a future webhook / Label-Studio
integration; it is intentionally NOT wired to anything external now.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import zipfile
from typing import Optional

from ocr_qa.models import DocumentReport, IngestedDocument, PageReport


def _page_payload(r: PageReport, image_filename: Optional[str]) -> dict:
    return {
        "page_no": r.page_no,
        "PQS": r.page_score,
        "verdict": r.verdict,
        "image": image_filename,
        "problems": r.problems,
        "metrics": [
            {
                "key": m.key,
                "name": m.name,
                "raw_value": m.raw_value,
                "score": m.score,
                "status": m.status,
                "hard_fail": m.hard_fail,
                "applicable": m.applicable,
                "reliability": m.reliability,
                "threshold": m.threshold,
                "what_it_measures": m.what_it_measures,
                "how_computed": m.how_computed,
                "justification": m.justification,
                "example": m.example,
                "evidence": m.evidence,
                "locations": m.locations,
            }
            for m in r.metrics
        ],
    }


def _review_markdown(doc: DocumentReport, faulty: list[PageReport],
                     images: dict[int, str]) -> str:
    lines = [
        f"# Human-Review Package — {doc.doc_id}",
        "",
        f"**Document verdict:** {doc.document_verdict}",
        "",
        f"{doc.verdict_reason}",
        "",
        f"**Faulty pages:** {', '.join(str(r.page_no) for r in faulty) or 'none'}",
        "",
        "---",
        "",
    ]
    for r in faulty:
        lines.append(f"## Page {r.page_no} — PQS {r.page_score:.0f}/100")
        img = images.get(r.page_no)
        if img:
            lines.append(f"![page {r.page_no}](pages/{os.path.basename(img)})")
        lines.append("")
        lines.append("### Faults on this page")
        for p in r.problems:
            lines.append(f"- {p}")
        lines.append("")
        lines.append("### Metric detail")
        for m in r.metrics:
            if not m.applicable:
                continue
            flag = " **[HARD]**" if m.hard_fail else ""
            lines.append(
                f"- **{m.name}** ({m.key}) — {m.score:.0f}/100 · {m.status}{flag}: "
                f"{m.justification}"
            )
            if m.example:
                lines.append(f"    - _Example:_ {m.example}")
            for e in m.evidence[:4]:
                lines.append(f"    - `{e}`")
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def build_review_package(
    doc: DocumentReport,
    ingested: Optional[IngestedDocument],
    out_path: str,
) -> str:
    """Write the faulty-pages review zip to ``out_path``; return the path."""
    faulty = [r for r in doc.page_reports if r.verdict == "review"]
    faulty.sort(key=lambda r: r.page_score)

    images: dict[int, str] = {}
    if ingested:
        for pi in ingested.pages:
            if pi.image_path and os.path.exists(pi.image_path):
                images[pi.page_no] = pi.image_path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # review.json
        payload = {
            "doc_id": doc.doc_id,
            "document_verdict": doc.document_verdict,
            "document_score": doc.document_score,
            "verdict_reason": doc.verdict_reason,
            "summary": doc.summary,
            "faulty_pages": [
                _page_payload(
                    r,
                    os.path.basename(images[r.page_no]) if r.page_no in images else None,
                )
                for r in faulty
            ],
        }
        zf.writestr("review.json", json.dumps(payload, indent=2, ensure_ascii=False))

        # review.csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["page_no", "PQS", "problem_1", "problem_2", "problem_3"])
        for r in faulty:
            probs = (r.problems + ["", "", ""])[:3]
            w.writerow([r.page_no, f"{r.page_score:.1f}", *probs])
        zf.writestr("review.csv", buf.getvalue())

        # review.md
        zf.writestr("review.md", _review_markdown(doc, faulty, images))

        # page images
        for r in faulty:
            img = images.get(r.page_no)
            if img and os.path.exists(img):
                zf.write(img, arcname=f"pages/{os.path.basename(img)}")

    return out_path


# --------------------------------------------------------------------------- #
# Future integration hook (NOT implemented now) — Sec 8.
# --------------------------------------------------------------------------- #
def send_for_review(package_path: str) -> dict:
    """MARKED HOOK for a future webhook / Label-Studio push.

    Deliberately a no-op stub so no external integration runs in this build.
    """
    return {
        "status": "not_sent",
        "reason": "external review integration not configured in this build",
        "package": package_path,
    }
