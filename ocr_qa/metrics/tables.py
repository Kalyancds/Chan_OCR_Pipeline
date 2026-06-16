"""M5 TSI — Table Structure Integrity (Sec 4, Group B).

For Table blocks only: header present? consistent column count across rows
(ragged-row rate), empty-cell ratio, stray colspans, numeric-cell plausibility.
If the page has no tables the metric reports 'not applicable' and its weight is
redistributed (Sec 5).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from ocr_qa.metrics.base import BaseMetric, DocContext, loc
from ocr_qa.models import Block, PageOCR

try:
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except Exception:  # pragma: no cover
    _HAS_BS4 = False

_NUM_RE = re.compile(r"^[\s]*[-+]?[\d.,]+\s*[%€$]?\s*$")


def _row_cols(row) -> int:
    cells = row.find_all(["td", "th"], recursive=False)
    if not cells:
        cells = row.find_all(["td", "th"])
    total = 0
    for c in cells:
        span = c.get("colspan")
        try:
            total += int(span) if span else 1
        except Exception:
            total += 1
    return total


def _analyze_table(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return {"valid": False}
    rows = table.find_all("tr")
    if not rows:
        return {"valid": False}

    has_header = bool(table.find("thead")) or bool(rows[0].find("th"))
    col_counts = [_row_cols(r) for r in rows]
    mode = Counter(col_counts).most_common(1)[0][0] if col_counts else 0
    ragged = sum(1 for c in col_counts if c != mode)

    all_cells = table.find_all(["td", "th"])
    empty = sum(1 for c in all_cells if not c.get_text(strip=True))

    stray_colspan = 0
    for c in all_cells:
        span = c.get("colspan")
        if span:
            try:
                if int(span) > max(mode, 1):
                    stray_colspan += 1
            except Exception:
                stray_colspan += 1

    return {
        "valid": True,
        "rows": len(rows),
        "has_header": has_header,
        "col_mode": mode,
        "col_counts": col_counts,
        "ragged": ragged,
        "empty": empty,
        "n_cells": len(all_cells),
        "stray_colspan": stray_colspan,
    }


class TSIMetric(BaseMetric):
    key = "TSI"
    name = "Table Structure Integrity"
    what_it_measures = (
        "Structural soundness of each <table>: a header row, a consistent "
        "number of columns per row, a low empty-cell ratio, and no stray "
        "column spans."
    )

    def compute(
        self, page: PageOCR, native_text: Optional[str], ctx: DocContext
    ):
        tables = page.table_blocks()
        cm = page.chunk_meta or {}
        # Chunk corroboration is only attributable to a page when the chunk
        # covers exactly this page (single_page); otherwise table_count spans
        # siblings and would mis-attribute.
        chunk_single = cm.get("covered") and cm.get("single_page")
        chunk_has_table = bool(cm.get("has_table")) if chunk_single else None
        chunk_table_count = int(cm.get("table_count", 0)) if chunk_single else 0

        if not tables or not _HAS_BS4:
            # If the chunk says this page HAS a table but none was parsed, that
            # is a dropped table — flag it rather than reporting N/A.
            if not tables and chunk_has_table and chunk_table_count >= 1:
                return self.result(
                    raw_value=1.0,
                    score=20.0,
                    status="bad",
                    threshold={"chunk_table_count": chunk_table_count},
                    how_computed="output_chunks reports a table on this page but "
                    "no <table> block was parsed from output.json.",
                    example=(
                        f"output_chunks says this page has {chunk_table_count} "
                        f"table(s), yet output.json has no <table> block — the "
                        f"table was dropped during OCR."
                    ),
                    locations=[loc("dropped_table", "definite",
                                   snippet=f"chunk reports {chunk_table_count} table(s), 0 parsed")],
                    evidence=[
                        f"output_chunks reports table_count={chunk_table_count} "
                        f"(has_table=true) but 0 tables parsed — possible dropped "
                        f"table"
                    ],
                    justification=(
                        f"Chunk metadata reports {chunk_table_count} table(s) on "
                        f"this page, but none were parsed — likely a dropped table."
                    ),
                    hard_fail=True,  # DEFINITE: table content lost
                )
            return self.not_applicable("page has no tables" if not tables else "bs4 missing")

        evidence: list[str] = []
        defect_scores: list[float] = []
        locations: list[dict] = []

        for b in tables:
            info = _analyze_table(b.html)
            if not info.get("valid"):
                defect_scores.append(1.0)
                evidence.append(f"{b.block_id or 'table'}: could not parse <table>")
                locations.append(loc("table_unparseable", "definite", block=b,
                                     snippet="could not parse <table>"))
                continue

            n_rows = info["rows"]
            ragged_rate = info["ragged"] / n_rows if n_rows else 0.0
            empty_rate = info["empty"] / info["n_cells"] if info["n_cells"] else 0.0
            no_header_pen = 0.0 if info["has_header"] else 0.4
            stray_pen = min(0.3, 0.1 * info["stray_colspan"])

            defect = min(1.0, ragged_rate + 0.5 * empty_rate + no_header_pen + stray_pen)
            defect_scores.append(defect)
            if defect > 0:
                bits = []
                if not info["has_header"]:
                    bits.append("no header")
                if info["ragged"]:
                    bits.append(f"{info['ragged']} ragged row(s) vs {info['col_mode']} cols")
                if empty_rate > 0.2:
                    bits.append(f"{empty_rate*100:.0f}% empty cells")
                locations.append(loc("table_defect", "definite", block=b,
                                     snippet="; ".join(bits) or "table defect"))

            if not info["has_header"]:
                evidence.append(f"{b.block_id or 'table'}: no header row (thead/th)")
            if info["ragged"]:
                bad_rows = [
                    i + 1
                    for i, c in enumerate(info["col_counts"])
                    if c != info["col_mode"]
                ]
                evidence.append(
                    f"{b.block_id or 'table'}: ragged rows {bad_rows} "
                    f"(expected {info['col_mode']} cols)"
                )
            if empty_rate > 0.2:
                evidence.append(
                    f"{b.block_id or 'table'}: {empty_rate * 100:.0f}% empty cells"
                )
            if info["stray_colspan"]:
                evidence.append(
                    f"{b.block_id or 'table'}: {info['stray_colspan']} stray colspan(s)"
                )

        raw = sum(defect_scores) / len(defect_scores)
        n_unparseable = sum(1 for L in locations if L["kind"] == "table_unparseable")

        # Chunk corroboration: parsed table count vs chunk-reported count.
        chunk_note = ""
        if chunk_single and chunk_has_table and chunk_table_count != len(tables):
            evidence.append(
                f"output_chunks reports table_count={chunk_table_count} but "
                f"{len(tables)} table(s) parsed — count mismatch"
            )
            raw = min(1.0, raw + 0.25)
            chunk_note = (
                f" Chunk reports {chunk_table_count} table(s) vs {len(tables)} parsed."
            )

        score = round((1.0 - raw) * 100.0, 1)
        status = "good" if score >= 70 else ("warn" if score >= 50 else "bad")
        if not evidence:
            evidence = [f"{len(tables)} table(s) well-structured"]

        if raw > 0 and evidence:
            example = (
                f"Example: {evidence[0]}. Across {len(tables)} table(s) that is "
                f"a {raw * 100:.0f}% defect rate."
            )
        else:
            example = (
                f"The {len(tables)} table(s) each have a header row and the same "
                f"column count in every row, with few empty cells → 0 defects."
            )

        return self.result(
            raw_value=raw,
            score=score,
            status=status,
            threshold={"defect_rate_good_at": 0.0, "bad_at": 1.0},
            how_computed="per table: ragged-row rate + 0.5*empty-cell rate + "
            "no-header penalty + stray-colspan penalty, averaged over tables; "
            "cross-checked against output_chunks table_metrics when available.",
            example=example,
            locations=locations,
            evidence=evidence,
            justification=(
                f"Table defect rate {raw * 100:.1f}% across {len(tables)} "
                f"table(s).{chunk_note}"
                + (" Severe — table content lost/broken." if (n_unparseable or raw >= 0.6)
                   else "")
            ),
            # DEFINITE only when a table is unparseable or severely broken
            # (content clearly lost) — not for cosmetic header/ragged issues.
            hard_fail=bool(n_unparseable) or raw >= 0.6,
        )
