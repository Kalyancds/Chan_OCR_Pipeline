"""Parse the ACTUAL Chandra OCR output schema (Sec 2) into ``list[PageOCR]``.

Honors the real schema exactly:

* ``output.json``  : a LIST of Page objects, each with a nested ``children``
                     block tree -> recursively flattened to content blocks.
* block HTML       : stripped to plain text (BeautifulSoup+lxml, regex fallback);
                     tables / figures kept as structured blocks.
* ``inference_failed`` : preserved per block as a first-class signal.
* NO token/word/line confidence is read or assumed.
* ``output.metadata.json`` : ``page_stats[].num_blocks`` per page (BCC).
* ``output.md``    : split on ``{N}----...`` delimiter lines (JSON<->MD agree).

The parser is DEFENSIVE: missing keys are tolerated, it never raises on
malformed input, and it records per-page what was found vs missing in
``parse_log`` for surfacing in the UI / logs.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from ocr_qa.models import Block, ChandraChunk, PageOCR

logger = logging.getLogger("ocr_qa.parser")

# --------------------------------------------------------------------------- #
# HTML stripping
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised indirectly
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except Exception:  # pragma: no cover
    _HAS_BS4 = False

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")
_MULTINL_RE = re.compile(r"\n{3,}")


def _bs(html: str) -> Optional["BeautifulSoup"]:
    if not _HAS_BS4 or not html:
        return None
    for parser in ("lxml", "html.parser"):
        try:
            return BeautifulSoup(html, parser)
        except Exception:
            continue
    return None


def strip_html(html: str) -> str:
    """Return readable plain text for a block's HTML.

    Footnote markers (<sup>) are kept inline; <br> and block tags become
    newlines/spaces. Falls back to a regex strip if bs4 is unavailable.
    """
    if not html:
        return ""
    soup = _bs(html)
    if soup is not None:
        # <br> -> newline before extraction.
        for br in soup.find_all("br"):
            br.replace_with("\n")
        text = soup.get_text(separator=" ")
    else:
        text = _TAG_RE.sub(" ", html)
        # Minimal entity decode for the regex path.
        text = (
            text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace("&quot;", '"')
        )
    text = _WS_RE.sub(" ", text)
    text = _MULTINL_RE.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Block-tree flattening
# --------------------------------------------------------------------------- #
def _coerce_bbox(val: Any) -> list[float]:
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            return [float(x) for x in val]
        except Exception:
            return []
    return []


def _coerce_polygon(val: Any) -> list[list[float]]:
    if isinstance(val, (list, tuple)):
        out: list[list[float]] = []
        for pt in val:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                try:
                    out.append([float(pt[0]), float(pt[1])])
                except Exception:
                    pass
        return out
    return []


def _has_table(html: str) -> bool:
    return "<table" in (html or "").lower()


def _has_figure(block_type: str, html: str) -> bool:
    bt = (block_type or "").lower()
    if bt in {"figure", "picture", "image"}:
        return True
    return "<img" in (html or "").lower() or "img-description" in (html or "").lower()


def _flatten_blocks(
    node: dict[str, Any],
    order_counter: list[int],
    found: dict[str, int],
    out: list[Block],
) -> None:
    """Depth-first flatten of a page's block tree into leaf/content blocks.

    A node with children is recursed into. A node that is itself content
    (has html / is a known leaf type) is emitted. Tables are emitted whole
    (not flattened into rows).
    """
    if not isinstance(node, dict):
        return

    children = node.get("children")
    block_type = node.get("block_type") or node.get("type") or "Unknown"
    html = node.get("html") or ""
    is_table = _has_table(html) or block_type == "Table"

    # Tables are atomic content blocks: emit and DO NOT descend into row/cell
    # children even if Chandra nests them.
    if is_table:
        _emit(node, block_type, html, order_counter, found, out)
        return

    if isinstance(children, list) and children:
        for child in children:
            _flatten_blocks(child, order_counter, found, out)
        # A container that ALSO carries its own non-empty html (rare) is still
        # captured via its children; skip the container to avoid double count.
        return

    # Leaf node.
    _emit(node, block_type, html, order_counter, found, out)


def _emit(
    node: dict[str, Any],
    block_type: str,
    html: str,
    order_counter: list[int],
    found: dict[str, int],
    out: list[Block],
) -> None:
    text = strip_html(html)
    is_table = _has_table(html) or block_type == "Table"
    is_figure = _has_figure(block_type, html)

    # Skip truly empty PageHeader/PageFooter (html="") per Sec 2.
    if block_type in {"PageHeader", "PageFooter"} and not text.strip() and not html.strip():
        found["skipped_empty_headfoot"] = found.get("skipped_empty_headfoot", 0) + 1
        return

    block = Block(
        block_type=block_type,
        text=text,
        html=html,
        bbox=_coerce_bbox(node.get("bbox")),
        polygon=_coerce_polygon(node.get("polygon")),
        inference_failed=bool(node.get("inference_failed", False)),
        reading_order=order_counter[0],
        is_table=is_table,
        is_figure=is_figure,
        block_id=str(node.get("id", "")),
    )
    order_counter[0] += 1
    out.append(block)
    found["blocks"] = found.get("blocks", 0) + 1
    if block.inference_failed:
        found["inference_failed"] = found.get("inference_failed", 0) + 1


# --------------------------------------------------------------------------- #
# output.md page splitting
# --------------------------------------------------------------------------- #
# Lines like "{7}------------------------------------------------" (real Chandra)
# or a bare "7------..." — page number optionally wrapped in curly braces.
_MD_DELIM_RE = re.compile(r"^\s*\{?(\d+)\}?\s*-{6,}\s*$", re.MULTILINE)


def split_markdown(md: str) -> dict[int, str]:
    """Return {page_no: markdown_segment} from the page-delimited output.md.

    The delimiter line ``{N}-----`` PRECEDES (heads) page N's content in the
    Chandra format. Tolerant of a leading preamble before the first delimiter.
    """
    if not md:
        return {}
    matches = list(_MD_DELIM_RE.finditer(md))
    if not matches:
        return {1: md.strip()}
    segments: dict[int, str] = {}
    for i, m in enumerate(matches):
        page_no = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        segments[page_no] = md[start:end].strip()
    return segments


# --------------------------------------------------------------------------- #
# output.html  (optional 5th artefact: per-page rendered HTML)
# --------------------------------------------------------------------------- #
def split_html(output_html: Optional[str]) -> dict[int, str]:
    """Return {page_id: inner_html} from output.html.

    Real Chandra output.html is ``<body><div class="page" data-page-id="N">
    …blocks…</div>…</body>``. We return each page div's inner HTML keyed by its
    0-based data-page-id. Defensive: returns {} if unparseable.
    """
    if not output_html or not output_html.strip():
        return {}
    soup = _bs(output_html)
    out: dict[int, str] = {}
    if soup is not None:
        divs = soup.find_all("div", class_="page")
        for i, div in enumerate(divs):
            pid = div.get("data-page-id")
            try:
                key = int(pid) if pid is not None else i
            except Exception:
                key = i
            try:
                out[key] = div.decode_contents()
            except Exception:
                out[key] = str(div)
        if out:
            return out
    # Regex fallback: split on the page-div markers.
    for m in re.finditer(
        r'<div[^>]*class="page"[^>]*data-page-id="(\d+)"[^>]*>(.*?)</div>\s*(?=<div[^>]*class="page"|</body>|\Z)',
        output_html,
        re.DOTALL,
    ):
        try:
            out[int(m.group(1))] = m.group(2)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def parse_metadata(meta: Any) -> dict[int, int]:
    """Return {page_id: num_blocks} from metadata.page_stats (Sec 2).

    No confidence field is assumed; if absent we simply return block counts.
    """
    out: dict[int, int] = {}
    if not isinstance(meta, dict):
        return out
    stats = meta.get("page_stats")
    if not isinstance(stats, list):
        return out
    for s in stats:
        if not isinstance(s, dict):
            continue
        pid = s.get("page_id")
        nb = s.get("num_blocks")
        if pid is None or nb is None:
            continue
        try:
            out[int(pid)] = int(nb)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# output_chunks.json (optional 4th artefact)
# --------------------------------------------------------------------------- #
def parse_chunks(chunks_json: Any) -> list[ChandraChunk]:
    """Parse output_chunks.json into ChandraChunk objects. Defensive: accepts a
    bare list, a {"chunks": [...]} wrapper, or a single object."""
    if isinstance(chunks_json, dict):
        if isinstance(chunks_json.get("chunks"), list):
            items = chunks_json["chunks"]
        else:
            items = [chunks_json]
    elif isinstance(chunks_json, list):
        items = chunks_json
    else:
        return []

    out: list[ChandraChunk] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        ids = it.get("page_ids")
        if not isinstance(ids, list):
            ids = []
        clean_ids = []
        for x in ids:
            try:
                clean_ids.append(int(x))
            except Exception:
                pass
        tm = it.get("table_metrics")
        out.append(
            ChandraChunk(
                page_start=int(it.get("page_start", 0) or 0),
                page_end=int(it.get("page_end", 0) or 0),
                page_ids=clean_ids,
                chunk_title=str(it.get("chunk_title", "") or ""),
                header_path=it.get("header_path") or [],
                token_count=int(it.get("token_count", 0) or 0),
                content=str(it.get("content", "") or ""),
                table_metrics=tm if isinstance(tm, dict) else {},
                raw=it,
            )
        )
    return out


def attach_chunks(pages: list[PageOCR], chunks: list[ChandraChunk]) -> None:
    """Attach each page's covering chunk text + table stats, in place.

    Chandra chunk ``page_ids`` are 0-based while PageOCR page numbers may be
    1-based; the offset is detected from the data and applied, with direct-match
    fallbacks. A page covered by no chunk is marked covered=False so BCC can flag
    a possible chunk/content drop.
    """
    if not chunks:
        return

    # page_id -> chunks containing it
    id_map: dict[int, list[ChandraChunk]] = {}
    all_ids: list[int] = []
    for ch in chunks:
        for pid in ch.page_ids:
            id_map.setdefault(pid, []).append(ch)
            all_ids.append(pid)

    page_nos = [p.page_no for p in pages]
    offset = 0
    if all_ids and page_nos:
        offset = min(page_nos) - min(all_ids)  # e.g. 1-based pages, 0-based ids => 1

    for p in pages:
        matched: list[ChandraChunk] = []
        for candidate in (p.page_no - offset, p.page_no, p.page_no - 1):
            if candidate in id_map:
                matched = id_map[candidate]
                break
        if not matched:
            p.chunk_meta = {"chunks_present": True, "covered": False}
            continue
        ch = matched[0]
        tm = ch.table_metrics or {}
        p.chunk_text = "\n\n".join(c.content for c in matched) if len(matched) > 1 else ch.content
        p.chunk_meta = {
            "chunks_present": True,
            "covered": True,
            "title": ch.chunk_title,
            "token_count": ch.token_count,
            "n_pages": len(ch.page_ids),
            "single_page": len(ch.page_ids) == 1,
            "has_table": bool(tm.get("has_table", False)),
            "table_count": int(tm.get("table_count", 0) or 0),
            "table_density": float(tm.get("table_density", 0.0) or 0.0),
        }


# --------------------------------------------------------------------------- #
# Page-count reconciliation (preprocessing audit)
# --------------------------------------------------------------------------- #
def audit_page_counts(
    output_json: Any,
    output_md: Optional[str] = None,
    metadata: Any = None,
    output_chunks: Any = None,
    output_html: Optional[str] = None,
    source_pages: Optional[int] = None,
) -> dict[str, Any]:
    """Count pages in every artefact (and the source) BEFORE analysis so page
    imbalance is caught and reported up-front.

    Returns a dict with per-artefact counts, whether the OCR artefacts agree,
    the canonical OCR page count, the source count, the delta and a status +
    human-readable messages.
    """
    counts: dict[str, Optional[int]] = {}

    # output.json (Page blocks)
    json_pages = len(ChandraParser._coerce_pages(output_json))
    counts["output.json"] = json_pages

    # metadata (separate file, else embedded in output.json)
    meta = metadata
    if meta is None and isinstance(output_json, dict):
        meta = output_json.get("metadata")
    counts["metadata"] = len(parse_metadata(meta)) if meta is not None else None

    # output.md delimiters
    if output_md:
        counts["output.md"] = len(_MD_DELIM_RE.findall(output_md)) or None
    else:
        counts["output.md"] = None

    # output.html page divs
    counts["output.html"] = len(split_html(output_html)) or None if output_html else None

    # output_chunks coverage
    chunks = parse_chunks(output_chunks) if output_chunks is not None else []
    if chunks:
        ids = {pid for c in chunks for pid in c.page_ids}
        counts["output_chunks"] = (max(ids) + 1) if ids else None
    else:
        counts["output_chunks"] = None

    ocr_vals = [v for k, v in counts.items() if v is not None]
    ocr_consistent = len(set(ocr_vals)) <= 1
    ocr_pages = json_pages or (max(set(ocr_vals), key=ocr_vals.count) if ocr_vals else 0)

    messages: list[str] = []
    status = "ok"
    if not ocr_consistent:
        status = "ocr_inconsistent"
        pairs = ", ".join(f"{k}={v}" for k, v in counts.items() if v is not None)
        messages.append(
            f"OCR artefacts DISAGREE on page count ({pairs}) — the output files "
            f"are inconsistent with each other; re-export from Chandra."
        )
    else:
        messages.append(f"All OCR artefacts agree: {ocr_pages} pages.")

    delta = None
    if source_pages is not None:
        delta = ocr_pages - source_pages
        if delta != 0:
            status = "source_mismatch" if status == "ok" else status
            more = "more" if delta > 0 else "fewer"
            messages.append(
                f"OCR has {ocr_pages} pages but the source rendered {source_pages} "
                f"({abs(delta)} {more} in OCR). The original-image↔OCR pairing will "
                f"drift; re-OCR the exact same file, or rely on the HTML/Markdown "
                f"view (which always matches the OCR)."
            )
        else:
            messages.append(f"Source matches OCR: {source_pages} pages. ✅")

    return {
        "counts": counts,
        "ocr_consistent": ocr_consistent,
        "ocr_pages": ocr_pages,
        "source_pages": source_pages,
        "delta_ocr_minus_source": delta,
        "status": status,  # ok | source_mismatch | ocr_inconsistent
        "messages": messages,
    }


# --------------------------------------------------------------------------- #
# page id -> page number
# --------------------------------------------------------------------------- #
_PAGE_ID_RE = re.compile(r"/page/(\d+)/")


def _page_number(page_obj: dict[str, Any], fallback: int) -> int:
    pid = page_obj.get("id", "")
    m = _PAGE_ID_RE.search(str(pid))
    if m:
        return int(m.group(1))
    for key in ("page", "page_no", "page_id", "page_number"):
        if key in page_obj:
            try:
                return int(page_obj[key])
            except Exception:
                pass
    return fallback


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
class ChandraParser:
    """Stateless parser; ``parse`` accepts already-loaded objects so it is
    reusable by both the Mock (files) and Real (API response) clients."""

    def __init__(self) -> None:
        self.parse_log: list[dict[str, Any]] = []

    def parse(
        self,
        output_json: Any,
        output_md: Optional[str] = None,
        metadata: Any = None,
        output_chunks: Any = None,
        output_html: Optional[str] = None,
    ) -> list[PageOCR]:
        self.parse_log = []
        md_segments = split_markdown(output_md or "")
        html_segments = split_html(output_html)
        # Metadata may be a separate file OR embedded in output.json under the
        # "metadata" key (real Chandra). Prefer the explicit arg, fall back to
        # the embedded one.
        if metadata is None and isinstance(output_json, dict):
            metadata = output_json.get("metadata")
        meta_counts = parse_metadata(metadata)
        chunks = parse_chunks(output_chunks) if output_chunks is not None else []

        # output.json is a LIST of page objects (Sec 2). Tolerate a single
        # dict or a {"pages": [...]} wrapper.
        pages_raw = self._coerce_pages(output_json)

        pages: list[PageOCR] = []
        for idx, page_obj in enumerate(pages_raw, start=1):
            page_no = _page_number(page_obj, idx)
            found: dict[str, int] = {}
            blocks: list[Block] = []
            order_counter = [0]

            children = page_obj.get("children")
            # Chandra's metadata.num_blocks counts the Page block's DIRECT
            # children, so capture that for an apples-to-apples BCC comparison.
            direct_children = len(children) if isinstance(children, list) else None
            if isinstance(children, list):
                for child in children:
                    _flatten_blocks(child, order_counter, found, blocks)
            else:
                found["missing_children"] = 1
                # Last resort: treat the page html itself as one block.
                page_html = page_obj.get("html") or ""
                if page_html:
                    _emit(page_obj, "Page", page_html, order_counter, found, blocks)

            page_text = "\n".join(b.text for b in blocks if b.text.strip())

            # Per-page rendered HTML: prefer output.html, else the Page block's
            # own html, else concatenate the content blocks' html.
            html_text = html_segments.get(page_no)
            if not html_text:
                page_html = page_obj.get("html") if isinstance(page_obj, dict) else None
                if page_html and page_html.strip() and page_html.strip() not in {"<page>...</page>"}:
                    html_text = page_html
                else:
                    joined = "\n".join(b.html for b in blocks if b.html and b.html.strip())
                    html_text = joined or None

            page = PageOCR(
                page_no=page_no,
                text=page_text,
                blocks=blocks,
                md_text=md_segments.get(page_no),
                num_blocks_meta=meta_counts.get(page_no),
                num_blocks_json=direct_children,
                html_text=html_text,
                language=None,  # filled later by language detection
                raw=page_obj if isinstance(page_obj, dict) else {},
            )
            pages.append(page)

            self.parse_log.append(
                {
                    "page_no": page_no,
                    "blocks_found": found.get("blocks", 0),
                    "inference_failed": found.get("inference_failed", 0),
                    "skipped_empty_headfoot": found.get("skipped_empty_headfoot", 0),
                    "has_md": page_no in md_segments,
                    "meta_num_blocks": meta_counts.get(page_no),
                    "missing_children": bool(found.get("missing_children")),
                }
            )
            logger.debug("parsed page %s: %s", page_no, self.parse_log[-1])

        # Attach optional chunk artefact (cross-source text + table stats).
        if chunks:
            attach_chunks(pages, chunks)
            covered = sum(1 for p in pages if p.chunk_meta.get("covered"))
            for entry in self.parse_log:
                entry["chunk_covered"] = any(
                    p.page_no == entry["page_no"] and p.chunk_meta.get("covered")
                    for p in pages
                )
            logger.debug("attached %d chunks; %d/%d pages covered",
                         len(chunks), covered, len(pages))

        # Renumber to contiguous 1-based page numbers for natural display and
        # alignment with rendered page images (which are 1-based). All page-level
        # artefacts (md / metadata / chunk) are already stored on each page, so
        # this only changes the displayed index, not the data.
        pages.sort(key=lambda p: p.page_no)
        remap: dict[int, int] = {}
        for new_no, p in enumerate(pages, start=1):
            remap[p.page_no] = new_no
            p.page_no = new_no
        for entry in self.parse_log:
            if entry.get("page_no") in remap:
                entry["page_no"] = remap[entry["page_no"]]

        return pages

    @staticmethod
    def _coerce_pages(output_json: Any) -> list[dict[str, Any]]:
        # Form 1: a bare LIST of page objects (spec / fixtures).
        if isinstance(output_json, list):
            return [p for p in output_json if isinstance(p, dict)]
        # Form 2: a DICT wrapper. The REAL Chandra output is
        #   {"children": [<Page block>, ...], "metadata": {...}}
        # where each child is a block_type=="Page" object holding its own
        # content children. Also tolerate an explicit {"pages": [...]} wrapper.
        if isinstance(output_json, dict):
            for key in ("pages", "children"):
                v = output_json.get(key)
                if isinstance(v, list) and v:
                    page_blocks = [
                        c
                        for c in v
                        if isinstance(c, dict) and c.get("block_type") == "Page"
                    ]
                    if page_blocks:
                        return page_blocks
                    if key == "pages":
                        return [c for c in v if isinstance(c, dict)]
            # No Page blocks found: treat the whole dict as a single page so its
            # content children are flattened.
            return [output_json]
        return []


# --------------------------------------------------------------------------- #
# Convenience loaders (used by MockChandraClient)
# --------------------------------------------------------------------------- #
def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()
