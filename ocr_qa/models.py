"""Typed data models (pydantic v2) for the whole pipeline (Sec 3).

These are the only contracts that cross module boundaries. Ingestion produces
``IngestedDocument``; the Chandra parser produces ``list[PageOCR]``; metrics
produce ``MetricResult``; scoring produces ``PageReport`` / ``DocumentReport``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Literals
# --------------------------------------------------------------------------- #
MetricStatus = Literal["good", "warn", "bad", "na"]
Reliability = Literal["high", "low"]
PageVerdict = Literal["ok", "review"]
DocVerdict = Literal["PASSED", "FAILED"]
SourceType = Literal["pdf", "docx", "pptx", "zip_images", "unknown"]


# --------------------------------------------------------------------------- #
# Ingestion-side
# --------------------------------------------------------------------------- #
class PageImage(BaseModel):
    page_no: int
    image_path: str
    width: int = 0
    height: int = 0


class NativeText(BaseModel):
    page_no: int
    text: str


class IngestedDocument(BaseModel):
    doc_id: str
    source_type: SourceType = "unknown"
    pages: list[PageImage] = Field(default_factory=list)
    native_text: Optional[list[NativeText]] = None

    def native_for(self, page_no: int) -> Optional[str]:
        if not self.native_text:
            return None
        for nt in self.native_text:
            if nt.page_no == page_no:
                return nt.text
        return None


# --------------------------------------------------------------------------- #
# OCR-side (Chandra)
# --------------------------------------------------------------------------- #
class Block(BaseModel):
    block_type: str = "Unknown"
    text: str = ""
    html: str = ""
    bbox: list[float] = Field(default_factory=list)  # [x0,y0,x1,y1]
    polygon: list[list[float]] = Field(default_factory=list)  # [[x,y]*4]
    inference_failed: bool = False
    reading_order: int = 0
    is_table: bool = False
    is_figure: bool = False
    block_id: str = ""

    # Classification helpers ------------------------------------------------
    PROSE_TYPES: ClassVar[set[str]] = {
        "Text",
        "SectionHeader",
        "ListItem",
        "Footnote",
        "Caption",
    }
    CONTENT_TYPES: ClassVar[set[str]] = {"Text", "SectionHeader", "Table", "ListItem"}

    @property
    def is_prose(self) -> bool:
        if self.is_table or self.is_figure:
            return False
        return self.block_type in self.PROSE_TYPES

    @property
    def is_content(self) -> bool:
        """Content blocks gate the IFR HARD condition (Sec 5)."""
        return self.block_type in self.CONTENT_TYPES or self.is_table


class ChandraChunk(BaseModel):
    """A chunk from output_chunks.json: a semantic span over one or more pages
    (page_ids, 0-based in Chandra) with its own markdown content + table stats."""

    page_start: int = 0
    page_end: int = 0
    page_ids: list[int] = Field(default_factory=list)
    chunk_title: str = ""
    header_path: list[str] = Field(default_factory=list)
    token_count: int = 0
    content: str = ""
    table_metrics: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class PageOCR(BaseModel):
    page_no: int
    text: str = ""
    blocks: list[Block] = Field(default_factory=list)
    md_text: Optional[str] = None
    num_blocks_meta: Optional[int] = None
    num_blocks_json: Optional[int] = None  # direct children of the Page block
    language: Optional[str] = None
    raw: dict[str, Any] = Field(default_factory=dict)
    # From output_chunks.json (optional 4th Chandra artefact):
    chunk_text: Optional[str] = None
    chunk_meta: dict[str, Any] = Field(default_factory=dict)
    # From output.html (optional 5th artefact): the page's rendered HTML.
    html_text: Optional[str] = None

    def prose_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.is_prose and b.text.strip())

    def table_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_table]

    def figure_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_figure]

    def content_blocks(self) -> list[Block]:
        return [b for b in self.blocks if b.is_content]


# --------------------------------------------------------------------------- #
# Metric & report side
# --------------------------------------------------------------------------- #
class MetricResult(BaseModel):
    key: str
    name: str
    raw_value: float = 0.0
    score: float = 100.0  # 0-100
    status: MetricStatus = "good"
    threshold: dict[str, Any] = Field(default_factory=dict)
    what_it_measures: str = ""
    how_computed: str = ""
    example: str = ""  # plain-English worked example from THIS page
    evidence: list[str] = Field(default_factory=list)
    # Error localizations: each is {kind, confidence: 'definite'|'statistical',
    # block_id, bbox:[x0,y0,x1,y1], snippet}. Used to highlight WHERE the error is.
    locations: list[dict[str, Any]] = Field(default_factory=list)
    justification: str = ""
    reliability: Reliability = "high"
    applicable: bool = True  # False => dropped from weighting (Sec 5)
    hard_fail: bool = False  # this metric tripped a HARD condition (Sec 5)


class PageReport(BaseModel):
    page_no: int
    metrics: list[MetricResult] = Field(default_factory=list)
    page_score: float = 100.0  # PQS
    verdict: PageVerdict = "ok"
    problems: list[str] = Field(default_factory=list)
    image_path: Optional[str] = None
    token_count: int = 0

    def metric(self, key: str) -> Optional[MetricResult]:
        for m in self.metrics:
            if m.key == key:
                return m
        return None


class DocumentReport(BaseModel):
    doc_id: str
    document_score: float = 100.0  # DQS
    document_verdict: DocVerdict = "PASSED"
    verdict_reason: str = ""
    page_reports: list[PageReport] = Field(default_factory=list)
    faulty_pages: list[int] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
