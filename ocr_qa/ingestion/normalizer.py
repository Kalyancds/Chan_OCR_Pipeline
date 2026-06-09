"""Source ingestion & page rendering (Sec 1, Step 1).

Turns any of the four supported source types into an ``IngestedDocument``:
page images (rendered ~200 DPI) plus the native PDF text layer when the PDF is
digital-born. This is the LEFT side of the side-by-side review; it is matched to
the Chandra ``PageOCR`` by page number.

  * PDF        -> PyMuPDF renders each page + extracts native text.
  * DOCX/PPTX  -> LibreOffice headless converts to PDF first (detected; if the
                 binary is missing we degrade with a clear, user-facing error).
  * ZIP images -> each image becomes a page (no native text).

Defensive: a render failure for one page does not abort the others.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from typing import Optional

from ocr_qa.config import Config
from ocr_qa.models import IngestedDocument, NativeText, PageImage, SourceType

logger = logging.getLogger("ocr_qa.ingestion")

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_LIBREOFFICE_CANDIDATES = [
    "soffice",
    "soffice.exe",
    "libreoffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
]


class IngestionError(Exception):
    """User-facing ingestion failure (shown in the UI, never a stack trace)."""


def _import_pymupdf():
    """Import PyMuPDF, preferring the modern ``pymupdf`` name (the legacy
    ``fitz`` alias is sometimes shadowed by a broken namespace package)."""
    try:
        import pymupdf  # PyMuPDF >= 1.24

        if hasattr(pymupdf, "open"):
            return pymupdf
    except Exception:
        pass
    import fitz  # legacy alias

    if not hasattr(fitz, "open"):
        raise IngestionError(
            "PyMuPDF is not importable (the 'fitz' name is shadowed). "
            "Install it with: pip install --upgrade PyMuPDF"
        )
    return fitz


def detect_source_type(path: str) -> SourceType:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext == ".pptx":
        return "pptx"
    if ext == ".zip":
        return "zip_images"
    return "unknown"


def find_libreoffice() -> Optional[str]:
    for cand in _LIBREOFFICE_CANDIDATES:
        if os.path.isabs(cand):
            if os.path.exists(cand):
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


class Normalizer:
    def __init__(self, config: Optional[Config] = None, work_dir: Optional[str] = None):
        self.config = config or Config()
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="ocrqa_")
        os.makedirs(self.work_dir, exist_ok=True)

    # ---- public entry ----------------------------------------------------
    def ingest(self, path: str, doc_id: Optional[str] = None) -> IngestedDocument:
        if not os.path.exists(path):
            raise IngestionError(f"Source file not found: {path}")
        stype = detect_source_type(path)
        doc_id = doc_id or os.path.splitext(os.path.basename(path))[0]

        if stype == "pdf":
            return self._ingest_pdf(path, doc_id)
        if stype in ("docx", "pptx"):
            pdf_path = self._office_to_pdf(path)
            doc = self._ingest_pdf(pdf_path, doc_id)
            doc.source_type = stype
            return doc
        if stype == "zip_images":
            return self._ingest_zip(path, doc_id)
        raise IngestionError(
            f"Unsupported source type '{os.path.splitext(path)[1]}'. "
            "Supported: PDF, DOCX, PPTX, ZIP-of-images."
        )

    # ---- PDF -------------------------------------------------------------
    def _ingest_pdf(self, path: str, doc_id: str) -> IngestedDocument:
        fitz = _import_pymupdf()

        img_dir = os.path.join(self.work_dir, f"{doc_id}_pages")
        os.makedirs(img_dir, exist_ok=True)
        zoom = self.config.render_dpi / 72.0

        pages: list[PageImage] = []
        natives: list[NativeText] = []
        native_chars = 0

        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise IngestionError(f"Could not open PDF: {exc}") from exc

        for i in range(pdf.page_count):
            page_no = i + 1
            try:
                page = pdf.load_page(i)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img_path = os.path.join(img_dir, f"page_{page_no}.png")
                pix.save(img_path)
                pages.append(
                    PageImage(
                        page_no=page_no,
                        image_path=img_path,
                        width=pix.width,
                        height=pix.height,
                    )
                )
                text = page.get_text("text") or ""
                native_chars += len(text.strip())
                natives.append(NativeText(page_no=page_no, text=text))
            except Exception as exc:  # one page failing must not abort
                logger.warning("page %s render failed: %s", page_no, exc)
                pages.append(
                    PageImage(page_no=page_no, image_path="", width=0, height=0)
                )

        pdf.close()

        # Heuristic: digital-born if there is a meaningful text layer.
        digital_born = native_chars > 40 * max(1, len(pages))
        return IngestedDocument(
            doc_id=doc_id,
            source_type="pdf",
            pages=pages,
            native_text=natives if digital_born else None,
        )

    # ---- DOCX / PPTX -----------------------------------------------------
    def _office_to_pdf(self, path: str) -> str:
        soffice = find_libreoffice()
        if not soffice:
            raise IngestionError(
                "LibreOffice (soffice) was not found on this system, so DOCX/PPTX "
                "cannot be converted to PDF. Install LibreOffice, or export the "
                "document to PDF first and upload that."
            )
        out_dir = os.path.join(self.work_dir, "converted")
        os.makedirs(out_dir, exist_ok=True)
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, path],
                check=True,
                capture_output=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired as exc:
            raise IngestionError("LibreOffice conversion timed out.") from exc
        except subprocess.CalledProcessError as exc:
            raise IngestionError(
                f"LibreOffice conversion failed: {exc.stderr.decode(errors='ignore')[:200]}"
            ) from exc

        base = os.path.splitext(os.path.basename(path))[0]
        pdf_path = os.path.join(out_dir, base + ".pdf")
        if not os.path.exists(pdf_path):
            raise IngestionError("LibreOffice did not produce a PDF output.")
        return pdf_path

    # ---- ZIP of images ---------------------------------------------------
    def _ingest_zip(self, path: str, doc_id: str) -> IngestedDocument:
        img_dir = os.path.join(self.work_dir, f"{doc_id}_pages")
        os.makedirs(img_dir, exist_ok=True)
        try:
            from PIL import Image
        except Exception:  # pragma: no cover
            Image = None

        pages: list[PageImage] = []
        try:
            with zipfile.ZipFile(path) as zf:
                names = [
                    n
                    for n in zf.namelist()
                    if os.path.splitext(n)[1].lower() in _IMG_EXTS
                    and not n.endswith("/")
                ]
                names.sort()
                if not names:
                    raise IngestionError("ZIP contains no recognised image files.")
                for page_no, name in enumerate(names, start=1):
                    data = zf.read(name)
                    out = os.path.join(img_dir, f"page_{page_no}.png")
                    w = h = 0
                    if Image is not None:
                        try:
                            from io import BytesIO

                            im = Image.open(BytesIO(data)).convert("RGB")
                            im.save(out, "PNG")
                            w, h = im.size
                        except Exception:
                            with open(out, "wb") as fh:
                                fh.write(data)
                    else:
                        with open(out, "wb") as fh:
                            fh.write(data)
                    pages.append(
                        PageImage(page_no=page_no, image_path=out, width=w, height=h)
                    )
        except zipfile.BadZipFile as exc:
            raise IngestionError("Uploaded file is not a valid ZIP archive.") from exc

        return IngestedDocument(
            doc_id=doc_id, source_type="zip_images", pages=pages, native_text=None
        )

    # ---- save an uploaded buffer to disk (Streamlit helper) --------------
    def save_upload(self, uploaded, suffix: str = "") -> str:
        """Persist a Streamlit UploadedFile / bytes to the work dir; return path."""
        name = getattr(uploaded, "name", None) or f"upload_{uuid.uuid4().hex}{suffix}"
        path = os.path.join(self.work_dir, name)
        data = uploaded.read() if hasattr(uploaded, "read") else uploaded
        if isinstance(data, str):
            data = data.encode("utf-8")
        with open(path, "wb") as fh:
            fh.write(data)
        return path
