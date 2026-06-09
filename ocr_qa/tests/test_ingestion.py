"""M3 ingestion tests: PDF renders + native text extracted; ZIP-of-images
becomes pages; source-type detection; graceful degradation messages.
Run: pytest ocr_qa/tests/test_ingestion.py -q
"""

from __future__ import annotations

import io
import os
import zipfile

import pytest

from ocr_qa.ingestion.normalizer import (
    IngestionError,
    Normalizer,
    detect_source_type,
)


def test_detect_source_type():
    assert detect_source_type("a.pdf") == "pdf"
    assert detect_source_type("a.docx") == "docx"
    assert detect_source_type("a.pptx") == "pptx"
    assert detect_source_type("a.zip") == "zip_images"
    assert detect_source_type("a.txt") == "unknown"


@pytest.fixture
def norm(tmp_path):
    return Normalizer(work_dir=str(tmp_path))


def _make_pdf(path: str, pages: int = 2):
    from ocr_qa.ingestion.normalizer import _import_pymupdf

    fitz = _import_pymupdf()
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i + 1}: this is a digital-born PDF with a real text layer "
            f"used to validate native-text extraction and CER agreement.",
            fontsize=12,
        )
    doc.save(path)
    doc.close()


def test_ingest_pdf_renders_and_extracts_native(norm, tmp_path):
    pdf = str(tmp_path / "doc.pdf")
    _make_pdf(pdf, pages=2)
    ing = norm.ingest(pdf)
    assert ing.source_type == "pdf"
    assert len(ing.pages) == 2
    for pi in ing.pages:
        assert os.path.exists(pi.image_path)
        assert pi.width > 0 and pi.height > 0
    # digital-born => native text present
    assert ing.native_text is not None
    assert "digital-born" in ing.native_for(1)


def test_render_dpi_applied(norm, tmp_path):
    pdf = str(tmp_path / "doc.pdf")
    _make_pdf(pdf, pages=1)
    ing = norm.ingest(pdf)
    # default 200 DPI on a letter page (612x792 pt) => ~1700x2200 px
    assert ing.pages[0].width > 1200


def test_ingest_zip_images(norm, tmp_path):
    from PIL import Image

    zpath = str(tmp_path / "imgs.zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(zpath, "w") as zf:
        for n in range(1, 4):
            img = Image.new("RGB", (200, 280), (255, 255, 255))
            b = io.BytesIO()
            img.save(b, "PNG")
            zf.writestr(f"page{n:02d}.png", b.getvalue())
    ing = norm.ingest(zpath)
    assert ing.source_type == "zip_images"
    assert len(ing.pages) == 3
    assert ing.native_text is None
    for pi in ing.pages:
        assert os.path.exists(pi.image_path)


def test_zip_without_images_errors(norm, tmp_path):
    zpath = str(tmp_path / "empty.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("readme.txt", "no images here")
    with pytest.raises(IngestionError):
        norm.ingest(zpath)


def test_missing_file_errors(norm):
    with pytest.raises(IngestionError):
        norm.ingest("does_not_exist.pdf")
