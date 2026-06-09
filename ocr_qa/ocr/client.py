"""Chandra OCR clients (Sec 0 STEP 1).

Both clients return the SAME shape: ``list[PageOCR]`` via ``ChandraParser``.
From the caller's perspective the two OCR paths are indistinguishable
afterwards (the spec's core requirement).

* ``MockChandraClient``  -- "OCR already done": user supplies output.json,
                            output.md, output.metadata.json. Fully functional.
* ``RealChandraClient``  -- "Run Chandra OCR": STUB now. The transport is a
                            placeholder; the PARSE step (real) reuses the same
                            ChandraParser on whatever JSON/MD/metadata the API
                            would return.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

from ocr_qa.models import PageOCR
from ocr_qa.ocr.chandra_parser import (
    ChandraParser,
    load_json_file,
    load_text_file,
)

# Accept a path string or an already-read bytes/str/buffer (Streamlit upload).
FileLike = Union[str, bytes, Any]


@dataclass
class ChandraBundle:
    """The raw artefacts Chandra emits, pre-parse. ``output_chunks`` is the
    optional 4th artefact (chunk-level content + table stats)."""

    output_json: Any
    output_md: Optional[str]
    metadata: Any
    output_chunks: Any = None
    output_html: Optional[str] = None


def _read_json(src: FileLike) -> Any:
    import json

    if isinstance(src, str) and os.path.exists(src):
        return load_json_file(src)
    if isinstance(src, bytes):
        return json.loads(src.decode("utf-8"))
    if hasattr(src, "read"):  # Streamlit UploadedFile / file handle
        data = src.read()
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return json.loads(data)
    if isinstance(src, str):
        return json.loads(src)
    return src  # already a python object


def _read_text(src: FileLike) -> str:
    if src is None:
        return ""
    if isinstance(src, str) and os.path.exists(src):
        return load_text_file(src)
    if isinstance(src, bytes):
        return src.decode("utf-8")
    if hasattr(src, "read"):
        data = src.read()
        return data.decode("utf-8") if isinstance(data, bytes) else data
    if isinstance(src, str):
        return src
    return ""


class BaseChandraClient(ABC):
    def __init__(self) -> None:
        self.parser = ChandraParser()

    @abstractmethod
    def get_bundle(self, *args: Any, **kwargs: Any) -> ChandraBundle:
        ...

    def run(self, *args: Any, **kwargs: Any) -> list[PageOCR]:
        bundle = self.get_bundle(*args, **kwargs)
        return self.parser.parse(
            bundle.output_json,
            bundle.output_md,
            bundle.metadata,
            bundle.output_chunks,
            bundle.output_html,
        )

    @property
    def parse_log(self) -> list[dict[str, Any]]:
        return self.parser.parse_log


class MockChandraClient(BaseChandraClient):
    """OCR-already-done path. Requires output.json + output.md + metadata."""

    def get_bundle(
        self,
        output_json: FileLike,
        output_md: FileLike = None,
        metadata: FileLike = None,
        output_chunks: FileLike = None,
        output_html: FileLike = None,
    ) -> ChandraBundle:
        return ChandraBundle(
            output_json=_read_json(output_json),
            output_md=_read_text(output_md) if output_md is not None else None,
            metadata=_read_json(metadata) if metadata is not None else None,
            output_chunks=_read_json(output_chunks) if output_chunks is not None else None,
            output_html=_read_text(output_html) if output_html is not None else None,
        )

    def from_dir(self, directory: str) -> ChandraBundle:
        """Convenience: load the standard triple (+ optional chunks/html) from a folder."""
        oj = os.path.join(directory, "output.json")
        omd = os.path.join(directory, "output.md")
        omm = os.path.join(directory, "output.metadata.json")
        omc = os.path.join(directory, "output_chunks.json")
        omh = os.path.join(directory, "output.html")
        return self.get_bundle(
            oj,
            omd if os.path.exists(omd) else None,
            omm if os.path.exists(omm) else None,
            omc if os.path.exists(omc) else None,
            omh if os.path.exists(omh) else None,
        )

    def run_from_dir(self, directory: str) -> list[PageOCR]:
        b = self.from_dir(directory)
        return self.parser.parse(
            b.output_json, b.output_md, b.metadata, b.output_chunks, b.output_html
        )


class RealChandraClient(BaseChandraClient):
    """Run-Chandra-OCR path. STUB transport; real parse.

    ``send`` is intentionally not wired to a live endpoint yet. It raises a
    clear, user-facing message so the UI can degrade. Once an endpoint exists,
    only ``get_bundle`` needs to change -- the parse path is already real.
    """

    NOT_IMPLEMENTED_MSG = (
        "Run-Chandra-OCR is not wired to a live endpoint in this build. "
        "Use 'OCR already done' and upload output.json / output.md / "
        "output.metadata.json, or point RealChandraClient at your Chandra "
        "service."
    )

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__()
        self.endpoint = endpoint
        self.api_key = api_key

    def get_bundle(self, source_path: str, **kwargs: Any) -> ChandraBundle:  # noqa: D401
        raise NotImplementedError(self.NOT_IMPLEMENTED_MSG)
