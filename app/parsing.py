"""Parser preference chain: Marker -> PyMuPDF -> OCR, first success wins.

Marker is an optional dependency (it pulls large ML models); install it with
`uv add marker-pdf` to enable. When absent, the chain logs it as unavailable
and falls through. OCR needs the tesseract binary plus its tessdata files.
"""

import importlib
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Protocol

import pymupdf
import pymupdf4llm


class ParserUnavailable(Exception):
    """The parser's backing tool/library is not installed on this machine."""


class ParseFailure(Exception):
    """No parser produced usable output."""


@dataclass(frozen=True)
class ParseResult:
    markdown: str
    parser: str


class Parser(Protocol):
    name: str
    supported_formats: frozenset[str]

    def parse(self, path: Path) -> str: ...


class MarkerParser:
    # PDF-only: we drive marker's PdfConverter. Marker can convert EPUB via a
    # different converter; if that's ever wanted, extend this rather than
    # assuming PdfConverter accepts EPUB. EPUBs currently have a single-parser
    # chain (PyMuPDF).
    name = "marker"
    supported_formats = frozenset({"pdf"})

    def __init__(self) -> None:
        self._converter: Any = None

    def parse(self, path: Path) -> str:
        try:
            pdf_module = importlib.import_module("marker.converters.pdf")
            models_module = importlib.import_module("marker.models")
            output_module = importlib.import_module("marker.output")
        except ImportError as exc:
            raise ParserUnavailable("marker-pdf is not installed") from exc
        if self._converter is None:  # model load is expensive; reuse across jobs
            self._converter = pdf_module.PdfConverter(
                artifact_dict=models_module.create_model_dict()
            )
        rendered = self._converter(str(path))
        markdown, _, _ = output_module.text_from_rendered(rendered)
        return str(markdown)


class PyMuPDFParser:
    name = "pymupdf"
    supported_formats = frozenset({"pdf", "epub"})

    def parse(self, path: Path) -> str:
        return pymupdf4llm.to_markdown(str(path))


TESSDATA_CANDIDATES = (
    "/opt/homebrew/share/tessdata",
    "/usr/local/share/tessdata",
    "/usr/share/tessdata",
    "/usr/share/tesseract-ocr/*/tessdata",
)


def find_tessdata() -> str | None:
    configured = os.environ.get("TESSDATA_PREFIX")
    if configured and Path(configured).is_dir():
        return configured
    for pattern in TESSDATA_CANDIDATES:
        for match in sorted(glob(pattern)):
            if Path(match).is_dir():
                return match
    return None


class OcrParser:
    name = "ocr"
    supported_formats = frozenset({"pdf"})

    def parse(self, path: Path) -> str:
        if shutil.which("tesseract") is None:
            raise ParserUnavailable("tesseract binary not found")
        tessdata = find_tessdata()
        if tessdata is None:
            raise ParserUnavailable("tessdata directory not found; set TESSDATA_PREFIX")

        pages: list[str] = []
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            for number, page in enumerate(doc, start=1):
                textpage = page.get_textpage_ocr(full=True, dpi=300, tessdata=tessdata)
                text = page.get_text(textpage=textpage).strip()
                if text:
                    pages.append(f"## Page {number}\n\n{text}")
        return "\n\n".join(pages)


class ParserChain:
    def __init__(self, parsers: Sequence[Parser]) -> None:
        self._parsers = list(parsers)

    def extract(
        self, path: Path, file_format: str, log: Callable[[str], None]
    ) -> ParseResult:
        outcomes: list[str] = []

        def record(parser: Parser, outcome: str) -> None:
            log(f"{parser.name}: {outcome}")
            outcomes.append(f"{parser.name}: {outcome}")

        for parser in self._parsers:
            if file_format not in parser.supported_formats:
                record(parser, f"skipped (does not support {file_format})")
                continue
            log(f"{parser.name}: attempting")
            try:
                markdown = parser.parse(path)
            except ParserUnavailable as exc:
                record(parser, f"unavailable — {exc}")
                continue
            except Exception as exc:
                record(parser, f"failed — {type(exc).__name__}: {exc}")
                continue
            if not any(ch.isalnum() for ch in markdown):
                record(parser, "failed — produced no text content")
                continue
            log(f"{parser.name}: succeeded")
            return ParseResult(markdown=markdown, parser=parser.name)
        raise ParseFailure(
            f"no parser succeeded for {file_format} file; attempts: {'; '.join(outcomes)}"
        )


def build_default_chain() -> ParserChain:
    return ParserChain([MarkerParser(), PyMuPDFParser(), OcrParser()])
