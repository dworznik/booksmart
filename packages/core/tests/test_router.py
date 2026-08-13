"""The router, in isolation: which route a document takes, and what happens when
it fails. No database, no real extractors.

The whole point of this file is that there is *no fallback*. Measurement
measured the old chain's real failure mode as silently dropping text rather than
raising, so a fallback triggered by exceptions caught almost none of the cases it
was written for — while guaranteeing nobody could tell afterwards which extractor
produced an artifact. So every test here that would have asserted "falls through
to the next one" asserts "raises, naming the route" instead.
"""

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from booksmart_core.parsing.contract import ExtractorReport
from booksmart_core.parsing import (
    MIN_CHARS_PER_PAGE,
    ParseFailure,
    ParseResult,
    ExtractorRouter,
    ExtractorUnavailable,
    build_default_router,
    text_layer_density,
)


class FakeExtractor:
    def __init__(
        self,
        route: str,
        *,
        markdown: str = "# ok",
        error: Exception | None = None,
        declines: tuple[str, ...] = (),
    ) -> None:
        self.route = route
        self._markdown = markdown
        self._error = error
        self._declines = declines
        self.calls = 0

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return ParseResult(
            markdown=self._markdown,
            report=ExtractorReport(route=self.route, declines=self._declines),
        )


def router(**overrides: FakeExtractor) -> tuple[ExtractorRouter, dict[str, FakeExtractor]]:
    extractors = {
        "epub": FakeExtractor("epub"),
        "pdf": FakeExtractor("pdf"),
        "ocr": FakeExtractor("pdf-ocr"),
    } | overrides
    return ExtractorRouter(**extractors), extractors  # type: ignore[arg-type]


def extract(
    built: ExtractorRouter, path: Path, file_format: str = "pdf"
) -> tuple[ParseResult, list[str]]:
    log: list[str] = []
    return built.extract(path, file_format, log.append), log


def text_pdf(path: Path, *, pages: int = 3) -> Path:
    """A PDF with a healthy text layer on every page."""
    document = pymupdf.open()
    for number in range(pages):
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 520, 700),
            f"Page {number + 1}. " + "Ordinary prose, enough of it to read as a text layer. " * 20,
            fontsize=10,
        )
    document.save(path)
    document.close()
    return path


def scanned_pdf(path: Path, *, pages: int = 2) -> Path:
    """A PDF carrying only a raster image of text — no text layer at all."""
    source = pymupdf.open()
    page = source.new_page()
    page.insert_text((72, 150), "SCANNED BOOK", fontsize=36)
    pixmap = page.get_pixmap(dpi=120)
    source.close()

    document = pymupdf.open()
    for _ in range(pages):
        image_page = document.new_page()
        image_page.insert_image(image_page.rect, pixmap=pixmap)
    document.save(path)
    document.close()
    return path


class TestRouting:
    def test_an_epub_goes_to_the_epub_route(self, tmp_path: Path) -> None:
        built, extractors = router()

        result, log = extract(built, tmp_path / "book.epub", "epub")

        assert result.route == "epub"
        assert extractors["pdf"].calls == 0 and extractors["ocr"].calls == 0
        assert any("route: epub" in line for line in log)

    def test_a_pdf_with_a_healthy_text_layer_never_reaches_ocr(self, tmp_path: Path) -> None:
        built, extractors = router()

        result, log = extract(built, text_pdf(tmp_path / "book.pdf"))

        assert result.route == "pdf"
        assert extractors["ocr"].calls == 0
        assert any("characters per page" in line for line in log)

    def test_a_pdf_with_no_text_layer_goes_to_ocr(self, tmp_path: Path) -> None:
        built, extractors = router()

        result, log = extract(built, scanned_pdf(tmp_path / "scan.pdf"))

        assert result.route == "pdf-ocr"
        assert extractors["pdf"].calls == 0
        assert any("not carrying this book" in line for line in log)

    def test_an_unknown_format_is_refused_rather_than_guessed(self, tmp_path: Path) -> None:
        built, _ = router()

        with pytest.raises(ParseFailure, match="no route for 'mobi'"):
            extract(built, tmp_path / "book.mobi", "mobi")


class TestTheTextLayerProbe:
    def test_a_native_pdf_is_well_over_the_threshold(self, tmp_path: Path) -> None:
        assert text_layer_density(text_pdf(tmp_path / "book.pdf")) > MIN_CHARS_PER_PAGE

    def test_a_scan_is_at_zero(self, tmp_path: Path) -> None:
        assert text_layer_density(scanned_pdf(tmp_path / "scan.pdf")) < MIN_CHARS_PER_PAGE

    def test_a_document_with_no_pages_is_zero_rather_than_a_crash(
        self, tmp_path: Path
    ) -> None:
        """A PDF with an empty page tree divides by zero otherwise, and the
        routing decision is made before anything has had a chance to report the
        file as broken.

        Written as bytes because PyMuPDF refuses to *save* a zero-page document
        ("cannot save with zero pages") while opening one quite happily — which
        is the asymmetry that lets such a file exist in the wild at all.
        `booksmart-bench`'s `sources` verb has a branch for it for the same
        reason.
        """
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
            b"trailer<</Root 1 0 R>>\n"
        )

        assert text_layer_density(empty) == 0

    def test_an_image_cover_does_not_route_a_healthy_book_to_ocr(self, tmp_path: Path) -> None:
        """The probe samples pages spread through the document, not from the
        front — the first pages of a book are cover, half title and copyright,
        several of which are images in a perfectly healthy PDF."""
        path = tmp_path / "book.pdf"
        cover = pymupdf.open()
        page = cover.new_page()
        page.insert_text((72, 150), "COVER", fontsize=36)
        pixmap = page.get_pixmap(dpi=120)
        cover.close()

        document = pymupdf.open()
        for _ in range(3):
            document.new_page().insert_image(document[-1].rect, pixmap=pixmap)
        for number in range(60):
            body = document.new_page()
            body.insert_textbox(
                pymupdf.Rect(72, 72, 520, 700),
                f"Page {number}. " + "Ordinary body prose in a healthy book. " * 25,
                fontsize=10,
            )
        document.save(path)
        document.close()

        assert text_layer_density(path) > MIN_CHARS_PER_PAGE


class TestNoFallback:
    def test_a_failing_route_raises_and_names_the_route(self, tmp_path: Path) -> None:
        built, extractors = router(pdf=FakeExtractor("pdf", error=RuntimeError("bad header")))

        with pytest.raises(ParseFailure) as excinfo:
            extract(built, text_pdf(tmp_path / "book.pdf"))

        assert "pdf route failed" in str(excinfo.value)
        assert "bad header" in str(excinfo.value)
        assert extractors["ocr"].calls == 0, "there is no second attempt"

    def test_an_unavailable_route_raises_with_the_remedy(self, tmp_path: Path) -> None:
        """A missing tesseract is a different problem from a bad book, so the
        message says which — but it is just as terminal."""
        built, _ = router(ocr=FakeExtractor("pdf-ocr", error=ExtractorUnavailable("no tesseract")))

        with pytest.raises(ParseFailure, match="pdf-ocr route is unavailable"):
            extract(built, scanned_pdf(tmp_path / "scan.pdf"))

    def test_output_with_no_text_content_is_a_failure(self, tmp_path: Path) -> None:
        """A parse that silently produced nothing is far worse than one that
        raises: an empty book ingests, and then scores zero on everything."""
        built, _ = router(pdf=FakeExtractor("pdf", markdown="\n\n-----\n\n"))

        with pytest.raises(ParseFailure, match="produced no text content"):
            extract(built, text_pdf(tmp_path / "book.pdf"))

    def test_a_routes_own_words_survive_being_told_which_route_it_was(
        self, tmp_path: Path
    ) -> None:
        """An extractor that has already said something precise — an EPUB spine
        item missing from the zip — must not have it buried under a wrapper
        naming only the exception class. But "which route produced this" is the
        first question a bad artifact raises, so both have to be in the message."""
        built, _ = router(
            epub=FakeExtractor("epub", error=ParseFailure("spine item ch12 is absent"))
        )

        with pytest.raises(ParseFailure) as excinfo:
            extract(built, tmp_path / "book.epub", "epub")

        assert "epub route failed" in str(excinfo.value)
        assert "spine item ch12 is absent" in str(excinfo.value)


class TestWhatTheLogSays:
    def test_a_decline_reaches_the_log_as_well_as_the_report(self, tmp_path: Path) -> None:
        """The report is what the gate ratchets. The log is what somebody
        watching an ingest sees, and a book that quietly stopped fencing is the
        change worth noticing at the time."""
        built, _ = router(pdf=FakeExtractor("pdf", declines=("no font signal",)))

        _, log = extract(built, text_pdf(tmp_path / "book.pdf"))

        assert any("declined — no font signal" in line for line in log)

    def test_success_is_stated(self, tmp_path: Path) -> None:
        _, log = extract(*[router()[0], text_pdf(tmp_path / "book.pdf")])

        assert any("succeeded" in line for line in log)


class TestTheDefaultRouter:
    def test_it_covers_all_three_routes(self) -> None:
        built = build_default_router()

        assert set(built._by_route) == {"epub", "pdf", "pdf-ocr"}  # noqa: SLF001

    def test_it_is_built_fresh_each_time(self) -> None:
        """The module-level chain this replaced existed only to amortise marker's
        model load. Every remaining extractor is stateless, so a shared instance
        buys nothing and costs the ability to substitute one in a test."""
        assert build_default_router() is not build_default_router()
