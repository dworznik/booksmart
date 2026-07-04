"""Unit tests for the parser preference chain (no DB, no real parsers)."""

from pathlib import Path

import pytest

from app.parsing import ParseFailure, ParserChain, ParserUnavailable


class FakeParser:
    def __init__(
        self,
        name: str,
        formats: tuple[str, ...] = ("pdf",),
        markdown: str = "# ok",
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.supported_formats = frozenset(formats)
        self._markdown = markdown
        self._error = error
        self.calls = 0

    def parse(self, path: Path) -> str:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._markdown


def extract(chain: ParserChain, file_format: str = "pdf") -> tuple[object, list[str]]:
    log: list[str] = []
    result = chain.extract(Path(f"/nonexistent/book.{file_format}"), file_format, log.append)
    return result, log


def test_default_chain_preference_order_is_marker_pymupdf_ocr() -> None:
    from app.parsing import build_default_chain

    chain = build_default_chain()

    assert [parser.name for parser in chain._parsers] == ["marker", "pymupdf", "ocr"]


class TestParserChain:
    def test_first_success_wins_and_later_parsers_never_run(self) -> None:
        first = FakeParser("first")
        second = FakeParser("second")

        result, _ = extract(ParserChain([first, second]))

        assert result.parser == "first"  # type: ignore[attr-defined]
        assert result.markdown == "# ok"  # type: ignore[attr-defined]
        assert second.calls == 0

    def test_parse_failure_falls_through_to_next_parser(self) -> None:
        broken = FakeParser("broken", error=RuntimeError("boom"))
        working = FakeParser("working")

        result, log = extract(ParserChain([broken, working]))

        assert result.parser == "working"  # type: ignore[attr-defined]
        assert any("broken" in line and "boom" in line for line in log)

    def test_unavailable_parser_falls_through(self) -> None:
        missing = FakeParser("missing", error=ParserUnavailable("not installed"))
        working = FakeParser("working")

        result, log = extract(ParserChain([missing, working]))

        assert result.parser == "working"  # type: ignore[attr-defined]
        assert any("missing" in line and "unavailable" in line for line in log)

    def test_parser_not_supporting_format_is_skipped_without_calling(self) -> None:
        pdf_only = FakeParser("pdf-only", formats=("pdf",))
        epub_capable = FakeParser("epub-capable", formats=("pdf", "epub"))

        result, log = extract(ParserChain([pdf_only, epub_capable]), file_format="epub")

        assert result.parser == "epub-capable"  # type: ignore[attr-defined]
        assert pdf_only.calls == 0
        assert any("pdf-only" in line and "skip" in line for line in log)

    def test_output_without_text_content_counts_as_failure(self) -> None:
        blank = FakeParser("blank", markdown="\n\n-----\n\n")
        working = FakeParser("working")

        result, log = extract(ParserChain([blank, working]))

        assert result.parser == "working"  # type: ignore[attr-defined]
        assert any("blank" in line for line in log)

    def test_all_parsers_failing_raises_with_format_and_parser_names(self) -> None:
        first = FakeParser("first", error=RuntimeError("bad header"))
        second = FakeParser("second", error=ParserUnavailable("nope"))

        with pytest.raises(ParseFailure) as excinfo:
            extract(ParserChain([first, second]), file_format="epub")

        message = str(excinfo.value)
        assert "epub" in message
        assert "first" in message and "second" in message

    def test_every_attempt_is_logged(self) -> None:
        parsers = [
            FakeParser("a", error=RuntimeError("x")),
            FakeParser("b", error=ParserUnavailable("y")),
            FakeParser("c"),
        ]

        _, log = extract(ParserChain(parsers))

        joined = "\n".join(log)
        assert "a" in joined and "b" in joined and "c" in joined
        assert any("succeeded" in line for line in log)
