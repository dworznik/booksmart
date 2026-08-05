"""The parser comparison, and the guards that keep it honest.

Three things live here and only one of them is slow.

``TestMetrics`` checks the five counters say what they claim, against markdown
written by hand. They run always and need nothing installed.

``TestChainSelection`` pins the behaviour the comparison depends on — which
parser the chain picks, and what it does when one is unavailable. It runs
always, against a generated PDF, and proves nothing about parse *quality*.

``TestRealEval`` is the comparison. It is skipped unless a document is named,
and it writes a table to stdout:

    BOOKSMART_PARSER_EVAL_PDF=~/Downloads/bash.pdf \\
    uv run pytest packages/core/tests/test_parser_eval.py -k RealEval -s

The chain prefers ``marker`` over ``pymupdf``, but ``marker-pdf`` is undeclared
and absent from CI, so ``MarkerParser`` has never actually run and every PDF
this project has parsed went through ``pymupdf4llm``. Installing marker to find
out whether that matters is what booksmart#80 asks for; see
``docs/research/parser-comparison.md`` for how to install it on macOS and what
to record.

Which document matters more than it looks. ``pymupdf4llm``'s code detection is
font-dependent, not absent — it fenced 70 code blocks per 20 pages of the GNU
Bash manual and none at all over 40 pages of one typeset programming book. Run
this over both an easy document and an awkward one before concluding anything.
"""

import os
from pathlib import Path

import pytest

from booksmart_core.parsing import ParseFailure, ParserChain, build_default_chain

from . import parser_eval

EVAL_PDF = os.environ.get("BOOKSMART_PARSER_EVAL_PDF", "")
# Marker is an ML pipeline and far slower than pymupdf4llm, so a whole manual is
# a poor first run — long enough to be indistinguishable from a hang. Default to
# a slice; set to 0 for the whole document once the shape of the answer is known.
EVAL_PAGES = int(os.environ.get("BOOKSMART_PARSER_EVAL_PAGES", "20"))
# Where to write each parser's markdown, so a twenty-minute marker run leaves
# something a human can diff. Unset, nothing is written.
EVAL_DUMP = os.environ.get("BOOKSMART_PARSER_EVAL_DUMP", "")
# First page of the slice (0-based). Books keep their code deep in the text —
# the front of one corpus book yielded two code lines in twenty pages.
EVAL_START = int(os.environ.get("BOOKSMART_PARSER_EVAL_START", "0"))
# Comma-separated parser names to compare; unset means every available one.
# A format experiment (PDF vs EPUB through the same parser) needs only pymupdf,
# and marker costs twenty minutes a run once it is importable in the venv.
EVAL_PARSERS = os.environ.get("BOOKSMART_PARSER_EVAL_PARSERS", "")


class TestMetrics:
    """A counter that lies makes the comparison a measurement of nothing."""

    def test_it_counts_headings_fences_and_furniture(self) -> None:
        markdown = "\n".join(
            [
                "# Chapter 1",
                "S M A L L T A L K B E S T",
                "prose",
                "```",
                "code()",
                "```",
                "12",
                "## Section",
            ]
        )

        metrics = parser_eval.measure(markdown, parser="probe", pages=2, seconds=0.0)

        assert metrics.headings == 2
        assert metrics.code_fences == 1  # a pair, not two
        assert metrics.letter_spaced_runs == 1
        assert metrics.page_number_lines == 1

    def test_a_page_number_inside_a_sentence_is_not_furniture(self) -> None:
        """Only a line that is *nothing but* digits is page furniture."""
        metrics = parser_eval.measure(
            "there were 12 parsers", parser="probe", pages=1, seconds=0.0
        )

        assert metrics.page_number_lines == 0

    def test_an_unterminated_fence_does_not_count_as_a_block(self) -> None:
        metrics = parser_eval.measure("```\ncode()", parser="probe", pages=1, seconds=0.0)

        assert metrics.code_fences == 0

    def test_characters_per_page_survives_an_empty_document(self) -> None:
        assert parser_eval.measure("", parser="p", pages=0, seconds=0.0).characters_per_page == 0


class TestChainSelection:
    """What the chain does, as opposed to how well it does it."""

    def test_pymupdf_parses_a_generated_pdf(self, tmp_path: Path) -> None:
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf")

        result = build_default_chain().extract(path, "pdf", lambda _: None)

        assert result.markdown.strip()
        assert "Extracting Text" in result.markdown

    def test_the_winning_parser_is_reported(self, tmp_path: Path) -> None:
        """`Book.parser_used` is set from this, and it is the only record of
        which parser produced a corpus."""
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf")

        result = build_default_chain().extract(path, "pdf", lambda _: None)

        assert result.parser in parser_eval.available_parsers()

    def test_marker_is_preferred_when_it_is_installed(self, tmp_path: Path) -> None:
        """The chain's order is the policy. This documents it either way, so a
        reordering shows up as a failing test rather than a changed corpus."""
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf")

        result = build_default_chain().extract(path, "pdf", lambda _: None)

        if parser_eval.marker_is_installed():
            assert result.parser == "marker"
        else:
            assert result.parser == "pymupdf"

    def test_an_unavailable_parser_is_recorded_and_skipped(self, tmp_path: Path) -> None:
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf")
        log: list[str] = []

        build_default_chain().extract(path, "pdf", log.append)

        if not parser_eval.marker_is_installed():
            assert any("marker" in line and "unavailable" in line for line in log)

    def test_a_chain_with_nothing_available_fails_loudly(self, tmp_path: Path) -> None:
        """A parse that silently produced nothing would be far worse than one
        that raises: an empty book ingests, and scores zero on everything."""
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf")

        with pytest.raises(ParseFailure, match="no parser succeeded"):
            ParserChain([]).extract(path, "pdf", lambda _: None)

    def test_the_probe_document_carries_what_the_metrics_count(
        self, tmp_path: Path
    ) -> None:
        """The generated fixture is only useful if it really contains a heading,
        a listing, a letter-spaced header and a page number."""
        path = parser_eval.build_probe_pdf(tmp_path / "probe.pdf", pages=2)

        markdown = build_default_chain().extract(path, "pdf", lambda _: None).markdown
        metrics = parser_eval.measure(markdown, parser="probe", pages=2, seconds=0.0)

        assert metrics.characters_per_page > 200
        assert metrics.letter_spaced_runs >= 1
        # Headings, fences and page-number handling are what the comparison is
        # *for*, so they are reported rather than asserted — pinning today's
        # numbers here would turn an improvement in pymupdf4llm into a failure.
        print(f"\nprobe document via {parser_eval.available_parsers()}:\n{metrics.as_row()}")


class TestFirstPages:
    def test_it_truncates_and_keeps_the_text(self, tmp_path: Path) -> None:
        source = parser_eval.build_probe_pdf(tmp_path / "whole.pdf", pages=5)

        sliced = parser_eval.first_pages(source, 2, tmp_path / "slice.pdf")

        assert parser_eval.page_count(sliced) == 2
        import pymupdf

        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Extracting Text" in text

    def test_asking_for_more_pages_than_exist_is_the_whole_document(
        self, tmp_path: Path
    ) -> None:
        source = parser_eval.build_probe_pdf(tmp_path / "whole.pdf", pages=3)

        sliced = parser_eval.first_pages(source, 999, tmp_path / "slice.pdf")

        assert parser_eval.page_count(sliced) == 3

    def test_a_slice_can_start_mid_document(self, tmp_path: Path) -> None:
        """Books keep their code deep in the text; a slice from page one of a
        corpus book held two code lines in twenty pages, which quietly turned a
        code-handling comparison into a front-matter comparison."""
        source = parser_eval.build_probe_pdf(tmp_path / "whole.pdf", pages=6)

        sliced = parser_eval.first_pages(source, 2, tmp_path / "slice.pdf", start=3)

        assert parser_eval.page_count(sliced) == 2
        import pymupdf

        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Chapter 4" in text  # 0-based start=3 is the fourth page


    def test_an_epub_can_be_sliced_too(self, tmp_path: Path) -> None:
        """select() is PDF-only, so an EPUB slice goes through a whole-document
        conversion first. Without this, the first EPUB anyone pointed the eval
        at would crash inside the harness rather than produce a comparison."""
        import zipfile

        source = tmp_path / "book.epub"
        with zipfile.ZipFile(source, "w") as z:
            z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
            z.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container version="1.0" '
                'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                '<rootfile full-path="content.opf" '
                'media-type="application/oebps-package+xml"/></rootfiles></container>',
            )
            z.writestr(
                "content.opf",
                '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
                'version="3.0" unique-identifier="id"><metadata '
                'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
                '<dc:identifier id="id">x</dc:identifier><dc:language>en</dc:language>'
                '</metadata><manifest><item id="c1" href="ch1.xhtml" '
                'media-type="application/xhtml+xml"/></manifest>'
                '<spine><itemref idref="c1"/></spine></package>',
            )
            z.writestr(
                "ch1.xhtml",
                '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml"><body>'
                "<h1>Sliceable</h1>" + "<p>prose to fill pages. </p>" * 400 + "</body></html>",
            )

        sliced = parser_eval.first_pages(source, 2, tmp_path / "slice.epub")

        assert sliced.suffix == ".pdf"  # rendered form; select is PDF-only
        assert parser_eval.page_count(sliced) == 2
        import pymupdf

        doc = pymupdf.open(sliced)
        text = doc[0].get_text()
        doc.close()
        assert "Sliceable" in text


class TestRealEval:
    """The comparison. Needs a real document, and marker installed to be
    interesting."""

    @pytest.mark.skipif(not EVAL_PDF, reason="set BOOKSMART_PARSER_EVAL_PDF to a PDF")
    def test_compare_every_available_parser(self, tmp_path: Path) -> None:
        source = Path(EVAL_PDF).expanduser()
        assert source.is_file(), f"no such file: {source}"

        path = source
        if EVAL_PAGES:
            path = parser_eval.first_pages(
                source, EVAL_PAGES, tmp_path / source.name, start=EVAL_START
            )

        # The range goes into the directory name so a mid-book run cannot
        # overwrite a front-matter run — which one did, destroying the dumps
        # that evidenced a content-loss finding.
        dump = (
            Path(EVAL_DUMP).expanduser() / f"pages-{EVAL_START + 1}-{EVAL_START + EVAL_PAGES}"
            if EVAL_DUMP and EVAL_PAGES
            else Path(EVAL_DUMP).expanduser() / "all-pages"
            if EVAL_DUMP
            else None
        )
        wanted = [p.strip() for p in EVAL_PARSERS.split(",") if p.strip()] or None
        results = parser_eval.compare(path, parsers=wanted, dump_to=dump)

        sliced = parser_eval.page_count(path)
        scope = (
            f"pages {EVAL_START + 1}-{EVAL_START + sliced}" if EVAL_PAGES else "all"
        )
        print(f"\n{source.name} — {scope} of {parser_eval.page_count(source)} pages")
        print(parser_eval.render(results))
        if not parser_eval.marker_is_installed():
            print(
                "\nmarker is not installed, so this compares one parser against "
                "itself. See docs/research/parser-comparison.md."
            )
        if dump is not None:
            print(f"markdown written to {dump}")
        assert results, "no parser could be run at all"
