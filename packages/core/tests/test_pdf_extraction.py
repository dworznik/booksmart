"""The `pdf` route: the font ladder, the headings, the declines.

Every PDF here is generated from stated typography — a body face, a code face,
sizes, margins — because that is exactly what the rules read. A fixture that says
"body is 10pt Times and listings are 9pt Courier indented 24pt" is a readable
statement of the case a tier exists for; a checked-in book would be neither
readable nor licensable.

The counts these rules were fitted to live in the gate's
baseline. What lives here is the *shape* of each case, and the properties that
have to hold whatever the corpus turns out to contain — above all that a rule
which cannot see the signal declines rather than guesses (ADR 0003).
"""

from pathlib import Path

import pymupdf
import pytest

from booksmart_core.parsing.pdf import PdfExtractor, family_of, read_lines, read_typography
from booksmart_core.structure import detect_structure

# Faces MuPDF ships and can be asked for by name, so a fixture can state its
# typography without shipping a font file. "cour" is flagged monospaced; "helv"
# and "tiro" are not.
MONO = "cour"
SANS = "helv"
SERIF = "tiro"

BODY_POINTS = 10.0
PROSE = (
    "Ordinary body prose that runs the full measure of the column and wraps "
    "across several lines, so that the median body line length is the length of "
    "a line of prose and not of something else. "
)


def build_pdf(
    path: Path,
    pages: list[dict[str, object]],
    *,
    body_font: str = SERIF,
    body_points: float = BODY_POINTS,
) -> Path:
    """A PDF from a list of page descriptions.

    Each page is ``{"prose": n, "heading": (text, points), "code": (lines, font,
    points, indent)}``; every part is optional.
    """
    document = pymupdf.open()
    for page_spec in pages:
        page = document.new_page()
        y = 60.0
        heading = page_spec.get("heading")
        if heading is not None:
            assert isinstance(heading, tuple)
            text, points = heading
            page.insert_text((72, y), str(text), fontsize=float(points), fontname=body_font)
            y += float(points) * 2
        prose_lines = int(page_spec.get("prose", 6))  # type: ignore[arg-type]
        if prose_lines:
            height = prose_lines * body_points * 1.4
            page.insert_textbox(
                pymupdf.Rect(72, y, 520, y + height),
                PROSE * max(1, prose_lines // 2),
                fontsize=body_points,
                fontname=body_font,
            )
            y += height + body_points
        code = page_spec.get("code")
        if code is not None:
            assert isinstance(code, tuple)
            lines, font, points, indent = code
            assert isinstance(lines, list)
            for line in lines:
                leading = len(line) - len(line.lstrip(" "))
                page.insert_text(
                    (72 + float(indent) + leading * float(points) * 0.6, y),
                    line.lstrip(" "),
                    fontsize=float(points),
                    fontname=str(font),
                )
                y += float(points) * 1.4
    document.save(path)
    document.close()
    return path


LISTING = ["def parse(self):", "    if self.ready:", "        return 1", "    return 0"]


def extract(path: Path) -> tuple[str, object]:
    result = PdfExtractor().extract(path, lambda _: None)
    return result.markdown, result.report


def typography(path: Path) -> object:
    document = pymupdf.open(path)
    try:
        return read_typography(read_lines(document))
    finally:
        document.close()


def fence_bodies(markdown: str) -> list[str]:
    bodies: list[str] = []
    inside: list[str] | None = None
    for line in markdown.split("\n"):
        if line.startswith("```"):
            if inside is None:
                inside = []
            else:
                bodies.append("\n".join(inside))
                inside = None
            continue
        if inside is not None:
            inside.append(line)
    return bodies


class TestFamilyFolding:
    """Every rule rests on telling families apart, so folding too eagerly is as
    wrong as not folding at all."""

    def test_a_subset_prefix_is_stripped(self) -> None:
        assert family_of("ABCDEF+MinionPro-Regular") == "MinionPro"

    def test_style_suffixes_fold_into_one_family(self) -> None:
        assert family_of("NewCenturySchlbk-Roman") == family_of("NewCenturySchlbk-Bold")

    def test_a_style_word_inside_a_name_is_left_alone(self) -> None:
        """"TimesNewRoman" is one family. Folding on `roman` anywhere would make
        it "TimesNew", and then the italic and the upright are different books."""
        assert family_of("TimesNewRoman") == "TimesNewRoman"


class TestTierOne:
    def test_a_monospaced_family_is_a_code_family(self, tmp_path: Path) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 8, "code": (LISTING, MONO, 9.0, 0)} for _ in range(4)],
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown)
        assert report.rule_counts  # type: ignore[attr-defined]
        assert "mono" in next(iter(report.rule_counts))  # type: ignore[attr-defined]

    def test_the_listing_comes_out_verbatim_with_its_indentation(
        self, tmp_path: Path
    ) -> None:
        """There are no leading spaces in a PDF text stream — indentation is
        purely a left margin, so a listing read naively comes out flush left and
        its nesting is gone."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 8, "code": (LISTING, MONO, 9.0, 0)} for _ in range(4)],
        )

        markdown, _ = extract(path)

        assert any("    if self.ready:" in body for body in fence_bodies(markdown))
        assert any("        return 1" in body for body in fence_bodies(markdown))

    def test_a_lone_contrasting_line_is_not_a_listing(self, tmp_path: Path) -> None:
        """A variable name set in the code face mid-sentence is not code, and
        fencing it splits the paragraph around it."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (["setUp"], MONO, 9.0, 0)} for _ in range(4)],
        )

        markdown, _ = extract(path)

        assert fence_bodies(markdown) == []


class TestTierTwo:
    """Only if tier 1 found nothing. Both legs of its disjunction are
    load-bearing, and each is carried by a different pinned book."""

    def test_indentation_alone_catches_a_listing_at_the_body_size(
        self, tmp_path: Path
    ) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, SANS, BODY_POINTS, 72)} for _ in range(4)],
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown)
        assert "contrast" in next(iter(report.rule_counts))  # type: ignore[attr-defined]

    def test_size_alone_catches_a_listing_at_the_body_margin(
        self, tmp_path: Path
    ) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, SANS, 7.0, 0)} for _ in range(4)],
            body_points=12.0,
        )

        markdown, _ = extract(path)

        assert fence_bodies(markdown)

    def test_tier_two_does_not_run_when_tier_one_found_a_family(
        self, tmp_path: Path
    ) -> None:
        """A fitted rule that never runs on the books a generic rule already
        handles is the whole reason the ladder is ordered."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, MONO, 9.0, 40)} for _ in range(4)],
        )

        assert typography(path).tier == "mono"  # type: ignore[attr-defined]

    def test_a_long_lined_contrasting_family_is_not_a_listing(
        self, tmp_path: Path
    ) -> None:
        """A pull quote or a sidebar is a different family running the full
        measure. Short lines are what separate a listing from one."""
        document = pymupdf.open()
        for _ in range(4):
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(72, 60, 520, 300), PROSE * 4, fontsize=BODY_POINTS, fontname=SERIF
            )
            page.insert_textbox(
                pymupdf.Rect(72, 320, 520, 500), PROSE * 3, fontsize=BODY_POINTS, fontname=SANS
            )
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)

        assert fence_bodies(markdown) == []


class TestDeclining:
    def test_a_single_family_document_declines_its_code(self, tmp_path: Path) -> None:
        """One pinned book is 1,284 pages of one family, its code separated only
        by bold keywords and italic identifiers. Reconstructing indentation there
        gets the lines right and the indentation wrong — which is worse than no
        fence, because a fence asserts its body is verbatim and nothing
        downstream can discover the assertion is false."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, SERIF, BODY_POINTS, 0)} for _ in range(4)],
        )

        markdown, report = extract(path)

        assert "```" not in markdown
        assert any("no font signal" in reason for reason in report.declines)  # type: ignore[attr-defined]

    def test_a_declining_document_keeps_its_headings(self, tmp_path: Path) -> None:
        """Heading detection declines *independently*. The two rest on different
        evidence: family separates code from prose, size separates headings from
        body text — so a single-family book still typesets its chapter openers
        larger."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [
                {"heading": (f"Chapter {number}", 20.0), "prose": 10,
                 "code": (LISTING, SERIF, BODY_POINTS, 0)}
                for number in range(1, 5)
            ],
        )

        markdown, report = extract(path)

        assert "```" not in markdown
        assert [chapter.title for chapter in detect_structure(markdown)] == [
            "Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4"
        ]
        assert len(report.declines) == 1  # type: ignore[attr-defined]

    def test_a_document_with_no_size_contrast_declines_its_headings(
        self, tmp_path: Path
    ) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, MONO, 9.0, 0)} for _ in range(4)],
        )

        markdown, report = extract(path)

        assert detect_structure(markdown) == []
        assert any("no size contrast" in reason for reason in report.declines)  # type: ignore[attr-defined]

    def test_a_document_with_both_signals_declines_nothing(self, tmp_path: Path) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [
                {"heading": (f"Chapter {number}", 20.0), "prose": 10,
                 "code": (LISTING, MONO, 9.0, 0)}
                for number in range(1, 5)
            ],
        )

        _, report = extract(path)

        assert report.declines == ()  # type: ignore[attr-defined]


class TestHeadings:
    def test_sizes_rank_into_levels_largest_first(self, tmp_path: Path) -> None:
        path = build_pdf(
            tmp_path / "b.pdf",
            [
                {"heading": ("Part One", 24.0), "prose": 8},
                {"heading": ("Chapter One", 18.0), "prose": 8},
                {"heading": ("A Section", 13.0), "prose": 8},
                {"prose": 10},
            ],
        )

        markdown, _ = extract(path)

        assert "# Part One" in markdown
        assert "## Chapter One" in markdown
        assert "### A Section" in markdown

    def test_no_heading_is_emitted_from_a_code_family_span(self, tmp_path: Path) -> None:
        """The in-module guarantee ADR 0003 buys. A listing set larger than the
        body — a display listing on a chapter opener — is still a listing."""
        path = build_pdf(
            tmp_path / "b.pdf",
            [{"prose": 10, "code": (LISTING, MONO, 14.0, 0)} for _ in range(4)],
            body_points=10.0,
        )

        markdown, _ = extract(path)

        assert fence_bodies(markdown)
        assert not any(line.startswith("#") for line in markdown.split("\n"))

    def test_a_paragraph_set_larger_than_the_body_is_not_a_heading(
        self, tmp_path: Path
    ) -> None:
        """Size alone is what over-promoted. A heading is a *line*, so an
        epigraph or pull quote in a larger face is prose, not six headings."""
        document = pymupdf.open()
        for _ in range(4):
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(72, 60, 520, 240), PROSE * 4, fontsize=14.0, fontname=SERIF
            )
            page.insert_textbox(
                pymupdf.Rect(72, 260, 520, 700), PROSE * 8, fontsize=10.0, fontname=SERIF
            )
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)

        assert markdown.count("\n#") <= 1

    def test_a_bullet_glyph_is_not_a_heading(self, tmp_path: Path) -> None:
        """A title says something. A book whose bullets outsize its body would
        otherwise grow a chapter per list item — and since detect_structure takes
        the *minimum* level present, that demotes every real chapter."""
        document = pymupdf.open()
        for _ in range(4):
            page = document.new_page()
            page.insert_text((72, 60), "A Real Heading", fontsize=18.0, fontname=SERIF)
            y = 100.0
            for _ in range(4):
                page.insert_text((72, y), "•", fontsize=14.0, fontname=SERIF)
                page.insert_textbox(
                    pymupdf.Rect(90, y - 10, 520, y + 40), PROSE, fontsize=10.0, fontname=SERIF
                )
                y += 60
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)

        assert "•" not in "".join(
            line for line in markdown.split("\n") if line.startswith("#")
        )

    def test_a_heading_wrapped_across_two_lines_is_one_heading(
        self, tmp_path: Path
    ) -> None:
        """"Chapter 2" over "Composing Small Parts" is one chapter. Two
        headings there is a chapter tree with twice as many chapters as the book,
        every other one titleless."""
        document = pymupdf.open()
        for number in range(1, 5):
            page = document.new_page()
            page.insert_textbox(
                pymupdf.Rect(72, 60, 300, 140),
                f"Chapter {number} Composing Small Parts",
                fontsize=20.0,
                fontname=SERIF,
            )
            page.insert_textbox(
                pymupdf.Rect(72, 200, 520, 700), PROSE * 6, fontsize=10.0, fontname=SERIF
            )
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)
        chapters = detect_structure(markdown)

        assert len(chapters) == 4
        assert "Composing Small Parts" in chapters[0].title


class TestProse:
    def test_lines_join_into_a_paragraph(self, tmp_path: Path) -> None:
        path = build_pdf(tmp_path / "b.pdf", [{"prose": 8} for _ in range(3)])

        markdown, _ = extract(path)

        assert "runs the full measure of the column and wraps across" in markdown

    def test_a_word_broken_at_a_line_end_is_healed(self, tmp_path: Path) -> None:
        """Leaving the hyphen in makes one word two, and both of them wrong."""
        document = pymupdf.open()
        page = document.new_page()
        page.insert_text((72, 60), "the extrac-", fontsize=10.0, fontname=SERIF)
        page.insert_text((72, 74), "tion pipeline", fontsize=10.0, fontname=SERIF)
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)

        assert "extraction pipeline" in markdown

    def test_prose_that_reads_as_structure_is_escaped(self, tmp_path: Path) -> None:
        """What protects a declining book: unfenced code comes out as text that
        says what it says and structures nothing."""
        document = pymupdf.open()
        for _ in range(3):
            page = document.new_page()
            page.insert_text((72, 60), "# not a heading", fontsize=10.0, fontname=SERIF)
            page.insert_textbox(
                pymupdf.Rect(72, 100, 520, 700), PROSE * 6, fontsize=10.0, fontname=SERIF
            )
        path = tmp_path / "b.pdf"
        document.save(path)
        document.close()

        markdown, _ = extract(path)

        assert "\\# not a heading" in markdown
        assert detect_structure(markdown) == []


class TestTheProfileIsDocumentWide:
    def test_a_page_of_pure_listing_does_not_make_code_the_body(
        self, tmp_path: Path
    ) -> None:
        """Judged alone, a page with no prose on it has a code family as its
        dominant one — and then the listing is body text and the prose around it
        is code."""
        pages: list[dict[str, object]] = [
            {"prose": 10, "code": (LISTING, MONO, 9.0, 0)} for _ in range(3)
        ]
        pages.append({"prose": 0, "code": (LISTING * 8, MONO, 9.0, 0)})
        path = build_pdf(tmp_path / "b.pdf", pages)

        assert typography(path).dominant not in typography(path).code_families  # type: ignore[attr-defined]


class TestNoLayoutWrapper:
    def test_the_module_does_not_import_pymupdf4llm(self) -> None:
        """Dropping it takes a noncommercial term out of the dependency tree of an
        MIT-declared published package, and its layout helpers are
        deliberately not vendored — copying AGPL source into an MIT repo is a
        worse position than depending on it."""
        source = (
            Path(__file__).parents[1] / "src/booksmart_core/parsing/pdf.py"
        ).read_text()

        assert "pymupdf4llm" not in source.split('"""', 2)[2]

    def test_the_wrapper_is_not_installed_at_all(self) -> None:
        with pytest.raises(ImportError):
            __import__("pymupdf4llm")


def test_the_route_is_wired_into_the_default_router() -> None:
    from booksmart_core.parsing import build_default_router

    router = build_default_router()

    assert isinstance(router._by_route["pdf"], PdfExtractor)  # noqa: SLF001


def test_the_extractor_logs_what_it_decided(tmp_path: Path) -> None:
    """A full-corpus sweep is watched while it runs, and "which family did
    it pick" is the question a bad row raises first."""
    path = build_pdf(
        tmp_path / "b.pdf", [{"prose": 8, "code": (LISTING, MONO, 9.0, 0)} for _ in range(4)]
    )
    log: list[str] = []

    PdfExtractor().extract(path, log.append)

    assert any("code families" in line for line in log)
