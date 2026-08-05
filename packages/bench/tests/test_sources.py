"""Cataloguing the files a sources handover drops into an assets checkout.

The verb's job is to answer "is this the book truth thinks it is, and can the
pipeline read it" before anyone spends money ingesting it. Most of that is pure
— scoring a file's text against authored chapter titles, auditing an edition,
rewriting a pin — and those tests inject the extracted evidence directly.

Two tests build a real PDF with PyMuPDF instead, because "is there a text layer"
is a question about a file and cannot be answered against a stand-in.
"""

from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest

from booksmart_bench.sources import (
    MIN_CHARS_PER_PAGE,
    Artifact,
    catalogue,
    identify,
    normalise,
    parse_ordinal,
    pin,
    repin_line,
)
from booksmart_bench.truth import load_truth

FILLER = (
    "This page carries enough ordinary prose that a sampled character count "
    "reads as a text layer rather than as a scan of one. " * 3
)


def make_artifact(
    path: Path,
    *,
    text: str,
    pages: int = 100,
    chars_per_page: int = 2000,
    sha256: str = "0" * 64,
    front: str | None = None,
) -> Artifact:
    return Artifact(
        path=path,
        sha256=sha256,
        pages=pages,
        chars_per_page=chars_per_page,
        text=normalise(text),
        # Edition claims are read from the front matter; a test that only says
        # `text` means "this is what the file says, wherever it says it".
        front=normalise(text if front is None else front),
    )


def drop(assets: Path, *names: str) -> list[Path]:
    """Put stub files in sources/. catalogue() walks the directory, so the files
    have to be there even when the reader is injected."""
    sources = assets / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = sources / name
        path.write_bytes(b"%PDF-1.4 stub")
        paths.append(path)
    return paths


def write_pdf(path: Path, lines: list[str]) -> Path:
    doc = pymupdf.open()
    for line in lines:
        page = doc.new_page()
        page.insert_text((72, 72), line, fontsize=11)
        page.insert_textbox(pymupdf.Rect(72, 96, 520, 700), FILLER, fontsize=9)
    doc.save(path)
    doc.close()
    return path


class TestIdentify:
    def test_a_file_is_matched_by_the_chapter_titles_it_contains(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """Not by its filename, which is whatever the last person typed, and not
        by its metadata, which is part of what is being verified."""
        books = load_truth(write_truth("a-book")).books
        artifact = make_artifact(
            tmp_path / "whatever-they-called-it.pdf",
            # The fixture book's whole authored structure, chapter and sections.
            text="First Chapter Widgets And Sprockets Grommets",
        )

        match = identify(artifact, books)

        assert match.slug == "a-book"
        assert match.score == pytest.approx(1.0)

    def test_a_file_matching_nothing_is_not_guessed_at(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        books = load_truth(write_truth("a-book")).books
        artifact = make_artifact(tmp_path / "mystery.pdf", text="an unrelated document")

        assert identify(artifact, books).slug is None

    def test_section_titles_count_as_evidence_too(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """Chapters alone are too few to identify a short book, and for books
        whose chapters are titled with single common words they are no evidence
        at all."""
        assets = write_truth(
            "a-book",
            toc={
                "chapters": [
                    {
                        "id": "1",
                        "title": "State",
                        "sections": [{"id": "1.1", "title": "Collecting Temporary Variable"}],
                    }
                ]
            },
        )
        books = load_truth(assets).books
        artifact = make_artifact(tmp_path / "b.pdf", text="Collecting Temporary Variable")

        assert identify(artifact, books).slug == "a-book"

    def test_generic_one_word_titles_are_weak_evidence(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """A book whose chapters are called State, Classes and Patterns matches
        every programming book ever written. Scoring those hits as highly as a
        distinctive heading made it the runner-up against the whole corpus, and
        ate the margin that decides an ambiguous file."""
        assets = write_truth(
            "generic-book",
            toc={
                "chapters": [
                    {
                        "id": id_,
                        "title": title,
                        "sections": [{"id": f"{id_}.1", "title": section}],
                    }
                    for id_, title, section in (
                        ("1", "State", "Explaining Temporary Variable"),
                        ("2", "Classes", "Qualified Subclass Name"),
                        ("3", "Patterns", "Why Patterns Work"),
                    )
                ]
            },
        )
        books = load_truth(assets).books
        # A foreign book that happens to use all three common words and none of
        # the distinctive headings.
        artifact = make_artifact(tmp_path / "other.pdf", text="state classes patterns")

        assert identify(artifact, books).slug is None

    def test_two_books_scoring_alike_are_reported_as_ambiguous(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """Two editions of one book, or a book and its own draft, look nearly
        identical. Naming one of them would pin truth to the wrong artifact."""
        assets = write_truth("book-one", toc={"chapters": [{"id": "1", "title": "Shared Title"}]})
        write_truth("book-two", root=assets, toc={"chapters": [{"id": "1", "title": "Shared Title"}]})
        books = load_truth(assets).books
        artifact = make_artifact(assets / "sources" / "which.pdf", text="Shared Title")

        match = identify(artifact, books)

        assert match.slug is None
        assert match.runner_up is not None


class TestAudit:
    def test_a_scan_without_a_text_layer_is_a_problem(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """The pipeline would fall through to OCR, which is a different
        extraction path — so the book would be benchmarking a different
        parser from every other book in the corpus."""
        assets = write_truth("a-book")
        (path,) = drop(assets, "scan.pdf")
        artifact = make_artifact(
            path,
            text="First Chapter",
            chars_per_page=MIN_CHARS_PER_PAGE - 1,
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert not entry.ok
        assert "OCR" in " ".join(entry.problems)

    def test_an_epub_is_accepted_with_a_note_about_its_parser_chain(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        (path,) = drop(assets, "book.epub")
        artifact = make_artifact(path, text="First Chapter")

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert entry.ok
        assert "EPUB" in " ".join(entry.notes)

    def test_a_file_disagreeing_with_an_existing_pin_is_a_problem(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """The pin exists to catch exactly this: truth was authored against
        other bytes, so it may no longer describe what is on disk."""
        assets = write_truth("a-book")  # the default fixture is pinned to "0" * 64
        (path,) = drop(assets, "book.pdf")
        artifact = make_artifact(
            path, text="First Chapter", sha256="b" * 64
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert not entry.ok
        assert "already pinned" in " ".join(entry.problems)

    def test_a_missing_publication_year_is_a_note_not_a_problem(
        self, tmp_path: Path, write_truth: Callable[..., Path]
    ) -> None:
        """Plenty of legitimate files do not print a year where we look. It is
        worth saying and not worth blocking on."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "publication_year": 1999,
                "source": {"file": "sources/a-book.pdf", "sha256": "TBD-when-file-lands"},
            },
        )
        (path,) = drop(assets, "a-book.pdf")
        artifact = make_artifact(path, text="First Chapter")

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert entry.ok
        assert "1999" in " ".join(entry.notes)


class TestEditionClaims:
    """The verb's whole reason for existing. Every other check catches a file
    that cannot be used; these catch one that can be used and is wrong."""

    def test_a_file_claiming_a_different_edition_is_a_problem(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A later edition of the same book shares most of its headings, so
        content matching alone scores it highly and pins the wrong artifact."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "edition": "2nd",
                "source": {"file": "sources/a.pdf", "sha256": "TBD"},
            },
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(
            path, text="Placeholder Book Third Edition First Chapter Widgets And Sprockets Grommets"
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert not entry.ok
        assert "3rd" in " ".join(entry.problems) or "third" in " ".join(entry.problems).lower()

    def test_the_expected_edition_among_several_mentioned_is_confirmation(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A second edition routinely discusses its first in the same front
        matter. Finding an ordinal is not the same as finding a contradiction."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "edition": "2nd",
                "source": {"file": "sources/a.pdf", "sha256": "TBD"},
            },
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(
            path,
            text=(
                "Placeholder Book Second Edition. In the first edition of this book the "
                "starting program was different. First Chapter Widgets And Sprockets Grommets"
            ),
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert entry.ok

    def test_a_contradicting_copyright_year_is_a_problem(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Some books never print an edition statement, and then the copyright
        year is the only thing separating two editions."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "publication_year": 2019,
                "source": {"file": "sources/a.pdf", "sha256": "TBD"},
            },
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(
            path, text="Copyright 2013 First Chapter Widgets And Sprockets Grommets"
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert not entry.ok
        assert "2013" in " ".join(entry.problems)


class TestOrdinals:
    """A table of ordinals that stops short does not fail loudly — it disables
    the edition check, which is the one outcome this module exists to prevent."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("1st", 1), ("3rd", 3), ("7th", 7), ("12th", 12), ("third", 3), ("10", 10)],
    )
    def test_it_reads_the_shapes_book_yaml_and_title_pages_use(
        self, text: str, expected: int
    ) -> None:
        assert parse_ordinal(text) == expected

    @pytest.mark.parametrize("text", ["", "revised", "anniversary"])
    def test_it_declines_what_is_not_an_ordinal(self, text: str) -> None:
        assert parse_ordinal(text) is None

    def test_an_edition_past_the_sixth_is_still_checked(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "edition": "7th",
                "source": {"file": "sources/a.pdf", "sha256": "TBD"},
            },
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(
            path, text="Placeholder Book Eighth Edition First Chapter Widgets And Sprockets Grommets"
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        assert not entry.ok
        assert "8th" in " ".join(entry.problems)

    def test_an_unreadable_edition_says_so_rather_than_going_quiet(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Silence here is indistinguishable from a passing check."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "edition": "20th Anniversary",
                "source": {"file": "sources/a.pdf", "sha256": "TBD"},
            },
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(
            path, text="Placeholder Book First Chapter Widgets And Sprockets Grommets"
        )

        entry = catalogue(assets, load_truth(assets), read=lambda _: artifact).entries[0]

        # 20th parses, so this one is checked rather than skipped.
        assert "20" in " ".join(entry.notes) or "20th" in " ".join(entry.notes)


class TestNormalise:
    def test_soft_hyphens_are_removed_not_turned_into_word_breaks(self) -> None:
        """Some PDFs carry a soft hyphen at every possible break point. Treating
        one as punctuation splits the word around it, and a heading that differs
        from the authored one by an invisible character stops matching."""
        assert normalise("Refactor\u00ading") == "refactoring"

    def test_zero_width_characters_go_the_same_way(self) -> None:
        assert normalise("Extract\u200bFunction") == "extractfunction"


class TestCatalogue:
    def test_it_says_which_books_are_still_missing(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        # A distinct ToC, or the two books tie and the ambiguity guard correctly
        # declines to identify either — which is a different test.
        write_truth(
            "another-book",
            root=assets,
            toc={"chapters": [{"id": "1", "title": "An Entirely Different Chapter"}]},
        )
        (path,) = drop(assets, "a.pdf")
        artifact = make_artifact(path, text="First Chapter")

        result = catalogue(assets, load_truth(assets), read=lambda _: artifact)

        assert result.missing == ("another-book",)

    def test_two_files_claiming_one_book_are_contested_not_pinned(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        sources = assets / "sources"
        drop(assets, "one.pdf", "two.pdf")

        result = catalogue(
            assets,
            load_truth(assets),
            read=lambda path: make_artifact(path, text="First Chapter"),
        )

        assert "a-book" in result.contested
        assert result.identified == {}

    def test_a_file_needing_a_rename_says_what_to(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        drop(assets, "Some Book (2nd Edition).pdf")

        result = catalogue(
            assets,
            load_truth(assets),
            read=lambda path: make_artifact(path, text="First Chapter"),
        )

        assert result.entries[0].rename_to == "placeholder.pdf"

    def test_a_non_book_file_in_sources_is_ignored(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth("a-book")
        sources = assets / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        (sources / "notes.txt").write_text("where I got these")

        result = catalogue(assets, load_truth(assets), read=lambda path: make_artifact(path, text=""))

        assert result.entries == ()
        assert result.ignored == ("notes.txt",)


class TestRepinLine:
    def test_it_replaces_the_placeholder_and_drops_the_stale_comment(self) -> None:
        line = "  sha256: TBD-when-file-lands # blocked on the sources handover (#23)"

        assert repin_line(line, "c" * 64) == f"  sha256: {'c' * 64}"

    def test_it_keeps_a_comment_that_still_means_something(self) -> None:
        """Only the comment that says "not yet" is stale once it is."""
        line = "  sha256: TBD-when-file-lands # the printing, not the reissue"

        assert repin_line(line, "c" * 64) == f"  sha256: {'c' * 64} # the printing, not the reissue"

    def test_it_preserves_indentation(self) -> None:
        assert repin_line("    sha256: TBD", "c" * 64).startswith("    sha256: ")


class TestPin:
    def test_it_writes_the_hash_and_renames_the_file(
        self, write_truth: Callable[..., Path]
    ) -> None:
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "source": {"file": "sources/placeholder.pdf", "sha256": "TBD-when-file-lands"},
            },
        )
        sources = assets / "sources"
        drop(assets, "whatever.pdf")
        result = catalogue(
            assets,
            load_truth(assets),
            read=lambda path: make_artifact(path, text="First Chapter", sha256="d" * 64),
        )

        pin(assets, result)

        assert (sources / "placeholder.pdf").exists()
        assert not (sources / "whatever.pdf").exists()
        assert "d" * 64 in (assets / "truth" / "a-book" / "book.yaml").read_text()

    def test_it_leaves_a_contested_book_alone(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Pinning one of two candidates would be a coin toss recorded as fact."""
        assets = write_truth(
            "a-book",
            book={
                "title": "Placeholder Book",
                "area": "an-area",
                "source": {"file": "sources/placeholder.pdf", "sha256": "TBD-when-file-lands"},
            },
        )
        sources = assets / "sources"
        drop(assets, "one.pdf", "two.pdf")
        result = catalogue(
            assets,
            load_truth(assets),
            read=lambda path: make_artifact(path, text="First Chapter", sha256="d" * 64),
        )

        pin(assets, result)

        assert "TBD" in (assets / "truth" / "a-book" / "book.yaml").read_text()


class TestReadArtifact:
    """The one part that has to meet a real file."""

    def test_it_reads_pages_text_and_a_hash_from_a_real_pdf(self, tmp_path: Path) -> None:
        from booksmart_bench.sources import read_artifact

        path = write_pdf(tmp_path / "book.pdf", ["First Chapter", "Second Chapter"])

        artifact = read_artifact(path)

        assert artifact.pages == 2
        assert artifact.chars_per_page > MIN_CHARS_PER_PAGE
        assert "first chapter" in artifact.text
        assert len(artifact.sha256) == 64
        assert artifact.unreadable is None

    def test_a_pdf_with_no_text_layer_reports_its_character_count(
        self, tmp_path: Path
    ) -> None:
        from booksmart_bench.sources import read_artifact

        doc = pymupdf.open()
        doc.new_page()
        doc.new_page()
        doc.save(tmp_path / "blank.pdf")
        doc.close()

        artifact = read_artifact(tmp_path / "blank.pdf")

        assert artifact.pages == 2
        assert artifact.chars_per_page < MIN_CHARS_PER_PAGE

    def test_a_file_that_will_not_open_is_reported_not_raised(self, tmp_path: Path) -> None:
        """One unreadable file must not stop the other thirteen being catalogued."""
        from booksmart_bench.sources import read_artifact

        path = tmp_path / "truncated.pdf"
        path.write_bytes(b"%PDF-1.4 and then nothing useful")

        artifact = read_artifact(path)

        assert artifact.unreadable is not None
        assert artifact.sha256  # still hashed, so the file can be talked about
