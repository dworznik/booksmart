"""The record->location join: detected rows -> ToC node ids.

The one join the gap audit named, and the reason a run file can carry locations
alone. Exercised here on plain structures rather than a corpus — the join is
about titles and parentage, not about the database the titles arrived from.
"""

from collections.abc import Callable
from pathlib import Path

from booksmart_bench.locations import Detected, join
from booksmart_bench.truth import BookTruth, load_truth


def book_truth(write_truth: Callable[..., Path], toc: dict[str, object]) -> BookTruth:
    return load_truth(write_truth("a-book", toc=toc)).books["a-book"]


TWO_CHAPTERS: dict[str, object] = {
    "chapters": [
        {
            "id": "1",
            "title": "First Chapter",
            "sections": [
                {"id": "1.1", "title": "Widgets And Sprockets"},
                {"id": "1.2", "title": "Grommets"},
            ],
        },
        {
            "id": "2",
            "title": "Second Chapter",
            "sections": [{"id": "2.1", "title": "Grommets"}],
        },
    ]
}


def detected(*rows: tuple[str, str, str, str | None]) -> list[Detected]:
    """(record_id, title, kind, parent) in document order."""
    return [
        Detected(record_id=record_id, title=title, kind=kind, position=index, parent_id=parent)
        for index, (record_id, title, kind, parent) in enumerate(rows)
    ]


class TestChapters:
    def test_a_chapter_resolves_by_title(self, write_truth: Callable[..., Path]) -> None:
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(detected(("c1", "First Chapter", "chapter", None)), book)

        assert joined.node_for("c1") == "1"

    def test_numbering_and_punctuation_do_not_stop_a_match(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A detector reads the heading off the page ("1. First Chapter"); the
        author wrote the title. Same node."""
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(detected(("c1", "1. First Chapter!", "chapter", None)), book)

        assert joined.node_for("c1") == "1"

    def test_a_chapter_truth_does_not_know_is_unresolved(
        self, write_truth: Callable[..., Path]
    ) -> None:
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(detected(("c9", "An Interlude", "chapter", None)), book)

        assert joined.node_for("c9") is None
        assert any("An Interlude" in note for note in joined.unresolved)

    def test_two_chapters_with_one_title_match_in_document_order(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Position is the tiebreak: one detected node must not satisfy two
        authored ones."""
        book = book_truth(
            write_truth,
            {"chapters": [{"id": "1", "title": "Summary"}, {"id": "2", "title": "Summary"}]},
        )

        joined = join(
            detected(("c1", "Summary", "chapter", None), ("c2", "Summary", "chapter", None)),
            book,
        )

        assert (joined.node_for("c1"), joined.node_for("c2")) == ("1", "2")

    def test_front_matter_resolves_even_when_the_detector_calls_it_a_chapter(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """The detector classifies matter by keyword and the author by hand; a
        disagreement about *kind* must not cost the location. What the two
        classifications disagree about is the scorer's business, not the join's."""
        book = book_truth(
            write_truth,
            {
                "front_matter": [{"id": "preface", "title": "Preface"}],
                "chapters": [{"id": "1", "title": "First Chapter"}],
            },
        )

        joined = join(detected(("c0", "Preface", "chapter", None)), book)

        assert joined.node_for("c0") == "preface"


class TestSections:
    def test_a_section_resolves_within_its_own_chapter(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """Both chapters have a section called "Grommets"; the FK decides which
        one this is."""
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(
            detected(
                ("c1", "First Chapter", "chapter", None),
                ("s1", "Grommets", "section", "c1"),
                ("c2", "Second Chapter", "chapter", None),
                ("s2", "Grommets", "section", "c2"),
            ),
            book,
        )

        assert (joined.node_for("s1"), joined.node_for("s2")) == ("1.2", "2.1")

    def test_a_section_under_an_unresolved_chapter_falls_back_to_a_unique_title(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """A chapter whose heading the detector mangled would otherwise cost
        every section under it — which scores as a retrieval miss when it is
        really a join failure."""
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(
            detected(
                ("c1", "Chapter One (mangled)", "chapter", None),
                ("s1", "Widgets And Sprockets", "section", "c1"),
            ),
            book,
        )

        assert joined.node_for("s1") == "1.1"

    def test_the_fallback_refuses_an_ambiguous_title(
        self, write_truth: Callable[..., Path]
    ) -> None:
        """"Grommets" names a section in both chapters, so a section under an
        unresolved chapter cannot be attributed to either without guessing."""
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(
            detected(
                ("c1", "Chapter One (mangled)", "chapter", None),
                ("s1", "Grommets", "section", "c1"),
            ),
            book,
        )

        assert joined.node_for("s1") is None

    def test_a_section_the_chapter_does_not_have_is_unresolved(
        self, write_truth: Callable[..., Path]
    ) -> None:
        book = book_truth(write_truth, TWO_CHAPTERS)

        joined = join(
            detected(
                ("c2", "Second Chapter", "chapter", None),
                ("s1", "Widgets And Sprockets", "section", "c2"),
            ),
            book,
        )

        # 1.1 belongs to chapter 1, and the fallback only fires for sections
        # whose own chapter never resolved.
        assert joined.node_for("s1") is None


class TestClaiming:
    def test_a_node_is_claimed_once(self, write_truth: Callable[..., Path]) -> None:
        """Two detected chapters with the same title cannot both be node "1" —
        that would let one book's structure score above 100% recall."""
        book = book_truth(write_truth, {"chapters": [{"id": "1", "title": "First Chapter"}]})

        joined = join(
            detected(
                ("c1", "First Chapter", "chapter", None),
                ("c2", "First Chapter", "chapter", None),
            ),
            book,
        )

        assert joined.node_for("c1") == "1"
        assert joined.node_for("c2") is None
