"""Matching a title on the page against a title a container states.

The interesting cases are all invisible ones. A heading that differs from the
authored title by a character nothing renders stops matching silently — there is
no error, no log line, just a chapter that quietly is not there — and one real
book carries a zero-width space in several hundred of its lines.
"""

from booksmart_core.titles import normalise, titles_match


class TestNormalise:
    def test_case_and_punctuation_are_collapsed(self) -> None:
        assert normalise("Chapter 4: A Title!") == normalise("chapter 4  a title")

    def test_a_soft_hyphen_is_deleted_rather_than_spaced(self) -> None:
        """Some PDFs carry one at every legal break point. Collapsing it to a
        space splits the word around it, so the title stops matching on a
        character nobody can see."""
        assert normalise("hyphen\u00adation") == "hyphenation"

    def test_every_invisible_character_is_deleted(self) -> None:
        for invisible in ("\u200b", "\u200c", "\u200d", "\ufeff"):
            assert normalise(f"tit{invisible}le") == "title"

    def test_real_punctuation_is_still_a_separator(self) -> None:
        """The opposite of what happens to an invisible: a hyphen somebody typed
        separates two words, and deleting it would run them together."""
        assert normalise("well-known") == "well known"

    def test_a_title_of_nothing_but_punctuation_normalises_to_nothing(self) -> None:
        assert normalise("— · —") == ""


class TestTitlesMatch:
    def test_the_same_title_spelled_two_ways_matches(self) -> None:
        assert titles_match("Chapter 4: A Title", "chapter 4 — a title")

    def test_an_invisible_character_does_not_stop_a_match(self) -> None:
        assert titles_match("A\u200b Title", "A Title")

    def test_a_number_typeset_apart_from_its_title_matches_the_whole_entry(
        self,
    ) -> None:
        """Books typeset "Chapter 4" and its title as two separate headings while
        the outline spells the entry as one. Equality loses both halves."""
        assert titles_match("Chapter 4", "Chapter 4: A Title")

    def test_an_entry_truncated_shorter_than_the_page_matches(self) -> None:
        """Containment runs the other way just as often — a bookmark shortened to
        fit a pane, against the full title on the page."""
        assert titles_match("Chapter 4: A Title, and Its Longer Subtitle", "Chapter 4: A Title")

    def test_a_different_title_does_not_match(self) -> None:
        assert not titles_match("Chapter 4: A Title", "Chapter 5: Another")

    def test_a_word_from_the_end_of_an_entry_does_not_match_it(self) -> None:
        """Measured against a real book. Containment anywhere let a body line
        saying one word match a section entry ending in that word, and the single
        spurious match taught the caller the wrong thing about the whole
        document — it cost six of that document's seven headings."""
        entry = "1.3 Formulating Abstractions with Higher-Order Procedures"

        assert not titles_match("Procedures", entry)
        assert not titles_match("Higher-Order Procedures", entry)

    def test_a_word_that_only_begins_the_same_way_does_not_match(self) -> None:
        """Whole words, so "Chapter 4" does not begin "Chapter 40"."""
        assert not titles_match("Chapter 4", "Chapter 40: A Title")

    def test_nothing_matches_an_empty_side(self) -> None:
        """Otherwise every heading matches, since "" is contained in everything —
        and a decorative rule normalises to nothing."""
        assert not titles_match("", "Chapter 4")
        assert not titles_match("— · —", "Chapter 4")
