"""Matching a title on the page against a title a container states.

The interesting cases are all invisible ones. A heading that differs from the
authored title by a character nothing renders stops matching silently — there is
no error, no log line, just a chapter that quietly is not there — and one real
book carries a zero-width space in several hundred of its lines.
"""

from booksmart_core.titles import normalise, title_remainder, titles_match


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

    def test_an_accented_letter_is_a_letter(self) -> None:
        """Keyed on ASCII, "Café Society" normalised to "caf society" — the
        accented letters deleted as though they were punctuation."""
        assert normalise("Café Society") == "café society"
        assert normalise("Übung") == "übung"

    def test_a_title_in_another_script_does_not_normalise_to_nothing(self) -> None:
        """The silent case. A book whose titles are not Latin normalised every one
        of them to the empty string, which matches nothing — so a PDF reported its
        whole outline as unlocated, and an EPUB emitted every declared title in
        front of the line already carrying it."""
        assert normalise("Глава 4") == "глава 4"
        assert normalise("第四章") == "第四章"

    def test_case_folds_beyond_lowercasing(self) -> None:
        """`casefold`, not `lower`: German ß and SS are the same title set two
        ways, and a publisher's outline routinely sets in caps what the page
        does not."""
        assert normalise("STRASSE") == normalise("Straße")


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

    def test_titles_in_another_script_match_and_differ_like_any_other(self) -> None:
        assert titles_match("Глава 4", "Глава 4: Заголовок")
        assert not titles_match("Глава 4", "Глава 5")
        assert title_remainder("Глава 4", "Глава 4: Заголовок") == "заголовок"

    def test_a_different_title_does_not_match(self) -> None:
        assert not titles_match("Chapter 4: A Title", "Chapter 5: Another")

    def test_a_word_from_the_end_of_an_entry_does_not_match_it(self) -> None:
        """Measured against a real book. Containment anywhere let a body line
        saying one common word match a section entry that happened to end in it,
        and the single spurious match taught the caller the wrong thing about the
        whole document — it cost six of that document's seven headings."""
        entry = "2.4 Reading a Container Without Extracting It"

        assert not titles_match("It", entry)
        assert not titles_match("Without Extracting It", entry)

    def test_a_word_that_only_begins_the_same_way_does_not_match(self) -> None:
        """Whole words, so "Chapter 4" does not begin "Chapter 40"."""
        assert not titles_match("Chapter 4", "Chapter 40: A Title")

    def test_the_rest_of_a_title_is_what_the_line_did_not_say(self) -> None:
        """A title too long for its measure wraps, and this is what lets a caller
        pick the second half of it off the next line."""
        assert title_remainder("Chapter 4", "Chapter 4: A Title") == "a title"

    def test_there_is_no_rest_of_a_title_the_line_said_whole(self) -> None:
        assert title_remainder("Chapter 4: A Title", "Chapter 4: A Title") == ""
        assert title_remainder("Chapter 4: A Title and More", "Chapter 4: A Title") == ""

    def test_nothing_matches_an_empty_side(self) -> None:
        """Otherwise every heading matches, since "" is contained in everything —
        and a decorative rule normalises to nothing."""
        assert not titles_match("", "Chapter 4")
        assert not titles_match("— · —", "Chapter 4")
