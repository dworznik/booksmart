"""Unit tests for cutting passages — the book's own words, in windows.

Driven through `cut_passages` with hand-written Markdown and unsaved
`Chapter` / `Section` rows carrying the line numbers structure detection would
have recorded. Every prose fixture here is synthetic.
"""

import pytest

from booksmart_core.models import Chapter, Section
from booksmart_core.passages import cut_passages


def words(count: int, word: str = "word") -> str:
    """A paragraph of exactly ``count`` words."""
    return " ".join([word] * count)


def chapter(title: str, line: int, *sections: tuple[str, int]) -> Chapter:
    return Chapter(
        title=title,
        position=0,
        source_line=line,
        sections=[
            Section(title=section_title, position=position, source_line=section_line)
            for position, (section_title, section_line) in enumerate(sections)
        ],
    )


def test_a_passage_is_an_exact_slice_of_the_markdown_under_its_section() -> None:
    markdown = (
        "# Tides\n"
        "\n"
        "## Spring tides\n"
        "\n"
        "The moon pulls the sea twice a day.\n"
        "\n"
        "The sun adds its own, smaller pull.\n"
    )
    tides = chapter("Tides", 1, ("Spring tides", 3))

    passages = cut_passages([tides], markdown)

    assert [passage.text for passage in passages] == [
        "## Spring tides\n"
        "\n"
        "The moon pulls the sea twice a day.\n"
        "\n"
        "The sun adds its own, smaller pull."
    ]
    [passage] = passages
    assert markdown[passage.start_char : passage.end_char] == passage.text
    assert passage.chapter is tides
    assert passage.section is tides.sections[0]


def word_count(text: str) -> int:
    return len(text.split())


def test_paragraphs_pack_into_a_window_until_the_next_would_overflow_it() -> None:
    # 100 + 100 fit a 256-word window; a third hundred would not.
    markdown = f"# Tides\n\n{words(100, 'ebb')}\n\n{words(100, 'flow')}\n\n{words(100, 'slack')}\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    assert [passage.text for passage in passages] == [
        f"# Tides\n\n{words(100, 'ebb')}\n\n{words(100, 'flow')}",
        words(100, "slack"),
    ]


def test_no_passage_straddles_a_section_boundary() -> None:
    markdown = (
        "# Tides\n"
        "\n"
        "An opening word before any section.\n"
        "\n"
        "## Spring tides\n"
        "\n"
        "High water is highest.\n"
        "\n"
        "## Neap tides\n"
        "\n"
        "High water is lowest.\n"
    )
    tides = chapter("Tides", 1, ("Spring tides", 5), ("Neap tides", 9))

    passages = cut_passages([tides], markdown)

    assert [(passage.section, passage.text) for passage in passages] == [
        (None, "# Tides\n\nAn opening word before any section."),
        (tides.sections[0], "## Spring tides\n\nHigh water is highest."),
        (tides.sections[1], "## Neap tides\n\nHigh water is lowest."),
    ]


def test_a_heading_with_nothing_under_it_is_not_a_passage() -> None:
    # The chapter heading runs straight into its first section, as most do.
    markdown = "# Tides\n\n## Spring tides\n\nHigh water is highest.\n"
    tides = chapter("Tides", 1, ("Spring tides", 3))

    passages = cut_passages([tides], markdown)

    assert [passage.text for passage in passages] == [
        "## Spring tides\n\nHigh water is highest."
    ]


def test_a_paragraph_larger_than_a_window_is_cut_to_fit() -> None:
    # One unwrapped 600-word line: nothing to split on but the words.
    markdown = f"# Tides\n\n{words(600)}\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    assert [word_count(passage.text) for passage in passages] == [256, 256, 88]
    for passage in passages:
        assert markdown[passage.start_char : passage.end_char] == passage.text


@pytest.mark.parametrize("words_per_line", [None, 30])
def test_an_oversized_paragraph_is_cut_between_sentences_where_it_has_any(
    words_per_line: int | None,
) -> None:
    """Six 100-word sentences in one paragraph. A cut at the 256th word would
    land mid-sentence; the last full stop before it is at the 200th.

    The same whether the paragraph is one long line or hard-wrapped every 30
    words: a line break in prose is where something wrapped it, not a place the
    author paused, so it is not a place to cut either."""
    paragraph = " ".join([words(99) + " end."] * 6).split()
    width = words_per_line or len(paragraph)
    wrapped = "\n".join(
        " ".join(paragraph[start : start + width]) for start in range(0, len(paragraph), width)
    )
    markdown = f"# Tides\n\n{wrapped}\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    # The first window also holds the two words of the heading above it.
    assert [word_count(passage.text) for passage in passages] == [202, 200, 200]
    assert all(passage.text.endswith("end.") for passage in passages)


def test_a_listing_larger_than_a_window_is_cut_at_its_line_breaks() -> None:
    # In code the line is the unit: 100 + 100 words of listing fit a window
    # beside the fence and the heading, and a third line would not.
    code = [words(100, f"line{number}") for number in range(4)]
    markdown = "# Tides\n\n```\n" + "\n".join(code) + "\n```\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    assert [passage.text for passage in passages] == [
        f"# Tides\n\n```\n{code[0]}\n{code[1]}",
        f"{code[2]}\n{code[3]}\n```",
    ]


def test_a_single_line_of_a_listing_larger_than_a_window_is_still_cut_to_fit() -> None:
    # A line of data, say, with no line break to cut at.
    markdown = f"# Tides\n\n```\n{words(600)}\n```\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    assert all(word_count(passage.text) <= 256 for passage in passages)
    assert sum(word_count(passage.text) for passage in passages) == word_count(markdown)


def test_a_fenced_listing_stays_whole_across_its_blank_lines() -> None:
    listing = "```\ndef ebb():\n    pass\n\n# not a heading\n\ndef flow():\n    pass\n```"
    markdown = f"# Tides\n\n{words(250)}\n\n{listing}\n\nAfter the listing.\n"

    passages = cut_passages([chapter("Tides", 1)], markdown)

    # The prose fills the first window; the listing opens the next one intact
    # rather than being cut at either of its blank lines.
    assert [passage.text for passage in passages] == [
        f"# Tides\n\n{words(250)}",
        f"{listing}\n\nAfter the listing.",
    ]


def test_text_before_the_first_chapter_yields_nothing() -> None:
    markdown = (
        "A title page, set before any heading.\n"
        "\n"
        "# Tides\n"
        "\n"
        "The sea rises.\n"
        "\n"
        "# Currents\n"
        "\n"
        "The sea moves.\n"
    )
    tides, currents = chapter("Tides", 3), chapter("Currents", 7)

    passages = cut_passages([tides, currents], markdown)

    assert [(passage.chapter, passage.text) for passage in passages] == [
        (tides, "# Tides\n\nThe sea rises."),
        (currents, "# Currents\n\nThe sea moves."),
    ]


def test_passages_never_overlap_and_come_in_reading_order() -> None:
    paragraphs = "\n\n".join(words(70, f"p{number}") for number in range(40))
    markdown = f"# Tides\n\n{paragraphs}\n\n## Neap tides\n\n{paragraphs}\n"
    tides = chapter("Tides", 1, ("Neap tides", 83))

    passages = cut_passages([tides], markdown)

    assert len(passages) > 10
    assert all(word_count(passage.text) <= 256 for passage in passages)
    for earlier, later in zip(passages, passages[1:]):
        assert earlier.end_char <= later.start_char
    # Nothing but whitespace falls between the windows: every word of the
    # chapter is in exactly one passage.
    covered = sum(word_count(passage.text) for passage in passages)
    assert covered == word_count(markdown)
