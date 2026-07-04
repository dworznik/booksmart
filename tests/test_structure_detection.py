"""Unit tests for heading-tree structure detection over parsed Markdown (no DB)."""

from app.structure import detect_structure


def outline(markdown: str) -> list[tuple[str, list[str]]]:
    return [
        (chapter.title, [section.title for section in chapter.sections])
        for chapter in detect_structure(markdown)
    ]


class TestDetectStructure:
    def test_chapters_with_sections(self) -> None:
        markdown = (
            "# Chapter One: Modules \n"
            "## Deep Modules \n"
            "body\n"
            "## Shallow Modules \n"
            "# Chapter Two: Complexity \n"
            "## Symptoms \n"
        )

        assert outline(markdown) == [
            ("Chapter One: Modules", ["Deep Modules", "Shallow Modules"]),
            ("Chapter Two: Complexity", ["Symptoms"]),
        ]

    def test_line_numbers_recorded(self) -> None:
        markdown = "intro\n# First\n## Sub\ntext\n# Second\n"

        chapters = detect_structure(markdown)

        assert chapters[0].line == 2
        assert chapters[0].sections[0].line == 3
        assert chapters[1].line == 5

    def test_chapter_level_is_minimum_heading_level_present(self) -> None:
        markdown = "## Part A\n### Detail A1\n## Part B\n"

        assert outline(markdown) == [("Part A", ["Detail A1"]), ("Part B", [])]

    def test_headings_deeper_than_sections_are_ignored(self) -> None:
        markdown = "# Top\n## Mid\n### Deep\n#### Deeper\n# Second Top\n"

        assert outline(markdown) == [("Top", ["Mid"]), ("Second Top", [])]

    def test_headings_inside_code_fences_are_ignored(self) -> None:
        markdown = "# Real\n```\n# not a heading\n## nor this\n```\n## After\n# Also Real\n"

        assert outline(markdown) == [("Real", ["After"]), ("Also Real", [])]

    def test_no_headings_yields_empty_structure(self) -> None:
        assert detect_structure("just prose\nacross lines\n") == []

    def test_section_heading_before_first_chapter_is_ignored(self) -> None:
        markdown = "## Orphan\n# First Chapter\n## Kept\n# Second Chapter\n"

        assert outline(markdown) == [
            ("First Chapter", ["Kept"]),
            ("Second Chapter", []),
        ]

    def test_closing_hashes_and_whitespace_stripped_from_titles(self) -> None:
        markdown = "# Title ##  \n"

        assert outline(markdown) == [("Title", [])]

    def test_chapter_only_documents(self) -> None:
        markdown = "# Alpha\ntext\n# Beta\n"

        assert outline(markdown) == [("Alpha", []), ("Beta", [])]

    def test_lone_top_heading_is_book_title_not_chapter(self) -> None:
        markdown = "# The Book Title\n## Chapter One\n### Section A\n## Chapter Two\n"

        assert outline(markdown) == [
            ("Chapter One", ["Section A"]),
            ("Chapter Two", []),
        ]

    def test_single_chapter_book_with_no_deeper_headings_keeps_chapter(self) -> None:
        markdown = "# Only Chapter\nbody text\n"

        assert outline(markdown) == [("Only Chapter", [])]

    def test_mismatched_fence_markers_do_not_confuse_detection(self) -> None:
        markdown = "~~~\n```\n# inside tilde fence\n~~~\n# Real Chapter\n"

        assert outline(markdown) == [("Real Chapter", [])]
