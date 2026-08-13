"""Unit tests for heading-tree structure detection over parsed Markdown (no DB)."""

from booksmart_core.structure import detect_structure


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


class TestEmphasisStripping:
    def test_bold_wrapped_titles_are_stripped(self) -> None:
        markdown = (
            "# **Chapter 1: Introduction**\n"
            "## **It's All About Complexity**\n"
            "# **Chapter 2: The Nature of Complexity**\n"
        )

        assert outline(markdown) == [
            ("Chapter 1: Introduction", ["It's All About Complexity"]),
            ("Chapter 2: The Nature of Complexity", []),
        ]

    def test_italic_and_underscore_wrappers_are_stripped(self) -> None:
        markdown = "# *Preface*\n# _Chapter 2: Nature_\n"

        assert outline(markdown) == [("Preface", []), ("Chapter 2: Nature", [])]

    def test_inner_emphasis_markers_are_removed(self) -> None:
        markdown = "# The **Deep** Module and the *Shallow* One\n"

        assert outline(markdown) == [("The Deep Module and the Shallow One", [])]

    def test_bold_italic_nesting_is_fully_stripped(self) -> None:
        markdown = "# ***Chapter 3***\n# ***Chapter 4***\n"

        assert outline(markdown) == [("Chapter 3", []), ("Chapter 4", [])]

    def test_intraword_underscores_survive(self) -> None:
        markdown = "# The snake_case_convention Explained\n"

        assert outline(markdown) == [("The snake_case_convention Explained", [])]


class TestChapterNumberTitleMerging:
    def test_bold_number_and_title_pair_merge_into_one_chapter(self) -> None:
        markdown = (
            "# **Chapter 4**\n"
            "# **Modules Should Be Deep**\n"
            "body\n"
            "# **Chapter 5**\n"
            "# **Information Hiding**\n"
        )

        assert outline(markdown) == [
            ("Chapter 4: Modules Should Be Deep", []),
            ("Chapter 5: Information Hiding", []),
        ]

    def test_merged_chapter_keeps_first_heading_source_line(self) -> None:
        markdown = "# Chapter 4\n# Modules Should Be Deep\n# Chapter 5\n# Interfaces\n"

        chapters = detect_structure(markdown)

        assert chapters[0].line == 1
        assert chapters[1].line == 3

    def test_part_roman_numeral_pairs_merge(self) -> None:
        markdown = "# Part I\n# Foundations\n# Part II\n# Practices\n"

        assert outline(markdown) == [
            ("Part I: Foundations", []),
            ("Part II: Practices", []),
        ]

    def test_consecutive_number_only_chapters_do_not_merge(self) -> None:
        markdown = "# Chapter 1\ntext\n# Chapter 2\ntext\n"

        assert outline(markdown) == [("Chapter 1", []), ("Chapter 2", [])]

    def test_bare_number_chapter_with_body_does_not_merge_into_next_heading(self) -> None:
        markdown = (
            "# Chapter 2\n"
            "A full chapter body follows the bare number heading.\n"
            "It runs for several lines before the next chapter starts.\n"
            "More prose here.\n"
            "# Epilogue\n"
        )

        assert outline(markdown) == [("Chapter 2", []), ("Epilogue", [])]

    def test_pair_separated_by_blank_line_still_merges(self) -> None:
        markdown = "# Chapter 4\n\n# Modules Should Be Deep\n# Chapter 5\n\n# Interfaces\n"

        assert outline(markdown) == [
            ("Chapter 4: Modules Should Be Deep", []),
            ("Chapter 5: Interfaces", []),
        ]

    def test_number_heading_followed_by_section_does_not_merge(self) -> None:
        markdown = "# Chapter 1\n## A Section\n# Chapter 2\n## Another\n"

        assert outline(markdown) == [
            ("Chapter 1", ["A Section"]),
            ("Chapter 2", ["Another"]),
        ]

    def test_sections_after_merged_pair_attach_to_merged_chapter(self) -> None:
        markdown = (
            "# Chapter 4\n"
            "# Modules Should Be Deep\n"
            "## Abstractions\n"
            "# Chapter 5\n"
            "# Information Hiding\n"
        )

        assert outline(markdown) == [
            ("Chapter 4: Modules Should Be Deep", ["Abstractions"]),
            ("Chapter 5: Information Hiding", []),
        ]


def kinds(markdown: str) -> list[tuple[str, str]]:
    return [(chapter.title, chapter.kind) for chapter in detect_structure(markdown)]


class TestMatterClassification:
    def test_front_matter_titles_are_marked(self) -> None:
        markdown = (
            "# Title Page\n# Copyright\n# Contents\n# Preface\n"
            "# Chapter 1: Introduction\n# Chapter 2: Complexity\n"
        )

        assert kinds(markdown) == [
            ("Title Page", "front_matter"),
            ("Copyright", "front_matter"),
            ("Contents", "front_matter"),
            ("Preface", "front_matter"),
            ("Chapter 1: Introduction", "chapter"),
            ("Chapter 2: Complexity", "chapter"),
        ]

    def test_back_matter_titles_are_marked(self) -> None:
        markdown = "# Chapter 1: Ideas\n# Appendix A: Notation\n# Bibliography\n# Index\n"

        assert kinds(markdown) == [
            ("Chapter 1: Ideas", "chapter"),
            ("Appendix A: Notation", "back_matter"),
            ("Bibliography", "back_matter"),
            ("Index", "back_matter"),
        ]

    def test_ambiguous_titles_classified_by_position(self) -> None:
        markdown = (
            "# Acknowledgments\n# Chapter 1: Ideas\n# Chapter 2: More\n# About the Author\n"
        )

        assert kinds(markdown) == [
            ("Acknowledgments", "front_matter"),
            ("Chapter 1: Ideas", "chapter"),
            ("Chapter 2: More", "chapter"),
            ("About the Author", "back_matter"),
        ]

    def test_preface_variants_match_by_prefix(self) -> None:
        markdown = "# Preface to the Second Edition\n# Chapter 1: Ideas\n"

        assert kinds(markdown) == [
            ("Preface to the Second Edition", "front_matter"),
            ("Chapter 1: Ideas", "chapter"),
        ]

    def test_leading_title_page_heading_before_front_matter_is_front_matter(self) -> None:
        markdown = "# A Philosophy of Software Design\n# Preface\n# Chapter 1: Ideas\n"

        assert kinds(markdown) == [
            ("A Philosophy of Software Design", "front_matter"),
            ("Preface", "front_matter"),
            ("Chapter 1: Ideas", "chapter"),
        ]

    def test_matter_lookalikes_with_longer_titles_stay_chapters(self) -> None:
        markdown = "# Notes on Distributed Systems\n# References and Borrowing\n"

        assert kinds(markdown) == [
            ("Notes on Distributed Systems", "chapter"),
            ("References and Borrowing", "chapter"),
        ]

    def test_ordinary_titles_stay_chapters(self) -> None:
        markdown = "# Introduction\n# The Indexing Problem\n"

        assert kinds(markdown) == [
            ("Introduction", "chapter"),
            ("The Indexing Problem", "chapter"),
        ]


class TestRealisticEpubOutline:
    def test_philosophy_style_epub_yields_clean_chapter_list(self) -> None:
        """The regression observed on A Philosophy of Software Design: bold
        headings, number/title pairs, and front matter inflating 21 chapters
        to 49 detected ones."""
        parts = [
            "# **A Philosophy of Software Design**",
            "# **Copyright**",
            "# **Contents**",
            "# **Preface**",
        ]
        for number, title in enumerate(
            ["Introduction", "The Nature of Complexity", "Working Code Isn't Enough"],
            start=1,
        ):
            parts.append(f"# **Chapter {number}**")
            parts.append(f"# **{title}**")
            parts.append("Body prose for the chapter.")
        parts.append("# **Summary of Design Principles**")
        parts.append("# **Index**")
        markdown = "\n".join(parts) + "\n"

        chapters = detect_structure(markdown)

        assert [(c.title, c.kind) for c in chapters] == [
            ("A Philosophy of Software Design", "front_matter"),
            ("Copyright", "front_matter"),
            ("Contents", "front_matter"),
            ("Preface", "front_matter"),
            ("Chapter 1: Introduction", "chapter"),
            ("Chapter 2: The Nature of Complexity", "chapter"),
            ("Chapter 3: Working Code Isn't Enough", "chapter"),
            ("Summary of Design Principles", "chapter"),
            ("Index", "back_matter"),
        ]
        body_chapters = [c for c in chapters if c.kind == "chapter"]
        assert len(body_chapters) == 4
