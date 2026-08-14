"""The extraction gate's counters, and the properties that keep them honest.

Every number here is a proxy, and a proxy that lies is worse than no proxy at
all — it is a green gate over a book that came out wrong. So each counter is
tested against a document whose answer was decided before the counter ran, and
the load-bearing one (`longest_dropped_run`) is tested against a fixture with a
*deliberate* omission, which is the only way to know it can see one.

Fast and corpus-free on purpose: this half of the gate runs on every push.
The other half — the pinned corpus sweep — is a human ritual and lives in
`booksmart-bench`.
"""

from pathlib import Path

import pytest

from booksmart_core.parse_metrics import (
    ExtractorReport,
    coverage,
    dropped_runs,
    measure,
    source_text_of,
    suspect_headings,
)

from . import documents


def report(route: str = "probe") -> ExtractorReport:
    return ExtractorReport(route=route)


class TestDroppedRuns:
    """Real loss is contiguous; furniture is short and scattered. The metric
    exists to tell those apart with no furniture model at all."""

    def test_output_carrying_everything_drops_nothing(self) -> None:
        source = "The parser reads the page. It writes markdown."

        assert dropped_runs(source, "The parser reads the page. It writes markdown.") == ()

    def test_a_removed_paragraph_is_reported_with_its_text(self) -> None:
        kept = "Alpha beta gamma delta. " * 5
        lost = "The appendix table nobody noticed had gone missing entirely. "
        source = kept + lost + kept

        runs = dropped_runs(source, kept + kept)

        assert len(runs) == 1
        assert "appendix table nobody noticed" in runs[0].excerpt
        # The run is measured in characters of source, not words.
        assert runs[0].characters >= len(lost) - 2

    def test_scattered_furniture_never_adds_up_to_one_long_run(self) -> None:
        """Running headers and page numbers are *supposed* to be dropped. Each
        is its own short run, and the longest of them stays short — which is why
        the gate can cap the run without modelling furniture."""
        pages = [
            f"PARSING CHAPTER THREE\nBody text on page {number} carrying real prose "
            f"about extraction and fences.\n{number}\n"
            for number in range(1, 21)
        ]
        source = "".join(pages)
        output = "\n\n".join(
            f"Body text on page {number} carrying real prose about extraction and fences."
            for number in range(1, 21)
        )

        runs = dropped_runs(source, output)

        assert runs, "the furniture really was dropped"
        assert max(run.characters for run in runs) < 40

    def test_a_word_reappearing_far_away_does_not_excuse_the_loss(self) -> None:
        """Multiset matching alone would call a dropped listing 'present'
        because its identifiers also occur in an index. Unique words anchor the
        comparison so a match has to be in roughly the right place."""
        preface = "one two three four five six seven eight nine ten. "
        listing = "handle request timeout retry backoff jitter deadline cancel. "
        index = "handle request timeout retry backoff jitter deadline cancel. "
        source = preface + listing + preface + index

        runs = dropped_runs(source, preface + preface + index)

        assert runs
        assert max(run.characters for run in runs) >= len(listing) - 2

    def test_reordered_output_is_not_loss(self) -> None:
        """The pipeline is allowed to move a footnote. Only absence counts."""
        first = "Alpha beta gamma. "
        second = "Delta epsilon zeta. "

        assert dropped_runs(first + second, second + first) == ()

    def test_the_run_points_at_what_actually_went_missing(self) -> None:
        """The excerpt is the deliverable, so it has to land on the hole rather
        than on some other stretch of the same words."""
        before = "Alpha bravo charlie delta echo foxtrot. "
        listing = "Kilo lima mike november oscar papa quebec. "
        after = "Romeo sierra tango uniform victor whiskey. "

        runs = dropped_runs(before + listing + after, before + after)

        assert len(runs) == 1
        assert "kilo lima mike" in runs[0].excerpt

    def test_a_word_broken_across_a_line_end_hyphen_is_still_the_word(self) -> None:
        """Otherwise every de-hyphenated word in the book reads as a small loss,
        and the noise floor swallows a real one."""
        assert dropped_runs("extrac-\ntion of struc-\nture", "extraction of structure") == ()

    def test_case_and_invisible_characters_do_not_count_as_loss(self) -> None:
        """Some PDFs carry a soft hyphen at every legal break point, and a
        heading arrives title-cased in one path and lower in the other."""
        assert dropped_runs("PARS\u00adING the WORD\u200b", "parsing the word") == ()

    def test_an_empty_output_loses_the_whole_document(self) -> None:
        source = "Alpha beta gamma delta epsilon."

        runs = dropped_runs(source, "")

        assert len(runs) == 1
        assert runs[0].characters >= len("Alpha beta gamma delta epsilon") - 1


class TestCoverage:
    """Reported, never gated — the pipeline is meant to drop furniture, so a
    number below 100 is normal and a target would be a wrong one."""

    def test_an_identical_output_covers_everything(self) -> None:
        assert coverage("alpha beta gamma", "alpha beta gamma") == 100.0

    def test_an_empty_output_covers_nothing(self) -> None:
        assert coverage("alpha beta gamma", "") == 0.0

    def test_an_empty_source_is_fully_covered_rather_than_undefined(self) -> None:
        """A source with no text layer at all divides by zero otherwise, and a
        crash in the gate reads as a failing book."""
        assert coverage("", "anything") == 100.0

    def test_half_a_document_covers_about_half_of_it(self) -> None:
        kept = "alpha beta gamma delta "
        lost = "epsilon zeta eta theta "

        assert 40.0 < coverage(kept + lost, kept) < 60.0


class TestSuspectHeadings:
    """A code-shaped ATX title is the measured symptom of fencing having failed
    upstream: `detect_structure` takes the minimum heading level present, so one
    of these can demote every real chapter in the book (ADR 0003)."""

    def test_a_call_expression_is_suspect(self) -> None:
        assert suspect_headings("# extract(path, file_format)") == ("extract(path, file_format)",)

    def test_a_language_keyword_opening_is_suspect(self) -> None:
        assert len(suspect_headings("## def parse(self):\n\n### class Parser:")) == 2

    def test_an_operator_is_suspect(self) -> None:
        assert suspect_headings("# a == b") == ("a == b",)

    def test_an_ordinary_chapter_title_is_not_suspect(self) -> None:
        markdown = "\n\n".join(
            [
                "# Chapter 4: Interfaces Should Be Narrow",
                "## Why Interfaces Matter",
                "### A Note on Naming (And Why It Is Hard)",
                "# Preface to the Second Edition",
            ]
        )

        assert suspect_headings(markdown) == ()

    @pytest.mark.parametrize(
        "title",
        [
            "From the Preface to the First Edition",
            "Class Invariants and Functional Languages",
            "Static Configuration",
            "Function singledispatch",
            "Class Properties",
            "Return of the King",
            "Import and Export",
        ],
    )
    def test_a_title_cased_keyword_is_english_not_code(self, title: str) -> None:
        """Every keyword in the list is also an English word. Matching them
        case-insensitively flagged five real chapter titles across three pinned
        books and zero real defects — code writes its keywords in lower case in
        every language this corpus holds."""
        assert suspect_headings(f"## {title}") == ()

    def test_a_heading_inside_a_fence_is_not_a_heading_at_all(self) -> None:
        """Fenced code is what the ADR buys; counting inside it would report the
        success as a failure."""
        markdown = "```python\n# extract(path)\ndef parse(self):\n```"

        assert suspect_headings(markdown) == ()


class TestMeasure:
    """The whole row, assembled."""

    def test_it_counts_fences_headings_and_the_share_of_code(self) -> None:
        markdown = "\n".join(
            [
                "# Chapter 1",
                "",
                "Prose about parsing.",
                "",
                "```python",
                "def parse(self):",
                "    return 1",
                "```",
                "",
                "| a | b |",
                "| --- | --- |",
                "| 1 | 2 |",
            ]
        )

        metrics = measure(markdown, source_text=markdown, report=report())

        assert metrics.fences == 1
        assert metrics.headings == 1
        assert metrics.blocks == {"heading": 1, "paragraph": 1, "code": 1, "table": 1}
        assert metrics.code_characters == len("def parse(self):\n    return 1\n")
        assert 0.0 < metrics.code_share < 1.0

    def test_an_unterminated_fence_is_not_a_block(self) -> None:
        metrics = measure("```\ncode()", source_text="code()", report=report())

        assert metrics.fences == 0

    def test_the_extractor_report_passes_through_untouched(self) -> None:
        """Rule counts and declines are the extractor's own testimony — no
        reader of the artifact can recover which rule fenced a block."""
        metrics = measure(
            "prose",
            source_text="prose",
            report=ExtractorReport(
                route="epub",
                rule_counts={"pre": 944},
                declines=("indentation is CSS-only",),
                skipped_documents=174,
            ),
        )

        assert metrics.route == "epub"
        assert metrics.rule_counts == {"pre": 944}
        assert metrics.declines == ("indentation is CSS-only",)
        assert metrics.skipped_documents == 174

    def test_the_row_round_trips_through_json(self) -> None:
        """The baseline is a committed JSON file, so a metric that cannot be
        written and read back is a metric that cannot be ratcheted."""
        metrics = measure("# Title\n\nprose", source_text="Title prose", report=report("pdf"))

        restored = type(metrics).from_dict(metrics.as_dict())

        assert restored == metrics


class TestAgainstARealDocument:
    """The counters are only worth anything if they survive a document that a
    parser actually read, rather than markdown written to please them."""

    def test_a_generated_pdf_round_trips_with_little_loss(self, tmp_path: Path) -> None:
        path = documents.build_probe_pdf(tmp_path / "probe.pdf", pages=2)
        source = source_text_of(path)

        metrics = measure(source, source_text=source, report=report())

        assert metrics.source_characters > 400
        assert metrics.longest_dropped_run == 0
        assert metrics.coverage_pct == 100.0

    def test_a_deliberately_truncated_output_is_caught(self, tmp_path: Path) -> None:
        """The one test that proves the gate can see a hole: three pages in, two
        pages out, and the missing page has to show up as one long run."""
        path = documents.build_probe_pdf(tmp_path / "probe.pdf", pages=3)
        whole = source_text_of(path)
        sliced = source_text_of(
            documents.first_pages(path, 2, tmp_path / "slice.pdf")
        )

        metrics = measure(sliced, source_text=whole, report=report())

        assert metrics.longest_dropped_run > 200
        assert metrics.coverage_pct < 80.0
        assert "Chapter 3" in metrics.longest_dropped_excerpt.title()
