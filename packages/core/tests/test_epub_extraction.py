"""The `epub` route: the container, the spine, the code rules, the declines.

Every EPUB here is built in the test from stated markup. That is not only the
no-book-text-in-this-repo rule — it is that each rule exists because one real
publisher's toolchain emits one specific shape, and a fixture spelling that shape
out is the only readable record of what the rule is *for*. The per-book counts the
rules were derived from live in the issue and in the gate's baseline; what lives
here is the markup.

Two things are asserted more strictly than they may look:

- **A fence body is verbatim.** A fence claims its contents are the code as the
  book set it, so nothing may normalise, re-indent or re-wrap it.
- **The heading set is the source's `<hN>` set and nothing else.** Measured exact
  on every pinned book. Every extra heading is a chapter the book does not
  have, and `detect_structure` takes the *minimum* level present — so one
  invented `<h1>` demotes every real chapter to a section.
"""

import re
import zipfile
from pathlib import Path

import pytest

from booksmart_core.parsing import ParseFailure
from booksmart_core.parsing.epub import EpubExtractor, read_spine
from booksmart_core.structure import detect_structure

CONTAINER = (
    '<?xml version="1.0"?>'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles></container>'
)


def build_epub(
    path: Path,
    documents: dict[str, str],
    *,
    spine: list[str] | None = None,
    manifest: dict[str, str] | None = None,
    omit: frozenset[str] = frozenset(),
) -> Path:
    """An EPUB from `{filename: body html}`.

    ``manifest`` and ``omit`` exist for the failure cases: a spine referencing an
    id the manifest never declares, and a manifest entry whose file is not in the
    zip.
    """
    items = manifest or {f"id{index}": name for index, name in enumerate(documents)}
    order = spine if spine is not None else list(items)
    opf = (
        '<?xml version="1.0"?>'
        '<package version="3.0" xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid">'
        '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>T</dc:title>'
        '<dc:identifier id="uid">x</dc:identifier><dc:language>en</dc:language></metadata>'
        "<manifest>"
        + "".join(
            f'<item id="{item_id}" href="{href}" media-type="application/xhtml+xml"/>'
            for item_id, href in items.items()
        )
        + "</manifest><spine>"
        + "".join(f'<itemref idref="{item_id}"/>' for item_id in order)
        + "</spine></package>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", opf)
        for name, body in documents.items():
            if name in omit:
                continue
            archive.writestr(
                f"OEBPS/{name}",
                '<?xml version="1.0" encoding="utf-8"?>'
                '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title>'
                "<style>p { margin: 0 }</style></head>"
                f"<body>{body}</body></html>",
            )
    return path


def extract(path: Path) -> tuple[str, object]:
    result = EpubExtractor().extract(path, lambda _: None)
    return result.markdown, result.report


def fence_bodies(markdown: str) -> list[str]:
    """Every fenced block's body, exactly as emitted."""
    bodies: list[str] = []
    lines = markdown.split("\n")
    index = 0
    while index < len(lines):
        if lines[index].startswith("```"):
            marker = lines[index].split(None, 1)[0] if lines[index].strip("`") else lines[index]
            marker = "`" * len(lines[index]) if set(lines[index]) == {"`"} else marker
            closing = next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if lines[candidate].strip() == marker.strip()
                ),
                None,
            )
            if closing is not None:
                bodies.append("\n".join(lines[index + 1 : closing]))
                index = closing + 1
                continue
        index += 1
    return bodies


class TestTheSpineIsTheReadingOrder:
    def test_documents_come_out_in_spine_order(self, tmp_path: Path) -> None:
        """Not manifest order, and not alphabetical. Spine order was measured
        correct in every pinned book."""
        path = build_epub(
            tmp_path / "b.epub",
            {"c.xhtml": "<p>third</p>", "a.xhtml": "<p>first</p>", "b.xhtml": "<p>second</p>"},
            manifest={"c": "c.xhtml", "a": "a.xhtml", "b": "b.xhtml"},
            spine=["a", "b", "c"],
        )

        markdown, _ = extract(path)

        assert markdown.index("first") < markdown.index("second") < markdown.index("third")

    def test_navigation_documents_are_never_opened(self, tmp_path: Path) -> None:
        """`toc.ncx` and `nav.xhtml` are ignored outright: one pinned book's NCX
        points at a file absent from its own manifest, and the spine is the one
        construct EPUB 2 and 3 express identically."""
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": "<p>body text</p>"},
            manifest={"a": "a.xhtml"},
            spine=["a"],
        )
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("OEBPS/nav.xhtml", "<html><body><h1>NAVIGATION</h1></body></html>")
            archive.writestr("OEBPS/toc.ncx", "<ncx><text>ALSO NAVIGATION</text></ncx>")

        markdown, _ = extract(path)

        assert "NAVIGATION" not in markdown

    def test_a_spine_item_missing_from_the_manifest_fails_the_book(self, tmp_path: Path) -> None:
        """A book with a silent hole looks complete, so nothing downstream would
        ever question it."""
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": "<p>one</p>"},
            manifest={"a": "a.xhtml"},
            spine=["a", "ghost"],
        )

        with pytest.raises(ParseFailure, match="ghost"):
            extract(path)

    def test_a_manifest_entry_with_no_file_fails_the_book(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": "<p>one</p>", "gone.xhtml": "<p>two</p>"},
            manifest={"a": "a.xhtml", "g": "gone.xhtml"},
            spine=["a", "g"],
            omit=frozenset({"gone.xhtml"}),
        )

        with pytest.raises(ParseFailure, match=re.escape("gone.xhtml")):
            extract(path)

    def test_an_empty_spine_fails_rather_than_producing_an_empty_book(
        self, tmp_path: Path
    ) -> None:
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<p>one</p>"}, spine=[])

        with pytest.raises(ParseFailure, match="empty spine"):
            extract(path)

    def test_a_percent_encoded_href_resolves(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub",
            {"a b.xhtml": "<p>spaced filename</p>"},
            manifest={"a": "a%20b.xhtml"},
            spine=["a"],
        )

        markdown, _ = extract(path)

        assert "spaced filename" in markdown

    def test_the_spine_resolves_to_zip_members(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<p>one</p>"}, manifest={"a": "a.xhtml"}, spine=["a"]
        )

        with zipfile.ZipFile(path) as archive:
            assert [item.href for item in read_spine(archive)] == ["OEBPS/a.xhtml"]


class TestMalformedMarkupIsToleratedNotDeleted:
    def test_an_undefined_entity_does_not_fail_the_document(self, tmp_path: Path) -> None:
        """One pinned book fails 25 of its 123 spine documents under a strict XML
        parser with `undefined entity &nbsp;`. `lxml`'s recover=True instead
        deletes the surrounding text, silently."""
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<p>before&nbsp;after</p>"})

        markdown, _ = extract(path)

        assert "before" in markdown and "after" in markdown

    def test_an_unclosed_element_does_not_swallow_the_rest_of_the_chapter(
        self, tmp_path: Path
    ) -> None:
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<p>first<p>second<h2>A Heading</h2><p>third"}
        )

        markdown, _ = extract(path)

        assert "first" in markdown and "second" in markdown and "third" in markdown
        assert "## A Heading" in markdown

    def test_a_stylesheet_never_reaches_the_artifact(self, tmp_path: Path) -> None:
        """A `<style>` body inlined into the artifact is thousands of characters an
        LLM then reads as prose."""
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<p>real text</p>"})

        markdown, _ = extract(path)

        assert "margin" not in markdown


class TestHeadings:
    def test_the_heading_set_is_exactly_the_source_hn_set(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": "<h1>Part One</h1><h2>Chapter One</h2><p>prose</p>"
                "<h3>A Section</h3><p>more prose</p><h6>An Aside</h6>"
            },
        )

        markdown, _ = extract(path)

        assert [line for line in markdown.split("\n") if line.startswith("#")] == [
            "# Part One",
            "## Chapter One",
            "### A Section",
            "###### An Aside",
        ]

    def test_emphasis_inside_a_heading_survives_into_the_artifact(self, tmp_path: Path) -> None:
        """`structure.py` unwraps emphasis from the *title* it persists; the GFM
        artifact carries what the book set."""
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<h2>The <em>Open</em> Principle</h2>"})

        markdown, _ = extract(path)

        assert "## The *Open* Principle" in markdown
        assert detect_structure(markdown)[0].title == "The Open Principle"

    def test_a_heading_spanning_lines_is_collapsed_to_one(self, tmp_path: Path) -> None:
        """An ATX heading is a line. A newline inside one ends the heading and
        starts a paragraph, so the second half of the title would be lost."""
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<h2>Modules Should\n   Be Deep</h2>"}
        )

        markdown, _ = extract(path)

        assert "## Modules Should Be Deep" in markdown


class TestTheCodeRules:
    def test_a_pre_element_is_fenced_verbatim(self, tmp_path: Path) -> None:
        body = "def parse(self):\n    if x:\n        return 1\n\n    return 2"
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": f"<pre>{body}</pre>"})

        markdown, report = extract(path)

        assert fence_bodies(markdown) == [body]
        assert report.rule_counts == {"pre": 1}  # type: ignore[attr-defined]

    def test_a_declared_language_becomes_the_fence_info_string(self, tmp_path: Path) -> None:
        """880 free fence info strings in one pinned book, read rather than
        guessed at."""
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": '<pre data-code-language="ts">const x: number = 1;</pre>'},
        )

        markdown, _ = extract(path)

        assert "```ts\nconst x: number = 1;\n```" in markdown

    def test_syntax_highlighting_spans_inside_a_pre_stay_text(self, tmp_path: Path) -> None:
        """30,591 of one book's 33,879 `<code>` elements are token spans inside
        `<pre>`. Wrapping each in backticks would make the listing unreadable —
        which is why inline versus block is decided by ancestry, never tag name."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<pre><code class="kr">interface</code> '
                '<code class="nx">State</code> <code class="p">{</code></pre>'
            },
        )

        markdown, _ = extract(path)

        assert fence_bodies(markdown) == ["interface State {"]

    def test_a_code_element_outside_a_pre_is_an_inline_span(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<p>Call <code>extract()</code> to begin.</p>"}
        )

        markdown, _ = extract(path)

        assert "Call `extract()` to begin." in markdown

    def test_a_paragraph_class_prefix_carries_a_listing(self, tmp_path: Path) -> None:
        """One pinned book contains no `<pre>` element at all: every listing is a
        `<p class="programlisting…">` with `<br/>` between lines. The prefix
        matters — the same book uses `programlisting1` through `4` for indent
        levels."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<p class="programlisting">void f() {<br/>  g();<br/>}</p>'
                '<p class="programlisting2">indented();</p>'
            },
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown) == ["void f() {\n  g();\n}", "indented();"]
        assert report.rule_counts == {"p.programlisting": 2}  # type: ignore[attr-defined]

    def test_a_code_table_becomes_one_fence_of_its_line_cells(self, tmp_path: Path) -> None:
        """Another book with no `<pre>` sets each line of a listing as a table row.
        The row's *other* cell is a gutter, and it must not reach the fence."""
        rows = "".join(
            f'<tr><td class="codeinfo"><span class="codeprefix"> </span></td>'
            f'<td class="codeline">{line}</td></tr>'
            for line in ("import sys", "", "def main():", "    return 0")
        )
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": f'<table class="processedcode">{rows}</table>'}
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown) == ["import sys\n\ndef main():\n    return 0"]
        assert report.rule_counts == {"table.processedcode": 1}  # type: ignore[attr-defined]

    def test_zero_width_spaces_are_stripped_from_a_reassembled_listing(
        self, tmp_path: Path
    ) -> None:
        """That publisher separates every highlighted token with a zero-width
        space — 4,153 of them in one book. A fence carrying those is code nobody
        can paste anywhere, and the lines are being reassembled from highlighting
        scaffolding rather than read from a `<pre>`, so there is no verbatim claim
        to break."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<table class="processedcode"><tr>'
                '<td class="codeinfo"> </td>'
                '<td class="codeline">\u200b<strong class="kw">from</strong>\u200b '
                "hypothesis \u200b<strong>import</strong>\u200b given</td></tr></table>"
            },
        )

        markdown, _ = extract(path)

        assert fence_bodies(markdown) == ["from hypothesis import given"]

    def test_a_boxed_paragraph_class_carries_a_listing(self, tmp_path: Path) -> None:
        """`within` is load-bearing: the same class outside the box is a caption."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<div class="boxa"><p class="pre-ex">def call<br/>  puts 1</p></div>'
                '<p class="pre-ex">a caption, not a listing</p>'
            },
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown) == ["def call\n  puts 1"]
        assert report.rule_counts == {"p.pre-ex": 1}  # type: ignore[attr-defined]

    def test_a_block_tt_is_a_listing_and_an_inline_one_is_not(self, tmp_path: Path) -> None:
        """543 of one book's `<tt class="calibre41">` elements are listings and the
        rest are inline mentions inside a paragraph. `parent_not` is what tells
        them apart, and it is ancestry rather than the tag name doing it."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<div><tt class="calibre41">currentFont.size = 16</tt></div>'
                '<p>Set <tt class="calibre41">size</tt> first.</p>'
            },
        )

        markdown, report = extract(path)

        assert fence_bodies(markdown) == ["currentFont.size = 16"]
        assert "Set `size` first." in markdown
        assert report.rule_counts == {"tt.calibre41": 1}  # type: ignore[attr-defined]

    def test_the_first_matching_rule_wins(self, tmp_path: Path) -> None:
        """`pre` is first because it is the only rule that is not
        publisher-specific."""
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": '<pre class="programlisting">x = 1</pre>'}
        )

        _, report = extract(path)

        assert report.rule_counts == {"pre": 1}  # type: ignore[attr-defined]

    def test_a_fence_outgrows_the_backticks_in_its_own_body(self, tmp_path: Path) -> None:
        """Books that show Markdown or shell examples contain backticks, and a
        three-backtick fence around a body containing three closes early — spilling
        the rest of the listing into prose."""
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<pre>run ```sh``` then stop</pre>"}
        )

        markdown, _ = extract(path)

        assert "````\nrun ```sh``` then stop\n````" in markdown


class TestNoChapterBoundaryFallsInsideAFence:
    def test_a_hash_comment_in_a_listing_is_not_a_heading(self, tmp_path: Path) -> None:
        """The whole of ADR 0003 in one assertion. Unfenced, these two comments
        become ATX headings — and since `detect_structure` takes the *minimum*
        level present, they would demote every real chapter to a section."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": "<h2>Chapter One</h2><p>prose</p>"
                "<pre># configure the parser\nparser = Parser()\n# run it\nparser.run()</pre>"
                "<h2>Chapter Two</h2><p>more prose</p>"
            },
        )

        markdown, _ = extract(path)
        chapters = detect_structure(markdown)

        assert [chapter.title for chapter in chapters] == ["Chapter One", "Chapter Two"]
        assert not any(chapter.sections for chapter in chapters)


class TestProseIsEscaped:
    def test_a_paragraph_opening_with_a_hash_does_not_become_a_heading(
        self, tmp_path: Path
    ) -> None:
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": "<h2>Real Chapter</h2><p># opens the file, in most shells</p>"},
        )

        markdown, _ = extract(path)

        assert "\\# opens the file" in markdown
        assert [chapter.title for chapter in detect_structure(markdown)] == ["Real Chapter"]

    @pytest.mark.parametrize(
        ("prose", "escaped"),
        [
            ("> quoted, but not a quote", "\\> quoted"),
            ("- not a bullet", "\\- not a bullet"),
            ("+ also not a bullet", "\\+ also not a bullet"),
            ("1. not a list item", "1\\. not a list item"),
            ("``` not a fence", "\\``` not a fence"),
            # A setext underline is a line of nothing but `=`; `=== text` is
            # ordinary prose, so only the bare form needs escaping.
            ("===", "\\==="),
        ],
    )
    def test_every_leading_marker_is_escaped(
        self, tmp_path: Path, prose: str, escaped: str
    ) -> None:
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": f"<p>{prose}</p>"})

        markdown, _ = extract(path)

        assert escaped in markdown

    def test_emphasis_markers_are_not_escaped(self, tmp_path: Path) -> None:
        """`*` is deliberately absent from the escape set: emphasis is emitted
        faithfully, and escaping it would break what `structure.py` unwraps."""
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<p>The <em>whole</em> point.</p>"})

        markdown, _ = extract(path)

        assert "The *whole* point." in markdown

    def test_a_fence_body_is_not_escaped(self, tmp_path: Path) -> None:
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<pre># a comment\n- a list</pre>"})

        markdown, _ = extract(path)

        assert fence_bodies(markdown) == ["# a comment\n- a list"]


class TestTheFurnitureFilter:
    def test_an_image_only_document_is_skipped_and_counted(self, tmp_path: Path) -> None:
        """Four of the pinned books ship a screenshot of every listing as an extra
        spine document — around seventy characters each, against thousands for real
        content. In one, 175 of 199 documents."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": "<h2>Chapter</h2><p>" + "Real content. " * 30 + "</p>",
                "shot.xhtml": '<p><img src="listing-4-2.png"/></p><p>Figure 4-2</p>',
            },
            manifest={"a": "a.xhtml", "s": "shot.xhtml"},
            spine=["a", "s"],
        )

        markdown, report = extract(path)

        assert report.skipped_documents == 1  # type: ignore[attr-defined]
        assert "Figure 4-2" not in markdown

    def test_a_short_document_with_no_image_is_kept(self, tmp_path: Path) -> None:
        """Both conjuncts matter. One pinned book is split into hundreds of Calibre
        fragments,
        many of them shorter than this, and a length-only rule would eat it."""
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<p>A short but real paragraph.</p>"})

        markdown, report = extract(path)

        assert report.skipped_documents == 0  # type: ignore[attr-defined]
        assert "A short but real paragraph." in markdown

    def test_an_illustrated_chapter_is_kept(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<h2>Chapter</h2><p><img src="diagram.png"/></p><p>'
                + "Real prose about the diagram. " * 20
                + "</p>"
            },
        )

        _, report = extract(path)

        assert report.skipped_documents == 0  # type: ignore[attr-defined]


class TestDeclines:
    def test_a_book_no_rule_matches_declines_with_its_evidence(self, tmp_path: Path) -> None:
        """One pinned book carries its 464 code lines as flush-left `<p>` elements
        whose indentation exists only in CSS. Reading them directly gets the lines
        right and the indentation entirely wrong, which is worse than not fencing:
        a fence asserts its body is verbatim, and nothing downstream can discover
        the assertion is false."""
        path = build_epub(
            tmp_path / "b.epub",
            {
                "a.xhtml": '<h2>Chapter</h2><p class="s6p">if (x == 1) {</p>'
                '<p class="s6p">doSomething();</p><p class="s6p">}</p>'
            },
        )

        markdown, report = extract(path)

        assert "```" not in markdown
        assert report.declines  # type: ignore[attr-defined]
        assert "no code rule matched" in report.declines[0]  # type: ignore[attr-defined]
        assert "code-shaped" in report.declines[0]  # type: ignore[attr-defined]

    def test_the_decline_says_how_much_code_shaped_prose_it_found(
        self, tmp_path: Path
    ) -> None:
        """On its own "no fences" cannot tell a book with no code from a book whose
        code is invisible here. The count separates them with no per-book
        knowledge."""
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<h2>A Chapter</h2><p>Ordinary prose only.</p>"}
        )

        _, report = extract(path)

        assert "0 prose line(s) are code-shaped" in report.declines[0]  # type: ignore[attr-defined]

    def test_headings_decline_independently_of_code(self, tmp_path: Path) -> None:
        """Structure and code rest on different evidence, so they decline apart.
        One pinned book is a Calibre build with no semantic heading anywhere, only
        generated `p` classes that happen to be typeset large — and guessing which
        numbered class is a chapter invents a tree that reads as authoritative."""
        path = build_epub(
            tmp_path / "b.epub",
            {"a.xhtml": '<p class="class_s6h1">Chapter One</p><pre>x = 1</pre>'},
        )

        markdown, report = extract(path)

        assert detect_structure(markdown) == []
        assert report.rule_counts == {"pre": 1}  # type: ignore[attr-defined]
        declines = report.declines  # type: ignore[attr-defined]
        assert len(declines) == 1
        assert "no <h1>-<h6> element" in declines[0]

    def test_a_book_with_both_signals_declines_nothing(self, tmp_path: Path) -> None:
        path = build_epub(
            tmp_path / "b.epub", {"a.xhtml": "<h2>Chapter</h2><pre>x = 1</pre>"}
        )

        _, report = extract(path)

        assert report.declines == ()  # type: ignore[attr-defined]


class TestNoMuPdfOnThisPath:
    def test_the_module_does_not_import_pymupdf(self) -> None:
        """MuPDF is not on the EPUB path at all — which also dissolves the known
        SIGSEGV, since the one corpus file that crashes it is an EPUB."""
        source = (
            Path(__file__).parents[1] / "src/booksmart_core/parsing/epub.py"
        ).read_text()

        assert "pymupdf" not in source

    def test_the_extractor_reads_an_epub_with_mupdf_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The stronger form of the same claim: break the import and the route
        still works."""
        import sys

        monkeypatch.setitem(sys.modules, "pymupdf", None)
        path = build_epub(tmp_path / "b.epub", {"a.xhtml": "<h2>Chapter</h2><pre>x = 1</pre>"})

        markdown, _ = EpubExtractor().extract(path, lambda _: None), None

        assert "## Chapter" in markdown.markdown  # type: ignore[union-attr]


class TestAnUnreadableContainerIsRefusedNotCrashed:
    """Every other failure this route hits is a `ParseFailure` naming the file.

    A truncated download bypassed that and escaped as `zipfile.BadZipFile`, which
    reads as a bug rather than as a bad file — and acquisition, which fetches
    candidates from a book site and expects a refusal it can act on, would not
    know to catch it.
    """

    def test_a_file_that_is_not_a_zip_is_a_parse_failure(self, tmp_path: Path) -> None:
        truncated = tmp_path / "half-downloaded.epub"
        truncated.write_bytes(b"PK\x03\x04 and then the connection dropped")

        with pytest.raises(ParseFailure, match="not a readable zip"):
            EpubExtractor().extract(truncated, lambda _: None)

    def test_an_empty_file_is_a_parse_failure(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.epub"
        empty.write_bytes(b"")

        with pytest.raises(ParseFailure):
            EpubExtractor().extract(empty, lambda _: None)

    def test_malformed_container_xml_names_the_file_that_is_broken(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "broken.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("META-INF/container.xml", "<container><unclosed>")

        with pytest.raises(ParseFailure, match="container.xml"):
            EpubExtractor().extract(path, lambda _: None)
