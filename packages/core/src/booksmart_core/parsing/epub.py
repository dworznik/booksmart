"""The `epub` route: read the OCF container directly.

MuPDF is not on this path at all, which is the point. It rendered an EPUB into
pages before anything could look at the markup, so the only signal left was
typographic — and it also SIGSEGVs on one of the pinned corpus files, which is an
EPUB.

`zipfile` and `xml.etree.ElementTree` for the container and the package document;
`html.parser` for content documents. No new dependency, which matters for a
published library.

Content documents go through an **HTML-tolerant** parser, never strict XML, even
though they claim to be XHTML. One pinned book fails 25 of its 123 spine documents
under a strict parser with `undefined entity &nbsp;`, and `lxml`'s `recover=True`
silently deletes the surrounding text rather than reporting it.

## Reading order is the spine, and only the spine

`toc.ncx` (EPUB 2) and `nav.xhtml` (EPUB 3) decide nothing about reading order.
Spine order was measured correct in every pinned book, and one of them has an NCX
pointing at a file absent from its own manifest. The spine is also the one
construct both EPUB versions express identically, so ignoring navigation for this
question dissolves most of the 2-vs-3 divergence instead of handling it.

An unreadable spine item **fails the whole book**. A book with a silent hole looks
complete, so nothing downstream would ever question it.

## Structure is the book's own `<hN>`, or else what the container declares

A different question, answered from different evidence. Where a book marks its
headings up, that markup is the answer and navigation is not consulted at all: a
fuzzy match against a nav document is not evidence enough to overrule a
publisher's `<h2>`, and navigation routinely omits, renames and reorders what the
book actually sets.

Where a book carries **no `<h1>`-`<h6>` at all** — a conversion whose headings
live in generated class names only — there is no semantic markup to overrule, and
the NCX or nav document is the publisher's own statement of the chapter tree. It
is read then, and only then. The objection that kept navigation out of reading
order does not reach this: an entry pointing outside the manifest is a skippable
entry, not a corrupt book, and what it costs is its own chapter.

## Code rules

Ordered predicates over a tag stack, not CSS selectors: there is no CSS engine
here. Which rules exist was decided by surveying the pinned corpus,
and each is carried by real books rather than invented:

- `pre` covers most of the pinned books and about 80% of all code blocks.
- Two books contain **no `<pre>` element at all** and would come out fenceless
  without a rule for the carrier their publisher used instead.

Inline versus block is decided by **ancestry, never tag name**: 30,591 of one
book's 33,879 `<code>` elements are syntax-highlighting token spans inside
`<pre>`, and 543 of another's `<tt>` elements are listings while the rest are
inline mentions.

Where no rule matches anywhere in a book, the route **declines** and says so,
with the count of code-shaped prose lines as evidence (ADR 0003). One pinned book
carries its 464 code lines as flush-left `<p>` elements whose indentation exists
only in CSS, across 28 generated classes with two competing properties, seven of
them negative. Reading those directly gets the lines right and the indentation
entirely wrong, which is worse than not fencing: a fence asserts its body is
verbatim, and nothing downstream can discover that the assertion is false.
"""

import posixpath
import re
import codecs
import zipfile
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

from booksmart_core.parsing.contract import ExtractorReport, ParseFailure, ParseResult
from booksmart_core.parsing.blocks import Block, looks_like_code, to_gfm
from booksmart_core.titles import normalise

CONTAINER = "META-INF/container.xml"
OCF_NS = "{urn:oasis:names:tc:opendocument:xmlns:container}"
OPF_NS = "{http://www.idpf.org/2007/opf}"
NCX_NS = "{http://www.daisy.org/z3986/2005/ncx/}"
NCX_MEDIA_TYPE = "application/x-dtbncx+xml"

# A spine document that is a picture of content rather than content. Four of the
# pinned books ship a screenshot of every listing as an extra spine document —
# 195 of them, around seventy characters each, against thousands for real
# content; in one book that is 175 of 199 documents. Both conjuncts matter: a
# length-only rule would eat another book's hundreds of Calibre-split fragments.
FURNITURE_CHARACTERS = 200

HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Elements that end a paragraph. A `<p>` inside a `<div>` inside a `<section>` is
# one paragraph, not three, so nesting is fine — what matters is that text either
# side of one of these does not run together into a single block.
BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "ul", "ol", "dl", "dt", "dd", "blockquote", "section",
        "article", "aside", "nav", "header", "footer", "figure", "figcaption",
        "table", "thead", "tbody", "tr", "td", "th", "hr", "br", "main", "body",
        *HEADINGS,
    }
)

# Never yields text worth having, and `<style>`/`<script>` bodies actively harm:
# a stylesheet inlined into the artifact is thousands of characters of noise that
# an LLM then summarises.
DROP_TAGS = frozenset({"head", "style", "script", "svg", "title", "meta", "link"})

VOID_TAGS = frozenset(
    {"br", "hr", "img", "image", "meta", "link", "area", "base", "col", "input", "source"}
)

# Inline markup, emitted faithfully. An EPUB `<h2>` may legitimately contain
# `<em>` or `<strong>`; structure.py unwraps emphasis from titles while the GFM
# artifact keeps it.
INLINE_MARKERS = {
    "em": "*", "i": "*", "cite": "*", "var": "*",
    "strong": "**", "b": "**",
    "code": "`", "tt": "`", "samp": "`", "kbd": "`",
}

# Zero-width space. One publisher separates every highlighted token in a code
# table with one — 4,153 of them in a single book — so a fence body read straight
# from those cells is code nobody can paste anywhere.
ZERO_WIDTH = "\u200b"


@dataclass
class Element:
    """One element of a content document, with the ancestors a rule needs."""

    tag: str
    classes: tuple[str, ...] = ()
    attrs: Mapping[str, str] = field(default_factory=dict)
    children: list["Element | str"] = field(default_factory=list)
    parent: "Element | None" = None

    def has_class(self, name: str) -> bool:
        return name in self.classes

    def ancestors(self) -> Iterator["Element"]:
        node = self.parent
        while node is not None:
            yield node
            node = node.parent

    def descendants(self) -> Iterator["Element"]:
        for child in self.children:
            if isinstance(child, Element):
                yield child
                yield from child.descendants()

    def text(self) -> str:
        """All text under this element, with `<br/>` as a line break.

        `<br/>` matters more than it looks: four of the pinned books separate the
        lines of a listing with it and carry no newline at all, so a reader
        honouring only `\\n` flattens each of them into one very long line.
        """
        parts: list[str] = []
        for child in self.children:
            if isinstance(child, str):
                parts.append(child)
            elif child.tag == "br":
                parts.append("\n")
            elif child.tag in DROP_TAGS:
                continue
            else:
                parts.append(child.text())
        return "".join(parts)


class _Reader(HTMLParser):
    """A content document as a tree.

    A tree rather than a SAX state machine because the rules are about ancestry —
    "a `<tt>` whose parent is not a `<p>`", "a `<p class="pre-ex">` inside a
    `<div class="boxa">`" — and a state machine expressing those is a state
    machine nobody can check.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Element(tag="#document")
        self._open: list[Element] = [self.root]

    @property
    def _current(self) -> Element:
        return self._open[-1]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name: (value or "") for name, value in attrs}
        element = Element(
            tag=tag,
            classes=tuple((values.get("class") or "").split()),
            attrs=values,
            parent=self._current,
        )
        self._current.children.append(element)
        if tag not in VOID_TAGS:
            self._open.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self._open.pop()

    def handle_endtag(self, tag: str) -> None:
        # Close to the nearest matching ancestor rather than assuming the document
        # is well-formed. Real books ship stray `</p>` and unclosed `<li>`, and a
        # reader that trusted the tags would reparent the rest of the chapter.
        for index in range(len(self._open) - 1, 0, -1):
            if self._open[index].tag == tag:
                del self._open[index:]
                return

    def handle_data(self, data: str) -> None:
        self._current.children.append(data)


# --- the code rules --------------------------------------------------------

Predicate = Callable[[Element], bool]


@dataclass(frozen=True)
class CodeRule:
    """A carrier of code, and how to read its body.

    ``cell_class`` is for the one publisher that sets each line of a listing as a
    table row: the body is that class of cell, one per row, and the *other* cell
    in the row is a line-number gutter that must not reach the fence.
    """

    name: str
    matches: Predicate
    cell_class: str = ""
    # Strip zero-width spaces from the body. Off for `<pre>`, where the publisher
    # has asserted the text is preformatted and the fence promises it verbatim;
    # on where the lines are being reassembled from highlighting scaffolding that
    # uses them as separators.
    strip_zero_width: bool = False


def _pre_element(element: Element) -> bool:
    return element.tag == "pre"


def _p_class_prefix(prefix: str) -> Predicate:
    def matches(element: Element) -> bool:
        return element.tag == "p" and any(
            name.startswith(prefix) for name in element.classes
        )

    return matches


def _table_class(name: str) -> Predicate:
    def matches(element: Element) -> bool:
        return element.tag == "table" and element.has_class(name)

    return matches


def _p_class_within(name: str, within: str) -> Predicate:
    def matches(element: Element) -> bool:
        return (
            element.tag == "p"
            and element.has_class(name)
            and any(ancestor.has_class(within) for ancestor in element.ancestors())
        )

    return matches


def _tag_class_parent_not(tag: str, name: str, parent_not: str) -> Predicate:
    def matches(element: Element) -> bool:
        return (
            element.tag == tag
            and element.has_class(name)
            and (element.parent is None or element.parent.tag != parent_not)
        )

    return matches


# Ordered: the first match wins. `pre` is first because it is the only rule that
# is not publisher-specific, and every book that has one uses it for everything.
#
# The four that follow are each carried by exactly one pinned book, and each of
# those books contains no `<pre>` at all. They are the price of reading real
# EPUBs: publishers with no `<pre>` in their toolchain put listings in `<p>`,
# `<tt>` or a table, and there is no generic markup left to key on once CSS is
# out of reach.
CODE_RULES: tuple[CodeRule, ...] = (
    CodeRule(name="pre", matches=_pre_element),
    CodeRule(name="p.programlisting", matches=_p_class_prefix("programlisting")),
    CodeRule(
        name="table.processedcode",
        matches=_table_class("processedcode"),
        cell_class="codeline",
        strip_zero_width=True,
    ),
    CodeRule(name="p.pre-ex", matches=_p_class_within("pre-ex", "boxa")),
    CodeRule(
        name="tt.calibre41",
        matches=_tag_class_parent_not("tt", "calibre41", "p"),
    ),
)

# Where a publisher declares the language of a listing. 880 free fence info
# strings in one pinned book, read rather than guessed.
LANGUAGE_ATTRS = ("data-code-language", "data-lang", "lang")


def _language_of(element: Element) -> str:
    for attribute in LANGUAGE_ATTRS:
        value = element.attrs.get(attribute, "").strip()
        if value and value.isascii() and value.replace("-", "").replace("+", "").isalnum():
            return value.lower()
    return ""


def _code_body(element: Element, rule: CodeRule) -> str:
    if rule.cell_class:
        lines = [
            cell.text()
            for cell in element.descendants()
            if cell.tag in {"td", "th"} and cell.has_class(rule.cell_class)
        ]
        body = "\n".join(line.rstrip("\n") for line in lines)
    else:
        body = element.text()
    if rule.strip_zero_width:
        body = body.replace(ZERO_WIDTH, "")
    return body.strip("\n")


# --- document -> blocks ----------------------------------------------------


@dataclass(frozen=True)
class NavPoint:
    """One entry of the chapter tree the container declares."""

    level: int
    title: str
    href: str  # a zip member name
    fragment: str  # an element id within it, or "" for the document itself


def _declared_anchors(
    document: Element, points: Sequence[NavPoint]
) -> tuple[list[NavPoint], dict[int, NavPoint], dict[int, NavPoint]]:
    """Where in this document each declared entry's heading goes.

    Three answers, because a navigation target is one of three things. An entry
    naming a file and nothing finer heads the whole document. An entry anchoring
    the line that carries the chapter's title makes *that line* the heading —
    which is what stops the title being said twice, once as a heading and again
    as the paragraph it was read from. An entry anchoring something chapter-sized
    is a destination rather than a title, and the heading goes in front of it.

    An entry whose anchor is not in the document is skipped. A stale id costs its
    own chapter and nothing else.
    """
    leading: list[NavPoint] = []
    titles: dict[int, NavPoint] = {}
    before: dict[int, NavPoint] = {}
    by_id: dict[str, Element] = {}
    for element in document.descendants():
        identifier = element.attrs.get("id", "")
        if identifier and identifier not in by_id:
            by_id[identifier] = element
    for point in points:
        if not point.fragment:
            leading.append(point)
            continue
        anchor = by_id.get(point.fragment)
        if anchor is None:
            continue
        target = _title_element(anchor, point.title)
        if target is None:
            before.setdefault(id(anchor), point)
        else:
            titles.setdefault(id(target), point)
    return leading, titles, before


def _title_element(element: Element, title: str) -> Element | None:
    """The element whose own text *is* this entry's title, if either of these is.

    The anchor itself where it carries the title. Its parent where the anchor is
    an empty `<a id="…"/>` inside the line that does, which is what a conversion
    emits — an id is cheaper to place than a heading.

    Nothing where neither says what the container says the chapter is called: the
    anchor is then a destination inside the chapter rather than its title, and the
    heading goes in front of it. Failing this way costs a title said twice, once
    as a heading and again as the paragraph it sits in. Failing the other way
    would promote a paragraph of prose to a heading and lose it as prose.

    Equality of the normalised titles, and deliberately not `titles_match`: the
    containment that test allows is what makes it right for *finding* a title on a
    page, and exactly what must not happen here — a paragraph containing the
    chapter's name is not the chapter's name.
    """
    wanted = normalise(title)
    if not wanted:
        return None
    if normalise(element.text()) == wanted:
        return element
    parent = element.parent
    # `<body>` is never a title, however short the document is; and it is not a
    # child of anything the walk descends through, so a heading put there is lost.
    if parent is None or parent.tag in {"body", "html", "#document"}:
        return None
    return parent if normalise(parent.text()) == wanted else None


def _blocks_of(
    document: Element, rules: Sequence[CodeRule], declared: Sequence[NavPoint]
) -> tuple[list[Block], dict[str, int]]:
    blocks: list[Block] = []
    fired: dict[str, int] = {}
    pending: list[str] = []
    leading, titles, before = _declared_anchors(document, declared)
    blocks.extend(
        Block(kind="heading", text=point.title, level=point.level) for point in leading
    )

    def flush() -> None:
        text = "".join(pending)
        pending.clear()
        if text.strip():
            blocks.append(Block(kind="paragraph", text=_tidy(text)))

    def walk(element: Element) -> None:
        for child in element.children:
            if isinstance(child, str):
                pending.append(child)
                continue
            if child.tag in DROP_TAGS:
                continue
            if child.tag == "br":
                pending.append("\n")
                continue
            rule = next((candidate for candidate in rules if candidate.matches(child)), None)
            if rule is not None:
                flush()
                body = _code_body(child, rule)
                if body.strip():
                    fired[rule.name] = fired.get(rule.name, 0) + 1
                    blocks.append(
                        Block(
                            kind="code",
                            text=body,
                            language=_language_of(child),
                            rule=rule.name,
                        )
                    )
                continue
            # After the code rules, never before: a destination landing on a
            # listing does not open a heading in the middle of it (ADR 0003).
            declared_title = titles.get(id(child))
            if declared_title is not None:
                flush()
                blocks.append(
                    Block(
                        kind="heading",
                        text=_tidy(_inline(child)) or declared_title.title,
                        level=declared_title.level,
                    )
                )
                continue
            declared_before = before.get(id(child))
            if declared_before is not None:
                flush()
                blocks.append(
                    Block(
                        kind="heading",
                        text=declared_before.title,
                        level=declared_before.level,
                    )
                )
            level = HEADINGS.get(child.tag)
            if level is not None:
                flush()
                title = _tidy(_inline(child))
                if title:
                    blocks.append(Block(kind="heading", text=title, level=level))
                continue
            if child.tag in INLINE_MARKERS and child.tag not in BLOCK_TAGS:
                pending.append(_inline(Element(tag="#inline", children=[child])))
                continue
            if child.tag in BLOCK_TAGS:
                flush()
                walk(child)
                flush()
                continue
            walk(child)

    walk(document)
    flush()
    return blocks, fired


def _inline(element: Element) -> str:
    """An element's text with its inline markup carried into GFM.

    Recursive, so `<strong><em>x</em></strong>` survives as `***x***` rather than
    losing the inner marker — and so a heading gets the same treatment prose does.
    A heading is where this matters most: an EPUB `<h2>` may legitimately contain
    `<em>`, and `structure.py` unwraps emphasis from the *title* it persists while
    the artifact keeps what the book set.
    """
    parts: list[str] = []
    for child in element.children:
        if isinstance(child, str):
            parts.append(child)
            continue
        if child.tag in DROP_TAGS:
            continue
        if child.tag == "br":
            parts.append("\n")
            continue
        marker = INLINE_MARKERS.get(child.tag, "")
        inner = _inline(child)
        parts.append(f"{marker}{inner}{marker}" if marker and inner.strip() else inner)
    return "".join(parts)


_WHITESPACE = re.compile(r"[ \t\r\f\v]+")


def _tidy(text: str) -> str:
    """Collapse a run of prose into what it says.

    Content documents are indented XHTML, so every paragraph arrives wrapped and
    padded. Newlines survive only where a `<br/>` put one, because that is the
    only place they mean anything.
    """
    lines = [_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line) if len(lines) > 1 else lines[0]


# --- the container ---------------------------------------------------------


@dataclass(frozen=True)
class SpineItem:
    idref: str
    href: str  # a zip member name


def read_text(archive: zipfile.ZipFile, href: str) -> str:
    """One content document as text, decoded the way it says to decode it.

    The OCF spec allows a content document to be UTF-8 **or UTF-16**, and a
    UTF-16 file decoded as UTF-8 is not slightly wrong — every other byte is a
    NUL, so the tag soup parses to nothing and the chapter silently disappears.
    A byte-order mark is the reliable signal and is what `utf-8-sig` and the
    UTF-16 codecs consume for us.

    Still `errors="replace"` at the end: one bad byte in one chapter is not a
    reason to refuse a book, and the damage is a glyph rather than a document.
    """
    raw = archive.read(href)
    for mark, codec in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if raw.startswith(mark):
            return raw.decode(codec, "replace")
    return raw.decode("utf-8", "replace")


def _xml(raw: bytes, where: str) -> "ElementTree.Element":
    """Parse one of the container's XML files, or say which one was malformed.

    Every other failure this route can hit is a ``ParseFailure`` naming the file;
    an ``ElementTree.ParseError`` escaping from here would be the one exception a
    caller had to know about separately, and the router turns anything else into
    a crash rather than a routed refusal.
    """
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ParseFailure(f"{where} is not readable XML ({exc})") from exc


@contextmanager
def open_container(path: Path) -> Iterator[zipfile.ZipFile]:
    """The EPUB's zip, with an unreadable one refused the way the route refuses.

    ``_package_path`` already turns a missing ``META-INF/container.xml`` into
    "this is not an EPUB container", so the contract is plain: a file this route
    cannot read is a ``ParseFailure``. A truncated download bypassed it, escaping
    as ``BadZipFile`` — which reads as a bug rather than as a bad file, and which
    acquisition would not know to catch.
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ParseFailure(
            f"{path.name} is not a readable zip, so it is not an EPUB container ({exc})"
        ) from exc
    with archive:
        yield archive


def _package_path(archive: zipfile.ZipFile) -> str:
    try:
        raw = archive.read(CONTAINER)
    except KeyError as exc:
        raise ParseFailure(
            f"no {CONTAINER} in the archive, so this is not an EPUB container"
        ) from exc
    root = _xml(raw, CONTAINER)
    rootfile = root.find(f"{OCF_NS}rootfiles/{OCF_NS}rootfile")
    if rootfile is None or not rootfile.get("full-path"):
        raise ParseFailure(f"{CONTAINER} names no package document")
    return unquote(str(rootfile.get("full-path")))


def _member(base: str, href: str) -> str:
    """An href as a zip member name, resolved against the directory it was in.

    The directory differs by who wrote the href: a manifest's are relative to the
    package document, a nav document's to the nav document, and the two are not
    always the same directory.
    """
    return posixpath.normpath(posixpath.join(base, href)) if base and href else href


@dataclass(frozen=True)
class _Package:
    """The package document, and where its hrefs are resolved from."""

    root: "ElementTree.Element"
    path: str
    base: str

    def member(self, href: str) -> str:
        return _member(self.base, href)


def _read_package(archive: zipfile.ZipFile) -> _Package:
    package_path = _package_path(archive)
    try:
        raw_package = archive.read(package_path)
    except KeyError as exc:
        raise ParseFailure(
            f"the package document {package_path!r} named by {CONTAINER} is not in the archive"
        ) from exc
    return _Package(
        root=_xml(raw_package, package_path),
        path=package_path,
        base=posixpath.dirname(package_path),
    )


def read_spine(archive: zipfile.ZipFile) -> tuple[SpineItem, ...]:
    """The spine, resolved to zip member names, in reading order.

    A missing manifest entry or a member absent from the archive raises here
    rather than being skipped: a book with a silent hole looks complete, and
    nothing downstream would question it.
    """
    package = _read_package(archive)
    package_path = package.path

    manifest = {
        item.get("id"): unquote(item.get("href") or "")
        for item in package.root.iter(f"{OPF_NS}item")
        if item.get("id")
    }
    members = set(archive.namelist())

    items: list[SpineItem] = []
    for reference in package.root.iter(f"{OPF_NS}itemref"):
        idref = reference.get("idref") or ""
        href = manifest.get(idref)
        if not href:
            raise ParseFailure(
                f"spine item {idref!r} has no entry in the manifest, so the book has a "
                f"hole where a chapter should be"
            )
        member = package.member(href)
        if member not in members:
            raise ParseFailure(
                f"spine item {idref!r} names {member!r}, which is not in the archive"
            )
        items.append(SpineItem(idref=idref, href=member))
    if not items:
        raise ParseFailure(f"{package_path} declares an empty spine")
    return tuple(items)


def read_navigation(archive: zipfile.ZipFile) -> tuple[NavPoint, ...]:
    """The chapter tree the container declares, in document order.

    EPUB 3 says it in a nav document and EPUB 2 in an NCX; a book may carry
    either or both, and the two say the same thing differently, so both are read.
    Nesting is the level, which is the whole reason to prefer this over a flat
    list of destinations.

    Anything malformed here yields nothing rather than failing the book. This is
    consulted only where the book has no headings of its own, so the worst case
    is the answer that was already going to be given.
    """
    package = _read_package(archive)
    members = set(archive.namelist())
    for member, kind in _navigation_documents(package, members):
        try:
            points = (
                _ncx_points(_xml(archive.read(member), member), posixpath.dirname(member))
                if kind == "ncx"
                else _nav_points(read_text(archive, member), posixpath.dirname(member))
            )
        except ParseFailure:
            continue
        if points:
            return points
    return ()


def _navigation_documents(package: _Package, members: set[str]) -> list[tuple[str, str]]:
    """The container's navigation documents, EPUB 3's first.

    Three ways of declaring one, because books use all three: `properties="nav"`
    on a manifest item (EPUB 3), the `toc` attribute of the spine (EPUB 2), and
    the NCX media type on its own for a book that declares neither.
    """
    spine = package.root.find(f"{OPF_NS}spine")
    ncx_id = (spine.get("toc") or "") if spine is not None else ""
    nav = ncx = ""
    for item in package.root.iter(f"{OPF_NS}item"):
        member = package.member(unquote(item.get("href") or ""))
        if member not in members:
            continue
        if "nav" in (item.get("properties") or "").split():
            nav = member
        elif item.get("media-type") == NCX_MEDIA_TYPE or (ncx_id and item.get("id") == ncx_id):
            ncx = member
    return [
        (member, kind) for member, kind in ((nav, "nav"), (ncx, "ncx")) if member
    ]


def _navigation_point(level: int, title: str, source: str, base: str) -> NavPoint:
    href, _, fragment = source.partition("#")
    # Clamped because Markdown has six levels and navigation may nest deeper.
    return NavPoint(
        level=min(max(level, 1), 6),
        title=title.strip(),
        href=_member(base, href),
        fragment=fragment,
    )


def _ncx_points(root: "ElementTree.Element", base: str) -> tuple[NavPoint, ...]:
    """An EPUB 2 `navMap`, whose nesting is `navPoint` inside `navPoint`."""

    def walk(parent: "ElementTree.Element", level: int) -> Iterator[NavPoint]:
        for point in parent.findall(f"{NCX_NS}navPoint"):
            label = point.find(f"{NCX_NS}navLabel/{NCX_NS}text")
            content = point.find(f"{NCX_NS}content")
            title = (label.text or "").strip() if label is not None else ""
            source = unquote(content.get("src") or "") if content is not None else ""
            if title and source:
                yield _navigation_point(level, title, source, base)
            yield from walk(point, level + 1)

    nav_map = root.find(f"{NCX_NS}navMap")
    return tuple(walk(nav_map, 1)) if nav_map is not None else ()


def _nav_points(markup: str, base: str) -> tuple[NavPoint, ...]:
    """An EPUB 3 nav document, whose nesting is `<ol>` inside `<li>`.

    Read through the same tolerant parser content documents get: a nav document
    is a content document, and one book's is no better formed than its chapters.
    """
    document = read_document(markup)
    toc = next(
        (
            element
            for element in document.descendants()
            if element.tag == "nav"
            and (
                "toc" in element.attrs.get("epub:type", "").split()
                or element.attrs.get("role") == "doc-toc"
            )
        ),
        None,
    )
    if toc is None:
        return ()
    points: list[NavPoint] = []

    def walk(element: Element, level: int) -> None:
        for child in element.children:
            if not isinstance(child, Element):
                continue
            if child.tag == "ol":
                walk(child, level + 1)
                continue
            source = child.attrs.get("href", "") if child.tag == "a" else ""
            if source:
                title = _tidy(child.text()).strip()
                if title:
                    points.append(_navigation_point(level, title, unquote(source), base))
                continue
            walk(child, level)

    walk(toc, 0)
    return tuple(points)


def read_document(markup: str) -> Element:
    """One content document as a tree.

    Public because reading an EPUB's structure is worth doing *without*
    extracting it: the bench harness censuses tags and class names to reject a
    conversion that lost its headings, before spending a download or an
    extraction on it. That census needs the same tolerant parse the extractor
    uses — content documents claim to be XHTML and one book fails a quarter of
    its documents under a strict parser — so the alternative was a second reader
    that disagreed with this one about what the book contains.

    ``errors="replace"`` rather than strict: one bad byte in one chapter is not a
    reason to refuse a book, and the failure it would cause is far from the cause.
    """
    reader = _Reader()
    reader.feed(markup)
    return reader.root


def read_member(archive: zipfile.ZipFile, href: str) -> Element:
    """One spine document of an open container, as a tree.

    Decoded through ``read_text``, so a UTF-16 content document reads as the
    chapter it is rather than as the empty one a UTF-8 decode makes of it.
    """
    return read_document(read_text(archive, href))


def _by_member(
    points: Sequence[NavPoint], spine: Sequence[SpineItem]
) -> dict[str, tuple[NavPoint, ...]]:
    """Declared entries grouped by the spine document they point into.

    An entry naming a document the spine does not carry is dropped here — the one
    thing that kept navigation out of reading order, reduced to what it actually
    is once the question is structure: an entry that names nothing readable.
    """
    order = {item.href for item in spine}
    grouped: dict[str, list[NavPoint]] = {}
    for point in points:
        if point.href in order:
            grouped.setdefault(point.href, []).append(point)
    return {href: tuple(found) for href, found in grouped.items()}


def is_furniture(document: Element) -> bool:
    """Whether a spine document is a picture of content rather than content."""
    has_image = any(child.tag in {"img", "image"} for child in document.descendants())
    return has_image and len(document.text().strip()) < FURNITURE_CHARACTERS


class EpubExtractor:
    route = "epub"

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult:
        with open_container(path) as archive:
            spine = read_spine(archive)
            log(f"epub: {len(spine)} spine document(s)")
            blocks, fired, skipped = self._read(archive, spine, {})
            if not any(block.kind == "heading" for block in blocks):
                # Only now, and only because there is nothing to overrule. The
                # second pass costs a re-parse of the spine, which a book with no
                # `<hN>` in it is rare enough to be worth.
                declared = _by_member(read_navigation(archive), spine)
                if declared:
                    log(
                        f"epub: no <h1>-<h6> in any spine document; taking the heading set "
                        f"from the {sum(len(points) for points in declared.values())} "
                        f"entries the container declares"
                    )
                    blocks, fired, skipped = self._read(archive, spine, declared)

        if skipped:
            log(f"epub: skipped {skipped} image-only spine document(s)")
        markdown = to_gfm(blocks)
        declines = (*self._code_decline(blocks, fired), *self._heading_decline(blocks))
        log(
            "epub: "
            + (", ".join(f"{name}={count}" for name, count in sorted(fired.items())) or "no fences")
        )
        return ParseResult(
            markdown=markdown,
            report=ExtractorReport(
                route=self.route,
                rule_counts=fired,
                declines=declines,
                skipped_documents=skipped,
            ),
        )

    @staticmethod
    def _read(
        archive: zipfile.ZipFile,
        spine: Sequence[SpineItem],
        declared: Mapping[str, Sequence[NavPoint]],
    ) -> tuple[list[Block], dict[str, int], int]:
        """Every spine document, in order, as blocks."""
        blocks: list[Block] = []
        fired: dict[str, int] = {}
        skipped = 0
        for item in spine:
            document = read_member(archive, item.href)
            if is_furniture(document):
                skipped += 1
                continue
            found, counts = _blocks_of(document, CODE_RULES, declared.get(item.href, ()))
            blocks.extend(found)
            for name, count in counts.items():
                fired[name] = fired.get(name, 0) + count
        return blocks, fired, skipped

    @staticmethod
    def _code_decline(blocks: Sequence[Block], fired: Mapping[str, int]) -> tuple[str, ...]:
        """Say so when no rule matched anywhere in the book.

        The count of code-shaped prose lines is what makes this worth reading. On
        its own "no fences" is ambiguous between a book with no code in it and a
        book whose code is carried by markup nothing here can see; the count
        separates the two without any per-book knowledge.
        """
        if fired:
            return ()
        code_shaped = sum(
            1 for block in blocks for line in block.text.split("\n") if looks_like_code(line)
        )
        return (
            f"no code rule matched any spine document; {code_shaped} prose line(s) are "
            f"code-shaped, so any fence here would be a guess",
        )

    @staticmethod
    def _heading_decline(blocks: Sequence[Block]) -> tuple[str, ...]:
        """Say so when neither the book nor its container declares any heading.

        Structure and code rest on different evidence, so they decline
        independently — a book can have perfectly good `<hN>` and no visible code,
        or the reverse. This is the reverse: one pinned book is a Calibre build
        with no semantic heading anywhere, only generated `p` classes that happen
        to be typeset large. What is left once its navigation has also been asked
        is a book whose structure could only be *guessed* — which of a build
        tool's numbered classes is a chapter, per book — and a wrong guess invents
        a chapter tree that reads as authoritative.
        """
        if any(block.kind == "heading" for block in blocks):
            return ()
        return (
            "no <h1>-<h6> element in any spine document and no navigation document "
            "declaring a chapter tree, so no structure is claimed; this build carries "
            "its headings in generated class names only",
        )
