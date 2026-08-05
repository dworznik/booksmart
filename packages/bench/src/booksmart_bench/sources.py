"""Cataloguing the book files a handover drops into an assets checkout.

Truth is authored against books this repo never sees, and the only thing tying
one to the other is the sha256 in `book.yaml`. Somebody has to put the files in
`sources/` and write those hashes down, and doing it by hand is a bad trade: the
work is tedious, and every way of getting it wrong is silent.

- The wrong *edition* is the expensive mistake. Chapter numbering moves between
  editions, so truth authored for one scores the other as a pipeline that lost
  half the book. Several titles in the set have a later edition that a search
  engine will offer first.
- A scan without a text layer ingests fine and benchmarks the OCR parser
  instead of the one every other book used.
- A file swapped later, after truth was authored, is invisible unless something
  compares the bytes to the pin.

So identification here is by *content*: a file is scored against the chapter
titles already authored in each book's `toc.yaml`. A filename is whatever the
last person typed and metadata is part of what is being checked, but a file
containing thirty of a book's thirty-six chapter titles is that book. A match
has to be both the best and clearly better than the runner-up — two editions of
one title look nearly identical, and naming one of them would pin truth to the
wrong artifact with no way to notice.

Read-only unless asked to pin. Nothing here ingests, so nothing here spends.
"""

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from booksmart_bench.truth import BookTruth, Truth

# What counts as a book file. Everything else in sources/ is somebody's notes.
BOOK_SUFFIXES = frozenset({".pdf", ".epub"})

# Below this many characters per sampled page the text layer is not carrying the
# book, and booksmart's parser chain would fall through to OCR.
MIN_CHARS_PER_PAGE = 200

# A file has to match this much of a book's authored chapter titles before it is
# called that book at all. Set low on purpose: a real ToC often renders with
# soft hyphens, ligatures or line breaks inside a title, so a correct file
# routinely misses a third of them.
MIN_SCORE = 0.4

# ...and it has to beat the runner-up by this much. Two editions of one book
# share most of their chapter titles, so a near-tie is the signal that matters.
MARGIN = 0.15

# Where a book states its own edition, ISBN and year. Claims are read from here
# and nowhere else: a bibliography citing another edition of the same title, or
# an author's preface recalling their first, is not the book identifying itself.
FRONT_PAGES = 14

_ALPHANUM = re.compile(r"[^a-z0-9]+")

# Characters that are in the text without being in the word. Some PDFs carry a
# soft hyphen at every legal break point, so collapsing one to a space splits the
# word around it and a heading differing from the authored one by an invisible
# character silently stops matching. Deleted rather than replaced, which is the
# opposite of what happens to real punctuation.
#
# Written as escapes, not as the characters themselves: a reader cannot see a
# soft hyphen in a diff, and any tool that strips them would change what this
# line means without changing how it looks.
_INVISIBLE = str.maketrans(
    dict.fromkeys("\u00ad\u200b\u200c\u200d\ufeff")  # soft hyphen, ZWSP, ZWNJ, ZWJ, BOM
)

# How a book states which edition it is, on a title page or in a copyright block.
# Digits are matched numerically rather than listed, so nothing here caps how
# many editions a book is allowed to have had — a table stopping at "sixth" made
# a seventh-edition book skip the edition check altogether, silently, which is
# the one outcome this whole module exists to prevent.
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
}
_EDITION_CLAIM = re.compile(
    r"\b(?:(" + "|".join(_ORDINAL_WORDS) + r")|(\d{1,2})(?:st|nd|rd|th))\s+edition\b"
    r"|\bedition\s+(\d{1,2})\b"
)
_ORDINAL_DIGITS = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?$")
_COPYRIGHT_YEAR = re.compile(r"\bcopyright\b[^a-z0-9]{0,12}((?:19|20)\d{2})")

# An edition claim counts only this close to the book's own title. Front matter
# is full of other books' editions — a publisher's list of related titles, an
# advertisement for the same author's other work — and counting those made a
# first-edition file look like it was claiming to be a third.
EDITION_WINDOW = 120

# A copyright year is not a publication year. Reprints and the gap between
# going to press and shipping put these one apart routinely; several editions
# apart is a different book.
YEAR_TOLERANCE = 1


def normalise(text: str) -> str:
    """Lower-cased, punctuation collapsed — the form both sides are compared in."""
    return _ALPHANUM.sub(" ", text.translate(_INVISIBLE).lower()).strip()


@dataclass(frozen=True)
class Artifact:
    """The evidence one file offers about itself."""

    path: Path
    sha256: str
    pages: int
    chars_per_page: int
    # Outline plus every page, normalised. One haystack rather than several,
    # because every question asked of it is "does this appear".
    text: str
    # The front matter alone, where a book states which edition it is. Kept apart
    # from `text` so an edition claim cannot be picked up from a bibliography.
    front: str = ""
    # Set when the file could not be read at all, and then nothing else is
    # meaningful. Reported rather than raised: one bad file must not stop the
    # other thirteen being catalogued.
    unreadable: str | None = None


@dataclass(frozen=True)
class Candidate:
    slug: str
    score: float


@dataclass(frozen=True)
class Match:
    slug: str | None
    score: float
    runner_up: Candidate | None


@dataclass(frozen=True)
class Catalogued:
    """One file, identified or not, with everything worth saying about it."""

    artifact: Artifact
    match: Match
    notes: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    rename_to: str | None = None

    @property
    def slug(self) -> str | None:
        return self.match.slug

    @property
    def ok(self) -> bool:
        return self.slug is not None and not self.problems


@dataclass(frozen=True)
class Catalogue:
    entries: tuple[Catalogued, ...] = ()
    # slug -> the single file that is unambiguously it and has no problems.
    identified: Mapping[str, Catalogued] = field(default_factory=dict)
    # slug -> the several files that all claim to be it.
    contested: Mapping[str, tuple[Catalogued, ...]] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


def read_artifact(path: Path) -> Artifact:
    """Open one file and take everything from it that identification needs.

    The hash is computed even for a file that will not open, so a report can
    still name the bytes somebody has to go and replace.
    """
    digest = sha256_of(path)
    try:
        doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 - any failure to open means unusable
        return Artifact(
            path=path,
            sha256=digest,
            pages=0,
            chars_per_page=0,
            text="",
            unreadable=f"{type(exc).__name__}: {exc}",
        )

    with doc:
        if doc.needs_pass:
            return Artifact(
                path=path,
                sha256=digest,
                pages=doc.page_count,
                chars_per_page=0,
                text="",
                unreadable="password-protected",
            )
        pages = doc.page_count
        if pages == 0:
            return Artifact(
                path=path, sha256=digest, pages=0, chars_per_page=0, text="", unreadable="no pages"
            )
        # Every page, not a sample. A sample of two dozen pages reads under a
        # tenth of a long book, and a book whose PDF carries no outline then has
        # most of its headings on pages nobody opened — which showed up as a
        # correct file that could not be told apart from the wrong one. Eleven
        # seconds on the longest book in the set is a fair price for that, and
        # this verb runs once per handover.
        #
        # Page text goes through _text() because pymupdf ships py.typed while
        # leaving its own methods unannotated, and one suppression reads better
        # than one per call site. Same reason core's parsing.py has one.
        body = [_text(doc, index) for index in range(pages)]
        outline = " ".join(entry[1] for entry in doc.get_toc())

    return Artifact(
        path=path,
        sha256=digest,
        pages=pages,
        chars_per_page=int(sum(len(text) for text in body) / pages),
        text=normalise(" ".join([outline, *body])),
        front=normalise(" ".join(body[:FRONT_PAGES])),
    )


def _text(doc: pymupdf.Document, index: int) -> str:
    return str(doc[index].get_text())  # type: ignore[no-untyped-call]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def identify(artifact: Artifact, books: Mapping[str, BookTruth]) -> Match:
    """Which book this file is, or nothing if the evidence does not say.

    Scored on how much of a book's authored structure appears in the file's text,
    weighted by how much each heading is worth as evidence. A heading of one
    common word is worth almost nothing: a book with chapters called State,
    Classes and Patterns otherwise matches every programming book ever written,
    and scoring those hits at face value made it the runner-up against the whole
    corpus — eating the margin that decides the genuinely ambiguous files.

    Sections count as well as chapters. Some books here have eight chapters and
    some have thirty-six, and a fraction over eight is too coarse to separate two
    candidates.
    """
    scored = sorted(
        (
            Candidate(slug=slug, score=_score(artifact, book))
            for slug, book in books.items()
            if _weighted_titles(book)
        ),
        key=lambda candidate: (-candidate.score, candidate.slug),
    )
    if not scored:
        return Match(slug=None, score=0.0, runner_up=None)

    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    decided = best.score >= MIN_SCORE and (
        runner_up is None or best.score - runner_up.score >= MARGIN
    )
    return Match(slug=best.slug if decided else None, score=best.score, runner_up=runner_up)


def _weighted_titles(book: BookTruth) -> list[tuple[str, int]]:
    """Each chapter and section title, with what a match on it is worth.

    The weight is the number of words of four letters or more — the same rough
    stand-in for a content word that the truth lint's kind guards use. So
    "Collecting Temporary Variable" is worth three and "State" is worth none,
    which is right: the first appears in one book and the second appears in all
    of them. A book whose every heading is a single common word ends up with a
    total weight near zero and simply cannot be identified this way, which is
    honest — its headings genuinely do not distinguish it.

    Front and back matter are left out. Preface, Index and Bibliography are
    everywhere and would only add noise on both sides of the fraction.
    """
    weighted: list[tuple[str, int]] = []
    for node in book.nodes.values():
        if node.kind not in {"chapter", "section"} or not node.title:
            continue
        normalised = normalise(node.title)
        weight = sum(1 for word in normalised.split() if len(word) >= 4)
        if weight:
            weighted.append((normalised, weight))
    return weighted


def _score(artifact: Artifact, book: BookTruth) -> float:
    titles = _weighted_titles(book)
    if not titles or not artifact.text:
        return 0.0
    total = sum(weight for _, weight in titles)
    hits = sum(weight for title, weight in titles if title in artifact.text)
    return hits / total


def audit(artifact: Artifact, book: BookTruth | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Everything wrong or worth saying about one file, given who it claims to be."""
    notes: list[str] = []
    problems: list[str] = []

    if artifact.unreadable:
        return (), (f"cannot be read ({artifact.unreadable})",)

    if artifact.chars_per_page < MIN_CHARS_PER_PAGE:
        problems.append(
            f"~{artifact.chars_per_page} characters per page of extractable text, under "
            f"{MIN_CHARS_PER_PAGE} — this reads as a scan with no text layer, and the "
            "pipeline would fall through to OCR, benchmarking a different parser from "
            "every other book in the corpus"
        )
    if artifact.path.suffix.lower() == ".epub":
        notes.append(
            "EPUB, so its parser chain is PyMuPDF alone with no marker pass and no OCR "
            "fallback; a corpus mixing formats measures parser choice alongside pipeline "
            "changes, so prefer a PDF if one exists"
        )

    if book is None:
        return tuple(notes), tuple(problems)

    if book.is_pinned:
        if book.source_sha256 == artifact.sha256:
            notes.append("already pinned to these bytes")
        else:
            problems.append(
                f"truth is already pinned to {(book.source_sha256 or '')[:16]}… and this file "
                f"is {artifact.sha256[:16]}…; truth was authored against other bytes, so it "
                "may no longer describe this file — re-check it before repinning"
            )

    edition_notes, edition_problems = _edition_claims(artifact, book)
    notes.extend(edition_notes)
    problems.extend(edition_problems)
    return tuple(notes), tuple(problems)


def parse_ordinal(text: str) -> int | None:
    """"3rd", "third", "3" -> 3. None when it is not an ordinal at all.

    Returning None matters: the caller reports the unread value rather than
    treating it as "no edition stated", so a `book.yaml` this cannot parse is
    visible instead of quietly disabling the check.
    """
    token = normalise(text).split(" ")[0] if text else ""
    if not token:
        return None
    if token in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[token]
    digits = _ORDINAL_DIGITS.match(token)
    return int(digits.group(1)) if digits else None


def _stated_editions(artifact: Artifact, book: BookTruth) -> set[int]:
    """Edition ordinals the file states *about itself*.

    Anchored on the book's own title, because front matter is full of other
    books' editions — a publisher's list of related titles, the same author's
    other work — and counting those made a first-edition file look as though it
    were claiming to be a third. Two words of title is enough of an anchor and
    survives the subtitle changing between editions, which it does.
    """
    words = normalise(book.title).split()
    if len(words) < 2 or not artifact.front:
        return set()
    anchor = " ".join(words[:2])

    stated: set[int] = set()
    for found in re.finditer(re.escape(anchor), artifact.front):
        window = artifact.front[
            max(0, found.start() - EDITION_WINDOW) : found.end() + EDITION_WINDOW
        ]
        for word, digit, after in _EDITION_CLAIM.findall(window):
            stated.add(_ORDINAL_WORDS[word] if word else int(digit or after))
    return stated


def _ordinal(number: int) -> str:
    """1 -> 1st. Said back in the form book.yaml writes it, so the two are
    comparable at a glance."""
    if 10 <= number % 100 <= 20:
        return f"{number}th"
    return f"{number}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th') }"


def _edition_claims(artifact: Artifact, book: BookTruth) -> tuple[list[str], list[str]]:
    """What the file says about which edition it is, against what truth expects.

    This is the check the verb exists for. Content matching cannot separate two
    editions of one book — a later edition keeps most of its headings, so it
    scores highly against truth authored for the earlier one and would be pinned
    without a murmur. What does separate them is that a book states its own
    edition and its own copyright year on the way in.

    A *contradiction* is a problem; silence is a note. A second edition routinely
    discusses its first in the same front matter, so finding an ordinal is not
    finding a disagreement — the expected one being absent while others are
    present is.

    The ISBN stays a note however it comes out. A single edition legitimately
    carries several (print, ebook, a publisher's own catalogue id), so a
    mismatch there is worth a human's glance and not worth blocking on.
    """
    notes: list[str] = []
    problems: list[str] = []

    stated_editions = _stated_editions(artifact, book)
    expected_edition = parse_ordinal(book.edition)
    if expected_edition and stated_editions:
        if expected_edition in stated_editions:
            notes.append(f"edition {book.edition} confirmed in the file")
        else:
            listed = ", ".join(_ordinal(number) for number in sorted(stated_editions))
            problems.append(
                f"the file states edition {listed} and truth is authored for "
                f"{book.edition} — chapter numbering moves between editions, so this would "
                "score as a pipeline that lost part of the book"
            )
    elif expected_edition:
        notes.append(f"edition {book.edition} not stated in the file — confirm by hand")
    elif book.edition:
        # Said out loud rather than skipped. An edition this cannot read means the
        # check silently does nothing, which is the failure it exists to catch.
        notes.append(
            f"edition {book.edition!r} could not be read as an ordinal, so the edition "
            "check did not run — confirm by hand"
        )

    stated_years = {int(year) for year in _COPYRIGHT_YEAR.findall(artifact.front)}
    expected_year = int(book.year) if book.year.isdigit() else None
    if expected_year and stated_years:
        nearest = min(stated_years, key=lambda year: abs(year - expected_year))
        if abs(nearest - expected_year) <= YEAR_TOLERANCE:
            notes.append(f"copyright {nearest} against an expected {expected_year}")
        else:
            problems.append(
                f"the file's copyright is {', '.join(str(y) for y in sorted(stated_years))} and "
                f"truth expects {expected_year} — that is more than a reprint apart, so this is "
                "probably a different edition"
            )
    elif expected_year:
        notes.append(f"year {expected_year} not stated in the file — confirm by hand")

    if book.isbn:
        if book.isbn in re.sub(r"[^0-9]", "", artifact.front):
            notes.append(f"ISBN {book.isbn} confirmed in the file")
        else:
            notes.append(
                f"ISBN {book.isbn} not found — one edition can carry several, so check this "
                "against the file rather than assuming either is wrong"
            )
    return notes, problems


def catalogue(
    assets: Path,
    truth: Truth,
    *,
    read: Callable[[Path], Artifact] = read_artifact,
) -> Catalogue:
    """Identify and audit every book file under ``<assets>/sources``."""
    sources = assets / "sources"
    if not sources.is_dir():
        return Catalogue(missing=tuple(sorted(truth.books)))

    files = sorted(p for p in sources.iterdir() if p.is_file())
    books = sorted(p for p in files if p.suffix.lower() in BOOK_SUFFIXES)
    ignored = tuple(p.name for p in files if p.suffix.lower() not in BOOK_SUFFIXES)

    entries: list[Catalogued] = []
    for path in books:
        artifact = read(path)
        match = identify(artifact, truth.books)
        book = truth.books.get(match.slug) if match.slug else None
        notes, problems = audit(artifact, book)
        if match.slug is None and not problems:
            problems = (
                *problems,
                _undecided(match),
            )
        rename = None
        if book is not None and book.source_file:
            expected = Path(book.source_file).name
            if path.name != expected:
                rename = expected
        entries.append(
            Catalogued(
                artifact=artifact,
                match=match,
                notes=notes,
                problems=problems,
                rename_to=rename,
            )
        )

    claims: dict[str, list[Catalogued]] = {}
    for entry in entries:
        if entry.ok and entry.slug is not None:
            claims.setdefault(entry.slug, []).append(entry)

    identified = {slug: group[0] for slug, group in claims.items() if len(group) == 1}
    contested = {slug: tuple(group) for slug, group in claims.items() if len(group) > 1}
    missing = tuple(slug for slug in sorted(truth.books) if slug not in identified)

    return Catalogue(
        entries=tuple(entries),
        identified=identified,
        contested=contested,
        missing=missing,
        ignored=ignored,
    )


def _undecided(match: Match) -> str:
    if match.runner_up is None or match.score == 0:
        return "matches no book's authored chapter titles"
    return (
        f"cannot be told apart from truth — best guess scores {match.score:.0%} against "
        f"{match.runner_up.score:.0%} for {match.runner_up.slug}, too close to call; "
        "identify it by hand and rename it to the file book.yaml expects"
    )


def pin(assets: Path, catalogued: Catalogue) -> tuple[str, ...]:
    """Rename each identified file to what truth expects, and write its hash.

    Only the unambiguous ones. A contested slug is left alone, because pinning
    one of two candidates records a coin toss as a fact.
    """
    actions: list[str] = []
    for slug, entry in sorted(catalogued.identified.items()):
        path = entry.artifact.path
        if entry.rename_to:
            target = path.with_name(entry.rename_to)
            path.rename(target)
            actions.append(f"{slug}: renamed {path.name} -> {target.name}")
            path = target
        book_yaml = assets / "truth" / slug / "book.yaml"
        original = book_yaml.read_text()
        updated = _repin(original, entry.artifact.sha256)
        if updated != original:
            book_yaml.write_text(updated)
            actions.append(f"{slug}: pinned {entry.artifact.sha256[:16]}…")
    return tuple(actions)


def _repin(document: str, sha256: str) -> str:
    lines = document.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if re.match(r"\s*sha256:", line):
            ending = "\n" if line.endswith("\n") else ""
            lines[index] = repin_line(line.rstrip("\n"), sha256) + ending
            break
    return "".join(lines)


# A comment on the pin that only makes sense while it is unpinned. Anything else
# an author wrote there is theirs and survives.
_STALE_COMMENT = re.compile(r"(?i)\b(tbd|handover|when the file lands|blocked)\b")


def repin_line(line: str, sha256: str) -> str:
    """Rewrite one ``sha256:`` line, keeping its indentation and any comment that
    still means something once the hash is real."""
    indent = line[: len(line) - len(line.lstrip())]
    _, _, remainder = line.partition("sha256:")
    _value, _, comment = remainder.partition("#")
    comment = comment.strip()
    if comment and not _STALE_COMMENT.search(comment):
        return f"{indent}sha256: {sha256} # {comment}"
    return f"{indent}sha256: {sha256}"
