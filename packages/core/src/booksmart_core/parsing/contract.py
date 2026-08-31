"""What every route promises, and the two ways it can fail.

Separate from the router so the extractors and the router can both depend on it
without depending on each other: routing knows the set of routes, an extractor
knows only its own.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

# Version of the text-extraction pipeline (routing + extractors + structure
# handling), stamped on every ingestion job as extraction_version. Bump when
# parsing behaviour changes so reprocessed runs are distinguishable in history.
#
# "1" was the marker -> pymupdf4llm -> OCR preference chain, whose `parser_used`
# vocabulary was `marker` | `pymupdf` | `ocr`. "2" is the router, whose vocabulary
# is the route names below. Rows written under "1" are left alone: `marker` and
# `pymupdf` are true statements about what produced those artifacts, and the
# version stamp is what says which vocabulary to read a row in.
#
# "3" keeps "2"'s route vocabulary and changes what a heading is. Both routes now
# read the chapter tree the container declares — a PDF's bookmark outline, an
# EPUB's NCX or nav document — where before both inferred one and neither asked;
# and where the PDF route still infers, a size has to head enough of the document
# to spend one of the six levels. Heading counts move on most documents, so a
# row's structure is only comparable with another row's at the same stamp.
EXTRACTION_VERSION = "3"

Route = Literal["epub", "pdf", "pdf-ocr"]


class ExtractorUnavailable(Exception):
    """The route's backing tool is not installed on this machine.

    Kept apart from `ParseFailure` because the remedy is different — install
    something, rather than look at the book — even though under the router both
    are terminal.
    """


class ParseFailure(Exception):
    """The document's route could not produce usable output."""


@dataclass(frozen=True)
class ExtractorReport:
    """An extractor's testimony about its own run.

    Everything here is invisible in the artifact. Which rule fenced a block,
    whether a signal was too weak to use, how many spine documents were dropped
    as furniture — none of it survives serialisation to GFM, and all of it is
    what a regression in the extractors looks like first.
    """

    route: str
    # Rule name -> blocks it produced ("pre", "table.processedcode",
    # "pdf:mono-family"). Ratcheted: a rule that stops firing is a regression
    # even when the character count holds up.
    rule_counts: Mapping[str, int] = field(default_factory=dict)
    # One line per signal the extractor refused to guess at, reason included.
    declines: tuple[str, ...] = ()
    # Spine documents skipped as furniture (image-only listing screenshots).
    skipped_documents: int = 0


@dataclass(frozen=True)
class ParseResult:
    """One document's artifact, plus what the extractor has to say about it.

    Markdown only. Whatever intermediate representation an extractor built to get
    here does not travel with the result — that is the parse contract ADR 0003
    records, and it is what lets a route be replaced without touching a stage.
    """

    markdown: str
    report: ExtractorReport

    @property
    def route(self) -> str:
        """What lands in `Book.parser_used`. The column keeps its old name and
        schema (there is no migration); under `EXTRACTION_VERSION` "2" the value
        it holds is a route name."""
        return self.report.route


class Extractor(Protocol):
    """One format, read one way.

    Takes a log callback rather than returning log lines, because the interesting
    ones arrive while a long book is being read and a caller watching a sweep
    wants them then. Everything a *later* reader needs is in the report.
    """

    route: str

    def extract(self, path: Path, log: Callable[[str], None]) -> ParseResult: ...
