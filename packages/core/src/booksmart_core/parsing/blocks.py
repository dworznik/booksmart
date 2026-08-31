"""The block IR the extractors build, and the GFM they serialise it to.

Internal to this package by design. ADR 0003: `Block` never crosses the parsing
module boundary — the parse Stage's contract is a GFM string, so an extractor can
change how it models a document without any downstream stage noticing.

Two rules carry most of the weight here.

**Code is emitted verbatim; prose is escaped.** A fence asserts that its body is
code exactly as the book set it, so nothing normalises it. Prose gets the opposite
treatment: a paragraph that happens to begin `# comment` or `1. First` would
otherwise become a heading or a list, and a line that is three backticks would
open a fence over the rest of the book. Escaping is correct GFM serialisation
rather than a heuristic, and it is what makes a *declining* extractor safe —
unfenced code comes out as text that says what it says and structures nothing.

**A fence is longer than the longest backtick run inside it.** Books that ship
Markdown or shell examples contain backticks, and a three-backtick fence around a
body containing three backticks closes early, spilling code into prose.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

Kind = Literal["heading", "paragraph", "code", "list", "table"]

# How many heading levels there are, which is a fact about GFM rather than a
# choice: `#######` is not a heading, it is a paragraph beginning with hashes.
# Named because both extractors have to clamp to it, and a level they clamp
# differently from what is serialised here would be a level that vanishes.
MAX_HEADING_LEVEL = 6

# Line-leading characters GFM reads as structure. `*` is deliberately absent:
# emphasis is emitted faithfully (an EPUB `<h2>` may contain `<em>`, and
# structure.py unwraps it), so escaping it here would break what it reads.
_LEADING = re.compile(r"^([#>+\-|]|=+$)")
_ORDERED = re.compile(r"^(\d+)([.)])")
_FENCE = re.compile(r"^(```+|~~~+)")
_BACKTICK_RUN = re.compile(r"`+")


@dataclass(frozen=True)
class Block:
    """One top-level block of a document.

    ``rule`` is which code rule produced a code block, and it is the reason the
    report can say a rule stopped firing — a thing no reader of the finished GFM
    could ever recover. It is deliberately not a confidence: a rule either
    matched or it did not.
    """

    kind: Kind
    text: str
    level: int = 0  # headings only, 1-6
    language: str = ""  # code only; the fence info string
    rule: str = ""  # code only; which rule matched


def escape_prose(text: str) -> str:
    """Make prose that cannot be read as structure.

    Per line, because that is the only place GFM's block markers bind. An ordered
    list marker is escaped after the digits (`1\\.`, the form GFM defines) rather
    than before them, which would not escape anything at all. `===` and `---`
    under a paragraph are setext headings, which is why a line of them is escaped
    too and not only the ATX form.

    Leading whitespace goes, rather than being preserved and escaped. Four spaces
    open an indented code block in GFM, and a paragraph's indentation is
    typography — the one place indentation carries meaning is inside a fence, and
    fences do not come through here.
    """
    escaped: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _FENCE.match(stripped) or _LEADING.match(stripped):
            stripped = "\\" + stripped
        else:
            ordered = _ORDERED.match(stripped)
            if ordered:
                stripped = f"{ordered.group(1)}\\{stripped[ordered.end(1):]}"
        escaped.append(stripped)
    return "\n".join(escaped)


def fence_for(body: str) -> str:
    """A fence long enough to survive the body's own backticks."""
    longest = max((len(run.group()) for run in _BACKTICK_RUN.finditer(body)), default=0)
    return "`" * max(3, longest + 1)


def to_gfm(blocks: Iterable[Block]) -> str:
    """One blank line between blocks, and nothing else invented."""
    rendered = [_render(block) for block in blocks]
    return "\n\n".join(part for part in rendered if part.strip()) + "\n"


def _render(block: Block) -> str:
    if block.kind == "heading":
        # Collapsed to one line: an ATX heading is a line, and a newline inside
        # one silently ends the heading and starts a paragraph.
        title = " ".join(block.text.split())
        return f"{'#' * max(1, min(block.level, MAX_HEADING_LEVEL))} {title}" if title else ""
    if block.kind == "code":
        fence = fence_for(block.text)
        body = block.text.rstrip("\n")
        return f"{fence}{block.language}\n{body}\n{fence}"
    if block.kind == "list":
        return "\n".join(f"- {escape_prose(item)}" for item in block.text.split("\n") if item.strip())
    if block.kind == "table":
        return block.text
    return escape_prose(block.text)


# A line of text that has the shape of source code. Each pattern is a shape prose
# does not have; together they are the countable half of "fencing failed here",
# per ADR 0003.
#
# Two callers, one definition, and the second is the reason this is a module-level
# name rather than a local. `suspect_headings` counts ATX titles that match, which
# is fencing having failed *loudly*. An extractor that matched no code rule at all
# counts prose lines that match, which is how its decline can say "this book has
# code I could not see" rather than the far weaker "I found no code".
_CODE_SHAPED = (
    # A call or a signature. No space before the paren, which is what separates
    # `extract(path)` from "A Note on Naming (And Why It Is Hard)" — prose puts
    # a space there and code does not.
    re.compile(r"[A-Za-z_]\w*\([^)]*\)"),
    # An operator no prose title carries.
    re.compile(r"(==|!=|<=|>=|=>|->|::|&&|\|\||\+=|-=|\*=|/=)"),
    # A statement opening. Case-sensitive, and that is the whole of what makes it
    # usable: every one of these words is also an English word, and a title-cased
    # one opens an ordinary heading. Measured against the corpus, matching
    # case-insensitively called "From the Preface to the First Edition", "Class
    # Properties", "Static Configuration" and "Function singledispatch" code —
    # five false positives across three books, against zero true ones, because
    # code writes its keywords in lower case in every language this corpus holds.
    re.compile(
        r"^(def|class|function|func|import|from|return|public|private|protected|static"
        r"|var|let|const|struct|enum|module|package|namespace|template|typedef|end)\b\s+\S"
    ),
    # A trailing block or statement terminator.
    re.compile(r"[;{}]$"),
)


def looks_like_code(line: str) -> bool:
    """Whether one line of text has the shape of source code rather than prose."""
    return any(pattern.search(line.strip()) for pattern in _CODE_SHAPED)
