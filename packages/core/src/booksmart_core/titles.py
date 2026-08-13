"""Comparing a title a book set with a title something else states.

Two sides that agree on the words and disagree on everything else: a container's
navigation says "Chapter 4: A Title", the page sets the number and the title as
two separate lines with a rule between them, and a hand-authored table of
contents says a third thing again. Every comparison between them runs through
here, so that two of them cannot drift apart into disagreeing about what "the
same title" means.

Text only, and no dependency: this is asked by both extractors, and the EPUB
route must stay free of MuPDF.
"""

import re

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


def normalise(text: str) -> str:
    """Lower-cased, punctuation collapsed — the form both sides are compared in."""
    return _ALPHANUM.sub(" ", text.translate(_INVISIBLE).lower()).strip()


def titles_match(heading: str, entry: str) -> bool:
    """Whether a heading on the page and an entry in a container's navigation are
    the same title.

    **Either one may begin the other, and equality is not required.** Navigation
    spells an entry as "Chapter 4: A Title" while books routinely typeset the
    number and the title as two separate headings, and the truncation runs the
    other way just as often — an outline entry shortened to fit a bookmark pane
    against the full title on the page. Requiring equality loses both, and each
    of them loses a whole chapter.

    It has to be the *beginning*, on a whole word. Containment anywhere was
    measured against real books and it is far too loose: an entry reading
    "1.3 Formulating Abstractions with Higher-Order Procedures" contains the word
    "procedures", so a body line saying only that matched it — and one spurious
    match is enough to teach a caller the wrong thing about the whole document.
    Two spellings of one title agree on how they start; what they disagree about
    is a number or a subtitle at the end.
    """
    left, right = normalise(heading), normalise(entry)
    if not left or not right:
        return False
    return left == right or left.startswith(f"{right} ") or right.startswith(f"{left} ")
