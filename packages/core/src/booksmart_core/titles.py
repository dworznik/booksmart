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

# `\W` under Unicode, plus the underscore it counts as a word character. A letter
# is a letter in every script: keyed on ASCII, an accented title lost the accented
# letters and a title in a non-Latin script normalised to nothing at all, which
# matches nothing and reads exactly like a book with no titles in it.
_ALPHANUM = re.compile(r"[\W_]+")

# Characters that are in the text without being in the word. Some PDFs carry a
# soft hyphen at every legal break point, so collapsing one to a space splits the
# word around it and a heading differing from the authored one by an invisible
# character silently stops matching. An EPUB may separate code tokens with
# zero-width spaces for the same invisible effect. Deleted rather than replaced,
# which is the opposite of what happens to real punctuation.
#
# Written as escapes, not as the characters themselves: a reader cannot see a
# soft hyphen in a diff, and any tool that strips them would change what this
# line means without changing how it looks.
INVISIBLE = str.maketrans(
    dict.fromkeys("\u00ad\u200b\u200c\u200d\ufeff")  # soft hyphen, ZWSP, ZWNJ, ZWJ, BOM
)


def normalise(text: str) -> str:
    """Lower-cased, punctuation collapsed — the form both sides are compared in."""
    return _ALPHANUM.sub(" ", text.translate(INVISIBLE).casefold()).strip()


def _begins(text: str, opening: str) -> bool:
    """Whether normalised ``text`` opens with ``opening``, on a whole word.

    On a whole word, so that "chapter 4" does not begin "chapter 40".
    """
    return text == opening or text.startswith(f"{opening} ")


def titles_match(heading: str, entry: str) -> bool:
    """Whether a heading on the page and an entry in a container's navigation are
    the same title.

    **Either one may begin the other, and equality is not required.** Navigation
    spells an entry as "Chapter 4: A Title" while books routinely typeset the
    number and the title as two separate headings, and the truncation runs the
    other way just as often — an outline entry shortened to fit a bookmark pane
    against the full title on the page. Requiring equality loses both, and each
    of them loses a whole chapter.

    It has to be the *beginning*. Containment anywhere was measured against a
    real book and it is far too loose: a section entry ending in a common word
    was matched by a body line saying only that word, and that one spurious match
    taught the caller the wrong thing about the whole document — it cost six of
    the document's seven headings. Two spellings of one title agree on how they
    start; what they disagree about is a number or a subtitle at the end.

    What this does not reach is a number on one side only — an entry reading
    "Introduction" against a page setting "1 Introduction". That case is visible
    rather than silent: it shows up as an entry the outline could not locate, and
    the extractor reports how many of those there were.
    """
    left, right = normalise(heading), normalise(entry)
    if not left or not right:
        return False
    return _begins(left, right) or _begins(right, left)


def title_remainder(heading: str, entry: str) -> str:
    """What is left of ``entry`` once ``heading`` has said the front of it.

    Empty where there is nothing left to look for — including where the heading
    said *more* than the entry does, which `titles_match` also accepts. A title
    too long for its measure wraps, and this is what lets a caller pick the rest
    of it up off the next line instead of losing half the chapter's name.
    """
    left, right = normalise(heading), normalise(entry)
    if not left or not right or not _begins(right, left):
        return ""
    return right[len(left) :].strip()
