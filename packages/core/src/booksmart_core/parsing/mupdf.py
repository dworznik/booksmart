"""Talking to MuPDF without being narrated at.

One function, in its own module, because it has three callers inside this package
and one outside it — and putting it in either the parse contract or the metrics
would make the dependency between those two run both ways.
"""

import pymupdf


def quiet_mupdf() -> None:
    """Silence MuPDF's narration of recoverable defects.

    It reports a missing EPUB stylesheet once per chapter, and one Calibre-produced
    corpus file emits megabytes of `unicode-range` complaints while opening
    perfectly well. Those go to MuPDF's own message stream, which shell
    redirection does not reach, so a full-corpus sweep drowns in them.

    Only the *display* is suppressed. Anything genuinely fatal still raises.
    """
    pymupdf.TOOLS.mupdf_display_errors(False)  # type: ignore[no-untyped-call]
    pymupdf.TOOLS.mupdf_display_warnings(False)  # type: ignore[no-untyped-call]
