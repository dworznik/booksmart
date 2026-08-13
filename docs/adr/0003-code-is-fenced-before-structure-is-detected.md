# Code is fenced before structure is detected; an unreliable signal declines

Extraction fences code first. Every structural reader downstream — heading
detection, ToC scoring, `iter_chapter_bodies`, LLM summarisation — only ever
sees code inside fences, and never has to ask whether a line that looks like
structure came from a listing. The extractors are free to carry a block IR
internally, but `Block` never crosses the parsing module boundary: the parse
Stage's contract stays a GFM string, so nothing downstream gains a dependency
on how a format was read. And where the signal that separates code from prose
is not reliable for a given document, the extractor declines and says so rather
than guessing — a book with no fences and a recorded reason is a book somebody
can act on.

This trades away the code provenance a downstream consumer might want. A
summariser cannot ask "which rule fenced this?", because the answer stopped at
the module boundary; it gets a fence or it gets escaped prose. It also trades
away coverage: a declining book is fenceless even where a heuristic would have
been right some of the time. We accept both. The first keeps one artifact
format for every route, and a widened contract can be added later against real
consumers instead of imagined ones. The second is the cheaper mistake —
indentation reconstructed wrongly reads as authoritative, and nothing
downstream can tell it from indentation that is right.

## Considered Options

- **Detect structure first, fence afterwards** — rejected: the dependency runs
  the other way. Unfenced code carrying `#` comments becomes ATX headings,
  which both invent chapters and split real chapter bodies at the comment. And
  `detect_structure` takes the *minimum* heading level present, so two stray
  comment lines are enough to demote every real chapter in the book to a
  section.
- **Widen `Block` so downstream stages see code provenance** — rejected here,
  not forever: it is a separate effort with its own consumers to identify, and
  doing it as part of this one would change the parse contract for every route
  on speculation.
- **Guess when the typographic signal is absent** — rejected: one corpus PDF has
  no font contrast at all, its listings set in the body family and differentiated
  only by bold keywords and italic identifiers, and one corpus EPUB carries its
  code indentation purely in CSS, across dozens of generated classes with two
  competing properties, several of them negative. A guess there gets the lines right and the indentation
  wrong, which is worse than no fence: a fence asserts that its body is code
  verbatim, and nothing reading the artifact can discover that the assertion is
  false.
