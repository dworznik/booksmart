# Passages are embedded on request and searched by name

The index held only what a model wrote about a book — chapter and section
summaries, and knowledge objects. Search could find the chapter that discusses
something and never the sentence that says it, and wording the book uses but no
summary repeats (an identifier, an error message, a coinage) was invisible to
both Branches of a Hybrid search. We now embed the text itself, as Passages:
windows of the parsed markdown, cut at paragraph boundaries inside one section,
packed toward 256 tokens with no overlap.

Three choices here are not the obvious ones, and each trades something away.

**Passages go in the same collection, and their payloads carry book text.**
`vectors.py` used to say the vector store held summaries and extracted content
and never raw book text. That is over, deliberately: a search hit that had to go
back to the filesystem for its words would not be a hit. What survives is the
half of the statement that mattered — the store never holds the *sole* copy.
`Book.parsed_path` is still where a book's words live, a Passage row holds a
character span into it and no text, and the collection can still be dropped and
rebuilt. A separate collection for passages would have kept the old sentence
true at the cost of a second Model lock to keep in step with the first, and of a
Hybrid search that could no longer rank a passage against a summary in one
query. The Model lock is untouched: it names models, not record types.

**Embedding passages is a Preference, off by default.** A book has roughly ten
times as much text as it has summary, so this multiplies what an ingest spends
on embedding by about that. An upgrade must not start charging for it. The cost
of the default is that the feature is invisible until someone turns it on.

**A search that names no record types does not return passages**, even once they
exist. They outnumber every other record in a book by an order of magnitude, so
including them unasked would change what an unqualified search returns for
every caller who enabled them for some other reason — and would do it silently.
The cost is a second opt-in: a caller has to enable passages *and* ask for them.

Both defaults are provisional in the same way hybrid-by-default was. Whether
passages should be embedded by default, or searched by default, is a question
about retrieval quality against cost that an eval answers and this change does
not.

## Consequences

- Cutting passages is a function the embeddings Stage calls, not a Stage. Passage rows
  and passage points are one output: cutting, writing rows and embedding in one
  run of one Stage is what keeps them matched under wholesale replacement
  (ADR 0002). It adds no Scope.
- Every run of the embeddings Stage replaces a book's passages, including a run
  with the Preference off, which leaves none. That is how they are removed.
- Passage rows cascade from their chapter and section. A span means something
  only against the parse its chapter was detected in, and the structure Stage
  deletes a book's chapters after every re-parse — so a re-parsed book has no
  passages until its embeddings Stage runs again, rather than passages whose
  offsets point into a file that has changed underneath them.
- A "token" in the window size is a whitespace-delimited word: no tokenizer, no
  model, the same boundaries on every machine. 256 is the `avg_len` of the
  locked BM25 Recipe (ADR 0004), which is the document length its normalisation
  is calibrated for; counting words rather than BM25's own stemmed,
  stopword-free terms makes that a near match and not an exact one.
- Search always filters on record type now. Before, a search with no filters
  sent none.
- A book's embeddings no longer fit one request. Passages take a book from
  about a hundred records to thousands, which reaches two bounds summaries
  never did: a vendor's cap on the tokens one embedding request may carry (now
  a Limit a provider can declare, `max_batch_tokens`), and a Qdrant server's
  cap on the size of one write. Both are split to fit rather than left to fail
  on the first long book.

## Considered Options

- **A separate collection for passages** — rejected above: two locks, and no
  single ranking across passages and summaries.
- **A Stage of its own for cutting passages** — rejected: its output would be rows that are
  only meaningful once embedded, with a window between the two Stages in which
  search resolves points to rows that no longer match them. It would also add a
  Scope for a step that calls no model and takes milliseconds.
- **Overlapping windows** — rejected: Fusion reads rank positions, and
  near-duplicate neighbours each take a position from a record that says
  something different.
- **Fixed-size windows that ignore structure** — rejected: a window straddling
  two sections belongs to neither, and the section is what a hit is shown by.
- **Storing the passage text in its row** — rejected: a second authoritative
  copy of the book in the database, to save a slice.
