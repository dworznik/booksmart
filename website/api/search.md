# Search

`search` is the read side of the pipeline: it embeds a query, ranks the vectors
in the [`VectorStore`](vectors.md), and resolves each hit back to the relational
row it was embedded from. It writes nothing and involves no Stage or Runner, so
a consumer can call it on any session. It is the function an HTTP search
endpoint would wrap.

Retrieval is **hybrid** by default: the query is embedded densely and
lexically, and the two rankings are fused. Pass `mode="dense"` for dense-only
retrieval when cosine scores comparable across queries matter more than exact
term matches. Which mode produced a result is carried on `SearchResults.mode`,
because the two kinds of score are not on the same scale.

Before any query is embedded, the collection's model lock is verified. A query
embedded by one model and compared against another model's vectors returns
plausible but wrong rankings, so a mismatch is refused rather than served.

::: booksmart_core.search
