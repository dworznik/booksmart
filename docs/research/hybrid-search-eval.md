# Hybrid vs dense-only retrieval on a booksmart-shaped corpus — findings

> Fixture eval for issue #40, answering the question the hybrid-search work could
> not settle from primary sources: **does fusing BM25 with dense retrieval
> actually beat dense retrieval alone on booksmart's corpus?**
>
> **Frozen at [`7a04ae3`](https://github.com/dworznik/booksmart/tree/7a04ae3)**
> (2026-08-03), the commit that merged this eval with the hybrid slices it
> measures. The numbers below are a record of what two embedding models did on
> that day against that corpus; they are not refreshed when either moves. The
> harness is checked in, so a later question gets a new run and a new note rather
> than an edit to this one:
>
> ```bash
> BOOKSMART_EVAL_EMBEDDING_PROVIDER=openai BOOKSMART_EVAL_API_KEY=$OPENAI_API_KEY \
>   uv run pytest packages/core/tests/test_hybrid_eval.py -k RealEval -s
> ```
>
> Corpus, query set and harness: `packages/core/tests/hybrid_eval.py`.
> `TestFixtureSet` in `test_hybrid_eval.py` enforces the property each query
> kind claims — an exact-term query's phrase must appear verbatim in its targets
> and nowhere else; a conceptual query must share no stemmed content word with
> its target, so BM25 provably cannot help it.

## Answer

**No — not on this corpus, and not with either shipped embedding model.**
Dense-only wins overall, and every point of the difference comes from the
conceptual queries. Hybrid never lost an exact-term or mixed query; it also
never won one.

| model | mode | hit@5 | MRR |
| --- | --- | --- | --- |
| `text-embedding-3-small` | hybrid | 0.94 | 0.847 |
| `text-embedding-3-small` | dense-only | 0.94 | **0.923** |
| `gemini-embedding-001` | hybrid | 1.00 | 0.944 |
| `gemini-embedding-001` | dense-only | 1.00 | **1.000** |

This is the opposite of the result the hybrid slices were built expecting, and
it is worth stating plainly rather than burying: on a 22-record corpus, both
embedding models already retrieve every coined term in the book — "temporal
decomposition", "tactical tornado", even "classitis" — at rank 1. There was no
lexical gap for BM25 to close, so fusion had only downside available to it.

## Why hybrid loses

RRF scores a record by the **rank** it reached in each branch, never by how
strongly it matched. The top hit of the sparse branch therefore collects the
full 1/(rank + k) whether it matched every query term or one incidental word.

Two concrete demotions, both under `gemini-embedding-001`:

```
Q: should i spend effort today to make tomorrow cheaper   (expects ch-tactical-strategic)
  hybrid: sec-general-purpose, ch-tactical-strategic, ...
  dense : ch-tactical-strategic, ko-change-amplification, ...
```

`sec-general-purpose` is the sparse branch's best match on the word **"make"**
and nothing else. That is enough to make it rank 1 in that branch, which is
enough to fuse it to rank 1 overall, displacing the record that actually answers
the question.

```
Q: how many things does a person have to hold in their head ... (expects ko-cognitive-load)
  hybrid: ch-complexity, ko-cognitive-load, ...
  dense : ko-cognitive-load, ch-complexity, ...
```

Same shape. The sparse branch always returns *something*, and on a query whose
vocabulary is deliberately disjoint from the target, that something is noise
promoted to rank 1.

## What this does and does not tell us

**Does**: on a single book's worth of summaries and knowledge objects, with a
current frontier embedding model, hybrid retrieval is not free. Unbounded RRF
over a sparse branch that is allowed to contribute its top hit unconditionally
costs conceptual recall.

**Does not**: it says nothing about a corpus of many books, where the dense
model's near-perfect scores here would degrade and lexical anchors would start
to matter; nor about queries containing genuinely out-of-vocabulary tokens —
version strings, API names, author surnames, error codes — which this corpus of
prose concepts does not contain. Both are the conditions under which hybrid
retrieval usually earns its keep, and neither is represented here. The corpus is
also small enough (22 records, limit=10) that the sparse branch returns nearly
half the corpus, which is close to the worst case for rank-only fusion.

## Recommendations

1. **Reconsider hybrid-by-default** (issue #39 shipped it as the default on the
   expectation this eval would confirm). On the present evidence a dense-only
   default with `--hybrid` opt-in would rank better for this product's current
   shape. This is a product decision, not a test failure, which is why the eval
   asserts nothing about it.
2. **Bound the sparse branch before tuning anything else.** The failure mode is
   entirely "weak lexical match gets full rank credit". A minimum BM25 score on
   the sparse prefetch, or a smaller branch limit than the dense one, targets it
   directly and needs no change to the fusion method.
3. **Try weighted fusion.** Qdrant offers DBSF (distribution-based score fusion)
   alongside RRF; it reads scores rather than ranks and so cannot promote a
   near-zero sparse match to the top on rank alone.
4. **Measure again before choosing a sparse model.** The BM25-vs-miniCOIL-vs-SPLADE
   question is not answerable from these numbers: BM25 was never the binding
   constraint here, the fusion policy was.
5. **Grow the corpus before trusting the conclusion too far.** Two or three
   books, and a query slice built from proper nouns and identifiers rather than
   concepts, would test the case hybrid is actually for.

## Generated tables

### `text-embedding-3-small` (the `Settings` default)


- Dense embedding model: `text-embedding-3-small`
- Sparse recipe: `Qdrant/bm25(k=1.2,b=0.75,avg_len=256.0,language=english)`
- Corpus: 22 fixture records · Queries: 18 · limit=10

## Per-query ranks

Rank of each expected record, best first. `—` means it was not in the top 10.

| kind | query | hybrid | dense-only | better |
| --- | --- | --- | --- | --- |
| exact-term | temporal decomposition | ko-temporal-decomposition @1 | ko-temporal-decomposition @1 | tie |
| exact-term | pass-through method | ko-pass-through-method @1 | ko-pass-through-method @1 | tie |
| exact-term | change amplification | ko-change-amplification @1, ch-complexity @2 | ko-change-amplification @1, ch-complexity @3 | tie |
| exact-term | unknown unknowns | ko-unknown-unknowns @1, ch-complexity @2 | ko-unknown-unknowns @1, ch-complexity @3 | tie |
| exact-term | conjoined methods | ko-conjoined-methods @1 | ko-conjoined-methods @1 | tie |
| exact-term | tactical tornado | ko-tactical-tornado @1 | ko-tactical-tornado @1 | tie |
| exact-term | classitis | ko-shallow-module @1 | ko-shallow-module @1 | tie |
| conceptual | a tiny tweak forces me to rewrite dozens of files | ko-change-amplification @4 | ko-change-amplification @2 | dense |
| conceptual | how many things does a person have to hold in their head before they can get anything done | ko-cognitive-load @2 | ko-cognitive-load @1 | dense |
| conceptual | a lot of capability reachable through only a handful of entry points | ko-deep-module @1, ch-modules-deep @3 | ko-deep-module @1, ch-modules-deep @3 | tie |
| conceptual | is it useful to weigh several rival approaches up front | ch-design-it-twice @1 | ch-design-it-twice @1 | tie |
| conceptual | make the awkward path just another ordinary path | ch-exceptions —, ko-exception-masking — | ch-exceptions @9, ko-exception-masking — | dense |
| conceptual | should i spend effort today to make tomorrow cheaper | ch-tactical-strategic @2 | ch-tactical-strategic @1 | dense |
| mixed | why is a shallow module bad for the people calling it | ko-shallow-module @1 | ko-shallow-module @1 | tie |
| mixed | comments that just repeat what the code already says | ch-comments-abstractions @1 | ch-comments-abstractions @1 | tie |
| mixed | information leakage between modules | ch-information-hiding @1, ko-temporal-decomposition @2 | ch-information-hiding @1, ko-temporal-decomposition @3 | tie |
| mixed | should an interface be general purpose or specific to my problem | sec-general-purpose @1 | sec-general-purpose @1 | tie |
| mixed | what makes code obscure to a new reader | ko-obscurity @1 | ko-obscurity @1 | tie |

## Aggregates

| slice | mode | hit@5 | MRR |
| --- | --- | --- | --- |
| all (n=18) | hybrid | 0.94 | 0.847 |
| all (n=18) | dense-only | 0.94 | 0.923 |
| exact-term (n=7) | hybrid | 1.00 | 1.000 |
| exact-term (n=7) | dense-only | 1.00 | 1.000 |
| conceptual (n=6) | hybrid | 0.83 | 0.542 |
| conceptual (n=6) | dense-only | 0.83 | 0.769 |
| mixed (n=5) | hybrid | 1.00 | 1.000 |
| mixed (n=5) | dense-only | 1.00 | 1.000 |

### `gemini-embedding-001`


- Dense embedding model: `gemini-embedding-001`
- Sparse recipe: `Qdrant/bm25(k=1.2,b=0.75,avg_len=256.0,language=english)`
- Corpus: 22 fixture records · Queries: 18 · limit=10

## Per-query ranks

Rank of each expected record, best first. `—` means it was not in the top 10.

| kind | query | hybrid | dense-only | better |
| --- | --- | --- | --- | --- |
| exact-term | temporal decomposition | ko-temporal-decomposition @1 | ko-temporal-decomposition @1 | tie |
| exact-term | pass-through method | ko-pass-through-method @1 | ko-pass-through-method @1 | tie |
| exact-term | change amplification | ko-change-amplification @1, ch-complexity @2 | ko-change-amplification @1, ch-complexity @2 | tie |
| exact-term | unknown unknowns | ko-unknown-unknowns @1, ch-complexity @2 | ko-unknown-unknowns @1, ch-complexity @2 | tie |
| exact-term | conjoined methods | ko-conjoined-methods @1 | ko-conjoined-methods @1 | tie |
| exact-term | tactical tornado | ko-tactical-tornado @1 | ko-tactical-tornado @1 | tie |
| exact-term | classitis | ko-shallow-module @1 | ko-shallow-module @1 | tie |
| conceptual | a tiny tweak forces me to rewrite dozens of files | ko-change-amplification @1 | ko-change-amplification @1 | tie |
| conceptual | how many things does a person have to hold in their head before they can get anything done | ko-cognitive-load @2 | ko-cognitive-load @1 | dense |
| conceptual | a lot of capability reachable through only a handful of entry points | ko-deep-module @1, ch-modules-deep @2 | ko-deep-module @1, ch-modules-deep @2 | tie |
| conceptual | is it useful to weigh several rival approaches up front | ch-design-it-twice @1 | ch-design-it-twice @1 | tie |
| conceptual | make the awkward path just another ordinary path | ch-exceptions @1, ko-exception-masking @10 | ch-exceptions @1, ko-exception-masking @4 | tie |
| conceptual | should i spend effort today to make tomorrow cheaper | ch-tactical-strategic @2 | ch-tactical-strategic @1 | dense |
| mixed | why is a shallow module bad for the people calling it | ko-shallow-module @1 | ko-shallow-module @1 | tie |
| mixed | comments that just repeat what the code already says | ch-comments-abstractions @1 | ch-comments-abstractions @1 | tie |
| mixed | information leakage between modules | ch-information-hiding @1, ko-temporal-decomposition @2 | ch-information-hiding @1, ko-temporal-decomposition @2 | tie |
| mixed | should an interface be general purpose or specific to my problem | sec-general-purpose @1 | sec-general-purpose @1 | tie |
| mixed | what makes code obscure to a new reader | ko-obscurity @1 | ko-obscurity @1 | tie |

## Aggregates

| slice | mode | hit@5 | MRR |
| --- | --- | --- | --- |
| all (n=18) | hybrid | 1.00 | 0.944 |
| all (n=18) | dense-only | 1.00 | 1.000 |
| exact-term (n=7) | hybrid | 1.00 | 1.000 |
| exact-term (n=7) | dense-only | 1.00 | 1.000 |
| conceptual (n=6) | hybrid | 1.00 | 0.833 |
| conceptual (n=6) | dense-only | 1.00 | 1.000 |
| mixed (n=5) | hybrid | 1.00 | 1.000 |
| mixed (n=5) | dense-only | 1.00 | 1.000 |
