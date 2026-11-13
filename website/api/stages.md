# Stages

A **Stage** is one step of ingestion and the unit of durability: it takes ids,
loads what it needs itself, replaces its output wholesale and commits. A
completed Stage is permanent, and a failed one never undoes the Stages before
it, so a Runner can retry a single Stage without repeating the rest.

Stages are **Run-blind**. They never read or write a Run record, which belongs
to whichever Runner is driving them. That is what lets the CLI's foreground
Runner and a durable, step-based executor share the same functions. Each Stage
returns a `StageReport` (its token spend, its counts and its own timings) for
the Runner to keep beside the Run.

The Stages run in a fixed order, each reading what the one before it wrote:

1. `run_parse`: the stored original becomes Markdown.
2. `run_structure`: the Markdown becomes a chapter and section tree.
3. `run_profile`: an LLM characterises the book.
4. `run_extraction`: an LLM pulls knowledge objects out of each chapter.
5. `run_summaries`: an LLM summarises each chapter.
6. `run_embeddings`: summaries and knowledge objects are embedded into the
   [`VectorStore`](vectors.md).

Providers (the LLM, the embedders, the vector store) are passed in as keyword
arguments rather than built inside a Stage, so a consumer chooses them and a
test can substitute fakes. See
[ADR 0002](https://github.com/dworznik/booksmart/blob/main/docs/adr/0002-stages-are-the-unit-of-durability.md)
for the reasoning.

::: booksmart_core.stages
