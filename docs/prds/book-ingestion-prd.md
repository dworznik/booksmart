# Product Requirements Document (PRD)

# Knowledge Repository --- Book Ingestion Pipeline (v1)

> **Frozen at [`b4f442a`](https://github.com/dworznik/booksmart/tree/b4f442a)**
> (2026-07-03): the requirements v1 was built against, kept as the record of what
> was asked for. It predates the workspace split and the removal of the HTTP
> server, so its surfaces and schemas are not the ones the code has now — read
> `CONTEXT.md` and `docs/adr/` for what holds today (see
> [`docs/agents/domain.md`](../agents/domain.md#living-docs-vs-frozen-docs)).

## Purpose

Build a reproducible ingestion pipeline that imports technical and
programming books into a structured knowledge repository. The goal of
the ingestion stage is **not** to answer questions directly, but to
transform books into a normalized, metadata-rich representation that can
later be compiled into best practices, skills, and specialized agents.

This PRD covers **only the ingestion subsystem**.

------------------------------------------------------------------------

# Goals

The ingestion pipeline must:

-   Import PDF and EPUB books.
-   Preserve the original source.
-   Extract structured text.
-   Identify the logical structure of the book.
-   Extract candidate knowledge objects.
-   Associate all extracted information with rich metadata.
-   Allow human hints before and after ingestion.
-   Be fully reproducible.
-   Support incremental reprocessing.
-   Keep complete provenance for every extracted object.

------------------------------------------------------------------------

# Non-goals

This project does **not** include:

-   RAG querying
-   Agent generation
-   Skill generation
-   User-facing chat
-   Recommendation engine
-   Practice compilation across books

Those are separate systems.

------------------------------------------------------------------------

# High-level Architecture

                   +-------------------+
                   |   User uploads    |
                   +---------+---------+
                             |
                             v
                  +----------------------+
                  |  Book Registration   |
                  +----------+-----------+
                             |
                   store original file
                             |
                             v
                  +----------------------+
                  | Text Extraction      |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Structure Detection  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Metadata Enrichment  |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Knowledge Extraction |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Repository           |
                  +----------------------+

------------------------------------------------------------------------

# Technology Stack

## Containerization

Docker Compose

## Database

PostgreSQL

Stores books, chapters, sections, extracted objects, ingestion jobs,
metadata, hints, and provenance.

## Vector Database

Qdrant

Stores embeddings for chapter summaries, section summaries, and
extracted knowledge objects. Raw book text is not stored solely inside
Qdrant.

## Object Storage

Filesystem (v1)

    storage/
    books/
    parsed/
    artifacts/
    logs/

Future versions may use S3-compatible storage.

## OCR / Parsing

Preferred order:

1.  Marker
2.  PyMuPDF
3.  OCR fallback

## LLM

Configurable (GPT-5.5, Claude, Gemini initially).

------------------------------------------------------------------------

# Services

-   API Service
-   Worker Service
-   PostgreSQL
-   Qdrant
-   Object Storage

------------------------------------------------------------------------

# Book Registration

Required:

-   title
-   author

Optional:

-   edition
-   publication year
-   ISBN

Automatic:

-   checksum
-   file hash
-   upload timestamp

------------------------------------------------------------------------

# Initial Hints

Optional:

-   Primary topic
-   Language
-   Framework
-   Methodology
-   Notes
-   Trust level
-   Intended use

------------------------------------------------------------------------

# Mutable Metadata

Metadata remains editable forever.

------------------------------------------------------------------------

# Ingestion Stages

1.  Store original file.
2.  Extract text.
3.  Detect structure.
4.  Generate book profile.
5.  Extract knowledge objects.
6.  Generate embeddings.
7.  Persist repository.

------------------------------------------------------------------------

# Knowledge Object Schema

-   id
-   book_id
-   chapter
-   section
-   type
-   title
-   content
-   summary
-   source_location
-   confidence
-   embedding_id
-   created_at

Supported types:

-   Practice
-   Principle
-   Tradeoff
-   Anti-pattern
-   Smell
-   Decision Rule
-   Definition
-   Glossary
-   Checklist

------------------------------------------------------------------------

# Provenance

Every object stores:

-   book
-   edition
-   chapter
-   section
-   page
-   paragraph
-   extraction model
-   extraction prompt version

------------------------------------------------------------------------

# Reprocessing

Support:

-   profile
-   extraction
-   embeddings
-   full rebuild

------------------------------------------------------------------------

# Versioning

Store extraction version, model version, prompt version, timestamp.

------------------------------------------------------------------------

# REST API

-   POST /books
-   GET /books
-   GET /books/{id}
-   PATCH /books/{id}
-   POST /books/{id}/ingest
-   POST /books/{id}/reprocess
-   GET /jobs/{id}

------------------------------------------------------------------------

# Docker Compose

Minimum:

-   knowledge-api
-   knowledge-worker
-   postgres
-   qdrant

Optional:

-   pgadmin
-   qdrant-ui

------------------------------------------------------------------------

# Acceptance Criteria

-   Accept PDF and EPUB.
-   Preserve originals.
-   Produce structured Markdown.
-   Detect document structure.
-   Generate book profiles.
-   Extract typed knowledge objects with provenance.
-   Generate embeddings.
-   Store relational data in PostgreSQL and vectors in Qdrant.
-   Allow metadata updates after ingestion.
-   Support incremental and full reprocessing.
-   Preserve ingestion history.
