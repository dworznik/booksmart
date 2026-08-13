"""Baseline schema (dialect-neutral).

Single squashed baseline that replaces the original 11-plus per-change
migrations. Those were Postgres-only — every timestamp defaulted with
``server_default=now()``, which SQLite cannot execute — so booksmart-core now
ships one history that migrates an empty database to head identically on both
SQLite and Postgres (CI exercises both). Timestamp defaults moved client-side
(see ``models._utcnow``); the only ``server_default`` values kept are plain
string literals (``chapters.kind='chapter'``, ``runs.scope='full'``), which are
portable.

The schema here is the post-squash end state of the old chain: the run record
is ``runs`` (né ``ingestion_jobs``, with no ``queued`` state and no
``started_at``), and the parsed-artifact pointer lives on
``books.parsed_path`` / ``books.parser_used``.

Existing Postgres deployment: it is already at the old head, so it must NOT
re-run this migration. Its ``alembic_version`` still points at the old ``0012``,
which no longer exists in this history, so a plain ``alembic stamp head`` errors
with "Can't locate revision '0012'". Purge the stale row and stamp this baseline
in one step (the physical schema already matches):

    alembic stamp --purge head

Fresh databases (every SQLite CLI install, and Postgres CI) migrate from empty.

Revision ID: 0001
Revises:
Create Date: 2026-07-07

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "books",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("author", sa.String(), nullable=False),
        sa.Column("edition", sa.String(), nullable=True),
        sa.Column("publication_year", sa.Integer(), nullable=True),
        sa.Column("isbn", sa.String(), nullable=True),
        sa.Column("primary_topic", sa.String(), nullable=True),
        sa.Column("language", sa.String(), nullable=True),
        sa.Column("framework", sa.String(), nullable=True),
        sa.Column("methodology", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column("trust_level", sa.String(), nullable=True),
        sa.Column("intended_use", sa.String(), nullable=True),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("file_format", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("checksum", sa.String(), nullable=False),
        sa.Column("file_hash", sa.String(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("parsed_path", sa.String(), nullable=True),
        sa.Column("parser_used", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "chapters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), server_default="chapter", nullable=False),
        sa.Column("source_line", sa.Integer(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_model", sa.String(), nullable=True),
        sa.Column("summary_prompt_version", sa.String(), nullable=True),
        sa.Column("embedding_id", sa.Uuid(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_chapters_book_id", "chapters", ["book_id"])

    op.create_table(
        "sections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_line", sa.Integer(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("summary_model", sa.String(), nullable=True),
        sa.Column("summary_prompt_version", sa.String(), nullable=True),
        sa.Column("embedding_id", sa.Uuid(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sections_chapter_id", "sections", ["chapter_id"])

    op.create_table(
        "book_profiles",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_book_profiles_book_id", "book_profiles", ["book_id"])

    op.create_table(
        "knowledge_objects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=True),
        sa.Column("section_id", sa.Uuid(), nullable=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("source_location", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("edition", sa.String(), nullable=True),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("paragraph", sa.Integer(), nullable=True),
        sa.Column("extraction_model", sa.String(), nullable=False),
        sa.Column("extraction_prompt_version", sa.String(), nullable=False),
        sa.Column("embedding_id", sa.Uuid(), nullable=True),
        sa.Column("embedding_model", sa.String(), nullable=True),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["section_id"], ["sections.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_knowledge_objects_book_id", "knowledge_objects", ["book_id"])
    op.create_index(
        "ix_knowledge_objects_book_id_type", "knowledge_objects", ["book_id", "type"]
    )

    op.create_table(
        "runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("scope", sa.String(), server_default="full", nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("parser_used", sa.String(), nullable=True),
        sa.Column("extraction_version", sa.String(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("prompt_version", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("runs")
    op.drop_index("ix_knowledge_objects_book_id_type", table_name="knowledge_objects")
    op.drop_index("ix_knowledge_objects_book_id", table_name="knowledge_objects")
    op.drop_table("knowledge_objects")
    op.drop_index("ix_book_profiles_book_id", table_name="book_profiles")
    op.drop_table("book_profiles")
    op.drop_index("ix_sections_chapter_id", table_name="sections")
    op.drop_table("sections")
    op.drop_index("ix_chapters_book_id", table_name="chapters")
    op.drop_table("chapters")
    op.drop_table("books")
