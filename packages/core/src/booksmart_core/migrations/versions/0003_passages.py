"""Passages: the windows of a book's own text that the embeddings Stage embeds.

Dialect-neutral, like the two revisions it follows.

A passage needs a row because search resolves every point back to one and
skips a point that has none — a passage held only in the vector store would be
embedded, paid for, and unfindable. The row is deliberately thin: where the
passage sits (book, chapter, section) and a character span into the parsed
markdown. The text itself is not a column. ``books.parsed_path`` is still the
one place a book's words live; a passage points into it.

Both structural foreign keys cascade. A span is only meaningful against the
markdown its chapter was detected in, and the structure Stage deletes a book's
chapters whenever that markdown is re-parsed — so a passage that outlived its
chapter would be a pair of offsets into a file that no longer says what they
were cut from.

Additive and backfill-free: passage embedding is a Preference that defaults to
off, so an existing deployment gains an empty table and nothing else.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-13
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "passages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("chapter_id", sa.Uuid(), nullable=False),
        sa.Column("section_id", sa.Uuid(), nullable=True),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("source_location", sa.String(), nullable=False),
        sa.Column("embedding_id", sa.Uuid(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["book_id"], ["books.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["section_id"], ["sections.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_passages_book_id", "passages", ["book_id"])


def downgrade() -> None:
    op.drop_index("ix_passages_book_id", table_name="passages")
    op.drop_table("passages")
