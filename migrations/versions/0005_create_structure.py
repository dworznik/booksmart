"""Create chapters and sections tables

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-04

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chapters",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("book_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("source_line", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(["chapter_id"], ["chapters.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_sections_chapter_id", "sections", ["chapter_id"])


def downgrade() -> None:
    op.drop_index("ix_sections_chapter_id", table_name="sections")
    op.drop_table("sections")
    op.drop_index("ix_chapters_book_id", table_name="chapters")
    op.drop_table("chapters")
