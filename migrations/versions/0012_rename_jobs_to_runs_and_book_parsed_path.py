"""Rename ingestion_jobs to runs; move parsed-artifact pointer onto books

Renames ``ingestion_jobs`` to ``runs`` (the Run vocabulary in CONTEXT.md),
drops the now-redundant ``started_at`` column (a Run is created the moment
execution starts, so ``created_at`` already marks the start and there is no
queued state before it), and adds ``books.parsed_path`` / ``books.parser_used``
so downstream stages resolve their input from the Book instead of querying
past successful runs.

The status column is left as a free-form string; the "shrink" from four
statuses to three (running/succeeded/failed, no queued) is behavioural.

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-07

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.rename_table("ingestion_jobs", "runs")
    op.drop_column("runs", "started_at")
    op.add_column("books", sa.Column("parsed_path", sa.String(), nullable=True))
    op.add_column("books", sa.Column("parser_used", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("books", "parser_used")
    op.drop_column("books", "parsed_path")
    op.add_column(
        "runs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.rename_table("runs", "ingestion_jobs")
