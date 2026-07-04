"""Add summary and embedding linkage columns

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-04

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDED_TABLES = ("chapters", "sections", "knowledge_objects")


def upgrade() -> None:
    op.add_column("chapters", sa.Column("summary", sa.Text(), nullable=True))
    op.add_column("sections", sa.Column("summary", sa.Text(), nullable=True))
    for table in EMBEDDED_TABLES:
        op.add_column(table, sa.Column("embedding_id", sa.Uuid(), nullable=True))
        op.add_column(table, sa.Column("embedding_model", sa.String(), nullable=True))
        op.add_column(table, sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    for table in EMBEDDED_TABLES:
        op.drop_column(table, "embedded_at")
        op.drop_column(table, "embedding_model")
        op.drop_column(table, "embedding_id")
    op.drop_column("sections", "summary")
    op.drop_column("chapters", "summary")
