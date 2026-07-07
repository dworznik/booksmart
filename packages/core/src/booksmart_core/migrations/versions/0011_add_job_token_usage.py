"""Add token usage totals to ingestion jobs

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-07

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ingestion_jobs", sa.Column("input_tokens", sa.Integer(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("output_tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "output_tokens")
    op.drop_column("ingestion_jobs", "input_tokens")
