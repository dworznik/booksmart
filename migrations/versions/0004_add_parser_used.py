"""Add parser_used to ingestion_jobs

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-04

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ingestion_jobs", sa.Column("parser_used", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "parser_used")
