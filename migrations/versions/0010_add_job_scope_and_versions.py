"""Add scope and version stamp columns to ingestion jobs

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-04

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Existing rows are whole-pipeline ingestion runs.
    op.add_column(
        "ingestion_jobs",
        sa.Column("scope", sa.String(), nullable=False, server_default="full"),
    )
    op.add_column("ingestion_jobs", sa.Column("extraction_version", sa.String(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("model_version", sa.String(), nullable=True))
    op.add_column("ingestion_jobs", sa.Column("prompt_version", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "prompt_version")
    op.drop_column("ingestion_jobs", "model_version")
    op.drop_column("ingestion_jobs", "extraction_version")
    op.drop_column("ingestion_jobs", "scope")
