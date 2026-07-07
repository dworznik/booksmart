"""Add book hint columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-04

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

HINT_COLUMNS = (
    "primary_topic",
    "language",
    "framework",
    "methodology",
    "notes",
    "trust_level",
    "intended_use",
)


def upgrade() -> None:
    for column in HINT_COLUMNS:
        op.add_column("books", sa.Column(column, sa.String(), nullable=True))


def downgrade() -> None:
    for column in reversed(HINT_COLUMNS):
        op.drop_column("books", column)
