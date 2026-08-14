"""Per-Stage spend and timing on a Run.

Dialect-neutral, like the baseline it follows: no ``server_default=now()``, and
``sa.JSON`` rather than ``JSONB`` so SQLite and Postgres share one history.

Additive and backfill-free. Existing rows keep NULL for ``runs.embedding_tokens``
and carry no ``run_stages`` at all, which is the truthful record: those runs
happened before anything measured where their spend went, and inventing a
distribution for them would be worse than saying nothing.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-14
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("embedding_tokens", sa.Integer(), nullable=True))
    op.create_table(
        "run_stages",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.Uuid(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("embedding_tokens", sa.Integer(), nullable=False),
        sa.Column("counts", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_run_stages_run_id", "run_stages", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_run_stages_run_id", table_name="run_stages")
    op.drop_table("run_stages")
    op.drop_column("runs", "embedding_tokens")
