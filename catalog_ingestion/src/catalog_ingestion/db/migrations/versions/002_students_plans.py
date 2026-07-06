"""Students and saved plans (app-side persistence for the backend).

Revision ID: 002
Revises: 001
Create Date: 2026-07-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "students",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("profile", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "plans",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("student_id", UUID(as_uuid=True), sa.ForeignKey("students.id", ondelete="CASCADE"), nullable=False),
        sa.Column("feedback", sa.Text, nullable=True),
        sa.Column("plan", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_plans_student_id", "plans", ["student_id"])


def downgrade() -> None:
    op.drop_index("ix_plans_student_id", table_name="plans")
    op.drop_table("plans")
    op.drop_table("students")
