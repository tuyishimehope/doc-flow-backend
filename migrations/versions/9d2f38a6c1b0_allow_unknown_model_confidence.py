"""Allow unknown model confidence values.

Revision ID: 9d2f38a6c1b0
Revises: 0656cbadef1e
Create Date: 2026-09-30 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "9d2f38a6c1b0"
down_revision: Union[str, Sequence[str], None] = "0656cbadef1e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "extracted_result",
        "confidence_score",
        existing_type=sa.Float(),
        nullable=True,
    )
    # The previous worker wrote 1.0 without producing a confidence estimate.
    op.execute("UPDATE extracted_result SET confidence_score = NULL")


def downgrade() -> None:
    op.execute(
        "UPDATE extracted_result SET confidence_score = 0 "
        "WHERE confidence_score IS NULL"
    )
    op.alter_column(
        "extracted_result",
        "confidence_score",
        existing_type=sa.Float(),
        nullable=False,
    )
