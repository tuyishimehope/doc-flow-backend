"""v9_add_prompt_version_column_on_message_schema

Revision ID: 7f7bccb2353e
Revises: c4e8a1d6f293
Create Date: 2026-10-20 08:24:23.073009

Records which prompt file wrote each assistant answer. ``chat_v1`` is the only prompt that has
ever produced a saved answer, so the existing answers are filled in now; once a second prompt
ships this can no longer be told from the data. User messages stay NULL.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7f7bccb2353e'
down_revision: Union[str, Sequence[str], None] = 'c4e8a1d6f293'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('message', sa.Column('prompt_version', sa.String(length=32), nullable=True))
    op.execute("UPDATE message SET prompt_version = 'chat_v1' WHERE role = 'assistant'")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('message', 'prompt_version')
