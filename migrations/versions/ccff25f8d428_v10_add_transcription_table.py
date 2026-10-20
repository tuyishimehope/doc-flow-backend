"""v10_add_transcription_table

Revision ID: ccff25f8d428
Revises: 7f7bccb2353e
Create Date: 2026-10-20 14:27:43.936312

Keeps the text a speech-to-text call returned, plus metadata (model, size of the audio in bytes,
how long the provider took). There is deliberately no column for the audio itself.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ccff25f8d428'
down_revision: Union[str, Sequence[str], None] = '7f7bccb2353e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('transcription',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('model', sa.String(length=100), nullable=False),
    sa.Column('audio_bytes', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('transcription')
