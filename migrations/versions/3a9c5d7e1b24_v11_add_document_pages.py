"""v11_add_document_pages

Revision ID: 3a9c5d7e1b24
Revises: ccff25f8d428
Create Date: 2026-10-22 09:00:00.000000

Keeps the text of each page of a PDF that was stored for questions, and lets a conversation
point at the document it is about.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '3a9c5d7e1b24'
down_revision: Union[str, Sequence[str], None] = 'ccff25f8d428'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'document_page',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('page', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['document.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'page', name='uq_document_page_document_id_page'),
    )
    op.add_column('conversation', sa.Column('document_id', sa.Integer(), nullable=True))
    op.create_index(
        op.f('ix_conversation_document_id'), 'conversation', ['document_id'], unique=False)
    op.create_foreign_key(
        'fk_conversation_document_id_document',
        'conversation', 'document', ['document_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('fk_conversation_document_id_document', 'conversation', type_='foreignkey')
    op.drop_index(op.f('ix_conversation_document_id'), table_name='conversation')
    op.drop_column('conversation', 'document_id')
    op.drop_table('document_page')
