"""v12_add_document_chunks

Revision ID: 8b2e4f6a1c35
Revises: 3a9c5d7e1b24
Create Date: 2026-10-23 09:00:00.000000

Real retrieval (plan_4, part C): the pgvector extension, a chunk table with one 768-number
vector per chunk, and document.index_status. The extension needs a Postgres that ships it
(the pgvector/pgvector image locally; Neon has it, enable it there too).
"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '8b2e4f6a1c35'
down_revision: Union[str, Sequence[str], None] = '3a9c5d7e1b24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.add_column(
        'document',
        sa.Column('index_status', sa.String(length=16), server_default='NONE', nullable=False),
    )
    op.create_table(
        'document_chunk',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('page', sa.Integer(), nullable=False),
        sa.Column('position', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(dim=768), nullable=False),
        sa.Column('embedding_model', sa.String(length=100), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['document.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('document_id', 'position', name='uq_document_chunk_document_id_position'),
    )
    op.create_index(
        'ix_document_chunk_user_id_document_id', 'document_chunk', ['user_id', 'document_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema. The extension is left installed: other objects may use it."""
    op.drop_index('ix_document_chunk_user_id_document_id', table_name='document_chunk')
    op.drop_table('document_chunk')
    op.drop_column('document', 'index_status')
