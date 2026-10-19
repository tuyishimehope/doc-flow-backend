"""v8_add_message_usage_columns

Revision ID: c4e8a1d6f293
Revises: 5b1e7c2a9d40
Create Date: 2026-10-19 12:00:00.000000

Token counts, provider and latency move out of the ``metadata`` JSON into real columns,
so usage can be summed, grouped by day and timed in SQL. Old rows are copied over.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c4e8a1d6f293'
down_revision: Union[str, Sequence[str], None] = '5b1e7c2a9d40'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Whole numbers only, so a malformed value in old JSON cannot break the cast.
_NUMBER = "^[0-9]{1,9}$"


def _from_json(key: str) -> str:
    return (
        f"CASE WHEN (metadata->>'{key}') ~ '{_NUMBER}' "
        f"THEN (metadata->>'{key}')::integer END"
    )


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('message', sa.Column('provider', sa.String(length=32), nullable=True))
    op.add_column('message', sa.Column('input_tokens', sa.Integer(), nullable=True))
    op.add_column('message', sa.Column('output_tokens', sa.Integer(), nullable=True))
    op.add_column('message', sa.Column('cached_tokens', sa.Integer(), nullable=True))
    op.add_column('message', sa.Column('reasoning_tokens', sa.Integer(), nullable=True))
    op.add_column('message', sa.Column('latency_ms', sa.Integer(), nullable=True))

    # Copy what the first version stored in the JSON (PostgreSQL only; other databases
    # have no such rows to carry over). Cut-off answers were saved with a JSON null, hence
    # the json_typeof check: only real objects have keys to copy.
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            f"""
            UPDATE message SET
                provider = NULLIF(metadata->>'provider', ''),
                input_tokens = {_from_json('input_tokens')},
                output_tokens = {_from_json('output_tokens')},
                latency_ms = {_from_json('latency_ms')}
            WHERE role = 'assistant' AND json_typeof(metadata) = 'object'
            """
        )
        # The JSON keeps only what has no column of its own.
        op.execute(
            """
            UPDATE message SET metadata = (metadata::jsonb - 'provider' - 'input_tokens'
                - 'output_tokens' - 'latency_ms')::json
            WHERE role = 'assistant' AND json_typeof(metadata) = 'object'
            """
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('message', 'latency_ms')
    op.drop_column('message', 'reasoning_tokens')
    op.drop_column('message', 'cached_tokens')
    op.drop_column('message', 'output_tokens')
    op.drop_column('message', 'input_tokens')
    op.drop_column('message', 'provider')
