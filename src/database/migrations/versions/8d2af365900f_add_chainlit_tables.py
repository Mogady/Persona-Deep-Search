"""add chainlit tables

Revision ID: 8d2af365900f
Revises: 6a4354cd05f6
Create Date: 2025-10-04 02:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8d2af365900f'
down_revision: Union[str, None] = '6a4354cd05f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create Chainlit tables
    op.create_table('users',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('identifier', sa.String(), nullable=False),
    sa.Column('metadata', sa.JSON(), nullable=False),
    sa.Column('createdAt', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('identifier')
    )

    op.create_table('threads',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('createdAt', sa.String(), nullable=True),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('userId', sa.String(), nullable=True),
    sa.Column('userIdentifier', sa.String(), nullable=True),
    sa.Column('tags', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('metadata', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['userId'], ['users.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('elements',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('threadId', sa.String(), nullable=True),
    sa.Column('type', sa.String(), nullable=True),
    sa.Column('url', sa.String(), nullable=True),
    sa.Column('chainlitKey', sa.String(), nullable=True),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('display', sa.String(), nullable=True),
    sa.Column('objectKey', sa.String(), nullable=True),
    sa.Column('size', sa.String(), nullable=True),
    sa.Column('page', sa.Integer(), nullable=True),
    sa.Column('language', sa.String(), nullable=True),
    sa.Column('forId', sa.String(), nullable=True),
    sa.Column('mime', sa.String(), nullable=True),
    sa.Column('props', sa.JSON(), nullable=True),
    sa.Column('playerConfig', sa.JSON(), nullable=True),
    sa.Column('autoPlay', sa.Boolean(), nullable=True),
    sa.ForeignKeyConstraint(['threadId'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('feedbacks',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('forId', sa.String(), nullable=False),
    sa.Column('threadId', sa.String(), nullable=False),
    sa.Column('value', sa.Integer(), nullable=False),
    sa.Column('comment', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['threadId'], ['threads.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('steps',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('threadId', sa.String(), nullable=False),
    sa.Column('parentId', sa.String(), nullable=True),
    sa.Column('command', sa.String(), nullable=True),
    sa.Column('streaming', sa.Boolean(), nullable=False),
    sa.Column('waitForAnswer', sa.Boolean(), nullable=True),
    sa.Column('isError', sa.Boolean(), nullable=True),
    sa.Column('metadata', sa.JSON(), nullable=False),
    sa.Column('tags', sa.ARRAY(sa.String()), nullable=True),
    sa.Column('input', sa.Text(), nullable=True),
    sa.Column('output', sa.Text(), nullable=True),
    sa.Column('createdAt', sa.String(), nullable=True),
    sa.Column('start', sa.String(), nullable=True),
    sa.Column('end', sa.String(), nullable=True),
    sa.Column('generation', sa.JSON(), nullable=True),
    sa.Column('showInput', sa.String(), nullable=True),
    sa.Column('defaultOpen', sa.Boolean(), nullable=True),
    sa.Column('language', sa.String(), nullable=True),
    sa.Column('feedback', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['threadId'], ['threads.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    # Drop Chainlit tables in reverse order
    op.drop_table('steps')
    op.drop_table('feedbacks')
    op.drop_table('elements')
    op.drop_table('threads')
    op.drop_table('users')
