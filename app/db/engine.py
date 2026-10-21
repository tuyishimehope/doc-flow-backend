from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    create_async_engine,
)

from app.core.config import settings

DATABASE_NAME = settings.DATABASE_NAME
DATABASE_PASSWORD = settings.DATABASE_PASSWORD
DATABASE_HOST = settings.DATABASE_HOST
DATABASE_PORT = settings.DATABASE_PORT
DATABASE_USER = settings.DATABASE_USER

# A failed query puts its SQL parameters into the error message, and with them the user's text
# (chat messages, transcripts). Hidden unless ENVIRONMENT is exactly "development".
HIDE_PARAMETERS = settings.ENVIRONMENT.strip().lower() != "development"

engine = create_async_engine(
    f"postgresql+asyncpg://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}",
    hide_parameters=HIDE_PARAMETERS,
)

sync_engine = create_engine(
    f"postgresql+psycopg2://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}",
    hide_parameters=HIDE_PARAMETERS,
)
