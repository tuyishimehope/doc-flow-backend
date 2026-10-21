from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import sessionmaker

from app.db.engine import engine, sync_engine

SessionLocal = async_sessionmaker(
    autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
)

SyncSession = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)
