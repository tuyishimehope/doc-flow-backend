"""Wait for application dependencies before running migrations or serving traffic."""

import asyncio
import logging
import time

import redis
from sqlalchemy import text

from app.core.config import settings
from app.core.minio import minio_client
from app.db.engine import engine

logger = logging.getLogger(__name__)


async def check_database() -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


def check_redis() -> None:
    client = redis.Redis.from_url(settings.broker_host, socket_connect_timeout=2, socket_timeout=2)
    try:
        client.ping()
    finally:
        client.close()


def check_minio() -> None:
    minio_client.bucket_exists(settings.MINIO_BUCKET)


async def wait_for_dependencies() -> None:
    deadline = time.monotonic() + settings.DEPENDENCY_STARTUP_TIMEOUT_SECONDS
    pending = {"PostgreSQL", "Redis", "MinIO"}

    while pending and time.monotonic() < deadline:
        if "PostgreSQL" in pending:
            try:
                await check_database()
                pending.remove("PostgreSQL")
            except Exception:
                pass

        if "Redis" in pending:
            try:
                await asyncio.to_thread(check_redis)
                pending.remove("Redis")
            except Exception:
                pass

        if "MinIO" in pending:
            try:
                await asyncio.to_thread(check_minio)
                pending.remove("MinIO")
            except Exception:
                pass

        if pending:
            logger.info("Waiting for application dependencies", extra={"pending": sorted(pending)})
            await asyncio.sleep(2)

    if pending:
        raise RuntimeError(
            "Dependencies did not become ready before startup timeout: "
            + ", ".join(sorted(pending))
        )
