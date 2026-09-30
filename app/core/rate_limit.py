import logging

from fastapi import HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

logger = logging.getLogger(__name__)

redis_client = Redis.from_url(
    settings.broker_host, socket_connect_timeout=2, socket_timeout=2)


async def check_rate_limit(key: str, limit: int, window_seconds: int) -> None:
    """Allow `limit` calls per `key` in a fixed window, else raise 429.

    Fails open: a Redis outage should not lock everyone out of logging in.
    """
    if not settings.rate_limit_enabled:
        return

    redis_key = f"rate_limit:{key}"
    try:
        async with redis_client.pipeline(transaction=True) as pipe:
            count, _ = await pipe.incr(redis_key).expire(redis_key, window_seconds, nx=True).execute()
    except RedisError:
        logger.warning("Rate limit skipped because Redis is unavailable", exc_info=True)
        return

    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Please try again later.",
            headers={"Retry-After": str(window_seconds)},
        )
