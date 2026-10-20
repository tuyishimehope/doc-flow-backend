"""At most N chat answers streaming at the same moment, per user.

Think of a cloakroom with N tickets per person. A ticket is taken before a stream starts and
handed back when it ends. A ticket that is never handed back (the server died, the connection
dropped without a clean close) is thrown away by the next request once it is older than
``ttl_seconds``, so nobody stays locked out.

The tickets of one user live in a Redis sorted set: member = a random id, score = the time the
stream started.
"""

import asyncio
import logging
import time
import uuid
from typing import Any

from redis.exceptions import RedisError

from app.core.rate_limit import redis_client

logger = logging.getLogger(__name__)

# Releases that must finish even though the request that started them was cancelled.
_background: set[asyncio.Task[Any]] = set()


def _key(user_id: int) -> str:
    return f"chat:streams:{user_id}"


async def acquire_stream_slot(
    user_id: int, *, limit: int, ttl_seconds: int
) -> str | None:
    """A slot id when the user may start a stream, or None when `limit` are already running.

    Fails open: when Redis is down the stream is allowed, and the returned id is a ticket that
    was never stored (giving it back removes nothing).
    """
    key = _key(user_id)
    slot_id = str(uuid.uuid4())
    try:
        now = time.time()
        # Add first, count second, all in one transaction: two requests arriving together
        # cannot both see room for the last place.
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(
                key, "-inf", f"({now - ttl_seconds}"
            )  # throw away dead tickets
            pipe.zadd(key, {slot_id: now})
            pipe.zcard(key)
            pipe.expire(
                key, ttl_seconds * 2
            )  # a user who never returns leaves no key behind
            _, _, count, _ = await pipe.execute()
        if count > limit:
            await redis_client.zrem(key, slot_id)  # a refused request keeps no ticket
            return None
        return slot_id
    except RedisError:
        logger.warning(
            "Stream limit skipped because Redis is unavailable", exc_info=True
        )
        return slot_id


async def release_stream_slot(user_id: int, slot_id: str | None) -> None:
    """Give the ticket back when the stream ends. Never raises, and finishes even if cancelled.

    It runs in a ``finally`` block, so an exception here would hide the real error. It also runs
    when the client leaves, which cancels the request: the removal is its own task, so the
    cancellation cannot stop it.
    """
    if slot_id is None:
        return
    task = asyncio.ensure_future(_remove(user_id, slot_id))
    _background.add(task)
    task.add_done_callback(_background.discard)
    await asyncio.shield(task)


async def _remove(user_id: int, slot_id: str) -> None:
    try:
        await redis_client.zrem(_key(user_id), slot_id)
    except RedisError:
        logger.warning(
            "Could not give a stream slot back; it expires by itself", exc_info=True
        )
