"""One daily token budget per user, shared by every AI feature.

A feature (chat today, voice later) calls ``check_daily_budget`` before it spends anything and
saves its token counts with the answer. The sum below reads those saved counts, so there is
exactly one place that decides "how much has this user used today" and one setting
(``CHAT_DAILY_TOKEN_BUDGET``) that limits it.

A day is a UTC day, the same day ``/chat/stats`` uses.

Known limits, on purpose for now:
- The check happens before the call, so the request that crosses the limit still finishes.
  Its size is capped by ``CHAT_MAX_OUTPUT_TOKENS`` and the history limits.
- Only answers saved with their usage are counted. The plain ``POST /chat`` saves nothing,
  and a cancelled or failed stream saves no usage, so neither adds to the total yet.
"""

import math
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.service.chat.crud import get_tokens_used_since
from app.service.chat.stats import window_start


class DailyBudgetExceededError(Exception):
    """The user has used today's tokens. Carries what the caller needs to explain it."""

    def __init__(self, used: int, limit: int, resets_at: datetime, retry_after: int) -> None:
        super().__init__(f"{used} of {limit} tokens used")
        self.used = used
        self.limit = limit
        self.resets_at = resets_at
        self.retry_after = retry_after


def next_reset(now: datetime) -> datetime:
    """The next midnight UTC."""
    return window_start(1, now) + timedelta(days=1)


async def tokens_used_today(
    user_id: int, db_session: AsyncSession, *, now: datetime | None = None
) -> int:
    """Where the day's total comes from. A feature whose cost is not saved on a chat message
    adds its share here, so it counts against the same budget."""
    now = now or datetime.now(timezone.utc)
    return await get_tokens_used_since(user_id, window_start(1, now), db_session)


async def check_daily_budget(
    user_id: int, db_session: AsyncSession, *, now: datetime | None = None
) -> None:
    """Raise ``DailyBudgetExceededError`` when the user has reached the daily limit.

    A limit of 0 turns the budget off.
    """
    limit = settings.CHAT_DAILY_TOKEN_BUDGET
    if limit <= 0:
        return
    now = now or datetime.now(timezone.utc)
    used = await tokens_used_today(user_id, db_session, now=now)
    if used < limit:
        return
    resets_at = next_reset(now)
    retry_after = max(1, math.ceil((resets_at - now).total_seconds()))
    raise DailyBudgetExceededError(used, limit, resets_at, retry_after)
