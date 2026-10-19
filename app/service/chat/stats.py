"""Usage numbers for one user: tokens per day, latency percentiles and error rate."""

import math
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.service.chat.crud import count_answers_by_status, get_latencies, get_usage_per_day
from app.service.chat.schema import MessageStatus

TOKEN_FIELDS = ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens")


def percentile(sorted_values: Sequence[int], p: float) -> int | None:
    """Nearest-rank percentile of an ascending list: the smallest value that at least
    ``p`` percent of the values are less than or equal to. ``None`` for no values."""
    if not sorted_values:
        return None
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def _as_date(value: Any) -> date:
    # PostgreSQL returns a date, SQLite a string.
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def window_start(days: int, now: datetime) -> datetime:
    """Midnight UTC of the first of the last ``days`` days, today included."""
    first_day = now.astimezone(timezone.utc).date() - timedelta(days=days - 1)
    return datetime.combine(first_day, time.min, tzinfo=timezone.utc)


async def get_chat_stats(
    user_id: int, days: int, db_session: AsyncSession, *, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    since = window_start(days, now)

    rows = await get_usage_per_day(user_id, since, db_session)
    by_status = await count_answers_by_status(user_id, since, db_session)
    latencies = await get_latencies(user_id, since, db_session)

    by_day = {_as_date(row.day): row for row in rows}
    per_day = []
    for offset in range(days):  # every day of the window, so a chart has no gaps
        day = since.date() + timedelta(days=offset)
        row = by_day.get(day)
        per_day.append(
            {"day": day, "answers": row.answers if row else 0}
            | {field: int(getattr(row, field)) if row else 0 for field in TOKEN_FIELDS}
        )

    answers = sum(by_status.values())
    errors = by_status.get(MessageStatus.ERROR.value, 0)
    return {
        "days": days,
        "answers": answers,
        "complete": by_status.get(MessageStatus.COMPLETE.value, 0),
        "errors": errors,
        "cancelled": by_status.get(MessageStatus.CANCELLED.value, 0),
        "error_rate": round(errors / answers, 4) if answers else 0.0,
        "latency": {
            "samples": len(latencies),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
        },
        **{field: sum(item[field] for item in per_day) for field in TOKEN_FIELDS},
        "per_day": per_day,
    }
