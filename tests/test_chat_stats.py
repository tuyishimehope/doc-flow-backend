"""GET /api/v1/chat/stats: tokens per day, p50/p95 latency and error rate."""

from datetime import datetime, timedelta, timezone

import pytest

from app.models.schema import Conversation, Message
from app.service.chat.stats import percentile, window_start
from tests.conftest import auth_header, create_test_user, login_user

URL = "/api/v1/chat/stats"


# ---- the maths, no database ----------------------------------------------------------------


@pytest.mark.parametrize(
    "values, p, expected",
    [
        ([], 50, None),
        ([7], 50, 7),
        ([7], 95, 7),
        ([10, 20, 30, 40], 50, 20),  # the smallest value with at least half of them at or below it
        ([10, 20, 30, 40], 95, 40),
        (list(range(1, 101)), 50, 50),
        (list(range(1, 101)), 95, 95),
        (list(range(1, 101)), 100, 100),
    ],
)
def test_percentile_is_the_nearest_rank(values, p, expected):
    assert percentile(values, p) == expected


def test_the_window_is_whole_days_in_utc_ending_today():
    noon = datetime(2026, 10, 19, 12, 30, tzinfo=timezone.utc)
    assert window_start(1, noon) == datetime(2026, 10, 19, tzinfo=timezone.utc)
    assert window_start(7, noon) == datetime(2026, 10, 13, tzinfo=timezone.utc)
    # A time given in another zone is converted first: 01:00 on the 20th in UTC+3 is still the 19th.
    plus3 = timezone(timedelta(hours=3))
    assert window_start(1, datetime(2026, 10, 20, 1, 0, tzinfo=plus3)) == datetime(2026, 10, 19, tzinfo=timezone.utc)


# ---- helpers ---------------------------------------------------------------------------------


async def user_id(client, headers) -> int:
    return (await client.get("/api/v1/users/me", headers=headers)).json()["id"]


async def make_conversation(db_session, owner_id: int) -> Conversation:
    row = Conversation(user_id=owner_id, title="stats")
    db_session.add(row)
    await db_session.flush()
    return row


def answer(conversation, when: datetime, *, status="complete", role="assistant", **usage) -> Message:
    return Message(conversation_id=conversation.id, role=role, content="x", status=status, created_at=when, **usage)


def hours_ago(hours: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


@pytest.fixture
async def mine(client, auth_headers, db_session) -> Conversation:
    return await make_conversation(db_session, await user_id(client, auth_headers))


# ---- the endpoint ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_it_needs_a_login(client):
    assert (await client.get(URL)).status_code == 401


@pytest.mark.anyio
async def test_no_answers_gives_zeros_for_every_day_of_the_window(client, auth_headers):
    body = (await client.get(URL, headers=auth_headers)).json()
    assert body["days"] == 7 and len(body["per_day"]) == 7
    assert (body["answers"], body["errors"], body["error_rate"]) == (0, 0, 0.0)
    assert body["latency"] == {"samples": 0, "p50_ms": None, "p95_ms": None}
    assert all(day["answers"] == 0 and day["input_tokens"] == 0 for day in body["per_day"])
    days = [day["day"] for day in body["per_day"]]
    assert days == sorted(days) and days[-1] == datetime.now(timezone.utc).date().isoformat()


@pytest.mark.anyio
async def test_tokens_are_summed_per_day_and_in_total(client, auth_headers, db_session, mine):
    db_session.add_all(
        [
            answer(mine, hours_ago(1), input_tokens=10, output_tokens=4, cached_tokens=6, reasoning_tokens=1, latency_ms=100),
            answer(mine, hours_ago(1.5), input_tokens=5, output_tokens=2, latency_ms=300),  # no cached/reasoning
            answer(mine, hours_ago(1), input_tokens=None, output_tokens=None, latency_ms=200),  # provider sent no usage
            answer(mine, hours_ago(24 * 2), input_tokens=100, output_tokens=50, latency_ms=900),
        ]
    )
    await db_session.flush()

    body = (await client.get(URL, headers=auth_headers, params={"days": 5})).json()

    assert (body["answers"], body["complete"]) == (4, 4)
    assert (body["input_tokens"], body["output_tokens"]) == (115, 56)
    assert (body["cached_tokens"], body["reasoning_tokens"]) == (6, 1)
    expected: dict[str, list[int]] = {}  # day -> [answers, input tokens], worked out from the timestamps
    for hours, tokens in [(1, 10), (1.5, 5), (1, 0), (48, 100)]:
        day = hours_ago(hours).date().isoformat()
        counts = expected.setdefault(day, [0, 0])
        counts[0] += 1
        counts[1] += tokens
    shown = {day["day"]: [day["answers"], day["input_tokens"]] for day in body["per_day"]}
    assert len(shown) == 5
    assert {day: counts for day, counts in shown.items() if counts != [0, 0]} == expected
    assert list(shown.values()).count([0, 0]) == 5 - len(expected)  # quiet days are there as zeros


@pytest.mark.anyio
async def test_latency_percentiles_use_finished_answers_only(client, auth_headers, db_session, mine):
    latencies = list(range(100, 2100, 100))  # 100, 200 ... 2000 ms: twenty answers
    db_session.add_all([answer(mine, hours_ago(1), latency_ms=ms) for ms in latencies])
    db_session.add(answer(mine, hours_ago(1), status="error", latency_ms=99999))  # must not count
    db_session.add(answer(mine, hours_ago(1), latency_ms=None))  # unknown latency must not count
    await db_session.flush()

    body = (await client.get(URL, headers=auth_headers)).json()

    assert body["latency"] == {"samples": 20, "p50_ms": 1000, "p95_ms": 1900}


@pytest.mark.anyio
async def test_error_rate_counts_errors_but_not_answers_the_user_cut_off(client, auth_headers, db_session, mine):
    db_session.add_all(
        [answer(mine, hours_ago(1)) for _ in range(6)]
        + [answer(mine, hours_ago(1), status="error") for _ in range(2)]
        + [answer(mine, hours_ago(1), status="cancelled") for _ in range(2)]
    )
    await db_session.flush()

    body = (await client.get(URL, headers=auth_headers)).json()

    assert (body["answers"], body["complete"], body["errors"], body["cancelled"]) == (10, 6, 2, 2)
    assert body["error_rate"] == 0.2


@pytest.mark.anyio
async def test_only_the_users_assistant_answers_inside_the_window_count(client, auth_headers, db_session, mine):
    stranger = await create_test_user(client, email="other@example.com")
    stranger_headers = auth_header(await login_user(client, email="other@example.com"))
    theirs = await make_conversation(db_session, await user_id(client, stranger_headers))
    db_session.add_all(
        [
            answer(mine, hours_ago(1), input_tokens=1),
            answer(mine, hours_ago(1), role="user", input_tokens=1000),  # the user's own message
            answer(mine, hours_ago(24 * 30), input_tokens=1000),  # too old for a 7 day window
            answer(theirs, hours_ago(1), input_tokens=1000),  # somebody else's answer
        ]
    )
    await db_session.flush()

    mine_stats = (await client.get(URL, headers=auth_headers)).json()
    their_stats = (await client.get(URL, headers=stranger_headers)).json()

    assert (mine_stats["answers"], mine_stats["input_tokens"]) == (1, 1)
    assert (their_stats["answers"], their_stats["input_tokens"]) == (1, 1000)
    assert stranger  # the second user exists


@pytest.mark.anyio
@pytest.mark.parametrize("days", [0, -1, 91, "many"])
async def test_the_number_of_days_is_limited(client, auth_headers, days):
    assert (await client.get(URL, headers=auth_headers, params={"days": days})).status_code == 422


@pytest.mark.anyio
async def test_a_longer_window_reaches_further_back(client, auth_headers, db_session, mine):
    db_session.add(answer(mine, hours_ago(24 * 20), input_tokens=7))
    await db_session.flush()

    assert (await client.get(URL, headers=auth_headers, params={"days": 7})).json()["answers"] == 0
    thirty = (await client.get(URL, headers=auth_headers, params={"days": 30})).json()
    assert (thirty["answers"], thirty["input_tokens"], len(thirty["per_day"])) == (1, 7, 30)
