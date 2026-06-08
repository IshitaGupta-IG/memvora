from collections import defaultdict, deque
from time import monotonic

from fastapi import HTTPException


_events: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def check_rate_limit(user_id: str, action: str, limit: int, window_seconds: int) -> None:
    key = (user_id, action)
    now = monotonic()
    events = _events[key]

    while events and now - events[0] > window_seconds:
        events.popleft()

    if len(events) >= limit:
        raise HTTPException(status_code=429, detail=f"Too many {action} requests. Please try again later.")

    events.append(now)
