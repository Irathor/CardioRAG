"""Redis-backed fixed-window rate limiter (project improvement round).

Same interface as `RateLimiter` (`rate_limit.py`) - `.allow(client_id) ->
bool` - but the counter lives in Redis instead of process memory, so every
API replica pointed at the same Redis instance shares one budget per
client. The in-memory limiter is correct only for a single process; this
is the fix for when the API ever runs as more than one.

Uses the standard INCR-then-conditionally-EXPIRE recipe, not a Lua script:
INCR is atomic and returns a value unique to the caller that triggered it,
so exactly one caller ever observes count == 1 for a given window - there
is no race where two different requests both believe they're the first
and both set (or skip) the expiry.
"""

_KEY_PREFIX = "cardiorag:ratelimit:"


class RedisRateLimiter:
    def __init__(self, redis_client, max_requests: int, window_seconds: int = 60):
        self._redis = redis_client
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    def allow(self, client_id: str) -> bool:
        key = f"{_KEY_PREFIX}{client_id}"
        count = self._redis.incr(key)
        if count == 1:
            self._redis.expire(key, self._window_seconds)
        return count <= self._max_requests
