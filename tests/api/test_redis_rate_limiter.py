from cardiorag.api.redis_rate_limiter import RedisRateLimiter


class _FakeRedis:
    """In-memory stand-in matching only the two calls RedisRateLimiter
    actually makes (incr, expire) - enough to test the counting/window
    logic without a real Redis server. TTL expiry itself is NOT simulated
    here (see test_redis_rate_limiter_real.py for that, against a real
    server) - this only proves the increment/threshold logic and that
    expire is requested exactly once per window."""

    def __init__(self):
        self._counts: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    def incr(self, key: str) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))


def test_allows_requests_up_to_the_limit():
    fake = _FakeRedis()
    limiter = RedisRateLimiter(fake, max_requests=3)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True


def test_rejects_requests_beyond_the_limit():
    fake = _FakeRedis()
    limiter = RedisRateLimiter(fake, max_requests=2)

    limiter.allow("client-a")
    limiter.allow("client-a")

    assert limiter.allow("client-a") is False


def test_tracks_each_client_independently():
    fake = _FakeRedis()
    limiter = RedisRateLimiter(fake, max_requests=1)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True  # separate key, separate budget
    assert limiter.allow("client-a") is False


def test_sets_expiry_only_once_per_window_not_on_every_request():
    fake = _FakeRedis()
    limiter = RedisRateLimiter(fake, max_requests=5, window_seconds=30)

    limiter.allow("client-a")
    limiter.allow("client-a")
    limiter.allow("client-a")

    assert fake.expire_calls == [("cardiorag:ratelimit:client-a", 30)]


def test_keys_are_namespaced_to_avoid_colliding_with_other_redis_users():
    fake = _FakeRedis()
    limiter = RedisRateLimiter(fake, max_requests=5)

    limiter.allow("some-client-ip")

    assert "cardiorag:ratelimit:some-client-ip" in fake._counts
