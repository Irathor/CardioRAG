from cardiorag.api.rate_limit import RateLimiter


def test_allows_requests_up_to_the_limit():
    limiter = RateLimiter(max_requests=3, window_seconds=60.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True


def test_rejects_requests_beyond_the_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60.0)

    limiter.allow("client-a")
    limiter.allow("client-a")

    assert limiter.allow("client-a") is False


def test_tracks_each_client_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True  # separate budget, not shared
    assert limiter.allow("client-a") is False


def test_window_resets_after_it_elapses(monkeypatch):
    limiter = RateLimiter(max_requests=1, window_seconds=60.0)
    fake_time = [1000.0]
    monkeypatch.setattr("cardiorag.api.rate_limit.time.monotonic", lambda: fake_time[0])

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False  # still inside the window

    fake_time[0] += 61.0  # past the window

    assert limiter.allow("client-a") is True
