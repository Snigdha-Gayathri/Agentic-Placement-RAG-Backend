from security.config import SecurityConfig
from security.rate_limiter import SlidingWindowRateLimiter


def test_rate_limiter_blocks_when_threshold_exceeded() -> None:
    limiter = SlidingWindowRateLimiter(SecurityConfig(rate_limit=2, rate_window_seconds=60))
    assert limiter.allow("127.0.0.1", "s1").allowed
    assert limiter.allow("127.0.0.1", "s1").allowed
    blocked = limiter.allow("127.0.0.1", "s1")
    assert not blocked.allowed
    assert blocked.retry_after_seconds > 0
