"""Sliding-window rate limiter for per-IP and per-session controls."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from .config import SecurityConfig


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: int


class SlidingWindowRateLimiter:
    """In-memory rate limiter suitable for single-instance deployment."""

    def __init__(self, config: SecurityConfig) -> None:
        self._limit = config.rate_limit
        self._window = config.rate_window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, ip: str, session_id: str | None = None) -> RateLimitDecision:
        key = f"{ip}:{session_id or 'anonymous'}"
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            queue = self._events[key]
            while queue and queue[0] < cutoff:
                queue.popleft()

            if len(queue) >= self._limit:
                retry = max(1, int(self._window - (now - queue[0])))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry)

            queue.append(now)
            return RateLimitDecision(allowed=True, retry_after_seconds=0)
