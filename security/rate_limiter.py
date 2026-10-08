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
    """In-memory rate limiter with bounded storage and IP/user identity enforcement."""

    def __init__(self, config: SecurityConfig, max_keys: int = 2000) -> None:
        self._limit = config.rate_limit
        self._window = config.rate_window_seconds
        self._max_keys = max_keys
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(
        self,
        ip: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> RateLimitDecision:
        # Rate limit primarily by authenticated identity or client IP to prevent session spoofing
        key = user_id or ip or "127.0.0.1"
        now = time.time()
        cutoff = now - self._window

        with self._lock:
            # Memory safety: if store exceeds max_keys, prune stale entries
            if len(self._events) > self._max_keys:
                self._prune_stale_keys(cutoff)

            queue = self._events[key]
            while queue and queue[0] < cutoff:
                queue.popleft()

            if len(queue) >= self._limit:
                retry = max(1, int(self._window - (now - queue[0])))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry)

            queue.append(now)
            return RateLimitDecision(allowed=True, retry_after_seconds=0)

    def _prune_stale_keys(self, cutoff: float) -> None:
        """Evict keys that have had no activity within the window."""
        stale_keys = [k for k, q in self._events.items() if not q or q[-1] < cutoff]
        for k in stale_keys:
            self._events.pop(k, None)
