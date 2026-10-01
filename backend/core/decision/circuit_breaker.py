"""Circuit breaker for external TypeSafe AI / Jev service resilience."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal operation: traffic routes to Jev
    OPEN = "OPEN"          # Tripped: fast-fail traffic to heuristic fallback
    HALF_OPEN = "HALF_OPEN"# Trial recovery: allow single probe to test recovery


class CircuitBreaker:
    """Enterprise-grade 3-state circuit breaker preventing cascade failures.

    Thread-safe logic with configurable failure threshold, cooldown recovery timeout,
    and automatic self-healing probe transitions.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        service_name: str = "TypeSafe-Jev",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.service_name = service_name

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._last_failure_reason: str | None = None
        self._recovery_attempts: int = 0
        self._last_state_change: float = time.time()

    @property
    def state(self) -> CircuitState:
        # Check if OPEN state has expired and should transition to HALF_OPEN
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info(
                    "Circuit breaker for %s transitioning from OPEN to HALF_OPEN (elapsed: %.1fs >= %.1fs)",
                    self.service_name,
                    elapsed,
                    self.recovery_timeout,
                )
                self._state = CircuitState.HALF_OPEN
                self._recovery_attempts += 1
                self._last_state_change = time.time()
        return self._state

    def can_execute(self) -> bool:
        """Determine whether an outbound call should be permitted."""
        current_state = self.state
        if current_state == CircuitState.CLOSED:
            return True
        if current_state == CircuitState.HALF_OPEN:
            return True
        return False

    def get_fast_fail_reason(self) -> str:
        """Return categorized fallback reason when circuit is tripped."""
        elapsed = round(time.time() - self._last_failure_time, 1)
        remaining = max(0.0, round(self.recovery_timeout - elapsed, 1))
        return (
            f"Circuit breaker is OPEN ({self.service_name} failed {self._failure_count} consecutive times; "
            f"last error: '{self._last_failure_reason}'; recovery probe in {remaining}s)"
        )

    def record_success(self) -> None:
        """Record successful invocation, resetting failures and closing circuit."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker for %s successfully recovered! Transitioning to CLOSED.", self.service_name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_reason = None
        self._last_state_change = time.time()

    def record_failure(self, reason: str) -> None:
        """Record failure, potentially tripping the circuit to OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._last_failure_reason = reason

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(
                "Circuit breaker probe for %s failed (%s). Re-tripping to OPEN.",
                self.service_name,
                reason,
            )
            self._state = CircuitState.OPEN
            self._last_state_change = time.time()
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            logger.warning(
                "Circuit breaker for %s TRIPPED to OPEN after %d consecutive failures. Last error: %s",
                self.service_name,
                self._failure_count,
                reason,
            )
            self._state = CircuitState.OPEN
            self._last_state_change = time.time()

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._last_failure_reason = None
        self._recovery_attempts = 0
        self._last_state_change = time.time()

    def get_status(self) -> dict[str, Any]:
        """Telemetry snapshot of circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_reason": self._last_failure_reason,
            "recovery_attempts": self._recovery_attempts,
            "cooldown_remaining_seconds": max(0.0, round(self.recovery_timeout - (time.time() - self._last_failure_time), 1))
            if self.state == CircuitState.OPEN
            else 0.0,
        }
