import time
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import TypeVar

from clinical_agent_harness.harness.errors import (
    CircuitOpenError,
    RetryableError,
)


T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 30.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.monotonic() - self._last_failure_time
            >= self.recovery_timeout_seconds
        ):
            self._state = CircuitState.HALF_OPEN

        return self._state

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:

        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                "Circuit breaker is open; operation blocked."
            )

        try:
            result = await operation()

        except (RetryableError, TimeoutError):
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

            raise

        else:
            self._failure_count = 0
            self._last_failure_time = None
            self._state = CircuitState.CLOSED

            return result