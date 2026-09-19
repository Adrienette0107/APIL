from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitStatus:
    provider: str
    state: CircuitState
    failure_count: int
    opened_at: float | None = None


class CircuitBreaker:
    """
    In-memory circuit breaker for provider execution.

    CLOSED:
        Requests are allowed.

    OPEN:
        Requests are blocked until recovery_timeout expires.

    HALF_OPEN:
        One recovery request is allowed.
        Additional requests remain blocked until that request
        succeeds or fails.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_timeout = max(1.0, recovery_timeout)

        self._states: dict[str, CircuitStatus] = {}
        self._half_open_in_flight: set[str] = set()

    def _get_status(self, provider: str) -> CircuitStatus:
        if provider not in self._states:
            self._states[provider] = CircuitStatus(
                provider=provider,
                state=CircuitState.CLOSED,
                failure_count=0,
                opened_at=None,
            )

        return self._states[provider]

    def allow_request(self, provider: str) -> bool:
        status = self._get_status(provider)

        if status.state == CircuitState.CLOSED:
            return True

        if status.state == CircuitState.OPEN:
            if status.opened_at is None:
                return False

            elapsed = time.monotonic() - status.opened_at

            if elapsed >= self.recovery_timeout:
                status.state = CircuitState.HALF_OPEN

                if provider in self._half_open_in_flight:
                    return False

                self._half_open_in_flight.add(provider)
                return True

            return False

        # HALF_OPEN
        if provider in self._half_open_in_flight:
            return False

        self._half_open_in_flight.add(provider)
        return True

    def record_success(self, provider: str) -> None:
        status = self._get_status(provider)

        status.state = CircuitState.CLOSED
        status.failure_count = 0
        status.opened_at = None

        self._half_open_in_flight.discard(provider)

    def record_failure(self, provider: str) -> None:
        status = self._get_status(provider)

        self._half_open_in_flight.discard(provider)

        status.failure_count += 1

        if status.failure_count >= self.failure_threshold:
            status.state = CircuitState.OPEN
            status.opened_at = time.monotonic()

    def get_status(self, provider: str) -> CircuitStatus:
        status = self._get_status(provider)

        return CircuitStatus(
            provider=status.provider,
            state=status.state,
            failure_count=status.failure_count,
            opened_at=status.opened_at,
        )

    def reset(self, provider: str) -> None:
        self._states[provider] = CircuitStatus(
            provider=provider,
            state=CircuitState.CLOSED,
            failure_count=0,
            opened_at=None,
        )

        self._half_open_in_flight.discard(provider)