from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    """
    State of a provider circuit.
    """

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
    Lightweight in-memory circuit breaker.

    CLOSED:
        Requests are allowed.

    OPEN:
        Provider has failed repeatedly.
        Requests are temporarily blocked.

    HALF_OPEN:
        Recovery timeout has passed.
        A request is allowed to check recovery.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ) -> None:

        self.failure_threshold = max(
            1,
            failure_threshold,
        )

        self.recovery_timeout = max(
            1.0,
            recovery_timeout,
        )

        self._states: dict[
            str,
            CircuitStatus,
        ] = {}

    def _get_status(
        self,
        provider: str,
    ) -> CircuitStatus:

        if provider not in self._states:

            self._states[provider] = (
                CircuitStatus(
                    provider=provider,
                    state=CircuitState.CLOSED,
                    failure_count=0,
                    opened_at=None,
                )
            )

        return self._states[provider]

    def allow_request(
        self,
        provider: str,
    ) -> bool:

        status = self._get_status(provider)

        # Normal state.
        if status.state == CircuitState.CLOSED:
            return True

        # Provider has failed too many times.
        if status.state == CircuitState.OPEN:

            if status.opened_at is None:
                return False

            elapsed = (
                time.monotonic()
                - status.opened_at
            )

            # Recovery period has passed.
            if (
                elapsed
                >= self.recovery_timeout
            ):

                status.state = (
                    CircuitState.HALF_OPEN
                )

                return True

            return False

        # HALF_OPEN.
        #
        # Allow a recovery request.
        return True

    def record_success(
        self,
        provider: str,
    ) -> None:

        status = self._get_status(provider)

        status.state = CircuitState.CLOSED
        status.failure_count = 0
        status.opened_at = None

    def record_failure(
        self,
        provider: str,
    ) -> None:

        status = self._get_status(provider)

        status.failure_count += 1

        if (
            status.failure_count
            >= self.failure_threshold
        ):

            status.state = CircuitState.OPEN
            status.opened_at = time.monotonic()

    def get_status(
        self,
        provider: str,
    ) -> CircuitStatus:

        status = self._get_status(provider)

        return CircuitStatus(
            provider=status.provider,
            state=status.state,
            failure_count=status.failure_count,
            opened_at=status.opened_at,
        )

    def reset(
        self,
        provider: str,
    ) -> None:

        self._states[provider] = CircuitStatus(
            provider=provider,
            state=CircuitState.CLOSED,
            failure_count=0,
            opened_at=None,
        )