from __future__ import annotations

import os

from backend.app.services.circuit_breaker import (
    CircuitBreaker,
)


def _get_int(
    name: str,
    default: int,
) -> int:

    try:
        return int(
            os.getenv(
                name,
                str(default),
            )
        )
    except (TypeError, ValueError):
        return default


def _get_float(
    name: str,
    default: float,
) -> float:

    try:
        return float(
            os.getenv(
                name,
                str(default),
            )
        )
    except (TypeError, ValueError):
        return default


# One shared circuit breaker for the APIL process.
circuit_breaker = CircuitBreaker(
    failure_threshold=_get_int(
        "APIL_CIRCUIT_FAILURE_THRESHOLD",
        3,
    ),
    recovery_timeout=_get_float(
        "APIL_CIRCUIT_RECOVERY_TIMEOUT",
        30.0,
    ),
)