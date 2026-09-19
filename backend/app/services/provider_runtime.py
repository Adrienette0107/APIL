from __future__ import annotations

import os

from backend.app.services.circuit_breaker import CircuitBreaker


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


# ------------------------------------------------------------
# Shared APIL circuit breaker
# ------------------------------------------------------------
#
# This object is created once when the application process
# starts and is reused by ProviderExecutor instances.
#
# This is important because circuit-breaker state must survive
# across individual requests.
#
shared_circuit_breaker = CircuitBreaker(
    failure_threshold=_get_int(
        "APIL_CIRCUIT_FAILURE_THRESHOLD",
        3,
    ),
    recovery_timeout=_get_float(
        "APIL_CIRCUIT_RECOVERY_TIMEOUT",
        30.0,
    ),
)


# Backward-compatible alias.
#
# Existing code may still import:
#
#     from backend.app.services.provider_runtime import circuit_breaker
#
# Keeping this alias prevents unnecessary breakage while we
# gradually clean up the project.
circuit_breaker = shared_circuit_breaker