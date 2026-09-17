from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from backend.app.core.provider_config import (
    ProviderConfig,
    get_provider_configs,
)
from backend.app.services.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)


@dataclass
class ProviderHealth:
    provider: str
    enabled: bool
    configured: bool
    circuit_state: str
    failure_count: int
    priority: int
    fallback_enabled: bool
    default_model: str | None


class ProviderHealthService:
    """
    Provides the current operational status of
    registered APIL providers.
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:

        self.configs: dict[
            str,
            ProviderConfig,
        ] = get_provider_configs()

        self.circuit_breaker = (
            circuit_breaker
            or CircuitBreaker()
        )

    @staticmethod
    def _is_configured(
        config: ProviderConfig,
    ) -> bool:

        # Ollama does not require an API key.
        if config.name == "ollama":
            return bool(config.base_url)

        return bool(config.api_key)

    def get_provider_health(
        self,
        provider: str,
    ) -> ProviderHealth | None:

        config = self.configs.get(provider)

        if config is None:
            return None

        circuit = self.circuit_breaker.get_status(
            provider
        )

        return ProviderHealth(
            provider=provider,
            enabled=config.enabled,
            configured=self._is_configured(config),
            circuit_state=circuit.state.value,
            failure_count=circuit.failure_count,
            priority=config.priority,
            fallback_enabled=config.fallback_enabled,
            default_model=config.default_model,
        )

    def get_all_health(
        self,
    ) -> list[ProviderHealth]:

        results: list[ProviderHealth] = []

        for provider in self.configs:

            health = self.get_provider_health(
                provider
            )

            if health is not None:
                results.append(health)

        results.sort(
            key=lambda item: item.priority
        )

        return results

    def get_summary(self) -> dict[str, Any]:

        providers = self.get_all_health()

        return {
            "total": len(providers),
            "enabled": sum(
                1
                for provider in providers
                if provider.enabled
            ),
            "configured": sum(
                1
                for provider in providers
                if provider.configured
            ),
            "circuit_open": sum(
                1
                for provider in providers
                if provider.circuit_state
                == CircuitState.OPEN.value
            ),
        }