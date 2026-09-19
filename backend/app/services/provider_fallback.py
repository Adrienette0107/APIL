from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from backend.app.core.errors import ProviderExecutionError
from backend.app.core.provider_config import (
    ProviderConfig,
    get_provider_configs,
)
from backend.app.services.provider_executor import (
    ProviderExecutionResult,
    ProviderExecutor,
)
from backend.app.services.provider_runtime import (
    shared_circuit_breaker,
)

logger = logging.getLogger("apil.provider_fallback")


class ProviderFallbackManager:
    """
    Central provider fallback coordinator.

    Responsibilities:
        1. Validate the primary provider.
        2. Execute through ProviderExecutor.
        3. Allow ProviderExecutor to handle retries/timeouts.
        4. Move to another provider only when the failure is retryable.
        5. Select a valid model for each provider.
        6. Return one normalized ProviderExecutionResult.

    Architecture:

        APIL Pipeline
             |
             v
        Fallback Manager
             |
             +---- Primary Provider
             |          |
             |          v
             |     ProviderExecutor
             |
             +---- Fallback Provider
                        |
                        v
                   ProviderExecutor
    """

    def __init__(
        self,
        executor: ProviderExecutor | None = None,
    ) -> None:

        self.configs: dict[str, ProviderConfig] = (
            get_provider_configs()
        )

        self.executor = (
            executor
            if executor is not None
            else ProviderExecutor(
                configs=self.configs,
                circuit_breaker=shared_circuit_breaker,
            )
        )

    # ------------------------------------------------------------------
    # Fallback providers
    # ------------------------------------------------------------------

    def get_fallback_providers(
        self,
        primary_provider: str,
    ) -> list[str]:
        """
        Return enabled fallback providers ordered by priority.

        Lower priority number means higher fallback priority.
        """

        primary_provider = (
            primary_provider.strip().lower()
            if primary_provider
            else ""
        )

        candidates: list[ProviderConfig] = []

        for provider_name, config in self.configs.items():

            provider_name = (
                provider_name.strip().lower()
            )

            if provider_name == primary_provider:
                continue

            if not config.enabled:
                continue

            if not config.fallback_enabled:
                continue

            candidates.append(config)

        candidates.sort(
            key=lambda config: (
                config.priority,
                config.name,
            )
        )

        return [
            config.name
            for config in candidates
        ]

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def get_model_for_provider(
        self,
        provider: str,
        requested_model: str,
        is_fallback: bool = False,
    ) -> str:
        """
        Resolve the model that should be sent to a provider.

        Rules:

        Primary provider:
            - explicit model is preserved
            - "auto" uses provider default

        Fallback provider:
            - always use that provider's configured default model
              when one exists

        This prevents cases such as:

            openai:gpt-4.1
                    |
                    v
                 Ollama

        from accidentally becoming:

            ollama:gpt-4.1
        """

        provider = (
            provider.strip().lower()
            if provider
            else ""
        )

        requested_model = (
            requested_model.strip()
            if requested_model
            else ""
        )

        config = self.configs.get(
            provider
        )

        if config is None:
            return ""

        # Fallback providers should use their own model.
        if is_fallback:

            if config.default_model:
                return config.default_model.strip()

            return ""

        # Primary provider with automatic selection.
        if (
            not requested_model
            or requested_model.lower() == "auto"
        ):

            if config.default_model:
                return config.default_model.strip()

            return ""

        # Primary provider with an explicit model.
        return requested_model

    # ------------------------------------------------------------------
    # Provider execution
    # ------------------------------------------------------------------

    async def execute_with_fallback(
        self,
        primary_provider: str,
        model: str,
        operation_factory: Callable[
            [str, str],
            Awaitable[Any],
        ],
    ) -> ProviderExecutionResult:
        """
        Execute the requested provider and fall back when appropriate.

        operation_factory(provider, model) must return an awaitable
        provider-specific response.

        ProviderExecutor handles:

            timeout
            retry
            backoff
            circuit breaker
            response normalization
            error normalization

        This class handles:

            provider ordering
            fallback decisions
            fallback model selection
        """

        primary_provider = (
            primary_provider.strip().lower()
            if primary_provider
            else ""
        )

        model = (
            model.strip()
            if model
            else ""
        )

        # ------------------------------------------------------------------
        # Validate primary provider
        # ------------------------------------------------------------------

        if not primary_provider:
            raise ProviderExecutionError(
                code="provider_not_specified",
                message="No provider was specified.",
                status_code=400,
                category="provider_not_specified",
                retryable=False,
            )

        primary_config = self.configs.get(
            primary_provider
        )

        if primary_config is None:
            raise ProviderExecutionError(
                code="provider_not_registered",
                message=(
                    f"Provider '{primary_provider}' "
                    "is not registered."
                ),
                status_code=502,
                provider=primary_provider,
                category="provider_not_registered",
                retryable=False,
            )

        if not primary_config.enabled:
            raise ProviderExecutionError(
                code="provider_disabled",
                message=(
                    f"Provider '{primary_provider}' "
                    "is disabled."
                ),
                status_code=503,
                provider=primary_provider,
                category="provider_disabled",
                retryable=False,
            )

        # ------------------------------------------------------------------
        # Build provider execution order
        # ------------------------------------------------------------------

        providers = [
            primary_provider,
            *self.get_fallback_providers(
                primary_provider
            ),
        ]

        last_error: ProviderExecutionError | None = None

        # ------------------------------------------------------------------
        # Execute providers
        # ------------------------------------------------------------------

        for index, provider in enumerate(
            providers
        ):

            provider_config = self.configs.get(
                provider
            )

            if provider_config is None:
                continue

            if not provider_config.enabled:
                continue

            is_fallback = index > 0

            provider_model = (
                self.get_model_for_provider(
                    provider=provider,
                    requested_model=model,
                    is_fallback=is_fallback,
                )
            )

            if not provider_model:
                logger.warning(
                    "provider_skipped_no_model",
                    extra={
                        "provider": provider,
                        "fallback": is_fallback,
                    },
                )
                continue

            logger.info(
                "provider_attempt",
                extra={
                    "provider": provider,
                    "model": provider_model,
                    "fallback": is_fallback,
                    "attempt_index": index,
                },
            )

            try:

                result = await self.executor.execute(
                    provider=provider,
                    model=provider_model,
                    operation=(
                        lambda
                        p=provider,
                        m=provider_model:
                        operation_factory(
                            p,
                            m,
                        )
                    ),
                )

                logger.info(
                    "provider_success",
                    extra={
                        "provider": provider,
                        "model": provider_model,
                        "fallback": is_fallback,
                        "attempts": result.attempts,
                    },
                )

                return result

            except ProviderExecutionError as exc:

                if exc.provider is None:
                    exc.provider = provider

                if not exc.category:
                    exc.category = exc.code

                last_error = exc

                logger.warning(
                    "provider_failed",
                    extra={
                        "provider": provider,
                        "model": provider_model,
                        "category": exc.category,
                        "code": exc.code,
                        "retryable": exc.retryable,
                        "attempts": exc.attempts,
                        "fallback": is_fallback,
                    },
                )

                # ----------------------------------------------------------
                # Non-retryable errors stop immediately.
                #
                # Example:
                #   invalid request
                #   invalid model
                #   malformed input
                #
                # Trying another provider does not generally fix these.
                # ----------------------------------------------------------

                if not exc.retryable:
                    raise

                # ----------------------------------------------------------
                # Retryable error:
                # continue to next provider if available.
                # ----------------------------------------------------------

                if index < len(providers) - 1:

                    next_provider = providers[
                        index + 1
                    ]

                    logger.warning(
                        "provider_fallback_started",
                        extra={
                            "failed_provider": provider,
                            "failed_model": provider_model,
                            "next_provider": next_provider,
                        },
                    )

                    continue

                logger.error(
                    "provider_fallback_exhausted",
                    extra={
                        "last_provider": provider,
                        "last_model": provider_model,
                        "category": exc.category,
                    },
                )

        # ------------------------------------------------------------------
        # Every available provider failed.
        # ------------------------------------------------------------------

        if last_error is not None:
            raise last_error

        raise ProviderExecutionError(
            message=(
                f"All providers failed for primary provider "
                f"'{primary_provider}'."
            ),
            code="all_providers_failed",
            status_code=503,
            provider=primary_provider,
            category="provider_unavailable",
            retryable=False,
            attempts=total_attempts,
        )


fallback_manager = ProviderFallbackManager()

