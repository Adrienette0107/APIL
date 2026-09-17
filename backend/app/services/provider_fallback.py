from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from backend.app.core.provider_config import (
    ProviderConfig,
    get_provider_configs,
)

from backend.app.services.provider_executor import (
    ProviderExecutionError,
    ProviderExecutionResult,
    ProviderExecutor,
)

logger = logging.getLogger("apil.provider_fallback")


class ProviderFallbackManager:
    """
    Manages controlled fallback between APIL providers.

    Example:

        Ollama fails
             ↓
        fallback allowed?
             ↓
        OpenAI
             ↓
        Gemini
             ↓
        final APIL error
    """

    def __init__(
        self,
        executor: ProviderExecutor | None = None,
    ) -> None:

        self.configs: dict[
            str,
            ProviderConfig,
        ] = get_provider_configs()

        self.executor = (
            executor
            or ProviderExecutor()
        )

    def get_fallback_providers(
        self,
        primary_provider: str,
    ) -> list[str]:

        candidates: list[
            ProviderConfig
        ] = []

        for (
            provider_name,
            config,
        ) in self.configs.items():

            if provider_name == primary_provider:
                continue

            if not config.enabled:
                continue

            if not config.fallback_enabled:
                continue

            candidates.append(config)

        # Lower priority number = higher priority.
        candidates.sort(
            key=lambda item: item.priority
        )

        return [
            config.name
            for config in candidates
        ]

    def get_model_for_provider(
        self,
        provider: str,
        requested_model: str,
    ) -> str:

        config = self.configs.get(provider)

        if config is None:
            return requested_model

        # If the request uses auto, use the
        # provider's configured default model.
        if (
            not requested_model
            or requested_model == "auto"
        ):

            if config.default_model:
                return config.default_model

        return requested_model

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
        Execute against the primary provider.

        If a retryable provider failure occurs,
        APIL can move to the next eligible provider.

        operation_factory receives:

            provider
            model

        and must return an awaitable provider response.
        """

        providers = [
            primary_provider,
            *self.get_fallback_providers(
                primary_provider
            ),
        ]

        last_error: ProviderExecutionError | None = None

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

            provider_model = (
                self.get_model_for_provider(
                    provider,
                    model,
                )
            )

            try:

                logger.info(
                    "provider_attempt",
                    extra={
                        "provider": provider,
                        "model": provider_model,
                        "fallback": index > 0,
                    },
                )

                result = await (
                    self.executor.execute(
                        provider=provider,
                        model=provider_model,
                        operation=lambda
                        p=provider,
                        m=provider_model: (
                            operation_factory(
                                p,
                                m,
                            )
                        ),
                    )
                )

                return result

            except ProviderExecutionError as exc:

                last_error = exc

                logger.warning(
                    "provider_failed",
                    extra={
                        "provider": provider,
                        "category": exc.category,
                        "retryable": exc.retryable,
                    },
                )

                # Never fallback for errors such as:
                #
                # - invalid request
                # - authentication failure
                # - unsupported provider
                # - disabled provider
                #
                if not exc.retryable:
                    raise

                # Continue to next provider.
                if index < len(providers) - 1:

                    logger.warning(
                        "provider_fallback_started",
                        extra={
                            "failed_provider": provider,
                            "next_provider": providers[
                                index + 1
                            ],
                        },
                    )

        if last_error:
            raise last_error

        raise ProviderExecutionError(
            provider=primary_provider,
            message="No provider is available.",
            category="no_provider_available",
            retryable=False,
        )