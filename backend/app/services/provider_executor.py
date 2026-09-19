from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend.app.core.errors import ProviderExecutionError
from backend.app.core.provider_config import (
    ProviderConfig,
    get_provider_configs,
)
from backend.app.services.circuit_breaker import CircuitBreaker
from backend.app.services.provider_runtime import (
    shared_circuit_breaker,
)

logger = logging.getLogger("apil.provider_executor")


@dataclass
class ProviderExecutionResult:
    """
    Normalized result returned by ProviderExecutor.
    """

    provider: str
    model: str
    content: str
    raw_response: Any = None
    attempts: int = 1


class ProviderExecutor:
    """
    Central execution layer for all AI providers.

    Responsibilities:
        - provider configuration validation
        - timeout protection
        - retry handling
        - exponential backoff
        - circuit breaker integration
        - provider error normalization
        - response normalization
        - attempt tracking

    Provider-specific SDK logic stays inside provider adapters.

    Architecture:

        APIL Pipeline
              ↓
        ProviderFallbackManager
              ↓
        ProviderExecutor
              ↓
        AIProvider adapter
              ↓
        OpenAI / Gemini / Anthropic / Groq / Ollama
    """

    def __init__(
        self,
        configs: dict[str, ProviderConfig] | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:

        self.configs = (
            configs
            if configs is not None
            else get_provider_configs()
        )

        # Use the shared process-level circuit breaker.
        #
        # This is important. Creating a new breaker for every
        # request would destroy circuit-breaker state.
        self.circuit_breaker = (
            circuit_breaker
            if circuit_breaker is not None
            else shared_circuit_breaker
        )

    # ==================================================================
    # PROVIDER CONFIGURATION
    # ==================================================================

    def get_config(
        self,
        provider: str,
    ) -> ProviderConfig:

        provider = (
            provider.strip().lower()
            if provider
            else ""
        )

        config = self.configs.get(provider)

        if config is None:
            raise ProviderExecutionError(
                code="provider_not_registered",
                message=(
                    f"Provider '{provider}' is not registered."
                ),
                status_code=502,
                provider=provider,
                category="provider_not_registered",
                retryable=False,
            )

        if not config.enabled:
            raise ProviderExecutionError(
                code="provider_disabled",
                message=(
                    f"Provider '{provider}' is disabled."
                ),
                status_code=503,
                provider=provider,
                category="provider_disabled",
                retryable=False,
            )

        return config

    # ==================================================================
    # MAIN EXECUTION
    # ==================================================================

    async def execute(
        self,
        provider: str,
        model: str,
        operation: Callable[
            [],
            Awaitable[Any],
        ],
    ) -> ProviderExecutionResult:
        """
        Execute a provider operation with timeout, retry and
        circuit-breaker protection.

        Parameters
        ----------
        provider:
            Provider name, e.g. "ollama", "openai".

        model:
            Provider-specific model name.

        operation:
            Async callable that performs the actual provider request.

        Returns
        -------
        ProviderExecutionResult
        """

        provider = (
            provider.strip().lower()
            if provider
            else ""
        )

        model = (
            model.strip()
            if model
            else ""
        )

        config = self.get_config(provider)

        # --------------------------------------------------------------
        # Circuit breaker
        # --------------------------------------------------------------

        if not self.circuit_breaker.allow_request(
            provider
        ):
            logger.warning(
                "provider_circuit_open",
                extra={
                    "provider": provider,
                    "model": model,
                },
            )

            raise ProviderExecutionError(
                code="circuit_open",
                message=(
                    f"Provider '{provider}' is temporarily "
                    "unavailable because its circuit breaker "
                    "is open."
                ),
                status_code=503,
                provider=provider,
                category="circuit_open",
                retryable=True,
            )

        # --------------------------------------------------------------
        # Retry count
        # --------------------------------------------------------------

        try:
            configured_retries = int(
                config.max_retries
            )
        except (
            TypeError,
            ValueError,
        ):
            configured_retries = 0

        # Protect the system from accidental huge retry counts.
        max_attempts = max(
            1,
            min(
                configured_retries + 1,
                4,
            ),
        )

        last_error: ProviderExecutionError | None = None

        # ==============================================================
        # EXECUTION LOOP
        # ==============================================================

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            logger.info(
                "provider_execution_started",
                extra={
                    "provider": provider,
                    "model": model,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                },
            )

            try:

                # ------------------------------------------------------
                # Provider timeout
                # ------------------------------------------------------

                try:
                    timeout_seconds = float(
                        config.timeout
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    timeout_seconds = 120.0

                raw_response = await asyncio.wait_for(
                    operation(),
                    timeout=timeout_seconds,
                )

                # ------------------------------------------------------
                # Extract final text
                # ------------------------------------------------------

                content = self._extract_content(
                    raw_response
                )

                if not content:
                    raise ProviderExecutionError(
                        code="empty_provider_response",
                        message=(
                            f"Provider '{provider}' returned "
                            "an empty response."
                        ),
                        status_code=502,
                        provider=provider,
                        category="empty_provider_response",
                        retryable=True,
                        attempts=attempt,
                    )

                # ------------------------------------------------------
                # Success
                # ------------------------------------------------------

                self.circuit_breaker.record_success(
                    provider
                )

                logger.info(
                    "provider_execution_success",
                    extra={
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                    },
                )

                return ProviderExecutionResult(
                    provider=provider,
                    model=model,
                    content=content,
                    raw_response=raw_response,
                    attempts=attempt,
                )

            # ==========================================================
            # TIMEOUT
            # ==========================================================

            except asyncio.TimeoutError as exc:

                last_error = ProviderExecutionError(
                    code="provider_timeout",
                    message=(
                        f"Provider '{provider}' timed out "
                        f"after {config.timeout} seconds."
                    ),
                    status_code=504,
                    provider=provider,
                    category="provider_timeout",
                    retryable=True,
                    attempts=attempt,
                )

                self.circuit_breaker.record_failure(
                    provider
                )

                logger.warning(
                    "provider_timeout",
                    extra={
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )

                if attempt >= max_attempts:
                    raise last_error from exc

            # ==========================================================
            # OUR NORMALIZED PROVIDER ERROR
            # ==========================================================

            except ProviderExecutionError as exc:

                # Make sure provider metadata is always available.
                if exc.provider is None:
                    exc.provider = provider

                if not exc.category:
                    exc.category = exc.code

                if exc.attempts < attempt:
                    exc.attempts = attempt

                last_error = exc

                # ------------------------------------------------------
                # Non-retryable provider errors
                # ------------------------------------------------------

                if not exc.retryable:

                    logger.error(
                        "provider_non_retryable_error",
                        extra={
                            "provider": provider,
                            "model": model,
                            "category": exc.category,
                            "code": exc.code,
                            "attempt": attempt,
                        },
                    )

                    raise exc

                # ------------------------------------------------------
                # Retryable provider errors
                # ------------------------------------------------------

                self.circuit_breaker.record_failure(
                    provider
                )

                logger.warning(
                    "provider_retryable_error",
                    extra={
                        "provider": provider,
                        "model": model,
                        "category": exc.category,
                        "code": exc.code,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )

                if attempt >= max_attempts:
                    raise exc

            # ==========================================================
            # UNKNOWN / SDK ERROR
            # ==========================================================

            except Exception as exc:

                retryable = self._is_retryable_error(
                    exc
                )

                category = self._error_category(
                    exc
                )

                safe_message = (
                    self._safe_error_message(
                        exc
                    )
                )

                normalized_error = (
                    ProviderExecutionError(
                        code=category,
                        message=safe_message,
                        status_code=(
                            self._status_code_for_category(
                                category
                            )
                        ),
                        provider=provider,
                        category=category,
                        retryable=retryable,
                        attempts=attempt,
                    )
                )

                last_error = normalized_error

                if retryable:
                    self.circuit_breaker.record_failure(
                        provider
                    )

                logger.warning(
                    "provider_unknown_error",
                    extra={
                        "provider": provider,
                        "model": model,
                        "category": category,
                        "retryable": retryable,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )

                if not retryable:
                    raise normalized_error from exc

                if attempt >= max_attempts:
                    raise normalized_error from exc

            # ==========================================================
            # BACKOFF
            # ==========================================================

            delay = min(
                2 ** (attempt - 1),
                8,
            )

            logger.info(
                "provider_retry_backoff",
                extra={
                    "provider": provider,
                    "model": model,
                    "delay_seconds": delay,
                    "next_attempt": attempt + 1,
                },
            )

            await asyncio.sleep(delay)

        # ==================================================================
        # DEFENSIVE FINAL ERROR
        # ==================================================================

        if last_error is not None:
            raise last_error

        raise ProviderExecutionError(
            code="provider_execution_failed",
            message=(
                f"Provider '{provider}' execution failed."
            ),
            status_code=502,
            provider=provider,
            category="provider_execution_failed",
            retryable=False,
            attempts=max_attempts,
        )

    # ==================================================================
    # RESPONSE EXTRACTION
    # ==================================================================

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """
        Convert common provider response structures into
        a normalized string.

        Provider adapters normally already return strings,
        but this keeps the executor resilient to generic
        provider implementations.
        """

        if response is None:
            return ""

        # --------------------------------------------------------------
        # Plain string
        # --------------------------------------------------------------

        if isinstance(response, str):
            return response.strip()

        # --------------------------------------------------------------
        # Dictionary responses
        # --------------------------------------------------------------

        if isinstance(response, dict):

            for key in (
                "content",
                "response",
                "text",
                "output",
            ):
                value = response.get(key)

                if isinstance(value, str):
                    return value.strip()

            # message = {"content": "..."}
            message = response.get(
                "message"
            )

            if isinstance(message, str):
                return message.strip()

            if isinstance(message, dict):

                content = message.get(
                    "content"
                )

                if isinstance(content, str):
                    return content.strip()

            # OpenAI-style:
            #
            # {
            #   "choices": [
            #       {
            #           "message": {
            #               "content": "..."
            #           }
            #       }
            #   ]
            # }
            choices = response.get(
                "choices"
            )

            if isinstance(
                choices,
                list,
            ) and choices:

                first_choice = choices[0]

                if isinstance(
                    first_choice,
                    dict,
                ):

                    choice_message = (
                        first_choice.get(
                            "message"
                        )
                    )

                    if isinstance(
                        choice_message,
                        dict,
                    ):

                        content = (
                            choice_message.get(
                                "content"
                            )
                        )

                        if isinstance(
                            content,
                            str,
                        ):
                            return content.strip()

                    text = first_choice.get(
                        "text"
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        return text.strip()

        # --------------------------------------------------------------
        # Object-based SDK responses
        # --------------------------------------------------------------

        content = getattr(
            response,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(
            text,
            str,
        ):
            return text.strip()

        message = getattr(
            response,
            "message",
            None,
        )

        if message is not None:

            message_content = getattr(
                message,
                "content",
                None,
            )

            if isinstance(
                message_content,
                str,
            ):
                return message_content.strip()

        return ""

    # ==================================================================
    # RETRY CLASSIFICATION
    # ==================================================================

    @staticmethod
    def _is_retryable_error(
        exc: Exception,
    ) -> bool:

        error_name = (
            type(exc).__name__.lower()
        )

        message = str(exc).lower()

        retryable_class_names = (
            "timeout",
            "connection",
            "connect",
            "temporarily",
            "rate",
            "server",
            "serviceunavailable",
        )

        if any(
            keyword in error_name
            for keyword in retryable_class_names
        ):
            return True

        retryable_messages = (
            "timeout",
            "timed out",
            "connection reset",
            "connection refused",
            "connection error",
            "temporarily unavailable",
            "temporary failure",
            "rate limit",
            "too many requests",
            "429",
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "bad gateway",
            "service unavailable",
        )

        return any(
            keyword in message
            for keyword in retryable_messages
        )

    # ==================================================================
    # ERROR CATEGORY
    # ==================================================================

    @staticmethod
    def _error_category(
        exc: Exception,
    ) -> str:

        error_name = (
            type(exc).__name__.lower()
        )

        message = str(exc).lower()

        if (
            "timeout" in error_name
            or "timed out" in message
        ):
            return "provider_timeout"

        if (
            "connection" in error_name
            or "connection error" in message
            or "connection refused" in message
            or "connection reset" in message
        ):
            return "provider_connection_error"

        if (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        ):
            return "provider_rate_limited"

        if any(
            code in message
            for code in (
                "500",
                "502",
                "503",
                "504",
            )
        ):
            return "provider_server_error"

        if (
            "401" in message
            or "403" in message
            or "unauthorized" in message
            or "forbidden" in message
            or "authentication" in message
        ):
            return "provider_authentication_error"

        if (
            "400" in message
            or "bad request" in message
            or "invalid request" in message
        ):
            return "provider_bad_request"

        return "provider_error"

    # ==================================================================
    # STATUS CODE MAPPING
    # ==================================================================

    @staticmethod
    def _status_code_for_category(
        category: str,
    ) -> int:

        mapping = {
            "provider_timeout": 504,
            "provider_connection_error": 502,
            "provider_rate_limited": 429,
            "provider_server_error": 502,
            "provider_authentication_error": 502,
            "provider_bad_request": 400,
            "provider_error": 502,
        }

        return mapping.get(
            category,
            502,
        )

    # ==================================================================
    # SAFE ERROR MESSAGE
    # ==================================================================

    @staticmethod
    def _safe_error_message(
        exc: Exception,
    ) -> str:

        message = str(exc).strip()

        if not message:
            return "Provider execution failed."

        sensitive_keywords = (
            "api_key",
            "apikey",
            "authorization",
            "password",
            "secret",
            "access_token",
            "refresh_token",
            "bearer ",
            "token=",
        )

        lowered = message.lower()

        if any(
            keyword in lowered
            for keyword in sensitive_keywords
        ):
            return (
                "Provider execution failed due to "
                "a protected credential or "
                "authentication error."
            )

        # Prevent huge SDK exceptions from entering
        # API responses/logging.
        return message[:1000]