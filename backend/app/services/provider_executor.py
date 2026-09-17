from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from backend.app.core.provider_config import (
    ProviderConfig,
    get_provider_configs,
)
from backend.app.services.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)

from backend.app.services.provider_runtime import (
    circuit_breaker,
)

from backend.app.services.provider_runtime import (
    circuit_breaker,
)

logger = logging.getLogger("apil.provider_executor")


@dataclass
class ProviderExecutionResult:
    provider: str
    model: str
    content: str
    raw_response: Any = None
    attempts: int = 1


class ProviderExecutionError(Exception):
    """
    Standard APIL provider execution error.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        category: str = "provider_error",
        retryable: bool = False,
        attempts: int = 1,
    ):
        self.provider = provider
        self.category = category
        self.retryable = retryable
        self.attempts = attempts

        super().__init__(message)


class ProviderExecutor:
    """
    Central execution layer for APIL providers.

    Responsibilities:
    - Provider configuration
    - Timeout
    - Retry
    - Exponential backoff
    - Circuit breaker
    - Error normalization
    - Structured logging

    Provider-specific API logic remains inside adapters.
    """

    def __init__(self) -> None:
        self.configs = get_provider_configs()

        self.circuit_breaker = CircuitBreaker(
    failure_threshold=int(
        os.getenv(
            "APIL_CIRCUIT_FAILURE_THRESHOLD",
            "3",
        )
    ),
    recovery_timeout=float(
        os.getenv(
            "APIL_CIRCUIT_RECOVERY_TIMEOUT",
            "30",
        )
    ),
)

    def get_config(
        self,
        provider: str,
    ) -> ProviderConfig:

        config = self.configs.get(provider)

        if config is None:
            raise ProviderExecutionError(
                provider=provider,
                message=(
                    f"Provider '{provider}' "
                    "is not registered."
                ),
                category="unsupported_provider",
                retryable=False,
            )

        if not config.enabled:
            raise ProviderExecutionError(
                provider=provider,
                message=(
                    f"Provider '{provider}' "
                    "is disabled."
                ),
                category="provider_disabled",
                retryable=False,
            )

        return config

    async def execute(
        self,
        provider: str,
        model: str,
        operation: Callable[
            [],
            Awaitable[Any],
        ],
    ) -> ProviderExecutionResult:

        config = self.get_config(provider)

        # Check circuit breaker before making
        # a request to the provider.
        if not self.circuit_breaker.allow_request(
            provider
        ):
            raise ProviderExecutionError(
                provider=provider,
                message=(
                    f"Provider '{provider}' is "
                    "temporarily unavailable because "
                    "its circuit is open."
                ),
                category="circuit_open",
                retryable=True,
            )

        # max_retries means retries AFTER the
        # first attempt.
        max_attempts = max(
            1,
            min(
                config.max_retries + 1,
                4,
            ),
        )

        last_error: Optional[Exception] = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):

            try:

                logger.info(
                    "provider_execution_started",
                    extra={
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                    },
                )

                raw_response = await asyncio.wait_for(
                    operation(),
                    timeout=config.timeout,
                )

                # Successful provider request.
                self.circuit_breaker.record_success(
                    provider
                )

                content = self._extract_content(
                    raw_response
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

            except asyncio.TimeoutError as exc:

                last_error = exc

                self.circuit_breaker.record_failure(
                    provider
                )

                logger.warning(
                    "provider_execution_timeout",
                    extra={
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                    },
                )

                if attempt >= max_attempts:

                    raise ProviderExecutionError(
                        provider=provider,
                        message=(
                            f"Provider '{provider}' "
                            f"timed out after "
                            f"{attempt} attempt(s)."
                        ),
                        category="timeout",
                        retryable=True,
                        attempts=attempt,
                    ) from exc

            except Exception as exc:

                last_error = exc

                retryable = (
                    self._is_retryable_error(exc)
                )

                # Only count actual provider
                # failures against the circuit.
                if retryable:
                    self.circuit_breaker.record_failure(
                        provider
                    )

                logger.warning(
                    "provider_execution_error",
                    extra={
                        "provider": provider,
                        "model": model,
                        "attempt": attempt,
                        "retryable": retryable,
                    },
                )

                if (
                    not retryable
                    or attempt >= max_attempts
                ):

                    raise ProviderExecutionError(
                        provider=provider,
                        message=(
                            self._safe_error_message(
                                exc
                            )
                        ),
                        category=(
                            self._error_category(exc)
                        ),
                        retryable=retryable,
                        attempts=attempt,
                    ) from exc

            # Exponential backoff.
            #
            # Attempt 1 -> 1 second
            # Attempt 2 -> 2 seconds
            # Attempt 3 -> 4 seconds
            delay = min(
                2 ** (attempt - 1),
                8,
            )

            await asyncio.sleep(delay)

        raise ProviderExecutionError(
            provider=provider,
            message="Provider execution failed.",
            category="provider_error",
            retryable=True,
            attempts=max_attempts,
        ) from last_error

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:

        if response is None:
            return ""

        if isinstance(response, str):
            return response

        if isinstance(response, dict):

            # Common APIL/provider response fields.
            for key in (
                "content",
                "response",
                "text",
                "output",
                "message",
            ):

                value = response.get(key)

                if isinstance(value, str):
                    return value

                if isinstance(value, dict):

                    nested_content = value.get(
                        "content"
                    )

                    if isinstance(
                        nested_content,
                        str,
                    ):
                        return nested_content

            # OpenAI-style:
            choices = response.get("choices")

            if isinstance(choices, list):
                if choices:

                    first = choices[0]

                    if isinstance(
                        first,
                        dict,
                    ):

                        message = first.get(
                            "message"
                        )

                        if isinstance(
                            message,
                            dict,
                        ):

                            content = message.get(
                                "content"
                            )

                            if isinstance(
                                content,
                                str,
                            ):
                                return content

                        text = first.get("text")

                        if isinstance(
                            text,
                            str,
                        ):
                            return text

        # Object-style response.
        content = getattr(
            response,
            "content",
            None,
        )

        if isinstance(content, str):
            return content

        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(text, str):
            return text

        return str(response)

    @staticmethod
    def _is_retryable_error(
        error: Exception,
    ) -> bool:

        error_name = (
            error.__class__.__name__.lower()
        )

        retryable_names = {
            "timeouterror",
            "timeoutexception",
            "connectionerror",
            "connectionexception",
            "connecterror",
            "connectexception",
            "readerror",
            "readtimeout",
            "temporaryerror",
        }

        if error_name in retryable_names:
            return True

        message = str(error).lower()

        retryable_keywords = (
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
            "service unavailable",
            "too many requests",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
        )

        return any(
            keyword in message
            for keyword in retryable_keywords
        )

    @staticmethod
    def _error_category(
        error: Exception,
    ) -> str:

        message = str(error).lower()

        if "timeout" in message:
            return "timeout"

        if (
            "connection" in message
            or "connect" in message
        ):
            return "connection_error"

        if (
            "429" in message
            or "rate limit" in message
            or "too many requests" in message
        ):
            return "rate_limited"

        if any(
            code in message
            for code in (
                "500",
                "502",
                "503",
                "504",
            )
        ):
            return "provider_unavailable"

        if (
            "401" in message
            or "unauthorized" in message
            or "invalid api key" in message
        ):
            return "authentication_error"

        if (
            "400" in message
            or "invalid request" in message
            or "bad request" in message
        ):
            return "invalid_request"

        return "provider_error"

    @staticmethod
    def _safe_error_message(
        error: Exception,
    ) -> str:

        message = str(error).strip()

        if not message:
            return "Provider execution failed."

        sensitive_words = (
            "api_key",
            "apikey",
            "authorization",
            "bearer ",
            "password",
            "secret",
            "token",
        )

        lower_message = message.lower()

        if any(
            word in lower_message
            for word in sensitive_words
        ):
            return "Provider request failed."

        return message