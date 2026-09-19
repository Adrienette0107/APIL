from __future__ import annotations

import logging
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


logger = logging.getLogger("apil.groq")


class GroqProvider(AIProvider):
    """
    Groq provider adapter.

    This adapter handles only Groq-specific API communication
    and response normalization.

    ProviderExecutor handles:
        - timeout
        - retry
        - exponential backoff
        - circuit breaker

    ProviderFallbackManager handles:
        - fallback provider selection
    """

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:

        self.api_key = (
            api_key.strip()
            if api_key
            else None
        )

        self.base_url = (
            base_url.strip()
            if base_url
            else None
        )

        self.client: Any = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:

        if not self.api_key:
            raise ProviderExecutionError(
                code="provider_authentication_failed",
                message=(
                    "Groq provider is not configured."
                ),
                status_code=502,
                provider=self.name,
                category=(
                    "provider_authentication_failed"
                ),
                retryable=False,
            )

        if self.client is not None:
            return self.client

        try:
            from groq import AsyncGroq

        except ImportError as exc:

            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message=(
                    "Groq SDK is not installed."
                ),
                status_code=502,
                provider=self.name,
                category="provider_sdk_unavailable",
                retryable=False,
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
        }

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = AsyncGroq(
            **client_kwargs
        )

        return self.client

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_error(
        exc: Exception,
    ) -> tuple[
        str,
        str,
        int,
        bool,
    ]:
        """
        Convert Groq exceptions into APIL's normalized
        ProviderExecutionError categories.
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code == 400:

            return (
                "provider_bad_request",
                "provider_bad_request",
                400,
                False,
            )

        if status_code in {
            401,
            403,
        }:

            return (
                "provider_authentication_failed",
                "provider_authentication_failed",
                502,
                False,
            )

        if status_code == 404:

            return (
                "model_unavailable",
                "model_unavailable",
                502,
                False,
            )

        if status_code == 429:

            return (
                "provider_rate_limited",
                "provider_rate_limited",
                429,
                True,
            )

        if (
            isinstance(
                status_code,
                int,
            )
            and status_code >= 500
        ):

            return (
                "provider_server_error",
                "provider_server_error",
                502,
                True,
            )

        error_name = (
            type(exc).__name__.lower()
        )

        message = str(
            exc
        ).lower()

        if (
            "timeout" in error_name
            or "timed out" in message
        ):

            return (
                "provider_timeout",
                "provider_timeout",
                504,
                True,
            )

        if (
            "connection" in error_name
            or "connection error" in message
            or "connection refused" in message
        ):

            return (
                "provider_connection_error",
                "provider_connection_error",
                502,
                True,
            )

        return (
            "provider_error",
            "provider_error",
            502,
            True,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:

        client = self._get_client()

        selected_model = (
            model.strip()
            if model
            else ""
        )

        if not selected_model:

            raise ProviderExecutionError(
                code="model_not_specified",
                message=(
                    "No Groq model was specified."
                ),
                status_code=400,
                provider=self.name,
                category="model_not_specified",
                retryable=False,
            )

        if not messages:

            raise ProviderExecutionError(
                code="empty_prompt",
                message=(
                    "Groq received no messages."
                ),
                status_code=400,
                provider=self.name,
                category="empty_prompt",
                retryable=False,
            )

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:

            try:
                parsed_max_tokens = int(
                    max_tokens
                )
            except (
                TypeError,
                ValueError,
            ):
                parsed_max_tokens = None

            if (
                parsed_max_tokens is not None
                and parsed_max_tokens > 0
            ):
                kwargs["max_tokens"] = (
                    parsed_max_tokens
                )

        logger.info(
            "Groq request | "
            "model=%s | "
            "messages=%s",
            selected_model,
            len(messages),
        )

        try:

            response = (
                await client.chat.completions.create(
                    **kwargs
                )
            )

        except Exception as exc:

            (
                code,
                category,
                status_code,
                retryable,
            ) = self._classify_error(
                exc
            )

            logger.warning(
                "Groq request failed | "
                "model=%s | "
                "category=%s | "
                "retryable=%s",
                selected_model,
                category,
                retryable,
            )

            raise ProviderExecutionError(
                code=code,
                message=(
                    f"Groq provider error: "
                    f"{category}."
                ),
                status_code=status_code,
                provider=self.name,
                category=category,
                retryable=retryable,
            ) from exc

        # ------------------------------------------------------------------
        # Extract response
        # ------------------------------------------------------------------

        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:

            raise ProviderExecutionError(
                code="empty_provider_response",
                message=(
                    "Groq returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        first_choice = choices[0]

        message = getattr(
            first_choice,
            "message",
            None,
        )

        if message is None:

            raise ProviderExecutionError(
                code="invalid_provider_response",
                message=(
                    "Groq returned an invalid response."
                ),
                status_code=502,
                provider=self.name,
                category="invalid_provider_response",
                retryable=True,
            )

        content = getattr(
            message,
            "content",
            None,
        )

        if not isinstance(
            content,
            str,
        ):

            raise ProviderExecutionError(
                code="invalid_provider_response",
                message=(
                    "Groq returned an invalid response."
                ),
                status_code=502,
                provider=self.name,
                category="invalid_provider_response",
                retryable=True,
            )

        content = content.strip()

        if not content:

            raise ProviderExecutionError(
                code="empty_provider_response",
                message=(
                    "Groq returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        logger.info(
            "Groq response | "
            "model=%s | "
            "content_chars=%s",
            selected_model,
            len(content),
        )

        return content

