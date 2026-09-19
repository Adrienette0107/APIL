from __future__ import annotations

import logging
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


logger = logging.getLogger("apil.openai")


class OpenAIProvider(AIProvider):
    """
    OpenAI provider adapter.

    Responsibilities:
        - OpenAI SDK communication
        - OpenAI response extraction
        - OpenAI-specific error classification

    ProviderExecutor handles:
        - timeout
        - retries
        - backoff
        - circuit breaker

    ProviderFallbackManager handles:
        - fallback provider selection
    """

    name = "openai"

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

        self.client = None

    # ------------------------------------------------------------------
    # Client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """
        Lazily create the OpenAI client.

        Lazy loading keeps cloud SDK initialization out of APIL startup
        when OpenAI is not being used.
        """

        if not self.api_key:
            raise ProviderExecutionError(
                code="provider_authentication_failed",
                message=(
                    "OpenAI provider is not configured."
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
            from openai import AsyncOpenAI

        except ImportError as exc:

            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message=(
                    "OpenAI SDK is not installed."
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
            client_kwargs["base_url"] = (
                self.base_url
            )

        self.client = AsyncOpenAI(
            **client_kwargs
        )

        return self.client

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """
        Extract final text from an OpenAI response.
        """

        try:
            choices = response.choices
        except AttributeError:
            choices = None

        if not choices:
            return ""

        first_choice = choices[0]

        message = getattr(
            first_choice,
            "message",
            None,
        )

        if message is None:
            return ""

        content = getattr(
            message,
            "content",
            None,
        )

        # Standard text response.
        if isinstance(
            content,
            str,
        ):
            return content.strip()

        # Some newer/compatible response shapes may expose
        # structured content blocks.
        if isinstance(
            content,
            list,
        ):

            parts: list[str] = []

            for item in content:

                if isinstance(
                    item,
                    dict,
                ):

                    text = item.get(
                        "text"
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        parts.append(text)

                else:

                    text = getattr(
                        item,
                        "text",
                        None,
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        parts.append(text)

            return "\n".join(
                parts
            ).strip()

        return ""

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
        Convert an OpenAI exception into:

            code
            category
            HTTP status
            retryable
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code == 401:

            return (
                "provider_authentication_failed",
                "provider_authentication_failed",
                502,
                False,
            )

        if status_code == 403:

            return (
                "provider_authentication_failed",
                "provider_authentication_failed",
                502,
                False,
            )

        if status_code == 400:

            return (
                "provider_bad_request",
                "provider_bad_request",
                400,
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
                    "No OpenAI model was specified."
                ),
                status_code=400,
                provider=self.name,
                category="model_not_specified",
                retryable=False,
            )

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        logger.info(
            "OpenAI request | model=%s | messages=%s",
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
                "OpenAI request failed | "
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
                    f"OpenAI provider error: "
                    f"{category}."
                ),
                status_code=status_code,
                provider=self.name,
                category=category,
                retryable=retryable,
            ) from exc

        content = self._extract_content(
            response
        )

        if not content:

            raise ProviderExecutionError(
                code="empty_provider_response",
                message=(
                    "OpenAI returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        logger.info(
            "OpenAI response | "
            "model=%s | "
            "content_chars=%s",
            selected_model,
            len(content),
        )

        return content


