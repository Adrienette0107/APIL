
from __future__ import annotations

import logging
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


logger = logging.getLogger("apil.anthropic")


class AnthropicProvider(AIProvider):
    """
    Anthropic provider adapter.

    This adapter is responsible only for:
        - Anthropic SDK communication
        - converting APIL messages to Anthropic format
        - extracting Anthropic text responses
        - normalizing Anthropic-specific errors

    ProviderExecutor handles:
        - timeout
        - retry
        - exponential backoff
        - circuit breaker

    ProviderFallbackManager handles:
        - fallback provider selection
    """

    name = "anthropic"

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
                    "Anthropic provider is not configured."
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
            from anthropic import AsyncAnthropic

        except ImportError as exc:

            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message=(
                    "Anthropic SDK is not installed."
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

        self.client = AsyncAnthropic(
            **client_kwargs
        )

        return self.client

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[
        str | None,
        list[dict[str, str]],
    ]:
        """
        Convert APIL's canonical messages into Anthropic's format.

        Anthropic handles system instructions separately from the
        normal user/assistant message sequence.
        """

        system_parts: list[str] = []

        converted_messages: list[
            dict[str, str]
        ] = []

        for message in messages:

            if not isinstance(
                message,
                dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    "user",
                )
            ).strip().lower()

            content = message.get(
                "content",
                "",
            )

            if not isinstance(
                content,
                str,
            ):
                content = str(
                    content
                )

            content = content.strip()

            if not content:
                continue

            if role == "system":

                system_parts.append(
                    content
                )
                continue

            normalized_role = (
                "assistant"
                if role == "assistant"
                else "user"
            )

            converted_messages.append(
                {
                    "role": normalized_role,
                    "content": content,
                }
            )

        system_prompt = (
            "\n\n".join(
                system_parts
            ).strip()
            if system_parts
            else None
        )

        return (
            system_prompt,
            converted_messages,
        )

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
        Convert Anthropic exceptions into APIL's normalized
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
                    "No Anthropic model was specified."
                ),
                status_code=400,
                provider=self.name,
                category="model_not_specified",
                retryable=False,
            )

        (
            system_prompt,
            user_messages,
        ) = self._convert_messages(
            messages
        )

        if not user_messages:

            raise ProviderExecutionError(
                code="empty_prompt",
                message=(
                    "Anthropic received no user messages."
                ),
                status_code=400,
                provider=self.name,
                category="empty_prompt",
                retryable=False,
            )

        # Anthropic requires max_tokens.
        output_tokens = (
            max_tokens
            if max_tokens is not None
            and max_tokens > 0
            else 1024
        )

        kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": user_messages,
            "max_tokens": output_tokens,
            "temperature": temperature,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        logger.info(
            "Anthropic request | "
            "model=%s | "
            "messages=%s | "
            "system=%s | "
            "max_tokens=%s",
            selected_model,
            len(user_messages),
            bool(system_prompt),
            output_tokens,
        )

        try:

            response = await client.messages.create(
                **kwargs
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
                "Anthropic request failed | "
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
                    f"Anthropic provider error: "
                    f"{category}."
                ),
                status_code=status_code,
                provider=self.name,
                category=category,
                retryable=retryable,
            ) from exc

        # ------------------------------------------------------------------
        # Extract text blocks
        # ------------------------------------------------------------------

        content_parts: list[str] = []

        response_content = getattr(
            response,
            "content",
            [],
        )

        for block in response_content:

            text = getattr(
                block,
                "text",
                None,
            )

            if isinstance(
                text,
                str,
            ):

                text = text.strip()

                if text:
                    content_parts.append(
                        text
                    )

        content = "\n".join(
            content_parts
        ).strip()

        if not content:

            raise ProviderExecutionError(
                code="empty_provider_response",
                message=(
                    "Anthropic returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        logger.info(
            "Anthropic response | "
            "model=%s | "
            "content_chars=%s",
            selected_model,
            len(content),
        )

        return content


