
from __future__ import annotations

import asyncio
import logging
from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


logger = logging.getLogger("apil.gemini")


class GeminiProvider(AIProvider):
    """
    Google Gemini provider adapter.

    ProviderExecutor owns:
        - timeout
        - retry
        - backoff
        - circuit breaker

    ProviderFallbackManager owns:
        - fallback provider selection

    This adapter owns only Gemini-specific communication and
    response normalization.
    """

    name = "gemini"

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

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        messages: list[dict[str, Any]],
    ) -> str:
        """
        Convert APIL's canonical chat messages into a Gemini
        compatible text prompt.

        The original roles are preserved explicitly.
        """

        parts: list[str] = []

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
            ).strip()

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
                label = "System"

            elif role == "assistant":
                label = "Assistant"

            else:
                label = "User"

            parts.append(
                f"{label}: {content}"
            )

        return "\n\n".join(
            parts
        ).strip()

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
        Convert Gemini exceptions into APIL's normalized error model.
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code is None:
            status_code = getattr(
                exc,
                "code",
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
        ):

            return (
                "provider_connection_error",
                "provider_connection_error",
                502,
                True,
            )

        # Unknown provider errors are considered retryable because
        # they may represent transient Gemini failures.
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

        if not self.api_key:
            raise ProviderExecutionError(
                code="provider_authentication_failed",
                message=(
                    "Gemini provider is not configured."
                ),
                status_code=502,
                provider=self.name,
                category=(
                    "provider_authentication_failed"
                ),
                retryable=False,
            )

        selected_model = (
            model.strip()
            if model
            else ""
        )

        if not selected_model:
            raise ProviderExecutionError(
                code="model_not_specified",
                message=(
                    "No Gemini model was specified."
                ),
                status_code=400,
                provider=self.name,
                category="model_not_specified",
                retryable=False,
            )

        prompt = self._build_prompt(
            messages
        )

        if not prompt:
            raise ProviderExecutionError(
                code="empty_prompt",
                message=(
                    "Gemini received an empty prompt."
                ),
                status_code=400,
                provider=self.name,
                category="empty_prompt",
                retryable=False,
            )

        try:

            import google.generativeai as genai

        except ImportError as exc:

            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message=(
                    "Gemini SDK is not installed."
                ),
                status_code=502,
                provider=self.name,
                category="provider_sdk_unavailable",
                retryable=False,
            ) from exc

        try:

            genai.configure(
                api_key=self.api_key
            )

            generation_config: dict[
                str,
                Any,
            ] = {
                "temperature": temperature,
            }

            if max_tokens is not None:
                generation_config[
                    "max_output_tokens"
                ] = max_tokens

            model_obj = (
                genai.GenerativeModel(
                    selected_model
                )
            )

            logger.info(
                "Gemini request | "
                "model=%s | "
                "messages=%s | "
                "input_chars=%s",
                selected_model,
                len(messages),
                len(prompt),
            )

            response = (
                await asyncio.to_thread(
                    model_obj.generate_content,
                    prompt,
                    generation_config=(
                        generation_config
                    ),
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
                "Gemini request failed | "
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
                    f"Gemini provider error: "
                    f"{category}."
                ),
                status_code=status_code,
                provider=self.name,
                category=category,
                retryable=retryable,
            ) from exc

        # ------------------------------------------------------------------
        # Extract final response
        # ------------------------------------------------------------------

        content = getattr(
            response,
            "text",
            None,
        )

        if not isinstance(
            content,
            str,
        ):

            raise ProviderExecutionError(
                code="invalid_provider_response",
                message=(
                    "Gemini returned an invalid response."
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
                    "Gemini returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        logger.info(
            "Gemini response | "
            "model=%s | "
            "content_chars=%s",
            selected_model,
            len(content),
        )

        return content


