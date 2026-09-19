
from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from ollama import AsyncClient

from backend.app.core.config import settings
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


logger = logging.getLogger("apil.ollama")


class OllamaProvider(AIProvider):
    """
    Ollama provider adapter.

    This class is responsible only for Ollama-specific
    communication and response extraction.

    ProviderExecutor is responsible for:

        - timeout
        - retry
        - backoff
        - circuit breaker
        - normalized execution errors

    ProviderFallbackManager is responsible for:

        - fallback provider selection

    APIL Pipeline is responsible for:

        - prompt optimization
        - adaptive processing
        - response optimization
    """

    name = "ollama"

    def __init__(
        self,
        host: str,
        default_model: str,
    ) -> None:

        self.client = AsyncClient(
            host=host
        )

        self.default_model = (
            default_model
        )

    # ------------------------------------------------------------------
    # Ollama options
    # ------------------------------------------------------------------

    def _get_num_predict(
        self,
        max_tokens: int | None,
    ) -> int | None:
        """
        Resolve Ollama's output-token limit.

        APIL's max_tokens value takes precedence.

        If no explicit max_tokens is supplied, the configured
        OLLAMA_NUM_PREDICT value is used.

        None means Ollama may use its model/provider default.
        """

        configured = getattr(
            settings,
            "OLLAMA_NUM_PREDICT",
            None,
        )

        try:
            configured = (
                int(configured)
                if configured is not None
                else None
            )
        except (
            TypeError,
            ValueError,
        ):
            configured = None

        if max_tokens is not None:

            try:
                max_tokens = int(
                    max_tokens
                )
            except (
                TypeError,
                ValueError,
            ):
                max_tokens = None

        if (
            max_tokens is not None
            and max_tokens > 0
        ):

            if (
                configured is not None
                and configured > 0
            ):
                return min(
                    max_tokens,
                    configured,
                )

            return max_tokens

        if (
            configured is not None
            and configured > 0
        ):
            return configured

        return None

    # ------------------------------------------------------------------
    # Thinking removal
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_response(
        content: str,
    ) -> str:
        """
        Remove accidental Qwen/Ollama thinking markup.

        Normally think=False prevents thinking from being returned
        as final content. This cleanup is a defensive final layer
        in case a model still emits <think> tags.
        """

        content = content.strip()

        if not content:
            return ""

        # Complete thinking block:
        #
        # <think>
        # internal reasoning
        # </think>
        # final answer
        #
        if "<think>" in content:

            closing_tag = "</think>"

            if closing_tag in content:

                _, _, content = content.partition(
                    closing_tag
                )

                content = content.strip()

            else:
                # If an incomplete thinking block somehow reaches
                # the adapter, do not expose it to the user.
                content = content.replace(
                    "<think>",
                    "",
                ).strip()

        # Defensive cleanup for stray closing tags.
        content = content.replace(
            "</think>",
            "",
        ).strip()

        return content

    # ------------------------------------------------------------------
    # Response extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response(
        response: Any,
    ) -> tuple[str, str | None]:
        """
        Extract final content and optional thinking content.

        Returns:

            (content, thinking)

        Thinking is deliberately not returned as the final APIL
        response.
        """

        content: Any = None
        thinking: Any = None

        # Ollama Python SDK response object.
        message = getattr(
            response,
            "message",
            None,
        )

        if message is not None:

            content = getattr(
                message,
                "content",
                None,
            )

            thinking = getattr(
                message,
                "thinking",
                None,
            )

        # Dict-style response.
        if isinstance(
            response,
            dict,
        ):

            raw_message = response.get(
                "message"
            )

            if isinstance(
                raw_message,
                dict,
            ):

                if content is None:
                    content = raw_message.get(
                        "content"
                    )

                if thinking is None:
                    thinking = raw_message.get(
                        "thinking"
                    )

            if content is None:
                content = response.get(
                    "content"
                )

            if thinking is None:
                thinking = response.get(
                    "thinking"
                )

            # Some Ollama-compatible responses may expose
            # the final text under "response".
            if content is None:
                content = response.get(
                    "response"
                )

        if not isinstance(
            content,
            str,
        ):
            content = ""

        if not isinstance(
            thinking,
            str,
        ):
            thinking = None

        return (
            content.strip(),
            thinking.strip()
            if thinking
            else None,
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

        selected_model = (
            model.strip()
            if model
            else self.default_model
        )

        if not selected_model:
            raise ProviderExecutionError(
                code="model_not_specified",
                message=(
                    "No Ollama model was specified."
                ),
                status_code=400,
                provider=self.name,
                category="model_not_specified",
                retryable=False,
            )

        num_predict = (
            self._get_num_predict(
                max_tokens
            )
        )

        options: dict[str, Any] = {
            "temperature": temperature,
        }

        if num_predict is not None:
            options["num_predict"] = (
                num_predict
            )

        input_chars = sum(
            len(
                str(
                    message.get(
                        "content",
                        "",
                    )
                )
            )
            for message in messages
            if isinstance(
                message,
                dict,
            )
        )

        logger.info(
            "Ollama request | "
            "model=%s | "
            "think=%s | "
            "stream=%s | "
            "keep_alive=%s | "
            "num_predict=%s | "
            "messages=%s | "
            "input_chars=%s",
            selected_model,
            False,
            False,
            settings.OLLAMA_KEEP_ALIVE,
            num_predict,
            len(messages),
            input_chars,
        )

        started = time.perf_counter()

        try:

            response = await self.client.chat(
                model=selected_model,
                messages=messages,

                # Critical for Qwen3:
                # do not expose model reasoning as the
                # user-visible answer.
                think=False,

                stream=False,

                keep_alive=(
                    settings.OLLAMA_KEEP_ALIVE
                ),

                options=options,
            )

        except httpx.TimeoutException as exc:

            raise ProviderExecutionError(
                code="provider_timeout",
                message=(
                    "The Ollama provider timed out."
                ),
                status_code=504,
                provider=self.name,
                category="provider_timeout",
                retryable=True,
            ) from exc

        except httpx.RequestError as exc:

            raise ProviderExecutionError(
                code="provider_connection_error",
                message=(
                    "The Ollama provider is unavailable."
                ),
                status_code=502,
                provider=self.name,
                category="provider_connection_error",
                retryable=True,
            ) from exc

        except Exception as exc:

            provider_status = getattr(
                exc,
                "status_code",
                None,
            )

            # Missing Ollama model.
            if provider_status == 404:

                raise ProviderExecutionError(
                    code="model_unavailable",
                    message=(
                        f"Ollama model "
                        f"'{selected_model}' "
                        "is unavailable."
                    ),
                    status_code=502,
                    provider=self.name,
                    category="model_unavailable",
                    retryable=False,
                ) from exc

            # Authentication/authorization.
            if provider_status in {
                401,
                403,
            }:

                raise ProviderExecutionError(
                    code="provider_authentication_failed",
                    message=(
                        "The Ollama provider "
                        "rejected the request."
                    ),
                    status_code=502,
                    provider=self.name,
                    category=(
                        "provider_authentication_failed"
                    ),
                    retryable=False,
                ) from exc

            # Other Ollama failures are normalized here.
            # ProviderExecutor decides whether/how many times
            # they should be retried.
            raise ProviderExecutionError(
                code="provider_error",
                message=(
                    "The Ollama provider "
                    "returned an error."
                ),
                status_code=502,
                provider=self.name,
                category="provider_error",
                retryable=True,
            ) from exc

        # ------------------------------------------------------------------
        # Extract response
        # ------------------------------------------------------------------

        content, thinking = (
            self._extract_response(
                response
            )
        )

        duration_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        eval_count = getattr(
            response,
            "eval_count",
            None,
        )

        prompt_eval_count = getattr(
            response,
            "prompt_eval_count",
            None,
        )

        # Dict response metadata fallback.
        if isinstance(
            response,
            dict,
        ):

            if eval_count is None:
                eval_count = response.get(
                    "eval_count"
                )

            if prompt_eval_count is None:
                prompt_eval_count = response.get(
                    "prompt_eval_count"
                )

        logger.info(
            "Ollama response | "
            "model=%s | "
            "duration_ms=%s | "
            "content_chars=%s | "
            "thinking_chars=%s | "
            "eval_count=%s | "
            "prompt_eval_count=%s",
            selected_model,
            duration_ms,
            len(content),
            len(thinking or ""),
            eval_count,
            prompt_eval_count,
        )

        # ------------------------------------------------------------------
        # Final defensive cleanup
        # ------------------------------------------------------------------

        content = self._clean_response(
            content
        )

        if not content:
            raise ProviderExecutionError(
                code="empty_provider_response",
                message=(
                    "Ollama returned an empty response."
                ),
                status_code=502,
                provider=self.name,
                category="empty_provider_response",
                retryable=True,
            )

        return content



