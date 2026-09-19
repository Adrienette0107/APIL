import asyncio
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

    def __init__(self, host: str, default_model: str):
        self.client = AsyncClient(host=host)
        self.default_model = default_model
        self.timeout_seconds = settings.OLLAMA_TIMEOUT_SECONDS

    def _get_num_predict(self, prompt: str) -> int:
        """Return the configured provider ceiling when no semantic override exists."""

        del prompt
        return max(256, settings.OLLAMA_NUM_PREDICT)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:

        # Extract the actual prompt sent to the provider.
        prompt = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )

        # Dynamically determine generation length.
        configured_num_predict = self._get_num_predict(prompt)
        num_predict = configured_num_predict

        options = {
            "num_predict": num_predict,
            "temperature": temperature,
        }

        # Explicit max_tokens from the caller takes priority.
        if max_tokens is not None:
            options["num_predict"] = min(max_tokens, configured_num_predict)

        selected_model = model or self.default_model

        input_chars = sum(
            len(str(message.get("content", "")))
            for message in messages
            if isinstance(message, dict)
        )

        logger.info(
            "Ollama request | model=%s | think=%s | stream=%s | keep_alive=%s | "
            "num_predict=%s | messages=%s | input_chars=%s | input_tokens~=%s",
            selected_model,
            settings.OLLAMA_THINK,
            False,
            settings.OLLAMA_KEEP_ALIVE,
            options["num_predict"],
            len(messages),
            input_chars,
            (input_chars + 3) // 4,
        )

        started = time.perf_counter()

        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    model=selected_model,
                    messages=messages,
                    think=False,
                    stream=False,
                    keep_alive=settings.OLLAMA_KEEP_ALIVE,
                    options=options,
                ),
                timeout=self.timeout_seconds,
            )

        except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderExecutionError(
                code="provider_timeout",
                message="The AI provider timed out.",
                status_code=504,
            ) from exc

        except httpx.RequestError as exc:
            raise ProviderExecutionError(
                code="provider_unavailable",
                message="The AI provider is unavailable.",
            ) from exc

        except Exception as exc:
            provider_status = getattr(exc, "status_code", None)

            if provider_status == 404:
                raise ProviderExecutionError(
                    code="model_unavailable",
                    message="The selected AI model is unavailable.",
                    status_code=502,
                ) from exc

            if provider_status in {401, 403}:
                raise ProviderExecutionError(
                    code="provider_authentication_failed",
                    message="The AI provider rejected its credentials.",
                    status_code=502,
                ) from exc

            raise ProviderExecutionError(
                code="provider_error",
                message="The AI provider returned an error.",
            ) from exc

        message = getattr(response, "message", None)

        response_content = None
        response_thinking = None

        if message is not None:
            response_content = getattr(message, "content", None)
            response_thinking = getattr(message, "thinking", None)

        if isinstance(response, dict):
            raw_message = response.get("message")

            if isinstance(raw_message, dict):
                response_content = raw_message.get("content")
                response_thinking = raw_message.get("thinking")
            elif response_content is None:
                response_content = response.get("content")

        logger.info(
            "Ollama response | model=%s | duration_ms=%s | response_type=%s | "
            "content_chars=%s | thinking_chars=%s | eval_count=%s | "
            "prompt_eval_count=%s | load_duration_ms=%s | eval_duration_ms=%s",
            selected_model,
            int((time.perf_counter() - started) * 1000),
            type(response).__name__,
            len(response_content or "")
            if isinstance(response_content, str)
            else 0,
            len(response_thinking or "")
            if isinstance(response_thinking, str)
            else 0,
            getattr(response, "eval_count", None),
            getattr(response, "prompt_eval_count", None),
            round((getattr(response, "load_duration", 0) or 0) / 1e6),
            round((getattr(response, "eval_duration", 0) or 0) / 1e6),
        )

        # Only use the final content.
        # Never concatenate or return the provider's thinking field.
        content = response_content

        if not isinstance(content, str):
            content = ""

        content = content.strip()

        # Remove explicit thinking blocks if they appear inside content.
        if "<think>" in content and "</think>" in content:
            _, _, content = content.partition("</think>")
            content = content.strip()

        elif content.startswith("<think>"):
            content = content.replace("<think>", "", 1).strip()

        if not content:
            raise ProviderExecutionError(
                code="empty_provider_response",
                message="The AI provider returned an empty response.",
                status_code=502,
            )

        return content

