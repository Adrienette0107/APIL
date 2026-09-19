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
        """Choose a generation budget from the request structure, not a giant category map."""

        prompt_lower = (prompt or "").lower()
        ceiling = max(256, min(settings.OLLAMA_NUM_PREDICT, 4096))

        if any(phrase in prompt_lower for phrase in ("one sentence", "single sentence", "briefly", "in short", "concise")):
            return min(256, ceiling)

        if any(phrase in prompt_lower for phrase in ("json", "table", "bullet", "list", "exactly five", "as json")):
            return min(512, ceiling)

        if any(phrase in prompt_lower for phrase in ("detailed", "in depth", "comprehensive", "step by step", "full explanation", "thoroughly", "architecture")):
            return min(2048, ceiling)

        if any(phrase in prompt_lower for phrase in ("code", "program", "python", "javascript", "function", "class", "script", "debug", "implementation", "patch")):
            return min(1536, ceiling)

        if len(prompt) > 2500:
            return min(2048, ceiling)

        if len(prompt) > 1200:
            return min(1024, ceiling)

        return min(768, ceiling)

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

        # Dynamically determine generation length from the request itself rather than a large static keyword list.
        num_predict = self._get_num_predict(prompt)

        options = {
            "num_predict": num_predict,
        }

        # Explicit max_tokens from the caller takes priority.
        if max_tokens is not None:
            options["num_predict"] = max_tokens

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
                    think=settings.OLLAMA_THINK,
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

        logger.info(
            "Ollama response | model=%s | duration_ms=%s | response_type=%s | "
            "content_chars=%s | thinking_chars=%s | eval_count=%s | "
            "prompt_eval_count=%s | load_duration_ms=%s | eval_duration_ms=%s",
            selected_model,
            int((time.perf_counter() - started) * 1000),
            type(response).__name__,
            len(getattr(getattr(response, "message", None), "content", None) or ""),
            len(getattr(getattr(response, "message", None), "thinking", None) or ""),
            getattr(response, "eval_count", None),
            getattr(response, "prompt_eval_count", None),
            round((getattr(response, "load_duration", 0) or 0) / 1e6),
            round((getattr(response, "eval_duration", 0) or 0) / 1e6),
        )

        # Extract response content.
        if isinstance(response, dict):
            message = response.get("message")
            content = (
                message.get("content")
                if isinstance(message, dict)
                else None
            )
        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", None)

        if not isinstance(content, str) or not content.strip():
            raise ProviderExecutionError(
                code="invalid_provider_response",
                message="The AI provider returned an empty response.",
            )

        # Remove Qwen3 thinking content if present.
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        elif "<think>" in content:
            content = content.split("<think>", 1)[0].strip()

        if not content:
            raise ProviderExecutionError(
                code="invalid_provider_response",
                message="The AI provider returned no final answer.",
            )

        return content

