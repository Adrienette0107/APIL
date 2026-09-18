import asyncio
from typing import Any

import httpx
from ollama import AsyncClient

from backend.app.core.config import settings
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


class OllamaProvider(AIProvider):

    def __init__(self, host: str, default_model: str):
        self.client = AsyncClient(host=host)
        self.default_model = default_model
        self.timeout_seconds = settings.OLLAMA_TIMEOUT_SECONDS

    def _get_num_predict(self, prompt: str) -> int:
        """
        Dynamically determine the maximum number of generated tokens
        based on the complexity and expected output type of the prompt.
        """

        prompt_lower = prompt.lower()

        # Code / programming requests generally need more output.
        code_keywords = [
            "code",
            "program",
            "python",
            "javascript",
            "java",
            "c++",
            "api",
            "function",
            "class",
            "implement",
            "build",
            "debug",
            "sql",
            "html",
            "css",
            "react",
            "fastapi",
        ]

        # Detailed requests need more generation space.
        detailed_keywords = [
            "detailed",
            "in detail",
            "comprehensive",
            "step by step",
            "deep explanation",
            "explain everything",
            "thoroughly",
            "architecture",
            "complete",
        ]

        if any(keyword in prompt_lower for keyword in code_keywords):
            return 8192

        if any(keyword in prompt_lower for keyword in detailed_keywords):
            return 8192

        # Long prompts can indicate more complex tasks.
        if len(prompt) > 3000:
            return 8192

        if len(prompt) > 1000:
            return 6144

        # Normal questions.
        return 4096

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
        
        num_predict = self._get_num_predict(prompt)

        options = {
            "num_predict": num_predict,
}

        # Explicit max_tokens from the caller takes priority.
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        selected_model = model or self.default_model

        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    model=selected_model,
                    messages=messages,
                    think=False,
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

