from __future__ import annotations

from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


class OpenAIProvider(AIProvider):
    """OpenAI provider implementation."""

    name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.client = None

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        if not self.api_key:
            raise ProviderExecutionError(
                code="provider_authentication_failed",
                message="The AI provider is not configured.",
                status_code=502,
            )

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message="The AI provider SDK is not installed.",
            ) from exc

        if self.client is None:
            self.client = AsyncOpenAI(api_key=self.api_key)

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ProviderExecutionError(
                code="invalid_provider_response",
                message="The AI provider returned an empty response.",
            )

        return content
