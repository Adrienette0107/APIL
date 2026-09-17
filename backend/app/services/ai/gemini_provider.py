from __future__ import annotations

from typing import Any

from backend.app.core.errors import ProviderExecutionError
from backend.app.services.ai.base import AIProvider


class GeminiProvider(AIProvider):
    """Google Gemini provider implementation."""

    name = "gemini"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

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
            import google.generativeai as genai
        except ImportError as exc:
            raise ProviderExecutionError(
                code="provider_sdk_unavailable",
                message="The AI provider SDK is not installed.",
            ) from exc

        genai.configure(api_key=self.api_key)

        prompt = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        content = response.text
        if not isinstance(content, str) or not content.strip():
            raise ProviderExecutionError(
                code="invalid_provider_response",
                message="The AI provider returned an empty response.",
            )

        return content
