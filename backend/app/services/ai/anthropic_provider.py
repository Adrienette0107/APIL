from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

from backend.app.services.ai.base import AIProvider


class AnthropicProvider(AIProvider):
    """Anthropic provider implementation."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        if not self.client:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        response = self.client.messages.create(
            model=model,
            messages=[
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in messages
            ],
            max_tokens=1024,
        )

        return response.content[0].text or ""
