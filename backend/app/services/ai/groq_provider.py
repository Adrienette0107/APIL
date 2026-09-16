from __future__ import annotations

import os
from typing import Any

from groq import Groq

from backend.app.services.ai.base import AIProvider


class GroqProvider(AIProvider):
    """Groq provider implementation."""

    name = "groq"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        if not self.client:
            raise ValueError("GROQ_API_KEY is not configured.")

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )

        return response.choices[0].message.content or ""
