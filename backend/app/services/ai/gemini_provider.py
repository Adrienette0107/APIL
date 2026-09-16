from __future__ import annotations

import os
from typing import Any

import google.generativeai as genai

from backend.app.services.ai.base import AIProvider


class GeminiProvider(AIProvider):
    """Google Gemini provider implementation."""

    name = "gemini"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        prompt = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in messages
        )
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        return response.text or ""
