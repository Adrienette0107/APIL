import ollama

from backend.app.services.ai.base import AIProvider


class OllamaProvider(AIProvider):

    async def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:

        options = {
            "temperature": temperature,
        }

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        response = ollama.chat(
            model=model,
            messages=messages,
            options=options,
        )

        return response["message"]["content"]