from __future__ import annotations


class PromptOptimizer:

    def optimize(
        self,
        prompt: str,
        preferences: dict | None = None,
    ) -> dict:

        original_prompt = prompt.strip()

        if not original_prompt:
            raise ValueError("Prompt cannot be empty.")

        preferences = preferences or {}

        language = preferences.get("language", "English")
        level = preferences.get("level", "beginner")
        response_length = preferences.get(
            "response_length",
            "medium",
        )

        optimized_prompt = (
            "You are an AI assistant operating through APIL.\n\n"
            "Instructions:\n"
            f"- Respond in {language}.\n"
            f"- Adapt the explanation for a {level} user.\n"
            f"- Keep the response {response_length} in length.\n"
            "- Answer the user's actual request directly.\n"
            "- Be clear, relevant, and structured.\n"
            "- Do not add unnecessary information.\n\n"
            "User request:\n"
            f"{original_prompt}"
        )

        return {
            "status": "success",
            "original_prompt": original_prompt,
            "optimized_prompt": optimized_prompt,
            "optimization": {
                "language": language,
                "level": level,
                "response_length": response_length,
                "prompt_changed": (
                    original_prompt != optimized_prompt
                ),
            },
        }