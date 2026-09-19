from __future__ import annotations

from backend.app.core.provider_config import (
    get_provider_configs,
)
from backend.app.services.ai.anthropic_provider import (
    AnthropicProvider,
)
from backend.app.services.ai.base import AIProvider
from backend.app.services.ai.gemini_provider import (
    GeminiProvider,
)
from backend.app.services.ai.groq_provider import (
    GroqProvider,
)
from backend.app.services.ai.ollama_provider import (
    OllamaProvider,
)
from backend.app.services.ai.openai_provider import (
    OpenAIProvider,
)


class AIProviderRouter:
    """
    Provider adapter registry.

    Responsibilities:
        - create provider adapters
        - expose providers by normalized provider name
        - reject unknown providers

    This class does NOT handle:
        - retries
        - timeouts
        - fallback
        - circuit breakers
        - prompt optimization
        - response optimization

    Those responsibilities belong to the appropriate
    APIL layers.
    """

    def __init__(self) -> None:

        configs = get_provider_configs()

        self.providers: dict[
            str,
            AIProvider,
        ] = {}

        # --------------------------------------------------------------
        # Ollama
        # --------------------------------------------------------------

        ollama_config = configs.get(
            "ollama"
        )

        if ollama_config is not None:

            self.providers["ollama"] = (
                OllamaProvider(
                    host=(
                        ollama_config.base_url
                        or "http://127.0.0.1:11434"
                    ),
                    default_model=(
                        ollama_config.default_model
                        or "qwen3:4b"
                    ),
                )
            )

        # --------------------------------------------------------------
        # OpenAI
        # --------------------------------------------------------------

        openai_config = configs.get(
            "openai"
        )

        if openai_config is not None:

            self.providers["openai"] = (
                OpenAIProvider(
                    api_key=openai_config.api_key,
                    base_url=openai_config.base_url,
                )
            )

        # --------------------------------------------------------------
        # Gemini
        # --------------------------------------------------------------

        gemini_config = configs.get(
            "gemini"
        )

        if gemini_config is not None:

            self.providers["gemini"] = (
                GeminiProvider(
                    api_key=gemini_config.api_key,
                    base_url=gemini_config.base_url,
                )
            )

        # --------------------------------------------------------------
        # Anthropic
        # --------------------------------------------------------------

        anthropic_config = configs.get(
            "anthropic"
        )

        if anthropic_config is not None:

            self.providers["anthropic"] = (
                AnthropicProvider(
                    api_key=anthropic_config.api_key,
                    base_url=anthropic_config.base_url,
                )
            )

        # --------------------------------------------------------------
        # Groq
        # --------------------------------------------------------------

        groq_config = configs.get(
            "groq"
        )

        if groq_config is not None:

            self.providers["groq"] = (
                GroqProvider(
                    api_key=groq_config.api_key,
                    base_url=groq_config.base_url,
                )
            )

    # ------------------------------------------------------------------
    # Provider lookup
    # ------------------------------------------------------------------

    def get_provider(
        self,
        provider_name: str,
    ) -> AIProvider:

        normalized_name = (
            provider_name.strip().lower()
            if provider_name
            else ""
        )

        provider = self.providers.get(
            normalized_name
        )

        if provider is None:

            raise ValueError(
                f"Unsupported AI provider: "
                f"{normalized_name or provider_name}"
            )

        return provider

    # ------------------------------------------------------------------
    # Provider availability
    # ------------------------------------------------------------------

    def has_provider(
        self,
        provider_name: str,
    ) -> bool:

        normalized_name = (
            provider_name.strip().lower()
            if provider_name
            else ""
        )

        return normalized_name in self.providers

    def list_providers(self) -> list[str]:

        return sorted(
            self.providers.keys()
        )


provider_router = AIProviderRouter()


