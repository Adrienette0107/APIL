from backend.app.core.config import settings
from backend.app.services.ai.anthropic_provider import AnthropicProvider
from backend.app.services.ai.gemini_provider import GeminiProvider
from backend.app.services.ai.groq_provider import GroqProvider
from backend.app.services.ai.base import AIProvider
from backend.app.services.ai.ollama_provider import OllamaProvider
from backend.app.services.ai.openai_provider import OpenAIProvider


class AIProviderRouter:

    def __init__(self):
        self.providers: dict[str, AIProvider] = {
            "ollama": OllamaProvider(
                host=settings.OLLAMA_URL,
                default_model=settings.OLLAMA_MODEL,
            ),
            "openai": OpenAIProvider(settings.OPENAI_API_KEY),
            "gemini": GeminiProvider(settings.GEMINI_API_KEY),
            "anthropic": AnthropicProvider(settings.ANTHROPIC_API_KEY),
            "groq": GroqProvider(settings.GROQ_API_KEY),
        }

    def get_provider(
        self,
        provider_name: str,
    ) -> AIProvider:

        provider = self.providers.get(provider_name)

        if provider is None:
            raise ValueError(
                f"Unsupported AI provider: {provider_name}"
            )

        return provider


provider_router = AIProviderRouter()
