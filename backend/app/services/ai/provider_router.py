from backend.app.services.ai.base import AIProvider
from backend.app.services.ai.ollama_provider import OllamaProvider


class AIProviderRouter:

    def __init__(self):
        self.providers: dict[str, AIProvider] = {
            "ollama": OllamaProvider(),
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
