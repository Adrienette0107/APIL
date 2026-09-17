from backend.app.core.config import settings
from backend.app.services.adaptive.analyzer import PromptAnalysis
from backend.app.services.ai.registry import (
    MODEL_REGISTRY,
    get_model_config,
)

DEFAULT_MODEL = settings.OLLAMA_MODEL
ADVANCED_MODEL = "qwen3:latest"
DEFAULT_PROVIDER_MODELS = {
    "ollama": settings.OLLAMA_MODEL,
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "groq": "llama-3.3-70b-versatile",
}
SUPPORTED_PROVIDERS = frozenset(DEFAULT_PROVIDER_MODELS)


def select_model(
    analysis: PromptAnalysis,
    requested_model: str,
) -> tuple[str, str]:

    if requested_model != "auto":
        provider_name, _, model_name = requested_model.partition(":")

        if ":" in requested_model and provider_name not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider_name}"
            )

        config = get_model_config(requested_model)

        if config is None and provider_name in DEFAULT_PROVIDER_MODELS:
            return provider_name, model_name or DEFAULT_PROVIDER_MODELS[provider_name]

        if config is None:
            raise ValueError(
                f"Unsupported model: {requested_model}"
            )

        if not config.enabled:
            raise ValueError(
                f"Model is currently disabled: {requested_model}"
            )

        return config.provider, config.model

    if analysis.complexity == "low":
        selected = DEFAULT_MODEL
    else:
        selected = ADVANCED_MODEL

    config = MODEL_REGISTRY[selected]

    return config.provider, config.model
