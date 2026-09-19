from __future__ import annotations

from backend.app.core.config import settings
from backend.app.services.adaptive.analyzer import PromptAnalysis
from backend.app.services.ai.registry import (
    MODEL_REGISTRY,
    get_model_config,
)


# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------

DEFAULT_MODEL = settings.OLLAMA_MODEL
ADVANCED_MODEL = "qwen3:latest"

DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "ollama": settings.OLLAMA_MODEL,
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-3-5-sonnet-latest",
    "groq": "llama-3.3-70b-versatile",
}

SUPPORTED_PROVIDERS = frozenset(
    DEFAULT_PROVIDER_MODELS.keys()
)


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def select_model(
    analysis: PromptAnalysis,
    requested_model: str,
) -> tuple[str, str]:
    """
    Resolve the provider and model that APIL should execute.

    Supported forms:

        auto
        ollama:qwen3:4b
        ollama:qwen3:latest
        openai:gpt-4o-mini
        gemini:gemini-2.0-flash
        anthropic:claude-3-5-sonnet-latest
        groq:llama-3.3-70b-versatile

    Returns:

        (provider, model)

    Examples:

        ("ollama", "qwen3:4b")
        ("openai", "gpt-4o-mini")
    """

    requested_model = (
        requested_model.strip()
        if requested_model
        else "auto"
    )

    # -----------------------------------------------------------------------
    # Automatic model selection
    # -----------------------------------------------------------------------

    if requested_model.lower() == "auto":

        if analysis.complexity == "low":
            selected_model = DEFAULT_MODEL
        else:
            selected_model = ADVANCED_MODEL

        config = MODEL_REGISTRY.get(
            selected_model
        )

        if config is None:
            raise ValueError(
                f"Automatically selected model "
                f"'{selected_model}' is not registered."
            )

        if not config.enabled:
            raise ValueError(
                f"Automatically selected model "
                f"'{selected_model}' is disabled."
            )

        return (
            config.provider,
            config.model,
        )

    # -----------------------------------------------------------------------
    # Explicit provider:model selection
    # -----------------------------------------------------------------------

    if ":" in requested_model:

        provider_name, model_name = (
            requested_model.split(
                ":",1
            )
        )

        provider_name = provider_name.strip().lower()
        model_name = model_name.strip()

        if not provider_name:
            raise ValueError(
                "Provider name cannot be empty."
            )

        if not model_name:
            raise ValueError(
                "Model name cannot be empty."
            )

        if provider_name not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider_name}"
            )

        qualified_model = (
            f"{provider_name}:{model_name}"
        )

        config = get_model_config(
            qualified_model
        )

        # -------------------------------------------------------------------
        # Provider is supported even if the exact model is not registered.
        #
        # This is important for APIL's provider-independent architecture.
        # The provider adapter is responsible for actually accepting the
        # model name.
        # -------------------------------------------------------------------

        if config is None:

            return (
                provider_name,
                model_name,
            )

        if not config.enabled:
            raise ValueError(
                f"Model is currently disabled: "
                f"{qualified_model}"
            )

        return (
            config.provider,
            config.model,
        )

    # -----------------------------------------------------------------------
    # Bare model name
    #
    # Example:
    #
    #     qwen3:4b
    #
    # Note:
    # qwen3:4b contains ":" and therefore normally reaches the branch above.
    # A bare registered model such as "gpt-4o-mini" can still be resolved
    # through the registry.
    # -----------------------------------------------------------------------

    config = get_model_config(
        requested_model
    )

    if config is not None:

        if not config.enabled:
            raise ValueError(
                f"Model is currently disabled: "
                f"{requested_model}"
            )

        return (
            config.provider,
            config.model,
        )

    raise ValueError(
        f"Unsupported model: {requested_model}"
    )

