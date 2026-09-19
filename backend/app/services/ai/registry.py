from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """
    Configuration for a model supported by APIL.

    The provider and model are kept separate so the rest of APIL
    does not need to parse provider-specific model identifiers.
    """

    provider: str
    model: str
    capabilities: tuple[str, ...] = ()
    enabled: bool = True


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {

    # -----------------------------------------------------------------------
    # Ollama
    # -----------------------------------------------------------------------

    "qwen3:4b": ModelConfig(
        provider="ollama",
        model="qwen3:4b",
        capabilities=(
            "general",
            "conversation",
            "fast",
        ),
    ),

    "qwen3:latest": ModelConfig(
        provider="ollama",
        model="qwen3:latest",
        capabilities=(
            "general",
            "conversation",
            "reasoning",
        ),
    ),

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------

    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model="gpt-4o-mini",
        capabilities=(
            "general",
            "conversation",
            "fast",
        ),
    ),

    "gpt-4o": ModelConfig(
        provider="openai",
        model="gpt-4o",
        capabilities=(
            "general",
            "conversation",
            "reasoning",
        ),
    ),

    # -----------------------------------------------------------------------
    # Gemini
    # -----------------------------------------------------------------------

    "gemini-2.0-flash": ModelConfig(
        provider="gemini",
        model="gemini-2.0-flash",
        capabilities=(
            "general",
            "conversation",
            "fast",
        ),
    ),

    # -----------------------------------------------------------------------
    # Anthropic
    # -----------------------------------------------------------------------

    "claude-3-5-sonnet-latest": ModelConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-latest",
        capabilities=(
            "general",
            "conversation",
            "reasoning",
        ),
    ),

    # -----------------------------------------------------------------------
    # Groq
    # -----------------------------------------------------------------------

    "openai/gpt-oss-20b": ModelConfig(
        provider="groq",
        model="openai/gpt-oss-20b",
        capabilities=(
            "general",
            "conversation",
            "reasoning",
            "fast",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Registry lookup
# ---------------------------------------------------------------------------

def get_model_config(
    model: str,
) -> ModelConfig | None:
    """
    Return the registered configuration for a model.

    Supports both:

        qwen3:4b

    and:

        ollama:qwen3:4b

    Provider-qualified models are normalized before lookup.
    """

    if not model:
        return None

    model = model.strip()

    if not model:
        return None

    # Provider-qualified model:
    #
    #     ollama:qwen3:4b
    #     openai:gpt-4o
    #     groq:openai/gpt-oss-20b
    #
    if ":" in model:

        provider, model_name = model.split(
            ":",
            1,
        )

        provider = provider.strip().lower()
        model_name = model_name.strip()

        if not provider or not model_name:
            return None

        config = MODEL_REGISTRY.get(
            model_name
        )

        if config is None:
            return None

        # Prevent a model from being used with
        # the wrong provider.
        if config.provider != provider:
            return None

        return config

    # Bare model lookup.
    return MODEL_REGISTRY.get(model)


# ---------------------------------------------------------------------------
# Runtime registration
# ---------------------------------------------------------------------------

def register_model(
    name: str,
    config: ModelConfig,
) -> None:
    """
    Register or replace a model configuration at runtime.
    """

    if not name or not name.strip():
        raise ValueError(
            "Model registry name cannot be empty."
        )

    if not config.provider.strip():
        raise ValueError(
            "Model provider cannot be empty."
        )

    if not config.model.strip():
        raise ValueError(
            "Model name cannot be empty."
        )

    MODEL_REGISTRY[name.strip()] = config


# ---------------------------------------------------------------------------
# List models
# ---------------------------------------------------------------------------

def list_models(
    enabled_only: bool = True,
) -> list[ModelConfig]:
    """
    Return registered models.

    By default only enabled models are returned.
    """

    models = list(
        MODEL_REGISTRY.values()
    )

    if enabled_only:
        models = [
            model
            for model in models
            if model.enabled
        ]

    return models
