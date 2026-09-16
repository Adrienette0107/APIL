from backend.app.services.adaptive.analyzer import PromptAnalysis
from backend.app.services.ai.registry import (
    MODEL_REGISTRY,
    get_model_config,
)

DEFAULT_MODEL = "qwen3:4b"
ADVANCED_MODEL = "qwen3:latest"


def select_model(
    analysis: PromptAnalysis,
    requested_model: str,
) -> tuple[str, str]:

    if requested_model != "auto":
        config = get_model_config(requested_model)

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
