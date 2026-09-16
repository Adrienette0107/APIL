from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    capabilities: tuple[str, ...]
    enabled: bool = True


MODEL_REGISTRY = {
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
}


def get_model_config(model: str) -> ModelConfig | None:
    return MODEL_REGISTRY.get(model)
