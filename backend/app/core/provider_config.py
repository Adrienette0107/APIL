from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    enabled: bool
    base_url: Optional[str]
    api_key: Optional[str]
    default_model: Optional[str]
    timeout: float
    max_retries: int
    priority: int
    fallback_enabled: bool


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_provider_configs() -> dict[str, ProviderConfig]:
    return {
        "ollama": ProviderConfig(
            name="ollama",
            enabled=_get_bool("OLLAMA_ENABLED", True),
            base_url=os.getenv(
                "OLLAMA_URL",
                "http://127.0.0.1:11434",
            ),
            api_key=None,
            default_model=os.getenv(
                "OLLAMA_MODEL",
                "qwen3:4b",
            ),
            timeout=_get_float(
                "OLLAMA_TIMEOUT",
                120.0,
            ),
            max_retries=_get_int(
                "OLLAMA_MAX_RETRIES",
                2,
            ),
            priority=_get_int(
                "OLLAMA_PRIORITY",
                1,
            ),
            fallback_enabled=_get_bool(
                "OLLAMA_FALLBACK_ENABLED",
                True,
            ),
        ),

        "openai": ProviderConfig(
            name="openai",
            enabled=_get_bool(
                "OPENAI_ENABLED",
                False,
            ),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            default_model=os.getenv("OPENAI_MODEL"),
            timeout=_get_float(
                "OPENAI_TIMEOUT",
                60.0,
            ),
            max_retries=_get_int(
                "OPENAI_MAX_RETRIES",
                2,
            ),
            priority=_get_int(
                "OPENAI_PRIORITY",
                2,
            ),
            fallback_enabled=_get_bool(
                "OPENAI_FALLBACK_ENABLED",
                True,
            ),
        ),

        "gemini": ProviderConfig(
            name="gemini",
            enabled=_get_bool(
                "GEMINI_ENABLED",
                False,
            ),
            base_url=os.getenv("GEMINI_BASE_URL"),
            api_key=os.getenv("GEMINI_API_KEY"),
            default_model=os.getenv("GEMINI_MODEL"),
            timeout=_get_float(
                "GEMINI_TIMEOUT",
                60.0,
            ),
            max_retries=_get_int(
                "GEMINI_MAX_RETRIES",
                2,
            ),
            priority=_get_int(
                "GEMINI_PRIORITY",
                3,
            ),
            fallback_enabled=_get_bool(
                "GEMINI_FALLBACK_ENABLED",
                True,
            ),
        ),

        "anthropic": ProviderConfig(
            name="anthropic",
            enabled=_get_bool(
                "ANTHROPIC_ENABLED",
                False,
            ),
            base_url=os.getenv("ANTHROPIC_BASE_URL"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            default_model=os.getenv("ANTHROPIC_MODEL"),
            timeout=_get_float(
                "ANTHROPIC_TIMEOUT",
                60.0,
            ),
            max_retries=_get_int(
                "ANTHROPIC_MAX_RETRIES",
                2,
            ),
            priority=_get_int(
                "ANTHROPIC_PRIORITY",
                4,
            ),
            fallback_enabled=_get_bool(
                "ANTHROPIC_FALLBACK_ENABLED",
                True,
            ),
        ),

        "groq": ProviderConfig(
            name="groq",
            enabled=_get_bool(
                "GROQ_ENABLED",
                False,
            ),
            base_url=os.getenv("GROQ_BASE_URL"),
            api_key=os.getenv("GROQ_API_KEY"),
            default_model=os.getenv("GROQ_MODEL"),
            timeout=_get_float(
                "GROQ_TIMEOUT",
                60.0,
            ),
            max_retries=_get_int(
                "GROQ_MAX_RETRIES",
                2,
            ),
            priority=_get_int(
                "GROQ_PRIORITY",
                5,
            ),
            fallback_enabled=_get_bool(
                "GROQ_FALLBACK_ENABLED",
                True,
            ),
        ),
    }