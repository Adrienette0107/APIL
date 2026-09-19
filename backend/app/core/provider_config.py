from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from backend.app.core.config import BASE_DIR


# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------

load_dotenv(
    BASE_DIR / ".env",
    override=True,
)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    """
    Runtime configuration for one AI provider.

    Provider-specific SDK/API logic must remain inside the provider
    adapter. This class only describes how APIL should execute it.
    """

    name: str
    enabled: bool
    base_url: str | None
    api_key: str | None
    default_model: str | None
    timeout: float
    max_retries: int
    priority: int
    fallback_enabled: bool


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _get_bool(
    name: str,
    default: bool = False,
) -> bool:
    """
    Read a boolean environment variable.
    """

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_int(
    name: str,
    default: int = 0,
) -> int:
    """
    Read an integer environment variable safely.
    """

    try:
        return int(
            os.getenv(
                name,
                str(default),
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return default


def _get_float(
    name: str,
    default: float = 60.0,
) -> float:
    """
    Read a positive floating-point environment variable safely.
    """

    try:
        value = float(
            os.getenv(
                name,
                str(default),
            )
        )

        if value <= 0:
            return default

        return value

    except (
        TypeError,
        ValueError,
    ):
        return default


def _get_timeout(
    provider: str,
    default: float,
) -> float:
    """
    Read provider timeout.

    Preferred:
        PROVIDER_TIMEOUT_SECONDS

    Backward-compatible:
        PROVIDER_TIMEOUT
    """

    return _get_float(
        f"{provider}_TIMEOUT_SECONDS",
        _get_float(
            f"{provider}_TIMEOUT",
            default,
        ),
    )


def _get_provider_enabled(
    provider: str,
    api_key: str | None = None,
    default: bool = False,
) -> bool:
    """
    Determine whether a provider is enabled.

    Explicit PROVIDER_ENABLED always takes precedence.

    If PROVIDER_ENABLED is not specified:
        - API-key providers are enabled when a key exists.
        - Ollama can remain enabled without a key.
    """

    explicit = os.getenv(
        f"{provider}_ENABLED"
    )

    if explicit is not None:
        return _get_bool(
            f"{provider}_ENABLED",
            default,
        )

    if api_key:
        return True

    return default


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

def get_provider_configs() -> dict[str, ProviderConfig]:
    """
    Build the complete APIL provider configuration.

    Configuration is evaluated at runtime so environment changes are
    picked up whenever the application process starts/reloads.
    """

    # -----------------------------------------------------------------------
    # Ollama
    # -----------------------------------------------------------------------

    ollama_model = os.getenv(
        "OLLAMA_MODEL",
        "qwen3:4b",
    )

    ollama_url = os.getenv(
        "OLLAMA_URL",
        "http://127.0.0.1:11434",
    )

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------

    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    # -----------------------------------------------------------------------
    # Gemini
    # -----------------------------------------------------------------------

    gemini_api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    # -----------------------------------------------------------------------
    # Anthropic
    # -----------------------------------------------------------------------

    anthropic_api_key = os.getenv(
        "ANTHROPIC_API_KEY"
    )

    # -----------------------------------------------------------------------
    # Groq
    # -----------------------------------------------------------------------

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    return {

        # ===================================================================
        # OLLAMA
        # ===================================================================

        "ollama": ProviderConfig(
            name="ollama",

            enabled=_get_provider_enabled(
                "OLLAMA",
                default=True,
            ),

            base_url=ollama_url,

            api_key=None,

            default_model=ollama_model,

            timeout=_get_timeout(
                "OLLAMA",
                120.0,
            ),

            max_retries=max(
                0,
                _get_int(
                    "OLLAMA_MAX_RETRIES",
                    1,
                ),
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

        # ===================================================================
        # OPENAI
        # ===================================================================

        "openai": ProviderConfig(
            name="openai",

            enabled=_get_provider_enabled(
                "OPENAI",
                api_key=openai_api_key,
                default=False,
            ),

            base_url=os.getenv(
                "OPENAI_BASE_URL"
            ),

            api_key=openai_api_key,

            default_model=os.getenv(
                "OPENAI_MODEL",
                "gpt-4o-mini",
            ),

            timeout=_get_timeout(
                "OPENAI",
                60.0,
            ),

            max_retries=max(
                0,
                _get_int(
                    "OPENAI_MAX_RETRIES",
                    2,
                ),
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

        # ===================================================================
        # GEMINI
        # ===================================================================

        "gemini": ProviderConfig(
            name="gemini",

            enabled=_get_provider_enabled(
                "GEMINI",
                api_key=gemini_api_key,
                default=False,
            ),

            base_url=os.getenv(
                "GEMINI_BASE_URL"
            ),

            api_key=gemini_api_key,

            default_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.0-flash",
            ),

            timeout=_get_timeout(
                "GEMINI",
                60.0,
            ),

            max_retries=max(
                0,
                _get_int(
                    "GEMINI_MAX_RETRIES",
                    2,
                ),
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

        # ===================================================================
        # ANTHROPIC
        # ===================================================================

        "anthropic": ProviderConfig(
            name="anthropic",

            enabled=_get_provider_enabled(
                "ANTHROPIC",
                api_key=anthropic_api_key,
                default=False,
            ),

            base_url=os.getenv(
                "ANTHROPIC_BASE_URL"
            ),

            api_key=anthropic_api_key,

            default_model=os.getenv(
                "ANTHROPIC_MODEL",
                "claude-3-5-sonnet-latest",
            ),

            timeout=_get_timeout(
                "ANTHROPIC",
                60.0,
            ),

            max_retries=max(
                0,
                _get_int(
                    "ANTHROPIC_MAX_RETRIES",
                    2,
                ),
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

        # ===================================================================
        # GROQ
        # ===================================================================

        "groq": ProviderConfig(
            name="groq",

            enabled=_get_provider_enabled(
                "GROQ",
                api_key=groq_api_key,
                default=False,
            ),

            base_url=os.getenv(
                "GROQ_BASE_URL"
            ),

            api_key=groq_api_key,

            # Current Groq model available to your account.
            default_model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            ),

            timeout=_get_timeout(
                "GROQ",
                60.0,
            ),

            max_retries=max(
                0,
                _get_int(
                    "GROQ_MAX_RETRIES",
                    2,
                ),
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


# ---------------------------------------------------------------------------
# Provider lookup helpers
# ---------------------------------------------------------------------------

def get_provider_config(
    provider: str,
) -> ProviderConfig | None:
    """
    Return one provider's configuration.
    """

    if not provider:
        return None

    provider = provider.strip().lower()

    return get_provider_configs().get(
        provider
    )


def get_enabled_provider_configs() -> dict[
    str,
    ProviderConfig,
]:
    """
    Return only providers currently enabled.
    """

    configs = get_provider_configs()

    return {
        name: config
        for name, config in configs.items()
        if config.enabled
    }