
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AIProvider(ABC):
    """
    Canonical provider interface for APIL.

    Every GenAI provider adapter must implement the same generate()
    contract.

    APIL communicates with providers through this abstraction so
    provider-specific SDKs and response formats remain isolated
    inside their respective adapters.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate a text response.

        Parameters
        ----------
        messages:
            Chat messages in the canonical APIL format.

            Example:

                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Explain machine learning."
                    }
                ]

        model:
            Provider-specific model identifier.

        temperature:
            Sampling temperature.

        max_tokens:
            Optional maximum number of output tokens.

        Returns
        -------
        str
            A normalized plain-text response.

        Notes
        -----
        Provider adapters should:

            - communicate with their own provider SDK/API
            - convert the native response to str
            - avoid implementing APIL fallback logic
            - avoid implementing APIL retry logic
            - avoid implementing APIL circuit-breaker logic
            - avoid modifying the user's prompt

        ProviderExecutor is responsible for:

            - timeout
            - retry
            - backoff
            - circuit breaker
            - error normalization

        APIL Pipeline is responsible for:

            - prompt analysis
            - prompt optimization
            - adaptive processing
            - provider/model selection
            - response sanitization
            - response evaluation
            - response improvement
        """

        raise NotImplementedError

