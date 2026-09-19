from __future__ import annotations

from fastapi import HTTPException


class ProviderExecutionError(Exception):
    """
    Canonical APIL provider execution exception.

    This is shared by:
        - ProviderExecutor
        - ProviderFallbackManager
        - APIL Pipeline
        - API routes
    """

    def __init__(
        self,
        message: str,
        code: str = "provider_error",
        status_code: int = 502,
        provider: str | None = None,
        category: str | None = None,
        retryable: bool = False,
        attempts: int = 1,
    ) -> None:

        super().__init__(message)

        self.message = message
        self.code = code
        self.status_code = status_code

        # Provider information
        self.provider = provider

        # Backward/operational classification
        self.category = (
            category
            if category is not None
            else code
        )

        # Whether ProviderFallbackManager should
        # attempt another provider.
        self.retryable = retryable

        # Number of provider attempts performed.
        self.attempts = attempts

    def __str__(self) -> str:
        return self.message


def internal_server_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail="APIL was unable to process the request.",
    )