from fastapi import HTTPException


class ProviderExecutionError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 502,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def internal_server_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail="APIL was unable to process the request.",
    )
