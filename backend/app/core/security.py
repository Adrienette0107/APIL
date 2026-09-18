from fastapi import HTTPException, Request, Security
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from backend.app.core.config import settings


bearer_scheme = HTTPBearer(
    auto_error=False
)
api_key_scheme = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(
        bearer_scheme
    ),
    api_key: str | None = Security(api_key_scheme),
):
    configured_key = (settings.APIL_API_KEY or "").strip()
    if not configured_key:
        return

    if credentials:
        token = credentials.credentials.strip()
    elif api_key:
        token = api_key.strip()
    else:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "missing_authorization",
                "message": "Missing authorization header.",
                "request_id": request.state.request_id,
            },
        )

    if token != configured_key:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid API key.",
                "request_id": request.state.request_id,
            },
        )