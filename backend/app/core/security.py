from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import settings


bearer_scheme = HTTPBearer(
    auto_error=False
)


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(
        bearer_scheme
    ),
):
    configured_key = (settings.APIL_API_KEY or "").strip()

    # Authentication disabled when no API key is configured.
    if not configured_key:
        return

    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "missing_authorization",
                "message": "Missing authorization header.",
                "request_id": request.state.request_id,
            },
        )

    token = credentials.credentials.strip()

    if token != configured_key:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid API key.",
                "request_id": request.state.request_id,
            },
        )