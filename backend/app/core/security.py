from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Security

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
    if not settings.APIL_API_KEY:
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

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_authorization",
                "message": "Invalid authorization format.",
                "request_id": request.state.request_id,
            },
        )

    token = credentials.credentials

    if token != settings.APIL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "invalid_api_key",
                "message": "Invalid API key.",
                "request_id": request.state.request_id,
            },
        )