from fastapi import Header, HTTPException

from backend.app.core.config import settings


async def verify_api_key(
    authorization: str | None = Header(default=None),
):
    if not settings.APIL_API_KEY:
        return

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format",
        )

    token = authorization.replace("Bearer ", "", 1)

    if token != settings.APIL_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )