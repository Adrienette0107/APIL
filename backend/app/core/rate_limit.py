import hashlib
import time

from fastapi import Header, HTTPException

from backend.app.core.config import settings
from backend.app.services.redis_service import get_redis


async def check_rate_limit(
    authorization: str | None = Header(default=None),
):
    redis = get_redis()

    # Redis is optional during development
    if redis is None:
        if settings.APP_ENV == "development":
            return

        raise HTTPException(
            status_code=503,
            detail="Rate limiting service is unavailable.",
        )

    # Identify the client
    if authorization:
        client_identity = authorization
    else:
        client_identity = "anonymous"

    # Hash the identity so the actual API key
    # is never stored directly in Redis.
    client_hash = hashlib.sha256(
        client_identity.encode("utf-8")
    ).hexdigest()

    current_time = int(time.time())

    window = (
        current_time
        // settings.RATE_LIMIT_WINDOW_SECONDS
    )

    redis_key = (
        f"apil:rate:{client_hash}:{window}"
    )

    count = await redis.incr(redis_key)

    # Set expiration when the counter is created
    if count == 1:
        await redis.expire(
            redis_key,
            settings.RATE_LIMIT_WINDOW_SECONDS,
        )

    # Reject requests above the limit
    if count > settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                "Rate limit exceeded. "
                "Please try again later."
            ),
        )
