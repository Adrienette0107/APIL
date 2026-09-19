from fastapi import HTTPException, Request
from redis.exceptions import RedisError

from backend.app.core.config import settings
from backend.app.services.redis_service import get_redis


async def check_rate_limit(request: Request) -> None:
    """Apply a shared Redis-backed request limit without an in-memory bypass."""

    redis = get_redis()
    if redis is None:
        if settings.APP_ENV.lower() == "production":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "rate_limit_unavailable",
                    "message": "Rate limiting is unavailable.",
                    "request_id": request.state.request_id,
                },
            )
        return

    client_host = request.client.host if request.client else "unknown"
    key = f"apil:rate:{client_host}:{request.url.path}"

    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)
    except RedisError as exc:
        if settings.APP_ENV.lower() == "production":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "rate_limit_unavailable",
                    "message": "Rate limiting is unavailable.",
                    "request_id": request.state.request_id,
                },
            ) from exc
        return

    if count > settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "rate_limit_exceeded",
                "message": "Too many requests.",
                "request_id": request.state.request_id,
            },
            headers={
                "Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS),
            },
        )