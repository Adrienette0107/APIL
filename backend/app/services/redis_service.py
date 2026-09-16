from redis.asyncio import Redis

from backend.app.core.config import settings


redis_client: Redis | None = None


def get_redis() -> Redis | None:
    global redis_client

    if not settings.REDIS_URL:
        return None

    if redis_client is None:
        redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )

    return redis_client