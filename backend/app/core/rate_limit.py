from fastapi import Request


async def check_rate_limit(request: Request):
    """
    Rate limiting is disabled for the current APIL development phase.
    This keeps the API independent of Redis.
    """
    return True