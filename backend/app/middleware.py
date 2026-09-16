import logging
import time
import uuid

from fastapi import Request


logger = logging.getLogger("apil")


async def request_logging_middleware(
    request: Request,
    call_next,
):
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed | "
            "request_id=%s | "
            "status=%s | "
            "duration_ms=%.2f",
            request_id,
            response.status_code,
            duration_ms,
        )

        return response

    except Exception:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Request failed | "
            "request_id=%s | "
            "duration_ms=%.2f",
            request_id,
            duration_ms,
        )

        raise