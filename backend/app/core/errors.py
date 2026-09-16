from fastapi import HTTPException


def internal_server_error() -> HTTPException:
    return HTTPException(
        status_code=500,
        detail="APIL was unable to process the request.",
    )
