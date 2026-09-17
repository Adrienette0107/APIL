from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.chat import router as chat_router
from backend.app.api.health import router as health_router
from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
from backend.app.middleware import (
    request_logging_middleware,
)


setup_logging()

app = FastAPI(
    title="APIL - Adaptive Prompt Intelligence Layer",
    version=settings.APP_VERSION,
)

app.middleware("http")(
    request_logging_middleware
)

if settings.CORS_ORIGINS:
    origins = [
        origin.strip()
        for origin in settings.CORS_ORIGINS.split(",")
        if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["Authorization", "Content-Type"],
    )

app.include_router(chat_router)
app.include_router(health_router)


@app.get("/")
def root():
    return {
        "name": "APIL",
        "description": "Adaptive Prompt Intelligence Layer",
        "version": settings.APP_VERSION,
        "status": "running",
    }
from backend.app.api.optimization import (
    router as optimization_router,
)
app.include_router(
    optimization_router
)
