from fastapi import FastAPI

app = FastAPI(
    title="APIL",
    description="Adaptive Prompt Intelligence Layer",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "name": "APIL",
        "description": "Adaptive Prompt Intelligence Layer",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }
