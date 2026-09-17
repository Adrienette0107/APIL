from __future__ import annotations

from fastapi import APIRouter

from backend.app.services.prompt_optimizer import (
    PromptOptimizer,
)
from backend.app.services.output_verifier import (
    OutputVerifier,
)


router = APIRouter(
    prefix="/v1/optimize",
    tags=["Optimization"],
)


prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()


@router.post("/prompt")
async def optimize_prompt(payload: dict):

    prompt = payload.get("prompt", "")

    preferences = payload.get(
        "preferences"
    )

    result = prompt_optimizer.optimize(
        prompt=prompt,
        preferences=preferences,
    )

    return result


@router.post("/response")
async def verify_response(payload: dict):

    original_prompt = payload.get(
        "original_prompt",
        "",
    )

    response = payload.get(
        "response",
        "",
    )

    result = output_verifier.verify(
        original_prompt=original_prompt,
        response=response,
    )

    return result