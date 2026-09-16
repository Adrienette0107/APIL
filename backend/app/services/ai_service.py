from backend.app.services.adaptive.pipeline import (
    build_adaptive_messages,
    process_response,
)
from backend.app.services.ai.model_router import select_model
from backend.app.services.ai.provider_router import provider_router


async def generate_response(
    prompt: str,
    history: list,
    model: str = "auto",
    preferences: dict | None = None,
):
    messages, analysis = build_adaptive_messages(
        prompt=prompt,
        history=history,
        preferences=preferences,
    )

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    provider = provider_router.get_provider(provider_name)

    response = await provider.generate(
        messages=messages,
        model=selected_model,
    )

    response = process_response(
        response=response,
        preferences=preferences,
    )

    return (
        provider_name,
        selected_model,
        response,
    )
