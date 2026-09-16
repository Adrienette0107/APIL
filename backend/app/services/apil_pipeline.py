from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.conversation_service import (
    get_conversation_history,
)

from backend.app.services.preferences_service import (
    get_user_preferences,
)

from backend.app.services.adaptive.pipeline import (
    build_adaptive_messages,
    process_response,
)

from backend.app.services.ai.model_router import select_model
from backend.app.services.ai.provider_router import provider_router


async def process_chat(
    db: AsyncSession,
    prompt: str,
    user_id: str,
    conversation_id: str,
    model: str = "auto",
    preferences: dict | None = None,
):

    saved_preferences = await get_user_preferences(
        db=db,
        user_id=user_id,
    )

    if preferences:
        saved_preferences.update(preferences)

    history = await get_conversation_history(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        limit=20,
    )

    messages, analysis = build_adaptive_messages(
        prompt=prompt,
        history=history,
        preferences=saved_preferences,
    )

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    provider = provider_router.get_provider(
        provider_name
    )

    response = await provider.generate(
        messages=messages,
        model=selected_model,
    )

    response = process_response(
        response=response,
        preferences=saved_preferences,
    )

    return {
        "provider": provider_name,
        "model": selected_model,
        "response": response,
        "preferences": saved_preferences,
        "analysis": {
            "intent": analysis.intent,
            "complexity": analysis.complexity,
            "domain": analysis.domain,
            "output_type": analysis.output_type,
        },
    }
