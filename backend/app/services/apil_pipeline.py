import asyncio
import logging

from click import prompt
import httpx
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
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.output_verifier import OutputVerifier


logger = logging.getLogger("apil")

prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()

async def process_chat(
    db: AsyncSession,
    prompt: str,
    user_id: str,
    conversation_id: str,
    model: str = "auto",
    preferences: dict | None = None,
    request_id: str | None = None,
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
    optimization_result = prompt_optimizer.optimize(
    prompt=prompt,
    preferences=saved_preferences,
)

    optimized_prompt = optimization_result["optimized_prompt"]

    logger.info(
    "Prompt optimization completed | request_id=%s",
    request_id,
)

    messages, analysis = build_adaptive_messages(
        prompt=optimized_prompt,
        history=history,
        preferences=saved_preferences,
    )

    processed_prompt = next(
        (
            message["content"]
            for message in reversed(messages)
            if message.get("role") == "user"
        ),
        optimized_prompt,
    )

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    logger.info(
        "Prompt optimized | request_id=%s | provider=%s | model=%s",
        request_id,
        provider_name,
        selected_model,
    )

    provider = provider_router.get_provider(
        provider_name
    )

    logger.info(
        "Sending optimized prompt to provider | request_id=%s | provider=%s | model=%s",
        request_id,
        provider_name,
        selected_model,
    )

    try:
        response = await provider.generate(
            messages=messages,
            model=selected_model,
        )
    except ProviderExecutionError:
        raise
    except (asyncio.TimeoutError, httpx.TimeoutException) as exc:
        raise ProviderExecutionError(
            code="provider_timeout",
            message="The AI provider timed out.",
            status_code=504,
        ) from exc
    except Exception as exc:
        raise ProviderExecutionError(
            code="provider_error",
            message="The AI provider returned an error.",
        ) from exc

    logger.info(
        "Provider response received | request_id=%s | provider=%s | model=%s",
        request_id,
        provider_name,
        selected_model,
    )

    raw_response = response

    response = process_response(
    response=raw_response,
    preferences=saved_preferences,
)

    verification_result = output_verifier.verify(
    original_prompt=prompt,
    response=response,
    preferences=saved_preferences,
)

    logger.info(
    "Response verification completed | request_id=%s | verified=%s",
    request_id,
    verification_result["verified"],
)
    return {
    "provider": provider_name,
    "model": selected_model,

    # Original user input
    "original_prompt": prompt,

    # Prompt after APIL optimization
    "optimized_prompt": optimized_prompt,

    # Prompt actually processed by adaptive pipeline
    "processed_prompt": processed_prompt,

    # Raw model output
    "raw_response": raw_response,

    # Final APIL response
    "response": response,

    # Output verification
    "verification": verification_result,

    "preferences": saved_preferences,

    "analysis": {
        "intent": analysis.intent,
        "complexity": analysis.complexity,
        "domain": analysis.domain,
        "output_type": analysis.output_type,
    },
}
