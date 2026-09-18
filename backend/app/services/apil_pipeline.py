import asyncio
import logging

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
from backend.app.services.adaptive.analyzer import analyze_prompt
from backend.app.services.response_evaluator import evaluate_response
from backend.app.services.response_improver import improve_response
from backend.app.services.response_sanitizer import sanitize_model_output
from backend.app.core.config import settings


logger = logging.getLogger("apil")

prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()
max_improvement_attempts = getattr(settings, "APIL_MAX_IMPROVEMENT_ATTEMPTS", 1)


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
    prompt_dna = optimization_result["prompt_dna"]

    logger.info("Prompt optimization completed | request_id=%s", request_id)

    messages, analysis = build_adaptive_messages(
        prompt=optimized_prompt,
        history=history,
        preferences=saved_preferences,
        analysis=analyze_prompt(prompt),
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

    provider = provider_router.get_provider(provider_name)

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
    response = process_response(response=raw_response, preferences=saved_preferences)

    verification_result = output_verifier.verify(
        original_prompt=prompt,
        response=response,
        preferences=saved_preferences,
    )

    try:
        response_evaluation = evaluate_response(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            response=response,
            prompt_dna=prompt_dna,
            preferences=saved_preferences,
        )
    except Exception as exc:
        logger.warning("Response evaluation failed | request_id=%s | error=%s", request_id, exc)
        response_evaluation = {
            "passed": False,
            "overall_quality": "unknown",
            "relevance": "unknown",
            "completeness": "unknown",
            "instruction_following": "unknown",
            "format_compliance": "unknown",
            "constraint_compliance": "unknown",
            "clarity": "unknown",
            "audience_fit": "unknown",
            "length_fit": "unknown",
            "issues": ["Evaluation failed; using the original provider response as a safe fallback."],
            "missing_requirements": [],
            "improvement_needed": False,
            "improvement_instructions": [],
        }

    improvement_applied = False
    final_response = response
    improvement_error = None
    attempts = 0

    while response_evaluation.get("improvement_needed") and attempts < max_improvement_attempts:
        improved_result = await improve_response(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            response=final_response,
            evaluation=response_evaluation,
            prompt_dna=prompt_dna,
            provider_name=provider_name,
            model_name=selected_model,
        )
        if improved_result.get("improvement_applied"):
            improved_response = improved_result["response"]
            final_response = improved_response
            improvement_applied = True
            response_evaluation = evaluate_response(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                response=final_response,
                prompt_dna=prompt_dna,
                preferences=saved_preferences,
            )
            attempts += 1
            continue

        improvement_error = improved_result.get("error")
        final_response = improved_result.get("response", final_response)
        break

    if attempts >= max_improvement_attempts and response_evaluation.get("improvement_needed"):
        logger.info(
            "Quality gate stopped improvement loop | request_id=%s | max_attempts=%s",
            request_id,
            max_improvement_attempts,
        )

    logger.info(
        "Response evaluation completed | request_id=%s | passed=%s | improvement_applied=%s",
        request_id,
        response_evaluation.get("passed"),
        improvement_applied,
    )

    sanitized_response = sanitize_model_output(final_response)
    if not sanitized_response:
        sanitized_response = "I’m sorry, but I couldn’t generate a final answer."

    return {
        "provider": provider_name,
        "model": selected_model,
        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "prompt_dna": prompt_dna,
        "processed_prompt": processed_prompt,
        "raw_response": raw_response,
        "response": sanitized_response,
        "verification": verification_result,
        "response_evaluation": response_evaluation,
        "improvement_applied": improvement_applied,
        "improvement_error": improvement_error,
        "preferences": saved_preferences,
        "analysis": {
            "intent": analysis.intent,
            "complexity": analysis.complexity,
            "domain": analysis.domain,
            "output_type": analysis.output_type,
        },
    }
