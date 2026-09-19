import asyncio
import logging
import time

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


def get_generation_budget(
    prompt_dna: dict,
    preferences: dict | None = None,
) -> int:
    """Derive a bounded output budget from semantic request requirements."""

    preferences = preferences or {}
    length = (
        prompt_dna.get("desired_length")
        or prompt_dna.get("response_length")
        or preferences.get("response_length")
    )
    output_format = prompt_dna.get("output_format")
    has_code = bool(prompt_dna.get("code_requirements")) or output_format == "code"

    if length == "short" or output_format == "single sentence":
        return 384
    if has_code:
        return 3072 if length == "long" else 2048
    if length == "long" or prompt_dna.get("desired_depth") == "detailed":
        return 2048
    if output_format in {"json", "table", "bullet points", "steps", "summary"}:
        return 768
    if prompt_dna.get("desired_depth") == "simple":
        return 256
    return 768


async def process_chat(
    db: AsyncSession,
    prompt: str,
    user_id: str,
    conversation_id: str,
    model: str = "auto",
    preferences: dict | None = None,
    request_id: str | None = None,
):

    start_total = time.perf_counter()

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

    prompt_start = time.perf_counter()
    optimization_result = prompt_optimizer.optimize(
        prompt=prompt,
        preferences=saved_preferences,
    )
    prompt_processing_ms = int((time.perf_counter() - prompt_start) * 1000)

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
    generation_budget = get_generation_budget(
        prompt_dna,
        saved_preferences,
    )

    logger.info(
        "Sending optimized prompt to provider | request_id=%s | provider=%s | model=%s | max_tokens=%s",
        request_id,
        provider_name,
        selected_model,
        generation_budget,
    )

    provider_call_count = 0
    provider_start = time.perf_counter()
    try:
        response = await provider.generate(
            messages=messages,
            model=selected_model,
            max_tokens=generation_budget,
        )
        provider_call_count = 1
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
    provider_generation_ms = int((time.perf_counter() - provider_start) * 1000)

    logger.info(
        "Provider response received | request_id=%s | provider=%s | model=%s",
        request_id,
        provider_name,
        selected_model,
    )

    raw_response = response
    sanitized_provider_response = sanitize_model_output(
        raw_response,
        provider=provider_name,
    )
    if not sanitized_provider_response:
        raise ProviderExecutionError(
            code="invalid_provider_response",
            message="The AI provider returned no usable final answer.",
            status_code=502,
        )
    response = process_response(
        response=sanitized_provider_response,
        preferences=saved_preferences,
    )

    verification_result = output_verifier.verify(
        original_prompt=prompt,
        response=response,
        preferences=saved_preferences,
    )

    evaluation_start = time.perf_counter()
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
    response_evaluation_ms = int((time.perf_counter() - evaluation_start) * 1000)

    improvement_applied = False
    final_response = response
    improvement_error = None
    attempts = 0
    improvement_ms = 0
    improvement_attempted = False

    if response_evaluation.get("improvement_needed") and attempts < max_improvement_attempts:
        improvement_start = time.perf_counter()
        improved_result = await improve_response(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            response=final_response,
            evaluation=response_evaluation,
            prompt_dna=prompt_dna,
            provider_name=provider_name,
            model_name=selected_model,
        )
        improvement_ms += int((time.perf_counter() - improvement_start) * 1000)
        improvement_attempted = bool(improved_result.get("improvement_attempted", True))
        if improvement_attempted:
            provider_call_count += 1
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

        else:
            improvement_error = improved_result.get("failure_reason")
            final_response = improved_result.get("response", final_response)

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

    sanitized_response = sanitize_model_output(
        final_response,
        provider=provider_name,
    )
    if not sanitized_response:
        raise ProviderExecutionError(
            code="invalid_provider_response",
            message="The AI provider returned no usable final answer after sanitization.",
            status_code=502,
        )

    final_quality_start = time.perf_counter()
    final_quality_gate = evaluate_response(
        original_prompt=prompt,
        optimized_prompt=optimized_prompt,
        response=sanitized_response,
        prompt_dna=prompt_dna,
        preferences=saved_preferences,
    )
    final_quality_gate_ms = int((time.perf_counter() - final_quality_start) * 1000)

    total_ms = int((time.perf_counter() - start_total) * 1000)

    return {
        "provider": provider_name,
        "model": selected_model,
        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "prompt_dna": prompt_dna,
        "processed_prompt": processed_prompt,
        "raw_provider_response": raw_response,
        "raw_response": raw_response,
        "response": sanitized_response,
        "provider_call_count": provider_call_count,
        "verification": verification_result,
        "response_evaluation": response_evaluation,
        "improvement_applied": improvement_applied,
        "improvement_attempts": attempts,
        "improvement_attempted": improvement_attempted,
        "improvement_needed": response_evaluation.get("improvement_needed"),
        "improvement_error": improvement_error,
        "final_quality_gate": final_quality_gate,
        "final_response": sanitized_response,
        "timing": {
            "prompt_processing_ms": prompt_processing_ms,
            "provider_generation_ms": provider_generation_ms,
            "response_evaluation_ms": response_evaluation_ms,
            "response_improvement_ms": improvement_ms,
            "final_quality_gate_ms": final_quality_gate_ms,
            "total_ms": total_ms,
        },
        "preferences": saved_preferences,
        "analysis": {
            "intent": analysis.intent,
            "complexity": analysis.complexity,
            "domain": analysis.domain,
            "output_type": analysis.output_type,
        },
    }
