from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.errors import ProviderExecutionError
from backend.app.services.adaptive.analyzer import analyze_prompt
from backend.app.services.adaptive.pipeline import (
    build_adaptive_messages,
    process_response,
)
from backend.app.services.ai.model_router import select_model
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.conversation_service import (
    get_conversation_history,
)
from backend.app.services.output_improver import improve_response
from backend.app.services.output_verifier import OutputVerifier
from backend.app.services.preferences_service import (
    get_user_preferences,
)
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.provider_fallback import (
    ProviderFallbackManager,
)
from backend.app.services.response_evaluator import (
    evaluate_response,
)
from backend.app.services.response_sanitizer import (
    sanitize_model_output,
)


logger = logging.getLogger("apil")


# ----------------------------------------------------------------------
# Shared service instances
# ----------------------------------------------------------------------

prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()
fallback_manager = ProviderFallbackManager()

MAX_IMPROVEMENT_ATTEMPTS = min(
    1,
    max(
        0,
        int(
            getattr(
                settings,
                "APIL_MAX_IMPROVEMENT_ATTEMPTS",
                1,
            )
        ),
    ),
)


# ----------------------------------------------------------------------
# Generation budget
# ----------------------------------------------------------------------

def get_generation_budget(
    prompt_dna: dict[str, Any],
    preferences: dict[str, Any] | None = None,
) -> int:
    """
    Calculate a bounded generation budget from prompt requirements.
    """

    preferences = preferences or {}

    length = (
        prompt_dna.get("desired_length")
        or prompt_dna.get("response_length")
        or preferences.get("response_length")
    )

    output_format = prompt_dna.get(
        "output_format"
    )

    has_code = bool(
        prompt_dna.get("code_requirements")
    ) or output_format == "code"

    if (
        length == "short"
        or output_format == "single sentence"
    ):
        return 384

    if has_code:
        return (
            3072
            if length == "long"
            else 2048
        )

    if (
        length == "long"
        or prompt_dna.get(
            "desired_depth"
        ) == "detailed"
    ):
        return 2048

    if output_format in {
        "json",
        "table",
        "bullet points",
        "steps",
        "summary",
    }:
        return 768

    if prompt_dna.get(
        "desired_depth"
    ) == "simple":
        return 256

    return 768


# ----------------------------------------------------------------------
# Error normalization
# ----------------------------------------------------------------------

def _extract_provider_error(
    exc: Exception,
) -> ProviderExecutionError:

    if isinstance(
        exc,
        ProviderExecutionError,
    ):
        return exc

    return ProviderExecutionError(
        provider="unknown",
        message="The AI provider returned an error.",
        category="provider_error",
        retryable=False,
    )


# ----------------------------------------------------------------------
# Chat pipeline
# ----------------------------------------------------------------------

async def process_chat(
    db: AsyncSession,
    prompt: str,
    user_id: str,
    conversation_id: str,
    model: str = "auto",
    preferences: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:

    start_total = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. NORMALIZE INPUT
    # ------------------------------------------------------------------

    prompt = (
        prompt.strip()
        if isinstance(prompt, str)
        else str(prompt)
    )

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    preferences = preferences or {}

    # ------------------------------------------------------------------
    # 2. LOAD SAVED USER PREFERENCES
    # ------------------------------------------------------------------

    saved_preferences = await get_user_preferences(
        db=db,
        user_id=user_id,
    )

    saved_preferences = (
        saved_preferences
        or {}
    )

    saved_preferences.update(
        preferences
    )

    # ------------------------------------------------------------------
    # 3. LOAD CONVERSATION HISTORY
    # ------------------------------------------------------------------

    history = await get_conversation_history(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        limit=20,
    )

    # ------------------------------------------------------------------
    # 4. PROMPT ANALYSIS
    # ------------------------------------------------------------------

    analysis_start = time.perf_counter()

    analysis = analyze_prompt(
        prompt
    )

    analysis_ms = int(
        (
            time.perf_counter()
            - analysis_start
        )
        * 1000
    )

    # ------------------------------------------------------------------
    # 5. PROMPT OPTIMIZATION
    # ------------------------------------------------------------------

    prompt_start = time.perf_counter()

    optimization_result = (
        prompt_optimizer.optimize(
            prompt=prompt,
            preferences=saved_preferences,
        )
    )

    prompt_processing_ms = int(
        (
            time.perf_counter()
            - prompt_start
        )
        * 1000
    )

    optimized_prompt = (
        optimization_result.get(
            "optimized_prompt"
        )
        or prompt
    )

    prompt_dna = (
        optimization_result.get(
            "prompt_dna"
        )
        or {}
    )

    logger.info(
        "Prompt optimization completed | "
        "request_id=%s",
        request_id,
    )

    # ------------------------------------------------------------------
    # 6. BUILD ADAPTIVE PROVIDER MESSAGES
    #
    # The adaptive pipeline receives the already optimized prompt.
    # It must NOT perform another independent optimization.
    # ------------------------------------------------------------------

    messages, adaptive_analysis = (
        build_adaptive_messages(
            prompt=optimized_prompt,
            history=history,
            preferences=saved_preferences,
            analysis=analysis,
        )
    )

    processed_prompt = next(
        (
            message.get("content")
            for message in reversed(messages)
            if (
                message.get("role") == "user"
                and isinstance(
                    message.get("content"),
                    str,
                )
            )
        ),
        optimized_prompt,
    )

    # ------------------------------------------------------------------
    # 7. SELECT PROVIDER + MODEL
    # ------------------------------------------------------------------

    provider_name, selected_model = (
        select_model(
            analysis=adaptive_analysis,
            requested_model=model,
        )
    )

    logger.info(
        "Model selected | "
        "request_id=%s | provider=%s | model=%s",
        request_id,
        provider_name,
        selected_model,
    )

    # ------------------------------------------------------------------
    # 8. GENERATION BUDGET
    # ------------------------------------------------------------------

    generation_budget = get_generation_budget(
        prompt_dna=prompt_dna,
        preferences=saved_preferences,
    )

    logger.info(
        "Generation budget calculated | "
        "request_id=%s | max_tokens=%s",
        request_id,
        generation_budget,
    )

    # ------------------------------------------------------------------
    # 9. PROVIDER OPERATION
    # ------------------------------------------------------------------

    provider_call_count = 0

    async def provider_operation(
        current_provider: str,
        current_model: str,
    ) -> Any:

        nonlocal provider_call_count

        provider = provider_router.get_provider(
            current_provider
        )

        provider_call_count += 1

        logger.info(
            "Sending optimized prompt to provider | "
            "request_id=%s | provider=%s | model=%s | "
            "max_tokens=%s",
            request_id,
            current_provider,
            current_model,
            generation_budget,
        )

        return await provider.generate(
            messages=messages,
            model=current_model,
            max_tokens=generation_budget,
        )

    # ------------------------------------------------------------------
    # 10. PRIMARY PROVIDER + FALLBACK EXECUTION
    # ------------------------------------------------------------------

    provider_start = time.perf_counter()

    try:

        execution_result = (
            await fallback_manager.execute_with_fallback(
                primary_provider=provider_name,
                model=selected_model,
                operation_factory=provider_operation,
            )
        )

    except ProviderExecutionError:
        raise

    except Exception as exc:

        normalized_error = (
            _extract_provider_error(exc)
        )

        raise normalized_error from exc

    provider_generation_ms = int(
        (
            time.perf_counter()
            - provider_start
        )
        * 1000
    )

    # ------------------------------------------------------------------
    # 11. READ EXECUTION RESULT
    # ------------------------------------------------------------------

    raw_response = (
        execution_result.raw_response
        or ""
    )

    response = (
        execution_result.content
        or ""
    )

    actual_provider = (
        execution_result.provider
        or provider_name
    )

    actual_model = (
        execution_result.model
        or selected_model
    )

    logger.info(
        "Provider response received | "
        "request_id=%s | provider=%s | model=%s | "
        "attempts=%s",
        request_id,
        actual_provider,
        actual_model,
        execution_result.attempts,
    )

    # ------------------------------------------------------------------
    # 12. SANITIZE PROVIDER OUTPUT
    # ------------------------------------------------------------------

    sanitized_provider_response = (
        sanitize_model_output(
            response,
            provider=actual_provider,
        )
    )

    if not sanitized_provider_response:

        raise ProviderExecutionError(
            provider=actual_provider,
            message=(
                "The AI provider returned no usable "
                "final answer."
            ),
            category="invalid_provider_response",
            retryable=False,
        )

    # ------------------------------------------------------------------
    # 13. RESPONSE PROCESSING
    # ------------------------------------------------------------------

    response = process_response(
        response=sanitized_provider_response,
        preferences=saved_preferences,
    )

    # ------------------------------------------------------------------
    # 14. BASIC OUTPUT VERIFICATION
    # ------------------------------------------------------------------

    verification_result = (
        output_verifier.verify(
            original_prompt=prompt,
            response=response,
            preferences=saved_preferences,
        )
    )

    # ------------------------------------------------------------------
    # 15. RESPONSE QUALITY EVALUATION
    # ------------------------------------------------------------------

    evaluation_start = time.perf_counter()

    try:

        response_evaluation = (
            evaluate_response(
                original_prompt=prompt,
                optimized_prompt=optimized_prompt,
                response=response,
                prompt_dna=prompt_dna,
                preferences=saved_preferences,
            )
        )

    except Exception as exc:

        logger.warning(
            "Response evaluation failed | "
            "request_id=%s | error=%s",
            request_id,
            exc,
        )

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
            "final_answer_validity": "unknown",
            "issues": [
                "Response evaluation failed; final response cannot be trusted."
            ],
            "missing_requirements": [],
            "improvement_needed": False,
            "improvement_instructions": [],
            "evaluation_error": True,
        }

    response_evaluation_ms = int(
        (
            time.perf_counter()
            - evaluation_start
        )
        * 1000
    )

    # ------------------------------------------------------------------
    # 16. CONDITIONAL RESPONSE IMPROVEMENT
    # ------------------------------------------------------------------

    improvement_applied = False
    improvement_attempted = False
    improvement_error = None
    improvement_attempts = 0
    improvement_ms = 0

    final_response = response

    while (
        response_evaluation.get(
            "improvement_needed"
        )
        and improvement_attempts
        < MAX_IMPROVEMENT_ATTEMPTS
    ):

        improvement_attempts += 1

        improvement_start = (
            time.perf_counter()
        )

        try:

            improved_result = (
                await improve_response(
                    original_prompt=prompt,
                    optimized_prompt=optimized_prompt,
                    response=final_response,
                    evaluation=response_evaluation,
                    prompt_dna=prompt_dna,
                    provider_name=actual_provider,
                    model_name=actual_model,
                )
            )

        except Exception as exc:

            improvement_error = str(exc)

            logger.warning(
                "Response improvement failed | "
                "request_id=%s | error=%s",
                request_id,
                exc,
            )

            break

        finally:

            improvement_ms += int(
                (
                    time.perf_counter()
                    - improvement_start
                )
                * 1000
            )

        improvement_attempted = bool(
            improved_result.get(
                "improvement_attempted",
                True,
            )
        )

        # --------------------------------------------------------------
        # No improvement was produced
        # --------------------------------------------------------------

        if not improvement_attempted:

            improvement_error = (
                improved_result.get(
                    "failure_reason"
                )
            )

            break

        # --------------------------------------------------------------
        # Track the improvement provider call
        # --------------------------------------------------------------

        provider_call_count += int(
            improved_result.get(
                "provider_call_count",
                1,
            )
            or 0
        )

        # --------------------------------------------------------------
        # Improvement succeeded
        # --------------------------------------------------------------

        if improved_result.get(
            "improvement_applied"
        ):

            improved_response = (
                improved_result.get(
                    "response"
                )
            )

            if not improved_response:

                improvement_error = (
                    "Improved response was empty."
                )
                break

            sanitized_improved_response = (
                sanitize_model_output(
                    improved_response,
                    provider=(
                        improved_result.get(
                            "provider"
                        )
                        or actual_provider
                    ),
                )
            )

            if not sanitized_improved_response:

                improvement_error = (
                    "Improved response was empty "
                    "after sanitization."
                )
                break

            final_response = (
                sanitized_improved_response
            )

            improvement_applied = True

            # ----------------------------------------------------------
            # Re-evaluate improved response
            # ----------------------------------------------------------

            response_evaluation = (
                evaluate_response(
                    original_prompt=prompt,
                    optimized_prompt=optimized_prompt,
                    response=final_response,
                    prompt_dna=prompt_dna,
                    preferences=saved_preferences,
                )
            )

            if not response_evaluation.get(
                "improvement_needed"
            ):
                break

            improvement_applied = False
            improvement_error = (
                "improved_response_failed_quality_gate"
            )

        # --------------------------------------------------------------
        # Improvement attempted but rejected
        # --------------------------------------------------------------

        else:

            improvement_error = (
                improved_result.get(
                    "failure_reason"
                )
            )

            break

    # ------------------------------------------------------------------
    # 17. FINAL SANITIZATION
    # ------------------------------------------------------------------

    sanitized_response = (
        sanitize_model_output(
            final_response,
            provider=actual_provider,
        )
    )

    if not sanitized_response:

        raise ProviderExecutionError(
            provider=actual_provider,
            message=(
                "The AI provider returned no usable "
                "final answer after sanitization."
            ),
            category="invalid_provider_response",
            retryable=False,
        )

    # ------------------------------------------------------------------
    # 18. FINAL QUALITY GATE
    # ------------------------------------------------------------------

    final_quality_start = (
        time.perf_counter()
    )

    final_quality_gate = (
        evaluate_response(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            response=sanitized_response,
            prompt_dna=prompt_dna,
            preferences=saved_preferences,
        )
    )

    final_quality_gate_ms = int(
        (
            time.perf_counter()
            - final_quality_start
        )
        * 1000
    )

    if not final_quality_gate.get("passed"):
        raise ProviderExecutionError(
            provider=actual_provider,
            message=(
                "The generated response did not pass the final quality gate."
            ),
            code="invalid_final_response",
            status_code=502,
            category="invalid_final_response",
            retryable=False,
            attempts=execution_result.attempts,
        )

    # ------------------------------------------------------------------
    # 19. TOTAL TIMING
    # ------------------------------------------------------------------

    total_ms = int(
        (
            time.perf_counter()
            - start_total
        )
        * 1000
    )

    # ------------------------------------------------------------------
    # 20. FINAL APIL RESULT
    # ------------------------------------------------------------------

    return {
        "provider": actual_provider,
        "model": actual_model,

        "original_prompt": prompt,

        "optimized_prompt": optimized_prompt,

        "prompt_dna": prompt_dna,

        "processed_prompt": processed_prompt,

        "raw_provider_response": raw_response,

        "raw_response": raw_response,

        "response": sanitized_response,

        "final_response": sanitized_response,

        "provider_call_count": provider_call_count,

        "provider_attempts": (
            execution_result.attempts
        ),

        "verification": verification_result,

        "response_evaluation": response_evaluation,

        "improvement_applied": (
            improvement_applied
        ),

        "improvement_attempted": (
            improvement_attempted
        ),

        "improvement_attempts": (
            improvement_attempts
        ),

        "improvement_needed": (
            response_evaluation.get(
                "improvement_needed"
            )
        ),

        "improvement_error": improvement_error,

        "final_quality_gate": final_quality_gate,

        "timing": {
            "analysis_ms": analysis_ms,
            "prompt_processing_ms": (
                prompt_processing_ms
            ),
            "provider_generation_ms": (
                provider_generation_ms
            ),
            "response_evaluation_ms": (
                response_evaluation_ms
            ),
            "response_improvement_ms": (
                improvement_ms
            ),
            "final_quality_gate_ms": (
                final_quality_gate_ms
            ),
            "total_ms": total_ms,
        },

        "preferences": saved_preferences,

        "analysis": {
            "intent": adaptive_analysis.intent,
            "complexity": adaptive_analysis.complexity,
            "domain": adaptive_analysis.domain,
            "output_type": adaptive_analysis.output_type,
        },
    }