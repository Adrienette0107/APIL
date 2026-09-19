from __future__ import annotations

from typing import Any

from backend.app.services.adaptive.pipeline import (
    build_adaptive_messages,
    process_response,
)
from backend.app.services.ai.model_router import select_model
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.output_improver import improve_response
from backend.app.services.output_verifier import OutputVerifier
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.provider_fallback import (
    ProviderFallbackManager,
)
from backend.app.services.response_evaluator import evaluate_response


# ----------------------------------------------------------------------
# Shared service instances
# ----------------------------------------------------------------------

prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()
fallback_manager = ProviderFallbackManager()


# ----------------------------------------------------------------------
# Main AI generation service
# ----------------------------------------------------------------------

async def generate_response(
    prompt: str,
    history: list[dict[str, Any]] | None = None,
    model: str = "auto",
    preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:

    # ------------------------------------------------------------------
    # 0. Normalize inputs
    # ------------------------------------------------------------------

    if history is None:
        history = []

    if preferences is None:
        preferences = {}

    prompt = (
        prompt.strip()
        if isinstance(prompt, str)
        else str(prompt)
    )

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    # ------------------------------------------------------------------
    # 1. PROMPT OPTIMIZATION
    # ------------------------------------------------------------------

    optimization_result = prompt_optimizer.optimize(
        prompt=prompt,
        preferences=preferences,
    )

    optimized_prompt = (
        optimization_result.get(
            "optimized_prompt",
            prompt,
        )
        or prompt
    )

    prompt_dna = (
        optimization_result.get(
            "prompt_dna",
            {},
        )
        or {}
    )

    # ------------------------------------------------------------------
    # 2. ADAPTIVE MESSAGE CONSTRUCTION
    # ------------------------------------------------------------------

    messages, analysis = build_adaptive_messages(
        prompt=optimized_prompt,
        history=history,
        preferences=preferences,
    )

    # ------------------------------------------------------------------
    # 3. PROVIDER + MODEL SELECTION
    # ------------------------------------------------------------------

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    # ------------------------------------------------------------------
    # 4. PROVIDER OPERATION
    #
    # The provider itself is NOT executed directly here.
    #
    # ProviderFallbackManager
    #       ↓
    # ProviderExecutor
    #       ↓
    # Provider adapter
    #
    # This keeps one canonical execution path.
    # ------------------------------------------------------------------

    async def provider_operation(
        provider: str,
        provider_model: str,
    ) -> str:

        provider_instance = provider_router.get_provider(
            provider
        )

        return await provider_instance.generate(
            messages=messages,
            model=provider_model,
        )

    # ------------------------------------------------------------------
    # 5. GENERATE RESPONSE WITH FALLBACK
    # ------------------------------------------------------------------

    execution_result = (
        await fallback_manager.execute_with_fallback(
            primary_provider=provider_name,
            model=selected_model,
            operation_factory=provider_operation,
        )
    )

    raw_response = (
        execution_result.content
        or ""
    )

    # ------------------------------------------------------------------
    # 6. FIRST RESPONSE PROCESSING
    # ------------------------------------------------------------------

    processed_response = process_response(
        response=raw_response,
        preferences=preferences,
    )

    # ------------------------------------------------------------------
    # 7. BASIC OUTPUT VERIFICATION
    # ------------------------------------------------------------------

    verification_result = output_verifier.verify(
        original_prompt=prompt,
        response=processed_response,
        preferences=preferences,
    )

    # ------------------------------------------------------------------
    # 8. QUALITY EVALUATION
    #
    # This is separate from OutputVerifier.
    #
    # OutputVerifier:
    #   catches obvious invalid output
    #
    # ResponseEvaluator:
    #   checks quality, requirements, format,
    #   completeness, language, audience fit, etc.
    # ------------------------------------------------------------------

    evaluation_result = evaluate_response(
        original_prompt=prompt,
        response=processed_response,
        preferences=preferences,
    )

    # ------------------------------------------------------------------
    # 9. CONDITIONAL OUTPUT IMPROVEMENT
    #
    # The model is called again ONLY when the evaluator says
    # improvement is actually required.
    # ------------------------------------------------------------------

    improvement_result = await improve_response(
        original_prompt=prompt,
        optimized_prompt=optimized_prompt,
        response=processed_response,
        evaluation=evaluation_result,
        provider_name=execution_result.provider,
        model_name=execution_result.model,
        prompt_dna=prompt_dna,
    )

    final_response = (
        improvement_result.get(
            "response",
            processed_response,
        )
        or processed_response
    )

    # ------------------------------------------------------------------
    # 10. PROCESS IMPROVED RESPONSE
    # ------------------------------------------------------------------

    if improvement_result.get(
        "improvement_applied",
        False,
    ):

        final_response = process_response(
            response=final_response,
            preferences=preferences,
        )

    # ------------------------------------------------------------------
    # 11. FINAL BASIC VERIFICATION
    # ------------------------------------------------------------------

    final_verification = output_verifier.verify(
        original_prompt=prompt,
        response=final_response,
        preferences=preferences,
    )

    # ------------------------------------------------------------------
    # 12. FINAL QUALITY EVALUATION
    #
    # Re-evaluate only when an improvement was attempted.
    # This prevents unnecessary duplicate evaluation work.
    # ------------------------------------------------------------------

    if improvement_result.get(
        "improvement_applied",
        False,
    ):

        final_evaluation = evaluate_response(
            original_prompt=prompt,
            response=final_response,
            preferences=preferences,
        )

    else:

        final_evaluation = evaluation_result

    # ------------------------------------------------------------------
    # 13. FINAL NORMALIZED RESPONSE
    # ------------------------------------------------------------------

    return {
        "provider": execution_result.provider,
        "model": execution_result.model,
        "attempts": execution_result.attempts,

        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "prompt_dna": prompt_dna,

        "raw_response": raw_response,

        "response": final_response,

        "verification": final_verification,

        "evaluation": final_evaluation,

        "improvement": {
            "attempted": improvement_result.get(
                "improvement_attempted",
                False,
            ),
            "applied": improvement_result.get(
                "improvement_applied",
                False,
            ),
            "failure_reason": improvement_result.get(
                "failure_reason"
            ),
        },
    }

