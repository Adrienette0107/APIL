from backend.app.services.adaptive.pipeline import (
    build_adaptive_messages,
    process_response,
)
from backend.app.services.ai.model_router import select_model
from backend.app.services.ai.provider_router import provider_router
from backend.app.services.provider_fallback import (
    ProviderFallbackManager,
)
from backend.app.services.prompt_optimizer import PromptOptimizer
from backend.app.services.output_verifier import OutputVerifier


# Create one fallback manager for the AI service.
fallback_manager = ProviderFallbackManager()

prompt_optimizer = PromptOptimizer()
output_verifier = OutputVerifier()
async def generate_response(
    prompt: str,
    history: list,
    model: str = "auto",
    preferences: dict | None = None,
):

    # --------------------------------------------------
    # 1. OPTIMIZE USER PROMPT
    # --------------------------------------------------

    optimization_result = prompt_optimizer.optimize(
        prompt=prompt,
        preferences=preferences,
    )

    optimized_prompt = optimization_result["optimized_prompt"]

    # --------------------------------------------------
    # 2. RUN EXISTING ADAPTIVE ANALYSIS
    # --------------------------------------------------

    messages, analysis = build_adaptive_messages(
        prompt=optimized_prompt,
        history=history,
        preferences=preferences,
    )

    # --------------------------------------------------
    # 3. SELECT PROVIDER + MODEL
    # --------------------------------------------------

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    # --------------------------------------------------
    # 4. SEND OPTIMIZED PROMPT TO GENAI MODEL
    # --------------------------------------------------

    async def provider_operation(
        provider: str,
        provider_model: str,
    ):
        provider_instance = provider_router.get_provider(
            provider
        )

        return await provider_instance.generate(
            messages=messages,
            model=provider_model,
        )

    execution_result = (
        await fallback_manager.execute_with_fallback(
            primary_provider=provider_name,
            model=selected_model,
            operation_factory=provider_operation,
        )
    )

    # --------------------------------------------------
    # 5. GET RAW MODEL OUTPUT
    # --------------------------------------------------

    raw_response = execution_result.content

    # --------------------------------------------------
    # 6. EXISTING RESPONSE PROCESSING
    # --------------------------------------------------

    processed_response = process_response(
        response=raw_response,
        preferences=preferences,
    )

    # --------------------------------------------------
    # 7. VERIFY MODEL OUTPUT
    # --------------------------------------------------

    verification_result = output_verifier.verify(
        original_prompt=prompt,
        response=processed_response,
        preferences=preferences,
    )

    # --------------------------------------------------
    # 8. RETURN COMPLETE APIL RESULT
    # --------------------------------------------------

    return {
        "provider": execution_result.provider,
        "model": execution_result.model,
        "original_prompt": prompt,
        "optimized_prompt": optimized_prompt,
        "raw_response": raw_response,
        "response": processed_response,
        "verification": verification_result,
    }
    # --------------------------------------------------
    # 1. Build adaptive messages
    # --------------------------------------------------

    messages, analysis = build_adaptive_messages(
        prompt=prompt,
        history=history,
        preferences=preferences,
    )

    # --------------------------------------------------
    # 2. Select provider + model
    # --------------------------------------------------

    provider_name, selected_model = select_model(
        analysis=analysis,
        requested_model=model,
    )

    # --------------------------------------------------
    # 3. Execute through fallback + executor
    # --------------------------------------------------

    async def provider_operation(
        provider: str,
        provider_model: str,
    ):
        provider = provider_router.get_provider(
            provider
        )

        return await provider.generate(
            messages=messages,
            model=provider_model,
        )

    execution_result = (
        await fallback_manager.execute_with_fallback(
            primary_provider=provider_name,
            model=selected_model,
            operation_factory=provider_operation,
        )
    )

    # --------------------------------------------------
    # 4. Process final response
    # --------------------------------------------------

    response = process_response(
        response=execution_result.content,
        preferences=preferences,
    )

    # --------------------------------------------------
    # 5. Return actual provider + model used
    # --------------------------------------------------

    return (
        execution_result.provider,
        execution_result.model,
        response,
    )